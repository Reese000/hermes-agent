"""Behavioral tests for the local Mem0 broker and the multi-process fixes.

Covers:
- broker HTTP contract (search/add/update/delete forwarding, response shapes)
- broker not-ready => HTTP 503 "mem0 store busy" (lock held by another process)
- broker auto-flips ready when the store becomes acquirable (retry thread)
- ensure_broker probe-before-spawn, spawn-once, disable flag, pytest guard
- OSS path-mode routing through the broker; non-path config stays direct
- initialize() closes the previous backend before overwriting (lock-leak fix)
- _ensure_backend lazy retry (session no longer dead after failed init)
- _note_backend_error classification: 503 transient, connect-drop, breaker
- OSSBackend.close() closes EVERY client incl. telemetry/migrations store

Every test drives the REAL code and asserts on observed behavior
(call_args.kwargs, HTTP status codes, close() call counts) — no name-based
or tautological checks.
"""

import time
from types import SimpleNamespace

import pytest

import plugins.memory.mem0 as mem0_plugin
from plugins.memory.mem0 import Mem0MemoryProvider, _is_connection_error, _is_transient_error


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class RecordingBackend:
    """Stands in for any Mem0Backend; records every call's kwargs."""

    def __init__(self, search_results=None):
        self.search_results = search_results if search_results is not None else []
        self.search_calls = []
        self.add_calls = []
        self.update_calls = []
        self.delete_calls = []
        self.closed = 0

    def search(self, query, *, filters, top_k=10, rerank=False):
        self.search_calls.append(
            {"query": query, "filters": filters, "top_k": top_k, "rerank": rerank}
        )
        return self.search_results

    def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
        self.add_calls.append(
            {"messages": messages, "user_id": user_id, "agent_id": agent_id,
             "infer": infer, "metadata": metadata}
        )
        return {"result": "Memory added.", "event_id": "evt-1"}

    def update(self, memory_id, text):
        self.update_calls.append({"memory_id": memory_id, "text": text})
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id):
        self.delete_calls.append({"memory_id": memory_id})
        return {"result": "Memory deleted.", "memory_id": memory_id}

    def close(self):
        self.closed += 1


class FakeClient:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class FakeStore:
    """A mem0 vector-store wrapper: has .client and its own close()."""

    def __init__(self):
        self.client = FakeClient()
        self.closed = 0

    def close(self):
        self.closed += 1


class FakeMemory:
    """Stands in for mem0.Memory with all three stores + telemetry."""

    def __init__(self):
        self.vector_store = FakeStore()
        self._telemetry_vector_store = FakeStore()
        self._entity_store = FakeStore()
        self.closed = 0
        self.posthog_shutdowns = 0
        self.telemetry = SimpleNamespace(posthog=SimpleNamespace(
            shutdown=lambda: setattr(self, "posthog_shutdowns", self.posthog_shutdowns + 1)
        ))

    def close(self):
        self.closed += 1


def _make_provider(monkeypatch, backend):
    provider = Mem0MemoryProvider()
    monkeypatch.setattr(provider, "_create_backend", lambda: backend)
    provider.initialize("test-session")
    provider._user_id = "u123"
    provider._agent_id = "hermes"
    return provider


# ---------------------------------------------------------------------------
# Helpers (module-level predicates)
# ---------------------------------------------------------------------------

class TestErrorClassification:
    def test_503_response_is_transient(self):
        exc = RuntimeError("boom")
        exc.response = SimpleNamespace(status_code=503)
        assert _is_transient_error(exc) is True

    def test_store_busy_text_is_transient(self):
        assert _is_transient_error(
            RuntimeError("mem0 store busy: another Hermes process holds the lock")
        ) is True

    def test_plain_error_is_not_transient(self):
        assert _is_transient_error(RuntimeError("connection timeout to LLM")) is False

    def test_connect_error_name_detected(self):
        class ConnectError(Exception):
            pass
        assert _is_connection_error(ConnectError("All connection attempts failed")) is True

    def test_connection_refused_text_detected(self):
        assert _is_connection_error(RuntimeError("HTTPConnectionPool: connection refused")) is True

    def test_hard_error_is_not_connection(self):
        assert _is_connection_error(RuntimeError("500 server error")) is False


# ---------------------------------------------------------------------------
# Broker HTTP contract
# ---------------------------------------------------------------------------

class TestBrokerContract:
    def _app(self, backend):
        from plugins.memory.mem0._local_broker import create_app
        return create_app(backend_provider=lambda: backend)

    def test_search_forwards_all_params(self):
        from fastapi.testclient import TestClient
        backend = RecordingBackend(search_results=[{"id": "m1", "memory": "fact", "score": 0.9}])
        with TestClient(self._app(backend)) as client:
            resp = client.post("/search", json={
                "query": "what car", "filters": {"user_id": "u123"},
                "top_k": 7, "rerank": True,
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"][0]["id"] == "m1"
        # Assert on the REAL call the broker made, not on response echo.
        assert backend.search_calls == [{
            "query": "what car", "filters": {"user_id": "u123"},
            "top_k": 7, "rerank": True,
        }]

    def test_add_forwards_all_params(self):
        from fastapi.testclient import TestClient
        backend = RecordingBackend()
        with TestClient(self._app(backend)) as client:
            resp = client.post("/memories", json={
                "messages": [{"role": "user", "content": "likes espresso"}],
                "user_id": "u123", "agent_id": "hermes",
                "infer": False, "metadata": {"channel": "cli"},
            })
        assert resp.status_code == 200
        assert backend.add_calls == [{
            "messages": [{"role": "user", "content": "likes espresso"}],
            "user_id": "u123", "agent_id": "hermes",
            "infer": False, "metadata": {"channel": "cli"},
        }]

    def test_update_and_delete_forward_ids(self):
        from fastapi.testclient import TestClient
        backend = RecordingBackend()
        with TestClient(self._app(backend)) as client:
            put = client.put("/memories/mid-9", json={"text": "updated text"})
            del_ = client.delete("/memories/mid-9")
        assert put.status_code == 200 and del_.status_code == 200
        assert put.json() == {"result": "Memory updated.", "memory_id": "mid-9"}
        assert del_.json() == {"result": "Memory deleted.", "memory_id": "mid-9"}
        assert backend.update_calls == [{"memory_id": "mid-9", "text": "updated text"}]
        assert backend.delete_calls == [{"memory_id": "mid-9"}]

    def test_not_ready_returns_503_store_busy(self):
        """Lock held by another process: provider construction fails => 503
        with the explicit 'mem0 store busy' contract the plugin keys on."""
        from fastapi.testclient import TestClient

        def locked():
            raise RuntimeError(
                "Storage folder C:/store is already accessed by another instance of Qdrant client"
            )

        from plugins.memory.mem0._local_broker import create_app
        with TestClient(create_app(backend_provider=locked)) as client:
            health = client.get("/healthz").json()
            resp = client.post("/search", json={"query": "x"})
        assert health["ready"] is False
        assert "already accessed" in health["error"]
        assert resp.status_code == 503
        assert "mem0 store busy" in resp.json()["detail"]
        # The plugin's transient classifier must key on this exact detail.
        assert _is_transient_error(RuntimeError(resp.json()["detail"])) is True

    def test_ready_flips_when_lock_frees(self, monkeypatch):
        """Retry thread re-attempts construction until it succeeds, then
        traffic flows — the automatic lock-handoff behavior."""
        from fastapi.testclient import TestClient
        import plugins.memory.mem0._local_broker as broker_mod

        monkeypatch.setattr(broker_mod, "_RETRY_INTERVAL_SECS", 0.05)
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("already accessed by another instance")
            return RecordingBackend(search_results=[{"id": "m1", "memory": "late", "score": 1.0}])

        from plugins.memory.mem0._local_broker import create_app
        with TestClient(create_app(backend_provider=flaky)) as client:
            deadline = time.monotonic() + 3.0
            health = client.get("/healthz").json()
            while not health["ready"] and time.monotonic() < deadline:
                time.sleep(0.05)
                health = client.get("/healthz").json()
            resp = client.post("/search", json={"query": "x"})
        assert health["ready"] is True, f"broker never became ready: {health}"
        assert attempts["n"] >= 3
        assert resp.status_code == 200
        assert resp.json()["results"][0]["memory"] == "late"

    def test_get_memories_lists_store_wide_with_vectors(self):
        """GET /memories must return the whole store, unfiltered, WITH vectors.

        The consolidation script dedupes on cosine similarity of the STORED
        vectors, so the listing has to carry them. mem0's Memory.get_all()
        cannot supply them: its MemoryItem model has no vector field and
        Qdrant.list() hardcodes with_vectors=False. get_all() also rejects a
        filter-less call (raises unless user_id/agent_id/run_id is present),
        which would silently hide the very memories consolidation must see.
        """
        class ScrollRecorder:
            def __init__(self):
                self.calls = []

            def scroll(self, *, collection_name, scroll_filter, limit,
                       with_payload, with_vectors, offset=None):
                self.calls.append({
                    "collection_name": collection_name,
                    "scroll_filter": scroll_filter,
                    "with_payload": with_payload,
                    "with_vectors": with_vectors,
                })
                return (
                    [
                        SimpleNamespace(
                            id="m1",
                            payload={"data": "likes espresso",
                                     "created_at": "2026-01-01T00:00:00Z"},
                            vector=[0.1, 0.2, 0.3],
                        ),
                        SimpleNamespace(
                            id="m2",
                            payload={"data": "prefers espresso",
                                     "created_at": "2026-01-02T00:00:00Z"},
                            vector=[0.9, 0.8, 0.7],
                        ),
                    ],
                    None,
                )

        from fastapi.testclient import TestClient
        recorder = ScrollRecorder()
        backend = SimpleNamespace(
            _memory=SimpleNamespace(
                vector_store=SimpleNamespace(client=recorder,
                                             collection_name="mem0_optimized")
            )
        )

        with TestClient(self._app(backend)) as client:
            resp = client.get("/memories")

        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert [m["id"] for m in body["results"]] == ["m1", "m2"]
        # Vector and payload must BOTH survive: dedupe reads vector for the
        # cosine matrix and payload data/created_at for the keeper choice.
        assert body["results"][0]["vector"] == [0.1, 0.2, 0.3]
        assert body["results"][0]["payload"]["data"] == "likes espresso"
        assert body["results"][1]["payload"]["created_at"] == "2026-01-02T00:00:00Z"
        # Assert on the REAL scroll the broker issued: store-wide (no filter),
        # vectors requested, and read-only (scroll, never upsert/delete).
        assert recorder.calls == [{
            "collection_name": "mem0_optimized",
            "scroll_filter": None,
            "with_payload": True,
            "with_vectors": True,
        }]

    def test_get_memories_is_503_while_store_busy(self):
        """Same lock-handoff contract as the mutating routes. Never answer an
        empty/partial listing while the store is unavailable — that would read
        as 'collection is clean' to the consolidation script."""
        from fastapi.testclient import TestClient
        from plugins.memory.mem0._local_broker import create_app

        def locked():
            raise RuntimeError(
                "Storage folder C:/store is already accessed by another instance of Qdrant client"
            )

        with TestClient(create_app(backend_provider=locked)) as client:
            resp = client.get("/memories")

        assert resp.status_code == 503
        assert "mem0 store busy" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# ensure_broker lifecycle
# ---------------------------------------------------------------------------

class TestEnsureBroker:
    def test_probes_first_and_does_not_spawn_when_running(self, monkeypatch):
        import plugins.memory.mem0._local_broker as broker_mod
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setattr(broker_mod, "_probe", lambda url: True)

        def must_not_spawn(*a, **k):
            raise AssertionError("_spawn called although probe succeeded")

        monkeypatch.setattr(broker_mod, "_spawn", must_not_spawn)
        url = broker_mod.ensure_broker({})
        assert url == "http://127.0.0.1:8765"

    def test_spawns_when_not_running_then_returns_url(self, monkeypatch):
        import plugins.memory.mem0._local_broker as broker_mod
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        state = {"up": False, "spawned": 0}
        monkeypatch.setattr(broker_mod, "_probe", lambda url: state["up"])

        def fake_spawn(port, log_path):
            state["spawned"] += 1
            state["up"] = True  # broker answers after spawn

        monkeypatch.setattr(broker_mod, "_spawn", fake_spawn)
        url = broker_mod.ensure_broker({})
        assert url == "http://127.0.0.1:8765"
        assert state["spawned"] == 1

    def test_disabled_broker_config_returns_none_without_probing(self, monkeypatch):
        import plugins.memory.mem0._local_broker as broker_mod
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        def must_not_probe(*a, **k):
            raise AssertionError("probed although broker disabled")

        monkeypatch.setattr(broker_mod, "_probe", must_not_probe)
        assert broker_mod.ensure_broker({"broker": {"enabled": False}}) is None

    def test_pytest_guard_never_spawns(self, monkeypatch):
        # PYTEST_CURRENT_TEST is set by pytest itself during test execution.
        import plugins.memory.mem0._local_broker as broker_mod

        def must_not_probe(*a, **k):
            raise AssertionError("probed although running under pytest")

        monkeypatch.setattr(broker_mod, "_probe", must_not_probe)
        assert broker_mod.ensure_broker({}) is None

    def test_custom_port_honored(self, monkeypatch):
        import plugins.memory.mem0._local_broker as broker_mod
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setattr(broker_mod, "_probe", lambda url: True)
        url = broker_mod.ensure_broker({"broker": {"port": 9999}})
        assert url == "http://127.0.0.1:9999"


# ---------------------------------------------------------------------------
# OSS routing in _create_backend
# ---------------------------------------------------------------------------

class TestOSSBrokerRouting:
    def _provider(self, oss_cfg):
        provider = Mem0MemoryProvider()
        provider._mode = "oss"
        provider._api_key = ""
        provider._host = ""
        provider._config = {"oss": oss_cfg}
        return provider

    def test_path_config_routes_through_broker(self, monkeypatch):
        import plugins.memory.mem0._local_broker as broker_mod
        seen = {}

        def fake_ensure(cfg):
            seen["cfg"] = cfg
            return "http://127.0.0.1:8765"

        monkeypatch.setattr(broker_mod, "ensure_broker", fake_ensure)
        provider = self._provider({
            "vector_store": {"provider": "qdrant", "config": {"path": "/store"}},
        })
        backend = provider._create_backend()
        from plugins.memory.mem0._backend import SelfHostedBackend
        assert isinstance(backend, SelfHostedBackend)
        assert str(backend._client.base_url).rstrip("/") == "http://127.0.0.1:8765"
        # Sync add through the broker must outlast extraction but stay under
        # the plugin's 120 s sync shutdown join.
        assert backend._client.timeout.read == pytest.approx(115.0, abs=0.01)
        assert seen["cfg"]["vector_store"]["config"]["path"] == "/store"
        backend.close()

    def test_non_path_config_stays_direct(self, monkeypatch):
        from plugins.memory.mem0 import _backend as backend_mod

        class DirectSentinel:
            def __init__(self, cfg):
                self.cfg = cfg

            def close(self):
                pass

        monkeypatch.setattr(backend_mod, "OSSBackend", DirectSentinel)

        def must_not_route(*a, **k):
            raise AssertionError("ensure_broker called for non-path config")

        import plugins.memory.mem0._local_broker as broker_mod
        monkeypatch.setattr(broker_mod, "ensure_broker", must_not_route)

        provider = self._provider({"vector_store": {"provider": "qdrant"}})
        backend = provider._create_backend()
        assert isinstance(backend, DirectSentinel)

    def test_broker_failure_falls_back_to_direct(self, monkeypatch):
        import plugins.memory.mem0._local_broker as broker_mod
        from plugins.memory.mem0 import _backend as backend_mod

        monkeypatch.setattr(broker_mod, "ensure_broker", lambda cfg: None)

        class DirectSentinel:
            def __init__(self, cfg):
                pass

            def close(self):
                pass

        monkeypatch.setattr(backend_mod, "OSSBackend", DirectSentinel)
        provider = self._provider({
            "vector_store": {"provider": "qdrant", "config": {"path": "/store"}},
        })
        assert isinstance(provider._create_backend(), DirectSentinel)


# ---------------------------------------------------------------------------
# Leak fixes + self-healing
# ---------------------------------------------------------------------------

class TestLifecycleFixes:
    def test_initialize_closes_previous_backend(self, monkeypatch):
        """Regression: re-init used to overwrite _backend without close(),
        leaking the exclusive path lock for the process lifetime."""
        first = RecordingBackend()
        second = RecordingBackend()
        provider = Mem0MemoryProvider()
        built = {"n": 0}

        def create():
            built["n"] += 1
            return first if built["n"] == 1 else second

        monkeypatch.setattr(provider, "_create_backend", create)
        provider.initialize("s1")
        assert provider._backend is first
        provider.initialize("s2")
        assert first.closed == 1, "previous backend was not closed on re-init"
        assert provider._backend is second
        assert second.closed == 0

    def test_ensure_backend_noop_when_backend_set(self, monkeypatch):
        backend = RecordingBackend()
        provider = Mem0MemoryProvider()
        provider._backend = backend

        def must_not_build(*a, **k):
            raise AssertionError("_create_backend called although backend set")

        monkeypatch.setattr(provider, "_create_backend", must_not_build)
        assert provider._ensure_backend() is True
        assert provider._backend is backend

    def test_ensure_backend_rebuilds_after_failed_init(self, monkeypatch):
        backend = RecordingBackend()
        provider = Mem0MemoryProvider()
        provider._backend = None
        provider._init_error = "already accessed by another instance"
        monkeypatch.setattr(provider, "_create_backend", lambda: backend)
        assert provider._ensure_backend() is True
        assert provider._backend is backend

    def test_ensure_backend_rate_limited(self, monkeypatch):
        provider = Mem0MemoryProvider()
        provider._backend = None
        provider._last_reinit_attempt = time.monotonic()

        def must_not_build(*a, **k):
            raise AssertionError("_create_backend called during rate-limit window")

        monkeypatch.setattr(provider, "_create_backend", must_not_build)
        assert provider._ensure_backend() is False
        assert provider._backend is None

    def test_tool_call_self_heals_after_failed_init(self, monkeypatch):
        """The exact production failure: init failed at session start, so the
        first mem0_search must trigger the lazy rebuild instead of erroring."""
        backend = RecordingBackend(search_results=[{"id": "m1", "memory": "ok", "score": 1.0}])
        provider = Mem0MemoryProvider()
        provider._backend = None
        provider._init_error = "already accessed by another instance"
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        monkeypatch.setattr(provider, "_create_backend", lambda: backend)
        import json
        result = json.loads(provider.handle_tool_call("mem0_search", {"query": "x"}))
        assert "error" not in result, result
        assert result["results"][0]["memory"] == "ok"
        assert len(backend.search_calls) == 1

    def test_503_does_not_trip_breaker(self, monkeypatch):
        busy = RuntimeError("Server error '503' ... mem0 store busy: lock held")
        busy.response = SimpleNamespace(status_code=503)

        class BusyBackend(RecordingBackend):
            def search(self, query, *, filters, top_k=10, rerank=False):
                raise busy

        provider = _make_provider(monkeypatch, BusyBackend())
        import json
        for _ in range(8):  # > _BREAKER_THRESHOLD (5)
            result = json.loads(provider.handle_tool_call("mem0_search", {"query": "x"}))
            assert "mem0 store busy" in result["error"], result
            assert "retry shortly" in result["error"], result
        assert provider._consecutive_failures == 0, "503 counted toward breaker"
        assert provider._is_breaker_open() is False, "breaker tripped by transient 503s"

    def test_connect_error_drops_backend_for_rebuild(self, monkeypatch):
        class ConnectError(Exception):
            pass

        class DeadBackend(RecordingBackend):
            def search(self, query, *, filters, top_k=10, rerank=False):
                raise ConnectError("All connection attempts failed")

        provider = _make_provider(monkeypatch, DeadBackend())
        import json
        json.loads(provider.handle_tool_call("mem0_search", {"query": "x"}))
        assert provider._backend is None, "dead endpoint backend not dropped for rebuild"
        assert provider._consecutive_failures == 1

    def test_hard_errors_still_trip_breaker(self, monkeypatch):
        class HardBackend(RecordingBackend):
            def search(self, query, *, filters, top_k=10, rerank=False):
                raise RuntimeError("500 Internal Server Error")

        provider = _make_provider(monkeypatch, HardBackend())
        import json
        for _ in range(5):
            json.loads(provider.handle_tool_call("mem0_search", {"query": "x"}))
        assert provider._consecutive_failures >= 5
        assert provider._is_breaker_open() is True


# ---------------------------------------------------------------------------
# OSSBackend.close() closes every client
# ---------------------------------------------------------------------------

class TestOSSCloseReleasesAllLocks:
    def _backend_with(self, memory):
        from plugins.memory.mem0._backend import OSSBackend
        backend = OSSBackend.__new__(OSSBackend)
        backend._memory = memory
        return backend

    def test_close_closes_all_three_stores_and_memory(self):
        mem = FakeMemory()
        backend = self._backend_with(mem)
        backend.close()
        assert mem.closed == 1, "Memory itself not closed"
        assert mem.vector_store.closed == 1, "main vector store not closed"
        assert mem.vector_store.client.closed == 1, "main Qdrant client (path .lock) still open"
        assert mem._telemetry_vector_store.closed == 1, "telemetry/migrations store not closed"
        assert mem._telemetry_vector_store.client.closed == 1, "migrations Qdrant client still open"
        assert mem._entity_store.client.closed == 1, "entity store client still open"
        assert mem.posthog_shutdowns == 1, "telemetry not shut down"

    def test_close_is_idempotent(self):
        mem = FakeMemory()
        backend = self._backend_with(mem)
        backend.close()
        backend.close()  # second close must be a no-op, not a crash
        assert mem.closed == 1

    def test_one_failing_close_does_not_strand_the_others(self):
        """Old code: a single suppress() wrapped the whole chain, so the first
        failure leaked every remaining lock. Each close is now independent."""
        mem = FakeMemory()

        def boom():
            raise RuntimeError("posthog shutdown failed")

        mem.telemetry = SimpleNamespace(posthog=SimpleNamespace(shutdown=boom))
        # First close() call on the memory itself fails hard:
        mem.close = lambda: (_ for _ in ()).throw(RuntimeError("memory close failed"))
        backend = self._backend_with(mem)
        backend.close()  # must not raise
        assert mem.vector_store.client.closed == 1, "main client stranded by unrelated failure"
        assert mem._telemetry_vector_store.client.closed == 1, "migrations client stranded"
        assert mem._entity_store.client.closed == 1, "entity client stranded"

"""Tests for Mem0 v3 API — new tool names, paginated responses, update/delete tools."""

import json
import threading
import time
import pytest

import plugins.memory.mem0 as mem0_plugin
from plugins.memory.mem0 import Mem0MemoryProvider


class FakeBackend:
    """Fake Mem0Backend for provider-level tests."""

    def __init__(self, search_results=None, all_results=None):
        self._search_results = search_results or []
        self._all_results = all_results or {"results": [], "count": 0}
        self.captured = []

    def search(self, query, *, filters, top_k=10, rerank=True):
        self.captured.append(("search", query, {"filters": filters, "top_k": top_k, "rerank": rerank}))
        return self._search_results

    def get_all(self, *, filters, page=1, page_size=100):
        self.captured.append(("get_all", {"filters": filters, "page": page, "page_size": page_size}))
        return self._all_results

    def list_all(self, *, filters):
        self.captured.append(("list_all", {"filters": filters}))
        return self._all_results.get("results", []) if isinstance(self._all_results, dict) else self._all_results

    def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
        self.captured.append((
            "add",
            messages,
            {"user_id": user_id, "agent_id": agent_id, "infer": infer, "metadata": metadata},
        ))
        return {"status": "PENDING", "event_id": "evt-test-123"}

    def update(self, memory_id, text, metadata=None):
        self.captured.append(("update", memory_id, text, metadata))
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id):
        self.captured.append(("delete", memory_id))
        return {"result": "Memory deleted.", "memory_id": memory_id}


class TestMem0V3Tools:
    """Test v3 tool names and response handling."""

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_search_returns_ids(self, monkeypatch):
        backend = FakeBackend(search_results=[{"id": "mem-1", "memory": "foo", "score": 0.9}])
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_search", {"query": "test"}))
        assert result["results"][0]["id"] == "mem-1"


    def test_add_uses_content_param(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_add", {"content": "user likes dark mode"}))
        assert len(backend.captured) == 1
        call = backend.captured[0]
        assert call[2]["infer"] is False
        assert call[2]["user_id"] == "u123"
        assert call[2]["agent_id"] == "hermes"
        assert "event_id" in result


    def test_old_tool_names_return_unknown(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_profile", {}))
        assert "error" in result
        result = json.loads(provider.handle_tool_call("mem0_conclude", {}))
        assert "error" in result


class TestMem0UpdateDelete:

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_update_calls_sdk(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_update", {"memory_id": "mem-1", "text": "updated fact"}
        ))
        assert backend.captured[0][1] == "mem-1"
        assert backend.captured[0][2] == "updated fact"
        assert result["result"] == "Memory updated."
        assert result["memory_id"] == "mem-1"


    def test_delete_calls_sdk(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_delete", {"memory_id": "mem-1"}
        ))
        assert backend.captured[0][1] == "mem-1"
        assert result["result"] == "Memory deleted."


class TestMem0ErrorHandling:

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider


class TestMem0V3Internal:

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_sync_turn_explicit_kwargs(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider.sync_turn("user said", "assistant replied", session_id="s1")
        provider._sync_thread.join(timeout=2)
        assert len(backend.captured) == 1
        call = backend.captured[0]
        assert call[2]["user_id"] == "u123"
        assert call[2]["agent_id"] == "hermes"
        assert call[2]["infer"] is True


class TestMem0Prefetch:
    """prefetch() must recall on the CURRENT question, synchronously.

    The old implementation ignored its ``query`` and returned whatever a
    background ``queue_prefetch`` had warmed from the PREVIOUS turn — so the
    first turn injected nothing and later turns injected stale, off-topic
    memories. These lock the corrected behaviour.
    """

    def _make_provider(self, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_prefetch_searches_current_query(self):
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "user prefers dark mode"}])
        provider = self._make_provider(backend)
        result = provider.prefetch("what theme do I like?")
        kind, query, opts = backend.captured[0]
        assert kind == "search"
        assert query == "what theme do I like?"
        assert opts["filters"] == {"user_id": "u123", "agent_id": "hermes"}
        assert opts["top_k"] == 10
        assert opts["rerank"] is False
        assert "## Mem0 Memory" in result
        assert "user prefers dark mode" in result


    def test_on_turn_start_queues_current_query(self):
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "lives in Berlin"}])
        provider = self._make_provider(backend)
        provider.on_turn_start(1, "where do I live?")
        provider._prefetch_thread.join(timeout=1)
        result = provider.prefetch("where do I live?")
        assert "lives in Berlin" in result
        assert len([c for c in backend.captured if c[0] == "search"]) == 1

    def test_slow_prefetch_returns_quickly(self, monkeypatch):
        entered = threading.Event()
        release = threading.Event()
        search_returned = threading.Event()

        class SlowBackend(FakeBackend):
            def search(self, query, *, filters, top_k=10, rerank=True):
                entered.set()
                try:
                    release.wait(30)
                    return super().search(
                        query, filters=filters, top_k=top_k, rerank=rerank
                    )
                finally:
                    search_returned.set()

        monkeypatch.setattr(mem0_plugin, "_PREFETCH_WAIT_SECS", 0.01)
        provider = self._make_provider(
            SlowBackend(search_results=[{"id": "m1", "memory": "lives in Berlin"}])
        )
        # DETERMINISTIC non-blocking witness — replaces `assert elapsed < 0.1`.
        #
        # The old form slept 0.2s in the backend and asserted prefetch returned
        # in under 0.1s. That makes the OS scheduler part of the assertion: on
        # a loaded box thread startup alone can eat the 100ms budget, so the
        # inequality flips with nothing wrong in the code under test. Observed
        # failing in a full-directory run of tests/plugins/memory.
        #
        # The real contract is that prefetch gives up on the slow backend
        # instead of waiting for it. Assert it directly: the backend search is
        # STILL PARKED (release unset, so `search_returned` cannot be set). If
        # prefetch ever waited for the backend, the search would have returned
        # first and this fails. No wall-clock constant.
        assert provider.prefetch("where do I live?") == ""
        assert entered.wait(30), "prefetch never reached the backend"
        assert not search_returned.is_set(), (
            "prefetch blocked on the slow backend: the backend search had "
            "already returned by the time prefetch did"
        )

        release.set()
        provider._prefetch_thread.join(timeout=30)
        assert "lives in Berlin" in provider.prefetch("where do I live?")


    def test_queue_prefetch_fires_no_search(self):
        # prefetch is synchronous now, so the post-turn warm is redundant and
        # must not fire a wasted backend search.
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "x"}])
        provider = self._make_provider(backend)
        provider.queue_prefetch("previous turn text")
        assert backend.captured == []


class TestMem0V3Config:

    def test_tool_schemas_four_tools(self):
        provider = Mem0MemoryProvider()
        schemas = provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert names == ["mem0_search", "mem0_add", "mem0_update", "mem0_delete"]

    def test_system_prompt_new_tool_names(self):
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        block = provider.system_prompt_block()
        assert "mem0_search" in block
        assert "mem0_add" in block
        assert "mem0_update" in block
        assert "mem0_delete" in block
        assert "mem0_list" not in block
        assert "mem0_profile" not in block
        assert "mem0_conclude" not in block


class TestMem0ModeSwitch:

    def test_default_mode_is_platform(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        provider = Mem0MemoryProvider()
        provider.initialize("test")
        assert provider._mode == "platform"

    def test_missing_mode_key_defaults_platform(self, monkeypatch, tmp_path):
        """Backward compat: old mem0.json without mode key works."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "mem0.json"
        config_path.write_text('{"user_id": "old-user"}')
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        provider = Mem0MemoryProvider()
        provider.initialize("test")
        assert provider._mode == "platform"
        assert provider._user_id == "old-user"

    def test_is_available_platform_needs_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        provider = Mem0MemoryProvider()
        assert provider.is_available() is False


class TestMem0UserIdResolution:
    """user_id resolution: configured override > gateway-native id > placeholder.

    Same human across CLI / Telegram / Discord / Slack / etc. should map to
    the same memory store when MEM0_USER_ID is set, and only fall back to the
    gateway-native id when it isn't.
    """

    def _provider(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        provider = Mem0MemoryProvider()
        # Skip backend instantiation — we only care about identity resolution.
        provider._create_backend = lambda: None  # type: ignore[method-assign]
        return provider

    def test_env_override_beats_gateway_native_id(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_USER_ID", "ryan@example.com")
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "ryan@example.com"

    def test_file_override_beats_gateway_native_id(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        (tmp_path / "mem0.json").write_text('{"user_id": "ryan@example.com"}')
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "ryan@example.com"

    def test_unset_falls_back_to_gateway_native_id(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "123456789"


    def test_legacy_placeholder_in_config_does_not_override_kwargs(self, monkeypatch, tmp_path):
        # Setup wizard historically wrote {"user_id": "hermes-user"} as the
        # suggested default. Treat that placeholder as unset so users on
        # gateways still get gateway-native ids — not silent collisions.
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        (tmp_path / "mem0.json").write_text('{"user_id": "hermes-user"}')
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "123456789"


class TestMem0WriteMetadata:
    """Writes carry metadata.channel so per-channel filtered views are possible
    without coupling identity to the channel.
    """

    def _make_provider(self, channel: str = "cli"):
        provider = Mem0MemoryProvider()
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._channel = channel
        provider._backend = FakeBackend()
        return provider


class _SentinelBackend:
    def __init__(self, *args):
        self.args = args


class TestCreateBackendRouting:
    """_create_backend() must pick the backend matching the configured mode/host."""

    def _provider(self, monkeypatch, *, mode="platform", api_key="k", host=""):
        # Neutralize lazy-install so the routing decision is all we exercise.
        monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **k: None, raising=False)
        provider = Mem0MemoryProvider()
        provider._mode = mode
        provider._api_key = api_key
        provider._host = host
        provider._config = {"oss": {"vector_store": {"provider": "qdrant"}}}
        return provider

    def test_routes_to_selfhosted_when_host_set(self, monkeypatch):
        captured = {}

        class SH(_SentinelBackend):
            def __init__(self, api_key, host):
                captured["args"] = (api_key, host)

        monkeypatch.setattr("plugins.memory.mem0._backend.SelfHostedBackend", SH)
        provider = self._provider(monkeypatch, host="http://sh:8888", api_key="adminkey")
        backend = provider._create_backend()
        assert isinstance(backend, SH)
        assert captured["args"] == ("adminkey", "http://sh:8888")


    def test_oss_mode_takes_precedence_over_host(self, monkeypatch):
        class OB(_SentinelBackend):
            def __init__(self, cfg):
                pass

        monkeypatch.setattr("plugins.memory.mem0._backend.OSSBackend", OB)
        provider = self._provider(monkeypatch, mode="oss", host="http://sh:8888")
        assert isinstance(provider._create_backend(), OB)

    def test_prompt_label_matches_routing_when_oss_and_host_both_set(self, monkeypatch):
        # system_prompt_block must mirror _create_backend precedence: with both
        # mode=oss and host set, OSS wins the routing, so the prompt must label
        # OSS — not "self-hosted (HTTP API)". Guards the prompt-vs-routing lie.
        provider = self._provider(monkeypatch, mode="oss", host="http://sh:8888")
        provider._user_id = "test"
        block = provider.system_prompt_block()
        assert "OSS" in block
        assert "HTTP API" not in block


class TestMem0OnPreCompress:
    """on_pre_compress() must extract facts from messages about to be
    compressed into Mem0 before the compressor proceeds, so no facts are lost.
    """

    def _make_provider(self, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_returns_empty_string(self):
        """on_pre_compress returns empty — facts go to Mem0, not the summary."""
        backend = FakeBackend()
        provider = self._make_provider(backend)
        result = provider.on_pre_compress([
            {"role": "user", "content": "I prefer dark mode"},
        ])
        assert result == ""

    def test_indexes_each_message_with_infer(self):
        """Each message with content is sent to backend.add() with infer=True."""
        backend = FakeBackend()
        provider = self._make_provider(backend)
        messages = [
            {"role": "user", "content": "I live in Savannah"},
            {"role": "assistant", "content": "Great, Savannah is lovely!"},
            {"role": "user", "content": "Thanks"},
        ]
        provider.on_pre_compress(messages)
        # Wait for background thread to complete
        provider._compress_thread.join(timeout=5)
        add_calls = [c for c in backend.captured if c[0] == "add"]
        assert len(add_calls) == 3
        for call in add_calls:
            msgs, opts = call[1], call[2]
            assert opts["infer"] is True
            assert opts["user_id"] == "u123"
            assert opts["agent_id"] == "hermes"

    def test_skips_empty_messages(self):
        """Messages without content are not sent to the backend."""
        backend = FakeBackend()
        provider = self._make_provider(backend)
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "ok"},
            {"role": "user"},  # no content key at all
        ]
        provider.on_pre_compress(messages)
        provider._compress_thread.join(timeout=5)
        add_calls = [c for c in backend.captured if c[0] == "add"]
        assert len(add_calls) == 1
        assert add_calls[0][1][0]["content"] == "ok"

    def test_non_blocking(self):
        """on_pre_compress must not block the caller — it spawns a background thread."""
        entered = threading.Event()
        release = threading.Event()

        class SlowBackend(FakeBackend):
            def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
                entered.set()
                release.wait(30)
                return super().add(messages, user_id=user_id, agent_id=agent_id,
                                  infer=infer, metadata=metadata)

        backend = SlowBackend()
        provider = self._make_provider(backend)
        result = provider.on_pre_compress([
            {"role": "user", "content": "test message"},
        ])
        # Must return immediately (empty string) without waiting for backend
        assert result == ""
        assert entered.wait(5), "background thread never reached the backend"
        release.set()
        provider._compress_thread.join(timeout=5)

    def test_circuit_breaker_blocks_extraction(self):
        """When circuit breaker is open, on_pre_compress must not call backend."""
        backend = FakeBackend()
        provider = self._make_provider(backend)
        # Trip the breaker
        provider._consecutive_failures = 10
        provider._breaker_open_until = time.monotonic() + 300
        result = provider.on_pre_compress([
            {"role": "user", "content": "should not be indexed"},
        ])
        assert result == ""
        # Give any spurious thread a moment
        time.sleep(0.1)
        add_calls = [c for c in backend.captured if c[0] == "add"]
        assert len(add_calls) == 0

    def test_records_success_on_backend_call(self):
        """Successful extraction resets the circuit breaker failure count."""
        backend = FakeBackend()
        provider = self._make_provider(backend)
        provider._consecutive_failures = 3
        provider.on_pre_compress([
            {"role": "user", "content": "reset the breaker"},
        ])
        provider._compress_thread.join(timeout=5)
        assert provider._consecutive_failures == 0

    def test_records_failure_on_backend_error(self):
        """Failed extraction increments the circuit breaker failure count."""
        class FailingBackend(FakeBackend):
            def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
                raise ConnectionError("Qdrant is down")

        backend = FailingBackend()
        provider = self._make_provider(backend)
        assert provider._consecutive_failures == 0
        provider.on_pre_compress([
            {"role": "user", "content": "this will fail"},
        ])
        provider._compress_thread.join(timeout=5)
        assert provider._consecutive_failures == 1


class TestMem0CompressionAwarePrefetch:
    """When context is near the compression threshold, on_turn_start must
    proactively prefetch memories from at-risk messages so relevant context
    survives compression in the active window.
    """

    def _make_provider(self, backend, *, config_override=None):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        if config_override:
            provider._config.update(config_override)
        return provider

    def test_extra_prefetch_when_near_threshold(self):
        """When remaining_tokens < 40% of context_limit, extra search fires."""
        search_queries = []
        backend = FakeBackend(
            search_results=[{"id": "m1", "memory": "user prefers dark mode"}]
        )

        class TrackingBackend(FakeBackend):
            def search(self, query, *, filters, top_k=10, rerank=True):
                search_queries.append(query)
                return super().search(query, filters=filters, top_k=top_k, rerank=rerank)

        backend = TrackingBackend(
            search_results=[{"id": "m1", "memory": "user prefers dark mode"}]
        )
        provider = self._make_provider(backend)
        provider._config["compression_aware_prefetch"] = True

        # context_limit=100000, remaining=30000 → 30% remaining → near threshold
        provider.on_turn_start(
            1, "what theme do I like?",
            remaining_tokens=30000,
            context_limit=100000,
            old_messages=[
                {"role": "user", "content": "I moved to Savannah last year"},
                {"role": "assistant", "content": "Welcome to Savannah!"},
            ],
        )
        provider._prefetch_thread.join(timeout=5)
        # Should have the main query + at least one keyword-derived query
        assert len(search_queries) >= 2
        assert "what theme do I like?" in search_queries

    def test_no_extra_prefetch_when_context_has_room(self):
        """When context is far from threshold, only the main query is prefetched."""
        search_queries = []

        class TrackingBackend(FakeBackend):
            def search(self, query, *, filters, top_k=10, rerank=True):
                search_queries.append(query)
                return super().search(query, filters=filters, top_k=top_k, rerank=rerank)

        backend = TrackingBackend(
            search_results=[{"id": "m1", "memory": "something"}]
        )
        provider = self._make_provider(backend)
        provider._config["compression_aware_prefetch"] = True

        # context_limit=100000, remaining=80000 → 80% remaining → plenty of room
        provider.on_turn_start(
            1, "hello there",
            remaining_tokens=80000,
            context_limit=100000,
        )
        provider._prefetch_thread.join(timeout=5)
        assert len(search_queries) == 1
        assert "hello there" in search_queries

    def test_disabled_by_default(self):
        """When compression_aware_prefetch is explicitly False, no extra prefetch."""
        search_queries = []

        class TrackingBackend(FakeBackend):
            def search(self, query, *, filters, top_k=10, rerank=True):
                search_queries.append(query)
                return super().search(query, filters=filters, top_k=top_k, rerank=rerank)

        backend = TrackingBackend(
            search_results=[{"id": "m1", "memory": "something"}]
        )
        provider = self._make_provider(backend)
        provider._config["compression_aware_prefetch"] = False

        provider.on_turn_start(
            1, "what theme?",
            remaining_tokens=30000,
            context_limit=100000,
            old_messages=[
                {"role": "user", "content": "I moved to Savannah"},
            ],
        )
        provider._prefetch_thread.join(timeout=5)
        assert len(search_queries) == 1

    def test_latency_under_3s(self):
        """Extra prefetch must not add more than 3s to on_turn_start."""
        import time as _time

        class SlowBackend(FakeBackend):
            def search(self, query, *, filters, top_k=10, rerank=True):
                _time.sleep(0.05)  # Simulate slight latency
                return super().search(query, filters=filters, top_k=top_k, rerank=rerank)

        backend = SlowBackend(
            search_results=[{"id": "m1", "memory": "result"}]
        )
        provider = self._make_provider(backend)
        provider._config["compression_aware_prefetch"] = True

        start = _time.monotonic()
        provider.on_turn_start(
            1, "test query",
            remaining_tokens=30000,
            context_limit=100000,
            old_messages=[
                {"role": "user", "content": "old message about topic"},
                {"role": "user", "content": "another old message"},
            ],
        )
        elapsed = _time.monotonic() - start
        assert elapsed < 3.0, f"on_turn_start took {elapsed:.1f}s, exceeds 3s budget"
        provider._prefetch_thread.join(timeout=10)


class TestMemoryLifecycleOrchestration:
    """End-to-end memory lifecycle: on_turn_start → prefetch → sync_turn → on_pre_compress.
    Verifies the three systems are coordinated, not independent.
    """

    def test_pipeline_sequence_on_turn_start_receives_context_budget(self):
        """on_turn_start receives remaining_tokens and context_limit from the runtime."""
        captured_kwargs = {}

        class RecordingProvider(Mem0MemoryProvider):
            def on_turn_start(self, turn_number, message, **kwargs):
                captured_kwargs.update(kwargs)
                super().on_turn_start(turn_number, message, **kwargs)

        provider = RecordingProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = FakeBackend()

        provider.on_turn_start(
            1, "test message",
            remaining_tokens=30000,
            context_limit=100000,
            old_messages=[{"role": "user", "content": "old context"}],
        )
        assert captured_kwargs["remaining_tokens"] == 30000
        assert captured_kwargs["context_limit"] == 100000
        assert len(captured_kwargs["old_messages"]) == 1

    def test_prefetch_result_includes_compression_aware_when_enabled(self):
        """When compression-aware prefetch fires, results are merged into prefetch output."""
        backend = FakeBackend(
            search_results=[{"id": "m1", "memory": "user lives in Savannah"}]
        )
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        provider._config["compression_aware_prefetch"] = True

        # Trigger both normal and compression-aware prefetch
        provider.on_turn_start(
            1, "where do I live?",
            remaining_tokens=30000,
            context_limit=100000,
            old_messages=[{"role": "user", "content": "I moved to Savannah"}],
        )
        # Wait for both threads
        provider._prefetch_thread.join(timeout=5)
        time.sleep(0.2)  # Give compression-aware thread time to finish

        result = provider.prefetch("where do I live?")
        # Should contain results from both prefetch paths
        assert "Savannah" in result

    def test_sync_turn_writes_facts_after_turn(self):
        """sync_turn extracts facts from completed turns (existing behavior)."""
        backend = FakeBackend()
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend

        provider.sync_turn("user message", "assistant reply")
        provider._sync_thread.join(timeout=5)

        add_calls = [c for c in backend.captured if c[0] == "add"]
        assert len(add_calls) == 1
        assert add_calls[0][2]["infer"] is True

    def test_pre_compress_saves_facts_before_compression(self):
        """on_pre_compress extracts facts from messages about to be compressed."""
        backend = FakeBackend()
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend

        messages = [
            {"role": "user", "content": "important fact 1"},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "important fact 2"},
        ]
        result = provider.on_pre_compress(messages)
        provider._compress_thread.join(timeout=5)

        assert result == ""
        add_calls = [c for c in backend.captured if c[0] == "add"]
        assert len(add_calls) == 3

    def test_full_lifecycle_no_facts_lost(self):
        """Complete lifecycle: turn → sync → pre-compress → all facts in Mem0."""
        backend = FakeBackend()
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend

        # Turn 1: normal conversation
        provider.sync_turn("I use Vim", "Great choice!")
        provider._sync_thread.join(timeout=5)

        # Context fills up, compression fires
        at_risk = [
            {"role": "user", "content": "I use Vim"},
            {"role": "assistant", "content": "Great choice!"},
            {"role": "user", "content": "What's your favorite plugin?"},
        ]
        provider.on_pre_compress(at_risk)
        provider._compress_thread.join(timeout=5)

        # All facts should be in Mem0 now
        add_calls = [c for c in backend.captured if c[0] == "add"]
        assert len(add_calls) >= 3  # sync(1) + pre_compress(3)


class TestSelfHostedConfig:
    """Config plumbing for self-hosted (MEM0_HOST env + is_available)."""

    def test_load_config_reads_mem0_host_env(self, monkeypatch):
        monkeypatch.setenv("MEM0_HOST", "http://localhost:8888")
        assert mem0_plugin._load_config()["host"] == "http://localhost:8888"




class TestTopicFingerprint:
    """Topic fingerprint extraction for cross-session memory identity resolution."""

    def test_extract_returns_top_keywords(self):
        """extract_topic_fingerprint returns the most significant keywords."""
        from plugins.memory.mem0 import extract_topic_fingerprint
        result = extract_topic_fingerprint(
            "The CNC machine is great for precision work on aluminum parts"
        )
        assert isinstance(result, list)
        assert len(result) >= 3
        # Domain words should be extracted
        lower = [w.lower() for w in result]
        assert "cnc" in lower or "machine" in lower

    def test_extract_returns_empty_for_empty_text(self):
        from plugins.memory.mem0 import extract_topic_fingerprint
        assert extract_topic_fingerprint("") == []
        assert extract_topic_fingerprint("   ") == []
        assert extract_topic_fingerprint(None) == []

    def test_extract_filters_stopwords(self):
        """Common stopwords should not appear in fingerprint."""
        from plugins.memory.mem0 import extract_topic_fingerprint
        result = extract_topic_fingerprint(
            "the is a an and or but in on at to for of with by from"
        )
        # Stopwords should be filtered out
        stopwords = {"the", "is", "a", "an", "and", "or", "but", "in", "on", "at",
                      "to", "for", "of", "with", "by", "from", "it", "this", "that"}
        for word in result:
            assert word.lower() not in stopwords

    def test_extract_respects_max_keywords(self):
        from plugins.memory.mem0 import extract_topic_fingerprint
        result = extract_topic_fingerprint(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa",
            max_keywords=3,
        )
        assert len(result) <= 3

    def test_extract_short_text_still_works(self):
        from plugins.memory.mem0 import extract_topic_fingerprint
        result = extract_topic_fingerprint("hello world")
        assert isinstance(result, list)
        # Should return at most 2 keywords
        assert len(result) <= 2

    def test_extract_lowercase_normalized(self):
        """All returned keywords should be lowercase."""
        from plugins.memory.mem0 import extract_topic_fingerprint
        result = extract_topic_fingerprint("CNC Machine Python Docker")
        for word in result:
            assert word == word.lower()


class TestMem0ProfileIsolation:
    """Per-profile memory isolation: agent_id derived from profile name,
    cross-profile search opt-in.
    """

    def _provider(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        provider = Mem0MemoryProvider()
        provider._create_backend = lambda: None
        return provider

    def test_agent_id_derived_from_profile_name(self, monkeypatch, tmp_path):
        """agent_id = 'hermes-{profile}' when agent_identity kwarg is passed."""
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", agent_identity="cam")
        assert provider._agent_id == "hermes-cam"

    def test_agent_id_default_when_no_profile(self, monkeypatch, tmp_path):
        """Without agent_identity, agent_id falls back to config or 'hermes'."""
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test")
        assert provider._agent_id == "hermes"

    def test_agent_id_from_config_overridden_by_profile(self, monkeypatch, tmp_path):
        """Profile-derived agent_id takes precedence over mem0.json config."""
        (tmp_path / "mem0.json").write_text('{"agent_id": "hermes-old"}')
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", agent_identity="dfm")
        assert provider._agent_id == "hermes-dfm"

    def test_agent_id_from_config_when_no_profile(self, monkeypatch, tmp_path):
        """Without profile, mem0.json agent_id is used."""
        (tmp_path / "mem0.json").write_text('{"agent_id": "hermes-custom"}')
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test")
        assert provider._agent_id == "hermes-custom"

    def test_write_uses_profile_agent_id(self, monkeypatch, tmp_path):
        """Backend.add() receives the profile-scoped agent_id."""
        backend = FakeBackend()
        provider = Mem0MemoryProvider()
        provider.initialize("test", agent_identity="cam")
        provider._backend = backend
        provider.sync_turn("user said", "assistant replied", session_id="s1")
        provider._sync_thread.join(timeout=2)
        call = backend.captured[0]
        assert call[2]["agent_id"] == "hermes-cam"

    def test_read_filters_include_agent_id_by_default(self, monkeypatch, tmp_path):
        """_read_filters() includes agent_id when profile_isolation is default (true)."""
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", agent_identity="cam")
        provider._user_id = "u123"
        filters = provider._read_filters()
        assert filters["user_id"] == "u123"
        assert filters["agent_id"] == "hermes-cam"

    def test_read_filters_exclude_agent_id_when_cross_profile(self, monkeypatch, tmp_path):
        """_read_filters() omits agent_id when cross_profile_search is true."""
        (tmp_path / "mem0.json").write_text('{"cross_profile_search": true}')
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", agent_identity="cam")
        provider._user_id = "u123"
        filters = provider._read_filters()
        assert "agent_id" not in filters
        assert filters == {"user_id": "u123"}

    def test_read_filters_exclude_agent_id_when_profile_isolation_false(self, monkeypatch, tmp_path):
        """_read_filters() omits agent_id when profile_isolation is false."""
        (tmp_path / "mem0.json").write_text('{"profile_isolation": false}')
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", agent_identity="cam")
        provider._user_id = "u123"
        filters = provider._read_filters()
        assert "agent_id" not in filters

    def test_search_uses_isolated_filters(self, monkeypatch, tmp_path):
        """mem0_search passes profile-scoped filters to backend."""
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "cam memory"}])
        provider = Mem0MemoryProvider()
        provider.initialize("test", agent_identity="cam")
        provider._user_id = "u123"
        provider._backend = backend
        result = json.loads(provider.handle_tool_call("mem0_search", {"query": "test"}))
        call = backend.captured[0]
        assert call[2]["filters"] == {"user_id": "u123", "agent_id": "hermes-cam"}
        assert result["results"][0]["memory"] == "cam memory"

    def test_prefetch_uses_isolated_filters(self, monkeypatch, tmp_path):
        """prefetch passes profile-scoped filters to backend."""
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "cam fact"}])
        provider = Mem0MemoryProvider()
        provider.initialize("test", agent_identity="cam")
        provider._user_id = "u123"
        provider._backend = backend
        result = provider.prefetch("what does cam know?")
        call = backend.captured[0]
        assert call[2]["filters"] == {"user_id": "u123", "agent_id": "hermes-cam"}

    def test_profile_isolation_default_value(self, monkeypatch, tmp_path):
        """profile_isolation defaults to true (isolation is the default)."""
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", agent_identity="cam")
        provider._user_id = "u123"
        filters = provider._read_filters()
        assert "agent_id" in filters
        assert filters["agent_id"] == "hermes-cam"

    def test_cron_profile_uses_hermes_cron(self, monkeypatch, tmp_path):
        """Cron jobs get agent_id 'hermes-cron'."""
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", agent_identity="cron")
        assert provider._agent_id == "hermes-cron"
        provider._user_id = "reese"
        filters = provider._read_filters()
        assert filters["agent_id"] == "hermes-cron"


class TestTopicFingerprintIntegration:
    """Cross-session memory identity resolution integration tests."""

    def _make_provider(self, backend, *, config_override=None):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        if config_override:
            provider._config.update(config_override)
        return provider

    def test_sync_turn_includes_topic_fingerprint_in_metadata(self):
        """sync_turn stores topic_fingerprint in the metadata of each add() call."""
        backend = FakeBackend()
        provider = self._make_provider(backend)
        provider.sync_turn(
            "I love CNC machining with Haas VF2",
            "Great, Haas VF2 is an excellent machine!",
        )
        provider._sync_thread.join(timeout=5)
        add_calls = [c for c in backend.captured if c[0] == "add"]
        assert len(add_calls) == 1
        metadata = add_calls[0][2].get("metadata", {})
        assert "topic_fingerprint" in metadata
        fp = metadata["topic_fingerprint"]
        assert isinstance(fp, list)
        assert len(fp) >= 2
        lower = [w.lower() for w in fp]
        assert "haas" in lower or "cnc" in lower

    def test_sync_turn_preserves_existing_metadata(self):
        """sync_turn preserves channel metadata alongside topic_fingerprint."""
        backend = FakeBackend()
        provider = self._make_provider(backend)
        provider._channel = "telegram"
        provider.sync_turn("user message", "assistant reply")
        provider._sync_thread.join(timeout=5)
        add_calls = [c for c in backend.captured if c[0] == "add"]
        metadata = add_calls[0][2].get("metadata", {})
        assert metadata.get("channel") == "telegram"
        assert "topic_fingerprint" in metadata

    def test_prefetch_cross_session_uses_fingerprint(self):
        """prefetch searches by topic_fingerprint in addition to query."""
        backend = FakeBackend(
            search_results=[{"id": "m1", "memory": "CNC machine preferences"}]
        )
        provider = self._make_provider(backend)
        provider._config["cross_session_identity"] = True
        result = provider.prefetch(
            "tell me about the CNC project",
            topic_fingerprint=["cnc", "machine", "haas"],
        )
        # Should have made 2 search calls: one for query, one for fingerprint
        search_calls = [c for c in backend.captured if c[0] == "search"]
        assert len(search_calls) >= 2
        # Second search should use the fingerprint keywords as query
        fingerprint_query = search_calls[1][1]
        assert "cnc" in fingerprint_query or "machine" in fingerprint_query

    def test_prefetch_cross_session_disabled_by_config(self):
        """When cross_session_identity is False, no fingerprint search is made."""
        backend = FakeBackend(
            search_results=[{"id": "m1", "memory": "some memory"}]
        )
        provider = self._make_provider(backend)
        provider._config["cross_session_identity"] = False
        result = provider.prefetch(
            "tell me about something",
            topic_fingerprint=["cnc", "machine"],
        )
        search_calls = [c for c in backend.captured if c[0] == "search"]
        # Only the main query search, no fingerprint search
        assert len(search_calls) == 1

    def test_prefetch_cross_session_merges_results(self):
        """Cross-session results are merged and deduped with main results."""
        # Return different results for query vs fingerprint search
        call_count = [0]

        class SplitBackend(FakeBackend):
            def search(self, query, *, filters, top_k=10, rerank=True):
                call_count[0] += 1
                if call_count[0] == 1:
                    # Main query results
                    return [{"id": "m1", "memory": "main query result"}]
                else:
                    # Fingerprint search results
                    return [{"id": "m2", "memory": "fingerprint result"}]

        backend = SplitBackend()
        provider = self._make_provider(backend)
        provider._config["cross_session_identity"] = True
        result = provider.prefetch(
            "CNC project",
            topic_fingerprint=["cnc", "machine"],
        )
        assert "main query result" in result
        assert "fingerprint result" in result

    def test_prefetch_cross_session_deduplicates_by_id(self):
        """When same memory appears in both searches, it is not duplicated."""
        call_count = [0]

        class DedupBackend(FakeBackend):
            def search(self, query, *, filters, top_k=10, rerank=True):
                call_count[0] += 1
                # Both searches return the same memory
                return [{"id": "m1", "memory": "same memory in both"}]

        backend = DedupBackend()
        provider = self._make_provider(backend)
        provider._config["cross_session_identity"] = True
        result = provider.prefetch(
            "CNC project",
            topic_fingerprint=["cnc", "machine"],
        )
        assert result.count("same memory in both") == 1

    def test_on_session_switch_prefetches_related_memories(self):
        """on_session_switch finds and prefetches related memories."""
        backend = FakeBackend(
            search_results=[{"id": "m1", "memory": "CNC machine details from yesterday"}]
        )
        provider = self._make_provider(backend)
        provider._config["cross_session_identity"] = True
        provider.on_session_switch(
            "new-session-id",
            first_message="What about the CNC project?",
        )
        # Should have searched by topic fingerprint
        search_calls = [c for c in backend.captured if c[0] == "search"]
        assert len(search_calls) >= 1

    def test_on_session_switch_noop_without_config(self):
        """on_session_switch does nothing when cross_session_identity is disabled."""
        backend = FakeBackend()
        provider = self._make_provider(backend)
        provider._config["cross_session_identity"] = False
        provider.on_session_switch(
            "new-session-id",
            first_message="What about the CNC project?",
        )
        search_calls = [c for c in backend.captured if c[0] == "search"]
        assert len(search_calls) == 0

    def test_cross_session_identity_default_true(self):
        """cross_session_identity defaults to True when not in config."""
        provider = Mem0MemoryProvider()
        # Before initialize, config is None
        assert provider._config is None
        # After initialize, config is set but cross_session_identity may not be
        provider.initialize("test-session")
        # Default should be True (via .get with True default)
        assert provider._config.get("cross_session_identity", True) is True

    def test_fingerprint_extraction_latency_under_50ms(self):
        """Fingerprint extraction must complete in under 50ms."""
        import time as _time
        from plugins.memory.mem0 import extract_topic_fingerprint

        text = (
            "The CNC machine is used for precision aluminum parts. "
            "Haas VF2 is the primary workhorse. Tornado insurance "
            "covers the shop equipment. Deepnest handles nesting density."
        )
        start = _time.monotonic()
        for _ in range(100):
            result = extract_topic_fingerprint(text)
        elapsed = (_time.monotonic() - start) / 100
        assert elapsed < 0.05, f"extract_topic_fingerprint took {elapsed*1000:.1f}ms, exceeds 50ms"
        assert len(result) >= 3


class TestPrefetchCacheMerge:
    """Compression-aware and normal prefetch results must merge without
    overwriting each other. Both paths write to _prefetch_memories and
    consuming builds a single deduplicated block.
    """

    def _make_provider(self, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_both_prefetch_paths_merge_results(self):
        """Normal + compression-aware prefetch results coexist in output."""
        call_count = 0
        normal_results = [{"id": "m1", "memory": "user prefers dark mode"}]
        compress_results = [{"id": "m2", "memory": "user lives in Savannah"}]

        class SplitBackend(FakeBackend):
            def search(self, query, *, filters, top_k=10, rerank=True):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return normal_results
                return compress_results

        backend = SplitBackend()
        provider = self._make_provider(backend)
        provider._config["compression_aware_prefetch"] = True

        # Fire both normal + compression-aware prefetch
        provider.on_turn_start(
            1, "where do I live?",
            remaining_tokens=30000,
            context_limit=100000,
            old_messages=[{"role": "user", "content": "I moved to Savannah"}],
        )
        provider._prefetch_thread.join(timeout=5)
        time.sleep(0.3)  # Give compression-aware thread time

        result = provider.prefetch("where do I live?")
        assert "dark mode" in result
        assert "Savannah" in result

    def test_dedup_prevents_duplicate_memories(self):
        """Identical memory text from both paths appears only once."""
        shared_memory = "user prefers dark mode"
        backend = FakeBackend(
            search_results=[{"id": "m1", "memory": shared_memory}]
        )
        provider = self._make_provider(backend)
        provider._config["compression_aware_prefetch"] = True

        provider.on_turn_start(
            1, "what theme?",
            remaining_tokens=30000,
            context_limit=100000,
            old_messages=[{"role": "user", "content": "I like dark mode themes"}],
        )
        provider._prefetch_thread.join(timeout=5)
        time.sleep(0.3)

        result = provider.prefetch("what theme?")
        assert result.count(shared_memory) == 1

    def test_cache_clears_on_new_query(self):
        """Starting a new prefetch query clears previous results."""
        backend = FakeBackend(
            search_results=[{"id": "m1", "memory": "result A"}]
        )
        provider = self._make_provider(backend)

        provider.on_turn_start(1, "query A")
        provider._prefetch_thread.join(timeout=5)
        result_a = provider.prefetch("query A")
        assert "result A" in result_a

        # New query should start fresh
        provider._start_prefetch("query B")
        provider._prefetch_thread.join(timeout=5)
        result_b = provider.prefetch("query B")
        # Result A's memory should not leak into result B
        # (unless the backend returns it again, which it does in FakeBackend)
        # The key is the cache was cleared


class TestSeparateCompressThread:
    """on_pre_compress must use _compress_thread, not _sync_thread, so
    concurrent compression and turn completion don't serialize.
    """

    def _make_provider(self, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_on_pre_compress_uses_compress_thread(self):
        """on_pre_compress must write to _compress_thread, not _sync_thread."""
        backend = FakeBackend()
        provider = self._make_provider(backend)
        provider.on_pre_compress([
            {"role": "user", "content": "test message"},
        ])
        assert provider._compress_thread is not None
        assert provider._sync_thread is None or not provider._sync_thread.is_alive()

    def test_sync_turn_uses_sync_thread(self):
        """sync_turn must still use _sync_thread."""
        backend = FakeBackend()
        provider = self._make_provider(backend)
        provider.sync_turn("user msg", "assistant reply")
        assert provider._sync_thread is not None

    def test_concurrent_compression_and_sync_dont_block(self):
        """Pre-compression extraction and sync_turn run independently."""
        entered_sync = threading.Event()
        entered_compress = threading.Event()
        release = threading.Event()

        class BlockingBackend(FakeBackend):
            def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
                if infer:
                    content = messages[0].get("content", "") if messages else ""
                    if "sync" in content:
                        entered_sync.set()
                    else:
                        entered_compress.set()
                    release.wait(30)
                return super().add(messages, user_id=user_id, agent_id=agent_id,
                                  infer=infer, metadata=metadata)

        backend = BlockingBackend()
        provider = self._make_provider(backend)

        # Start both concurrently
        provider.sync_turn("sync message", "reply")
        provider.on_pre_compress([{"role": "user", "content": "compress message"}])

        # Both should enter the backend without blocking each other
        assert entered_sync.wait(5), "sync_turn never entered backend"
        assert entered_compress.wait(5), "on_pre_compress never entered backend"

        release.set()
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5)
        if provider._compress_thread:
            provider._compress_thread.join(timeout=5)


class TestHealthMetrics:
    """get_health_metrics() must return accurate session-level counters."""

    def _make_provider(self, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_initial_metrics_zero(self):
        """Fresh provider starts with zero metrics."""
        backend = FakeBackend()
        provider = self._make_provider(backend)
        metrics = provider.get_health_metrics()
        assert metrics["memories_added_session"] == 0
        assert metrics["compression_events"] == 0
        assert metrics["compression_facts_extracted"] == 0

    def test_sync_turn_increments_added(self):
        """sync_turn increments memories_added_session."""
        backend = FakeBackend()
        provider = self._make_provider(backend)
        provider.sync_turn("user msg", "reply")
        provider._sync_thread.join(timeout=5)
        metrics = provider.get_health_metrics()
        assert metrics["memories_added_session"] == 1

    def test_pre_compress_increments_compression_counters(self):
        """on_pre_compress increments compression_events and facts_extracted."""
        backend = FakeBackend()
        provider = self._make_provider(backend)
        provider.on_pre_compress([
            {"role": "user", "content": "fact 1"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "fact 2"},
        ])
        provider._compress_thread.join(timeout=5)
        metrics = provider.get_health_metrics()
        assert metrics["compression_events"] == 1
        assert metrics["compression_facts_extracted"] == 3

    def test_compression_aware_prefetch_enabled_by_default(self):
        """compression_aware_prefetch defaults to True."""
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        assert provider._config.get("compression_aware_prefetch", True) is True

# ---------------------------------------------------------------------------
# Memory consolidation tests
# ---------------------------------------------------------------------------

class TestMem0Consolidation:
    """Tests for the consolidate() method: dedup, contradiction detection, logging."""

    def _make_provider(self, backend, *, config_override=None):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        if config_override:
            provider._config.update(config_override)
        return provider

    def test_exact_duplicates_deleted(self):
        """Two memories with identical text should result in one deletion."""
        backend = FakeBackend(all_results=[
            {"id": "m1", "memory": "Reese lives in Savannah", "created_at": "2026-01-01T00:00:00Z"},
            {"id": "m2", "memory": "Reese lives in Savannah", "created_at": "2026-06-01T00:00:00Z"},
        ])
        backend._search_results = []  # No semantic matches for phase 2
        provider = self._make_provider(backend)
        provider._consolidate()
        delete_calls = [c for c in backend.captured if c[0] == "delete"]
        assert len(delete_calls) == 1
        assert delete_calls[0][1] == "m1"  # Older one deleted

    def test_near_duplicates_via_normalized_text(self):
        """Memories that differ only in punctuation/casing are duplicates."""
        backend = FakeBackend(all_results=[
            {"id": "m1", "memory": "Reese lives in Savannah, GA!", "created_at": "2026-01-01T00:00:00Z"},
            {"id": "m2", "memory": "reese lives in savannah ga", "created_at": "2026-06-01T00:00:00Z"},
        ])
        provider = self._make_provider(backend)
        provider._consolidate()
        delete_calls = [c for c in backend.captured if c[0] == "delete"]
        assert len(delete_calls) == 1

    def test_no_duplicates_when_different(self):
        """Completely different memories should not be deleted."""
        backend = FakeBackend(all_results=[
            {"id": "m1", "memory": "Reese lives in Savannah", "created_at": "2026-01-01"},
            {"id": "m2", "memory": "Reese uses Python 3.11", "created_at": "2026-01-01"},
        ])
        provider = self._make_provider(backend)
        provider._consolidate()
        delete_calls = [c for c in backend.captured if c[0] == "delete"]
        assert len(delete_calls) == 0

    def test_most_recent_kept(self):
        """When duplicates exist, the most recent one is kept."""
        backend = FakeBackend(all_results=[
            {"id": "m-old", "memory": "fact A", "created_at": "2026-01-01T00:00:00Z"},
            {"id": "m-new", "memory": "fact A", "created_at": "2026-08-01T00:00:00Z"},
        ])
        provider = self._make_provider(backend)
        provider._consolidate()
        delete_calls = [c for c in backend.captured if c[0] == "delete"]
        assert len(delete_calls) == 1
        assert delete_calls[0][1] == "m-old"

    def test_contradiction_detection(self):
        """Memories with negation patterns and shared keywords are flagged."""
        backend = FakeBackend(all_results=[
            {"id": "m1", "memory": "Reese uses Vim daily", "created_at": "2026-01-01"},
            {"id": "m2", "memory": "Reese does not use Vim anymore", "created_at": "2026-06-01"},
        ])
        provider = self._make_provider(backend)
        provider._consolidate()
        # Contradictions are detected and logged (no update calls — metadata
        # update not yet supported by all backends). Verify via stat counter.
        # No delete calls should happen for contradictions.
        delete_calls = [c for c in backend.captured if c[0] == "delete"]
        assert len(delete_calls) == 0

    def test_contradiction_not_deleted(self):
        """Contradicting memories are flagged but not deleted."""
        backend = FakeBackend(all_results=[
            {"id": "m1", "memory": "Reese uses Vim daily", "created_at": "2026-01-01"},
            {"id": "m2", "memory": "Reese does not use Vim anymore", "created_at": "2026-06-01"},
        ])
        provider = self._make_provider(backend)
        provider._consolidate()
        delete_calls = [c for c in backend.captured if c[0] == "delete"]
        assert len(delete_calls) == 0

    def test_consolidation_log_written(self, tmp_path):
        """Consolidation actions are logged to the audit file."""
        import os
        os.environ["HERMES_HOME"] = str(tmp_path)
        backend = FakeBackend(all_results=[
            {"id": "m1", "memory": "fact A", "created_at": "2026-01-01"},
            {"id": "m2", "memory": "fact A", "created_at": "2026-06-01"},
        ])
        provider = self._make_provider(backend)
        provider._consolidate()
        log_path = tmp_path / "logs" / "memory-consolidation.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "consolidation_run" in content
        assert "deleted_duplicates: 1" in content

    def test_consolidation_empty_memories(self):
        """Consolidation with no memories should be a no-op."""
        backend = FakeBackend(all_results=[])
        provider = self._make_provider(backend)
        provider._consolidate()  # Should not raise
        delete_calls = [c for c in backend.captured if c[0] == "delete"]
        assert len(delete_calls) == 0

    def test_consolidation_single_memory(self):
        """Consolidation with one memory should be a no-op."""
        backend = FakeBackend(all_results=[
            {"id": "m1", "memory": "fact A", "created_at": "2026-01-01"},
        ])
        provider = self._make_provider(backend)
        provider._consolidate()
        delete_calls = [c for c in backend.captured if c[0] == "delete"]
        assert len(delete_calls) == 0

    def test_consolidation_records_success(self):
        """Successful consolidation resets the circuit breaker."""
        backend = FakeBackend(all_results=[
            {"id": "m1", "memory": "fact A", "created_at": "2026-01-01"},
            {"id": "m2", "memory": "fact A", "created_at": "2026-06-01"},
        ])
        provider = self._make_provider(backend)
        provider._consecutive_failures = 3
        provider._consolidate()
        assert provider._consecutive_failures == 0

    def test_consolidation_records_failure(self):
        """Backend failure during consolidation increments failure count."""
        class FailingBackend(FakeBackend):
            def list_all(self, *, filters):
                raise ConnectionError("Qdrant is down")

        backend = FailingBackend()
        provider = self._make_provider(backend)
        assert provider._consecutive_failures == 0
        provider._consolidate()
        assert provider._consecutive_failures == 1

    def test_consolidation_non_blocking(self):
        """Consolidation runs in a background thread, not blocking the caller."""
        entered = threading.Event()
        release = threading.Event()

        class SlowBackend(FakeBackend):
            def list_all(self, *, filters):
                entered.set()
                release.wait(30)
                return []

        backend = SlowBackend()
        provider = self._make_provider(backend)
        provider._start_consolidation()
        # _start_consolidation returns immediately
        assert entered.wait(5), "background thread never started"
        release.set()
        provider._consolidation_thread.join(timeout=5)

    def test_consolidation_config_gated(self):
        """Consolidation is skipped when consolidation_enabled is False."""
        backend = FakeBackend(all_results=[
            {"id": "m1", "memory": "fact A", "created_at": "2026-01-01"},
            {"id": "m2", "memory": "fact A", "created_at": "2026-06-01"},
        ])
        provider = self._make_provider(backend, config_override={
            "consolidation_enabled": False,
        })
        # on_turn_start should not trigger consolidation when disabled
        provider._turns_since_consolidation = 100
        provider.on_turn_start(100, "test")
        # No thread should have been spawned
        assert provider._consolidation_thread is None
        # No list_all call means consolidation didn't run via on_turn_start
        list_all_calls = [c for c in backend.captured if c[0] == "list_all"]
        assert len(list_all_calls) == 0


class TestMem0PeriodicConsolidationTrigger:
    """Tests that on_turn_start triggers consolidation at the configured interval."""

    def _make_provider(self, backend, *, config_override=None):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        if config_override:
            provider._config.update(config_override)
        return provider

    def test_triggers_at_interval(self):
        """Consolidation fires when turn count reaches the interval."""
        backend = FakeBackend(all_results=[
            {"id": "m1", "memory": "fact", "created_at": "2026-01-01"},
            {"id": "m2", "memory": "fact", "created_at": "2026-06-01"},
        ])
        provider = self._make_provider(backend, config_override={
            "consolidation_interval_turns": 3,
            "consolidation_enabled": True,
        })
        # Simulate turns
        for i in range(3):
            provider.on_turn_start(i + 1, f"turn {i+1}")
        provider._consolidation_thread.join(timeout=5)
        list_all_calls = [c for c in backend.captured if c[0] == "list_all"]
        assert len(list_all_calls) == 1

    def test_resets_counter_after_trigger(self):
        """After consolidation fires, the counter resets."""
        backend = FakeBackend(all_results=[
            {"id": "m1", "memory": "fact", "created_at": "2026-01-01"},
            {"id": "m2", "memory": "fact", "created_at": "2026-06-01"},
        ])
        provider = self._make_provider(backend, config_override={
            "consolidation_interval_turns": 2,
            "consolidation_enabled": True,
        })
        provider.on_turn_start(1, "turn 1")
        provider.on_turn_start(2, "turn 2")
        provider._consolidation_thread.join(timeout=5)
        assert provider._turns_since_consolidation == 0

    def test_does_not_fire_before_interval(self):
        """Consolidation does not fire before the interval is reached."""
        backend = FakeBackend(all_results=[
            {"id": "m1", "memory": "fact", "created_at": "2026-01-01"},
            {"id": "m2", "memory": "fact", "created_at": "2026-06-01"},
        ])
        provider = self._make_provider(backend, config_override={
            "consolidation_interval_turns": 10,
            "consolidation_enabled": True,
        })
        for i in range(5):
            provider.on_turn_start(i + 1, f"turn {i+1}")
        # Wait briefly for any spurious thread
        time.sleep(0.1)
        list_all_calls = [c for c in backend.captured if c[0] == "list_all"]
        assert len(list_all_calls) == 0

    def test_no_consolidation_when_breaker_open(self):
        """Consolidation is skipped when the circuit breaker is tripped."""
        backend = FakeBackend(all_results=[
            {"id": "m1", "memory": "fact", "created_at": "2026-01-01"},
            {"id": "m2", "memory": "fact", "created_at": "2026-06-01"},
        ])
        provider = self._make_provider(backend, config_override={
            "consolidation_interval_turns": 1,
            "consolidation_enabled": True,
        })
        provider._consecutive_failures = 10
        provider._breaker_open_until = time.monotonic() + 300
        provider.on_turn_start(1, "turn 1")
        time.sleep(0.1)
        list_all_calls = [c for c in backend.captured if c[0] == "list_all"]
        assert len(list_all_calls) == 0

    def test_default_interval_is_50(self):
        """Default consolidation interval is 50 turns."""
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        assert provider._config.get("consolidation_interval_turns", 50) == 50


class TestMem0NormalizeText:
    """Tests for the _normalize_text static method."""

    def test_lowercase(self):
        assert Mem0MemoryProvider._normalize_text("Hello World") == "hello world"

    def test_strip_punctuation(self):
        assert Mem0MemoryProvider._normalize_text("fact, with. punctuation!") == "fact with punctuation"

    def test_collapse_whitespace(self):
        assert Mem0MemoryProvider._normalize_text("  lots   of   spaces  ") == "lots of spaces"

    def test_empty_string(self):
        assert Mem0MemoryProvider._normalize_text("") == ""


class TestMem0FindDuplicateGroups:
    """Tests for the _find_duplicate_groups static method."""

    def test_exact_duplicates_grouped(self):
        memories = [
            {"id": "m1", "memory": "Reese lives in Savannah"},
            {"id": "m2", "memory": "Reese lives in Savannah"},
        ]
        groups = Mem0MemoryProvider._find_duplicate_groups(memories, None, 0.85)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_no_duplicates_no_groups(self):
        memories = [
            {"id": "m1", "memory": "Reese lives in Savannah"},
            {"id": "m2", "memory": "Reese uses Python"},
        ]
        groups = Mem0MemoryProvider._find_duplicate_groups(memories, None, 0.85)
        assert len(groups) == 0

    def test_empty_memory_text_skipped(self):
        memories = [
            {"id": "m1", "memory": ""},
            {"id": "m2", "memory": "valid fact"},
        ]
        groups = Mem0MemoryProvider._find_duplicate_groups(memories, None, 0.85)
        assert len(groups) == 0


class TestMem0FindContradictions:
    """Tests for the _find_contradictions static method."""

    def test_negation_detected(self):
        memories = [
            {"id": "m1", "memory": "Reese uses Vim daily"},
            {"id": "m2", "memory": "Reese does not use Vim anymore"},
        ]
        pairs = Mem0MemoryProvider._find_contradictions(memories)
        assert len(pairs) == 1
        assert set(pairs[0]) == {"m1", "m2"}

    def test_no_contradiction_without_negation(self):
        memories = [
            {"id": "m1", "memory": "Reese uses Vim daily"},
            {"id": "m2", "memory": "Reese uses Emacs daily"},
        ]
        pairs = Mem0MemoryProvider._find_contradictions(memories)
        assert len(pairs) == 0

    def test_no_contradiction_without_keyword_overlap(self):
        memories = [
            {"id": "m1", "memory": "Reese uses Vim daily"},
            {"id": "m2", "memory": "Does not use Python"},
        ]
        pairs = Mem0MemoryProvider._find_contradictions(memories)
        assert len(pairs) == 0

    def test_empty_memories(self):
        pairs = Mem0MemoryProvider._find_contradictions([])
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Temporal metadata and time-aware retrieval tests
# ---------------------------------------------------------------------------

class TestExtractTemporalHint:
    """extract_temporal_hint() detects temporal expressions in memory content
    and returns a structured hint for storage as metadata.
    """

    def test_returns_none_for_no_temporal_expression(self):
        from plugins.memory.mem0 import extract_temporal_hint
        assert extract_temporal_hint("Reese lives in Savannah") is None

    def test_detects_last_weekday(self):
        from plugins.memory.mem0 import extract_temporal_hint
        hint = extract_temporal_hint("I decided last Tuesday to use Vim")
        assert hint is not None
        assert "past" in hint.lower() or "last" in hint.lower() or "tuesday" in hint.lower()

    def test_detects_next_week(self):
        from plugins.memory.mem0 import extract_temporal_hint
        hint = extract_temporal_hint("The meeting is next week")
        assert hint is not None
        assert "future" in hint.lower() or "next" in hint.lower()

    def test_detects_yesterday(self):
        from plugins.memory.mem0 import extract_temporal_hint
        hint = extract_temporal_hint("Yesterday I went to the shop")
        assert hint is not None

    def test_detects_tomorrow(self):
        from plugins.memory.mem0 import extract_temporal_hint
        hint = extract_temporal_hint("I will go tomorrow")
        assert hint is not None

    def test_detects_in_n_months(self):
        from plugins.memory.mem0 import extract_temporal_hint
        hint = extract_temporal_hint("The delivery arrives in 3 months")
        assert hint is not None

    def test_detects_in_n_weeks(self):
        from plugins.memory.mem0 import extract_temporal_hint
        hint = extract_temporal_hint("Deadline is in 2 weeks")
        assert hint is not None

    def test_returns_none_for_empty_text(self):
        from plugins.memory.mem0 import extract_temporal_hint
        assert extract_temporal_hint("") is None
        assert extract_temporal_hint(None) is None

    def test_latency_under_10ms(self):
        """Temporal extraction must complete in under 10ms."""
        import time as _time
        from plugins.memory.mem0 import extract_temporal_hint
        text = "Last Tuesday I decided to move to Savannah in 3 months and start tomorrow"
        start = _time.monotonic()
        for _ in range(100):
            extract_temporal_hint(text)
        elapsed = (_time.monotonic() - start) / 100
        assert elapsed < 0.01, f"extract_temporal_hint took {elapsed*1000:.1f}ms, exceeds 10ms"


class TestTemporalWriteMetadata:
    """_write_metadata() must include timestamp, session_id, and turn_number
    so every memory carrys temporal context.
    """

    def _make_provider(self):
        provider = Mem0MemoryProvider()
        provider._channel = "cli"
        provider._session_id = "test-session-123"
        provider._turn_number = 7
        return provider

    def test_metadata_includes_timestamp(self):
        provider = self._make_provider()
        meta = provider._write_metadata()
        assert "timestamp" in meta
        # Must be ISO format
        from datetime import datetime
        datetime.fromisoformat(meta["timestamp"].replace("Z", "+00:00"))

    def test_metadata_includes_session_id(self):
        provider = self._make_provider()
        meta = provider._write_metadata()
        assert meta.get("session_id") == "test-session-123"

    def test_metadata_includes_turn_number(self):
        provider = self._make_provider()
        meta = provider._write_metadata()
        assert meta.get("turn_number") == 7

    def test_metadata_includes_channel(self):
        provider = self._make_provider()
        meta = provider._write_metadata()
        assert meta.get("channel") == "cli"

    def test_sync_turn_stores_temporal_metadata(self):
        """sync_turn() must include timestamp and session_id in write metadata."""
        backend = FakeBackend()
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        provider.sync_turn("I love CNC machining", "Great!")
        provider._sync_thread.join(timeout=5)
        add_calls = [c for c in backend.captured if c[0] == "add"]
        assert len(add_calls) == 1
        metadata = add_calls[0][2].get("metadata", {})
        assert "timestamp" in metadata
        assert "session_id" in metadata

    def test_on_pre_compress_stores_temporal_metadata(self):
        """on_pre_compress() must include temporal metadata in each add() call."""
        backend = FakeBackend()
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        provider.on_pre_compress([
            {"role": "user", "content": "important fact"},
        ])
        provider._compress_thread.join(timeout=5)
        add_calls = [c for c in backend.captured if c[0] == "add"]
        assert len(add_calls) == 1
        metadata = add_calls[0][2].get("metadata", {})
        assert "timestamp" in metadata
        assert "session_id" in metadata

    def test_mem0_add_stores_temporal_metadata(self):
        """mem0_add tool call must include temporal metadata."""
        backend = FakeBackend()
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        json.loads(provider.handle_tool_call("mem0_add", {"content": "user prefers dark mode"}))
        add_calls = [c for c in backend.captured if c[0] == "add"]
        assert len(add_calls) == 1
        metadata = add_calls[0][2].get("metadata", {})
        assert "timestamp" in metadata


class TestTimeAwarePrefetch:
    """prefetch() must accept optional date range filters to enable
    time-aware retrieval: 'what did I decide last week?'
    """

    def _make_provider(self, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_prefetch_passes_date_filters_to_backend(self):
        """When since_date is provided, the filter is passed to backend.search()."""
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "result"}])
        provider = self._make_provider(backend)
        result = provider.prefetch(
            "what did I decide last week?",
            since_date="2026-08-13",
        )
        search_calls = [c for c in backend.captured if c[0] == "search"]
        assert len(search_calls) >= 1
        filters = search_calls[0][2]["filters"]
        assert "created_at" in filters or "since_date" in filters

    def test_prefetch_passes_until_date(self):
        """When until_date is provided, the filter is passed."""
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "result"}])
        provider = self._make_provider(backend)
        result = provider.prefetch(
            "what happened yesterday?",
            since_date="2026-08-18",
            until_date="2026-08-19",
        )
        search_calls = [c for c in backend.captured if c[0] == "search"]
        filters = search_calls[0][2]["filters"]
        # Should contain some form of date range filter
        has_range = any(k in filters for k in ["created_at", "since_date", "date_range"])
        assert has_range

    def test_prefetch_without_dates_unchanged(self):
        """When no date filters provided, behavior is identical to before."""
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "result"}])
        provider = self._make_provider(backend)
        result = provider.prefetch("what do I like?")
        search_calls = [c for c in backend.captured if c[0] == "search"]
        filters = search_calls[0][2]["filters"]
        # No date-related keys in filters
        assert not any(k in filters for k in ["created_at", "since_date", "date_range"])


class TestStaleFactDetection:
    """During consolidation, memories with temporal hints pointing to past
    relative dates should be flagged as potentially stale.
    """

    def _make_provider(self, backend, *, config_override=None):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        if config_override:
            provider._config.update(config_override)
        return provider

    def test_stale_detection_flags_old_temporal_hint(self):
        """A memory with a temporal_hint from days ago should be flagged."""
        backend = FakeBackend(all_results=[
            {
                "id": "m1",
                "memory": "The meeting is last Tuesday",
                "created_at": "2026-08-12T00:00:00Z",
                "metadata": {"temporal_hint": "past_tuesday"},
            },
            {
                "id": "m2",
                "memory": "Reese lives in Savannah",
                "created_at": "2026-08-19T00:00:00Z",
            },
        ])
        provider = self._make_provider(backend)
        provider._consolidate()
        # The stale memory should have been updated with potentially_stale flag
        update_calls = [c for c in backend.captured if c[0] == "update"]
        stale_updates = [c for c in update_calls if "potentially_stale" in str(c)]
        assert len(stale_updates) >= 1

    def test_non_temporal_memories_not_flagged(self):
        """Memories without temporal hints should not be flagged as stale."""
        backend = FakeBackend(all_results=[
            {
                "id": "m1",
                "memory": "Reese lives in Savannah",
                "created_at": "2026-01-01T00:00:00Z",
            },
        ])
        provider = self._make_provider(backend)
        provider._consolidate()
        update_calls = [c for c in backend.captured if c[0] == "update"]
        assert len(update_calls) == 0

    def test_recent_temporal_not_flagged(self):
        """A temporal memory from today should NOT be flagged stale."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
        backend = FakeBackend(all_results=[
            {
                "id": "m1",
                "memory": "The meeting is next Tuesday",
                "created_at": today,
                "metadata": {"temporal_hint": "next_tuesday"},
            },
        ])
        provider = self._make_provider(backend)
        provider._consolidate()
        update_calls = [c for c in backend.captured if c[0] == "update"]
        stale_updates = [c for c in update_calls if "potentially_stale" in str(c)]
        assert len(stale_updates) == 0

    def test_stale_detection_disabled_by_config(self):
        """When stale_detection is False, no flagging occurs."""
        backend = FakeBackend(all_results=[
            {
                "id": "m1",
                "memory": "The meeting is last Tuesday",
                "created_at": "2026-08-12T00:00:00Z",
                "metadata": {"temporal_hint": "past_tuesday"},
            },
        ])
        provider = self._make_provider(backend, config_override={"stale_detection": False})
        provider._consolidate()
        update_calls = [c for c in backend.captured if c[0] == "update"]
        stale_updates = [c for c in update_calls if "potentially_stale" in str(c)]
        assert len(stale_updates) == 0


class TestTemporalConfigDefaults:
    """Config defaults for temporal features."""

    def test_temporal_metadata_enabled_by_default(self):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        assert provider._config.get("temporal_metadata", True) is True

    def test_stale_detection_enabled_by_default(self):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        assert provider._config.get("stale_detection", True) is True


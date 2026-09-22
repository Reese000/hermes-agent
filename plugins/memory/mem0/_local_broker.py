"""Local Mem0 broker — single-process owner of the path-mode Qdrant locks.

Why this exists
---------------
Qdrant's local (path) mode takes an EXCLUSIVE file lock on ``<path>/.lock``
for the lifetime of each ``QdrantClient``. Hermes runs many concurrent
processes that each build a ``mem0.Memory`` at session init (desktop
backend, gateway, every kanban CLI worker), so whoever initializes first
holds the lock for its whole run and every other process fails with
``Storage folder ... is already accessed by another instance of Qdrant
client``. Docker/Qdrant-server is deliberately not used — this broker is
the no-Docker single-writer equivalent: ONE process owns the store, all
Hermes processes share it over loopback HTTP.

The HTTP contract matches
:class:`plugins.memory.mem0._backend.SelfHostedBackend` exactly (that
backend was written for the self-hosted Mem0 FastAPI server); the plugin
routes OSS path-mode configs through ``ensure_broker()``, which spawns this
module detached and returns the base URL.

Lock handoff: if an old-code process currently holds the path lock, the
broker starts with ``ready=false``, retries construction in a background
thread every 3 s, and flips to ``ready=true`` the moment the lock frees.
While not ready, endpoints return HTTP 503 with a clear message; the
plugin treats 503 as transient (no circuit-breaker penalty).

Run: ``python -m plugins.memory.mem0._local_broker [--port N]``
(spawned detached by ``ensure_broker``; HERMES_HOME must be set so
mem0.json resolves).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import suppress
from typing import Any, Callable, Optional

logger = logging.getLogger("plugins.memory.mem0.broker")

DEFAULT_PORT = 8765
# Lock-acquire retry cadence while an old-code process still owns the store.
_RETRY_INTERVAL_SECS = 3.0
# How long ensure_broker() waits for the spawned broker to answer /healthz.
SPAWN_WAIT_SECS = 10.0
# HTTP timeout for plugin->broker calls. Must cover mem0 fact extraction on
# add(infer=True) (10-90 s) but stay BELOW the plugin's sync-thread shutdown
# join (_SYNC_SHUTDOWN_WAIT_SECS = 120) so an over-long call times out in the
# sync thread BEFORE shutdown force-joins it. The broker-side handler finishes
# the insert even if this client disconnects, so a timeout here never loses
# the memory — strictly better than the old in-process close-mid-insert.
BROKER_TIMEOUT_SECS = 115.0


def _repo_root() -> str:
    # .../plugins/memory/mem0/_local_broker.py -> .../hermes-agent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _broker_url(port: int = DEFAULT_PORT) -> str:
    return f"http://127.0.0.1:{port}"


def _probe(url: str, timeout: float = 1.5) -> bool:
    """True when a broker answers /healthz (ready OR still acquiring)."""
    import httpx

    try:
        return httpx.get(f"{url}/healthz", timeout=timeout).status_code == 200
    except Exception:
        return False


def _spawn(port: int, log_path: str) -> Optional[subprocess.Popen]:
    """Start the broker detached; returns the Popen or None on immediate failure."""
    # Popen(env=...) rejects non-strings: get_hermes_home() returns a Path on
    # Windows (WindowsPath), which raised "environment can only contain strings"
    # and silently killed the spawn. Coerce every value defensively.
    env = {str(k): str(v) for k, v in os.environ.items()}
    root = _repo_root()
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        from hermes_constants import get_hermes_home

        env["HERMES_HOME"] = str(get_hermes_home())
    except Exception:
        pass
    log_file = None
    with suppress(Exception):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8")
    kwargs: dict[str, Any] = {
        "env": env,
        "cwd": root,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if log_file is not None:
        kwargs["stdout"] = log_file
        kwargs["stderr"] = log_file
    if sys.platform == "win32":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP — survive caller exit.
        kwargs["creationflags"] = 0x00000008 | 0x00000200
    else:
        kwargs["start_new_session"] = True
    cmd = [sys.executable, "-m", "plugins.memory.mem0._local_broker", "--port", str(port)]
    try:
        return subprocess.Popen(cmd, **kwargs)
    except Exception as exc:
        logger.error("mem0 broker spawn failed: %s", exc)
        if log_file is not None:
            with suppress(Exception):
                log_file.close()
        return None


def ensure_broker(oss_config: dict) -> Optional[str]:
    """Return the broker base URL, spawning it if needed.

    Returns None only when the broker is unreachable AND could not be
    spawned — the caller then falls back to the legacy direct path-mode
    backend (single-process semantics, previous behaviour).
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        # Never spawn a detached broker from inside the test suite (tests
        # exercise this via injected _probe/_spawn instead).
        return None
    broker_cfg = oss_config.get("broker") or {}
    if broker_cfg.get("enabled") is False:
        return None
    port = int(broker_cfg.get("port", DEFAULT_PORT))
    wait = float(broker_cfg.get("wait_secs", SPAWN_WAIT_SECS))
    url = _broker_url(port)

    if _probe(url):
        return url

    from hermes_constants import get_hermes_home

    log_path = os.path.join(str(get_hermes_home()), "logs", "mem0-broker.log")
    _spawn(port, log_path)

    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if _probe(url):
            return url
        time.sleep(0.25)
    return url if _probe(url) else None


def _load_backend():
    """Build the OSS backend from mem0.json (the same config the plugin uses)."""
    from hermes_constants import get_hermes_home

    from plugins.memory.mem0._backend import OSSBackend

    config_path = os.path.join(str(get_hermes_home()), "mem0.json")
    with open(config_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    oss = cfg.get("oss") or {}
    if not oss.get("vector_store", {}).get("config", {}).get("path"):
        raise RuntimeError("mem0.json oss.vector_store.config.path missing — broker requires path mode")
    return OSSBackend(oss)


def create_app(backend_provider: Callable[[], Any] = _load_backend):
    """FastAPI app wrapping one shared OSS backend.

    ``backend_provider`` is injectable for tests; in production it builds the
    real backend lazily at startup and holds it for the process life.
    ``ready`` flips True only once the backend exists; while another process
    holds the path lock, a background thread keeps retrying every 3 s.
    Response shapes mirror what SelfHostedBackend._json() parses.
    """
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="hermes-mem0-broker", docs_url=None, redoc_url=None)
    state: dict[str, Any] = {"backend": None, "ready": False, "lock": threading.Lock(), "error": ""}

    def _try_build() -> bool:
        if state["ready"]:
            return True
        with state["lock"]:
            if state["ready"]:
                return True
            try:
                state["backend"] = backend_provider()
                state["ready"] = True
                state["error"] = ""
                logger.info("mem0 broker: store acquired, ready")
                return True
            except Exception as exc:
                state["error"] = str(exc)
                logger.warning("mem0 broker: store not acquired yet: %s", exc)
                return False

    def _rebuilder() -> None:
        while not state["ready"]:
            if _try_build():
                return
            time.sleep(_RETRY_INTERVAL_SECS)

    @app.on_event("startup")
    def _startup() -> None:
        if not _try_build():
            threading.Thread(target=_rebuilder, daemon=True, name="mem0-broker-retry").start()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "ready": state["ready"], "error": state["error"]}

    def _require():
        if not state["ready"] or state["backend"] is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "mem0 store busy: another Hermes process holds the local Qdrant "
                    f"lock ({state['error'] or 'acquiring'}). The broker activates "
                    "automatically once it frees."
                ),
            )
        return state["backend"]

    @app.post("/search")
    def search(payload: dict) -> dict:
        backend = _require()
        results = backend.search(
            str(payload.get("query", "")),
            filters=dict(payload.get("filters") or {}),
            top_k=int(payload.get("top_k") or 10),
            rerank=bool(payload.get("rerank", False)),
        )
        return {"results": results or []}

    @app.post("/memories")
    def add(payload: dict) -> dict:
        backend = _require()
        result = backend.add(
            list(payload.get("messages") or []),
            user_id=str(payload.get("user_id") or ""),
            agent_id=str(payload.get("agent_id") or "hermes"),
            infer=bool(payload.get("infer", False)),
            metadata=dict(payload.get("metadata") or {}),
        )
        return result if isinstance(result, dict) else {"result": result}

    @app.put("/memories/{memory_id}")
    def update(memory_id: str, payload: dict) -> dict:
        backend = _require()
        return backend.update(memory_id, str(payload.get("text") or ""))

    @app.delete("/memories/{memory_id}")
    def delete(memory_id: str) -> dict:
        backend = _require()
        return backend.delete(memory_id)

    @app.get("/memories")
    def list_all() -> dict:
        """Read-only listing of EVERY memory, WITH vectors, no entity filter.

        Why this is not ``Memory.get_all()``: mem0's ``MemoryItem`` model has
        no vector field and ``Qdrant.list()`` hardcodes ``with_vectors=False``,
        so ``get_all()`` cannot return the vectors the consolidation script's
        0.85 cosine dedupe runs on. ``get_all()`` also RAISES unless the caller
        supplies user_id/agent_id/run_id, which would silently hide memories
        written under any other entity.

        This route only scrolls — it never mutates the store, and it never
        opens the path lock (the backend already owns it for this process).
        """
        backend = _require()
        memory = getattr(backend, "_memory", None)
        vector_store = getattr(memory, "vector_store", None)
        client = getattr(vector_store, "client", None)
        collection = getattr(vector_store, "collection_name", None)
        if client is None or not collection:
            raise HTTPException(
                status_code=500,
                detail=(
                    "mem0 backend exposes no in-process vector store "
                    f"(got {type(backend).__name__}); GET /memories requires "
                    "the OSS path-mode backend"
                ),
            )
        results: list = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                scroll_filter=None,
                limit=1000,
                with_payload=True,
                with_vectors=True,
                offset=offset,
            )
            for point in points:
                results.append(
                    {
                        "id": point.id,
                        "vector": point.vector,
                        "payload": dict(point.payload or {}),
                    }
                )
            if offset is None or not points:
                break
        return {"count": len(results), "results": results}

    return app


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="mem0-local-broker")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    app = create_app()

    import uvicorn

    # Bind loopback only: the broker is a local IPC seam, never a network service.
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

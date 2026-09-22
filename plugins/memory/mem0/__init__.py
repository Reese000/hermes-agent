"""Mem0 memory plugin — MemoryProvider interface.

Server-side LLM fact extraction, semantic search, and automatic deduplication
via the Mem0 Platform API (cloud) or OSS (self-hosted) via Memory.

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.

Configuration
-------------
Secret (lives in $HERMES_HOME/.env or the environment):
  MEM0_API_KEY       — Mem0 Platform API key (required for platform mode)
  MEM0_HOST          — Base URL of a self-hosted Mem0 server. When set, the
                       plugin talks to that server directly over HTTP
                       (X-API-Key auth) instead of the cloud API.

Behavioral settings (live in $HERMES_HOME/mem0.json, set via `hermes memory
setup`):
  mode               — Backend mode: "platform" (default) or "oss"
  host               — Self-hosted Mem0 server URL (alt: MEM0_HOST env var).
                       When set, routes to the self-hosted HTTP backend.
  user_id            — Canonical user identifier. When set, it is applied
                       uniformly across every gateway (CLI, Telegram, Slack,
                       Discord, …) so the same human gets one merged memory
                       store. When unset, the gateway-native id (e.g. Telegram
                       numeric id, Discord snowflake) is used instead.
  agent_id           — Agent identifier (default: hermes)

The matching MEM0_MODE / MEM0_USER_ID / MEM0_AGENT_ID environment variables are
still read as a backward-compatible fallback, but mem0.json is the canonical
home for these non-secret settings.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from agent.secret_scope import get_secret
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120
_PREFETCH_WAIT_SECS = 3
# Max seconds shutdown() waits for an in-flight mem0-sync thread before
# closing the backend. Server-side LLM fact extraction runs 10-90 s, so a
# 5 s join (the old behaviour) closed the Qdrant client mid-insert and
# silently dropped the memory ("Cannot send a request, as the client has
# been closed").
_SYNC_SHUTDOWN_WAIT_SECS = 120
# Per-message character cap applied by sync_turn() before the turn is ingested.
# Small OSS embedders (Ollama bge-small-zh-v1.5, all-minilm: 512 tokens) raise on
# oversized input, that failure was only logged, and the whole extraction was
# silently dropped (#106235/#37421). Override with ``sync_max_chars`` in
# mem0.json (e.g. 6000) for 8k-token embedders such as text-embedding-3-small.
_SYNC_MSG_MAX_CHARS = 450

_CLIENT_ERROR_TYPES = ("MemoryNotFoundError", "ValidationError")

# Sentinel returned when neither MEM0_USER_ID nor a gateway-native id is
# available. Treated as "no operator-configured user_id" by initialize() so
# that legacy mem0.json files written by the setup wizard (which historically
# wrote this exact placeholder) still allow gateway-native ids to flow
# through instead of silently overriding them with the placeholder.
_DEFAULT_USER_ID = "hermes-user"


def _is_client_error(exc: Exception) -> bool:
    """True for user-caused errors (bad ID, not found) that should NOT trip circuit breaker."""
    etype = type(exc).__name__
    if etype in _CLIENT_ERROR_TYPES:
        return True
    err_str = str(exc).lower()
    return "404" in err_str or "not found" in err_str or "valid uuid" in err_str


def _is_transient_error(exc: Exception) -> bool:
    """True for broker-busy responses (HTTP 503 while another Hermes process still
    holds the local Qdrant path lock). Environmental and self-healing: the broker
    activates automatically when the lock frees, so these must NOT count toward
    the circuit breaker (otherwise 5 back-to-back 503s would disable Mem0 for the
    cooldown long after the lock was released)."""
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    if status == 503:
        return True
    return "mem0 store busy" in str(exc).lower()


def _is_connection_error(exc: Exception) -> bool:
    """True when the backend endpoint itself is unreachable (the local broker or a
    self-hosted server died mid-session) — the backend must be dropped and rebuilt
    so the next call re-runs ensure_broker() and respawns the broker."""
    if type(exc).__name__ in ("ConnectError", "ConnectTimeout", "NewConnectionError"):
        return True
    err_str = str(exc).lower()
    return (
        "connection refused" in err_str
        or "all connection attempts failed" in err_str
        or "connect call failed" in err_str
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides.

    Environment variables provide defaults; mem0.json (if present) overrides
    individual keys.  This avoids a silent failure when the JSON file exists
    but is missing fields like ``api_key`` that the user set in ``.env``.
    """
    from hermes_constants import get_hermes_home

    config = {
        "mode": os.environ.get("MEM0_MODE", "platform"),
        "api_key": get_secret("MEM0_API_KEY", ""),
        "host": os.environ.get("MEM0_HOST", ""),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "oss": {},
    }
    # Only carry user_id when the operator explicitly configured one (env or
    # mem0.json). An absent key tells initialize() to fall back to the
    # gateway-native id from kwargs instead of overriding it with a placeholder.
    env_user_id = os.environ.get("MEM0_USER_ID")
    if env_user_id:
        config["user_id"] = env_user_id

    config_path = get_hermes_home() / "mem0.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


def _read_mem0_json(config_path) -> dict:
    """Read ``mem0.json`` as a dict; an absent, malformed or non-object file reads as {}.

    Single tolerant reader for the setup wizard, which pre-fills prompts from the
    existing file and merge-writes it back — a missing or hand-broken file must
    not abort the wizard mid-setup.
    """
    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _truncate_for_sync(text: str, max_chars: int) -> str:
    """Cap ``text`` at ``max_chars``, stopping on the last sentence boundary that fits.

    A mid-word cut would hand the extractor a fragment it reads as one garbled
    sentence; no boundary at all falls back to a plain cut (better than dropping
    the turn entirely).
    """
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    boundary = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if boundary == -1:
        return cut.rstrip()
    return cut[: boundary + 1]


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search the user's memories by meaning; returns facts ranked by "
        "relevance. Use this before answering any question that may depend on "
        "what you know about the user (preferences, facts, history, people, "
        "projects, past decisions). For multi-part or multi-hop questions, "
        "call it several times — vary the wording and run follow-up searches "
        "on what earlier results reveal; one search is rarely enough."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
            "rerank": {"type": "boolean", "description": "Rerank results for relevance (default: false, platform mode only)."},
        },
        "required": ["query"],
    },
}

ADD_SCHEMA = {
    "name": "mem0_add",
    "description": (
        "Store a durable fact about the user, verbatim (no LLM extraction). "
        "Call this the moment the user states a lasting preference, correction, "
        "decision, or personal detail worth recalling on future turns — don't "
        "wait to be asked to remember. Skip transient chit-chat and facts you've "
        "already stored."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to store."},
        },
        "required": ["content"],
    },
}

UPDATE_SCHEMA = {
    "name": "mem0_update",
    "description": (
        "Replace the text of an existing memory by its ID (take the ID from a "
        "mem0_search result). Use when a stored fact has changed "
        "or was wrong — correct it in place instead of adding a duplicate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to update."},
            "text": {"type": "string", "description": "New text content."},
        },
        "required": ["memory_id", "text"],
    },
}

DELETE_SCHEMA = {
    "name": "mem0_delete",
    "description": (
        "Delete a memory by its ID (take the ID from a mem0_search "
        "result). Use when a stored fact is obsolete or the user asks you to "
        "forget it; prefer mem0_update if the fact merely changed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to delete."},
        },
        "required": ["memory_id"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 memory with server-side extraction and semantic search.

    Supports Platform API (cloud) and OSS (self-hosted) modes via MEM0_MODE.
    """

    def __init__(self):
        self._config = None
        self._backend = None
        self._mode = "platform"
        self._api_key = ""
        self._host = ""
        self._user_id = _DEFAULT_USER_ID
        self._agent_id = "hermes"
        self._rerank_default = False
        self._channel = "cli"  # gateway channel name (cli/telegram/discord/...)
        self._sync_thread = None
        self._prefetch_thread = None
        self._prefetch_query = ""
        self._prefetch_result = ""
        self._prefetch_done = False
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._breaker_lock = threading.Lock()
        self._sync_lock = threading.Lock()
        self._sync_in_flight = threading.Event()
        self._prefetch_lock = threading.Lock()
        self._atexit_registered = False

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        mode = cfg.get("mode", "platform")
        if mode == "oss":
            return bool(cfg.get("oss", {}).get("vector_store"))
        # Platform needs an api_key; self-hosted needs a host (api_key optional
        # when the server runs with AUTH_DISABLED).
        return bool(cfg.get("api_key") or cfg.get("host"))

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(values)
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self):
        cfg = _load_config()
        mode = cfg.get("mode", "platform")
        api_key_required = mode != "oss"
        return [
            {"key": "api_key", "description": "Mem0 Platform API key", "secret": True, "required": api_key_required, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "host", "description": "Self-hosted Mem0 server URL (leave blank for cloud)", "required": False, "env_var": "MEM0_HOST"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "false", "choices": ["true", "false"]},
        ]

    def post_setup(self, hermes_home: str, config: dict) -> None:
        from ._setup import post_setup
        post_setup(hermes_home, config)

    def _create_backend(self):
        # Lazy-install the mem0 SDK on demand before either backend imports
        # it. ensure() honors security.allow_lazy_installs (default true) and,
        # on a sealed Docker venv, redirects the install to the durable
        # target. On failure we fall through so the import inside the backend
        # produces the canonical error, captured below.
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("memory.mem0", prompt=False)
        except ImportError:
            pass
        except Exception:
            pass
        try:
            if self._mode == "oss":
                from ._backend import OSSBackend
                oss_cfg = self._config.get("oss", {})
                # Path-mode Qdrant takes an EXCLUSIVE .lock per storage folder:
                # only ONE process in the system may open the store, while Hermes
                # runs many concurrent processes (desktop backend, gateway, cron,
                # kanban CLI workers) that all build a backend at session init.
                # A direct in-process client therefore worked for whoever won the
                # race and failed everyone else with "already accessed by another
                # instance". Route through the local broker instead: one long-lived
                # process owns both locks, everyone else speaks loopback HTTP
                # (local installation — no Docker, no mem0 server package: mem0
                # 2.0.x ships none). Falls back to the legacy direct client only
                # if the broker is explicitly disabled or could not be spawned.
                if oss_cfg.get("vector_store", {}).get("config", {}).get("path"):
                    try:
                        from ._local_broker import BROKER_TIMEOUT_SECS, ensure_broker
                        url = ensure_broker(oss_cfg)
                        if url:
                            from ._backend import SelfHostedBackend
                            return SelfHostedBackend(
                                self._api_key, url, timeout=BROKER_TIMEOUT_SECS
                            )
                    except Exception as broker_exc:
                        logger.warning(
                            "mem0 local broker unavailable, falling back to direct path client: %s",
                            broker_exc,
                        )
                return OSSBackend(oss_cfg)
            if self._host:
                from ._backend import SelfHostedBackend
                return SelfHostedBackend(self._api_key, self._host)
            from ._backend import PlatformBackend
            return PlatformBackend(self._api_key)
        except Exception as e:
            logger.error("Mem0 backend failed to initialize (%s mode): %s", self._mode, e)
            self._init_error = str(e)
            return None

    # Minimum gap between lazy re-init attempts in _ensure_backend().
    _RETRY_MIN_INTERVAL_SECS = 30.0

    def _ensure_backend(self) -> bool:
        """Rate-limited self-heal: (re)build the backend after a failed initialize()
        or after an endpoint death dropped it. Session init often failed simply
        because another process held the path lock at that instant — a permanently
        dead session until restart was the old (buggy) outcome; this retries on the
        next tool call instead. Broker spawn+wait is bounded (~10 s worst case) and
        only runs on this user-invoked path, never on the hot prefetch path."""
        if self._backend is not None:
            return True
        now = time.monotonic()
        if now - getattr(self, "_last_reinit_attempt", 0.0) < self._RETRY_MIN_INTERVAL_SECS:
            return False
        self._last_reinit_attempt = now
        backend = self._create_backend()
        if backend is None:
            return False
        self._backend = backend
        if not self._atexit_registered:
            atexit.register(self._shutdown_backend)
            self._atexit_registered = True
        logger.info("Mem0: backend initialized on late retry")
        return True

    def _note_backend_error(self, exc: Exception) -> None:
        """Classify a backend error once (single place for all call sites):
        client errors and broker-busy 503s are ignored for breaker purposes,
        endpoint-connection death drops the backend so the next tool call
        rebuilds it (respawning the broker if it died), everything else counts
        toward the circuit breaker."""
        if _is_client_error(exc) or _is_transient_error(exc):
            return
        if _is_connection_error(exc):
            self._backend = None
        self._record_failure()

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        with self._breaker_lock:
            if self._consecutive_failures < _BREAKER_THRESHOLD:
                return False
            if time.monotonic() >= self._breaker_open_until:
                self._consecutive_failures = 0
                return False
            return True

    def _format_error(self, prefix: str, exc: Exception) -> str:
        msg = f"{prefix}: {exc}"
        if _is_transient_error(exc):
            return (
                msg + " (mem0 store busy: another Hermes process holds the local "
                "Qdrant lock; it activates automatically once the lock frees — "
                "retry shortly)"
            )
        if self._mode == "oss":
            err_str = str(exc).lower()
            if "connection" in err_str or "refused" in err_str or "timeout" in err_str:
                vs = self._config.get("oss", {}).get("vector_store", {})
                msg += f" (check that {vs.get('provider', 'vector store')} is running)"
        return msg

    def _record_success(self):
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self):
        with self._breaker_lock:
            self._consecutive_failures += 1
            count = self._consecutive_failures
            if count >= _BREAKER_THRESHOLD:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            else:
                count = 0
        if count >= _BREAKER_THRESHOLD:
            hint = ""
            if self._mode == "oss":
                vs = self._config.get("oss", {}).get("vector_store", {})
                provider = vs.get("provider", "unknown")
                hint = f" Check that your {provider} vector store is running and reachable."
            logger.warning(
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.%s",
                count, _BREAKER_COOLDOWN_SECS, hint,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._mode = self._config.get("mode", "platform")
        self._api_key = self._config.get("api_key", "")
        self._host = self._config.get("host", "")
        # Resolution order for user_id:
        #   1. Operator-configured MEM0_USER_ID (env or $HERMES_HOME/mem0.json) —
        #      the canonical principal, applied across every gateway so the same
        #      human gets one merged memory store.
        #   2. Gateway-native id from kwargs (Telegram numeric id, Discord
        #      snowflake, etc.) — preserves per-platform isolation when no
        #      override is configured.
        #   3. Hardcoded fallback _DEFAULT_USER_ID (CLI with no auth).
        # The literal _DEFAULT_USER_ID string is treated as unset so users who
        # ran the setup wizard with the suggested default still get gateway-
        # native ids instead of being silently bucketed together.
        configured = self._config.get("user_id")
        if configured == _DEFAULT_USER_ID:
            configured = None
        self._user_id = configured or kwargs.get("user_id") or _DEFAULT_USER_ID
        self._agent_id = self._config.get("agent_id", "hermes")
        # Persisted rerank preference (setup wizard / mem0.json). Used as the
        # DEFAULT for mem0_search when the model doesn't pass ``rerank``
        # explicitly; per-call args still win. Platform-only feature — other
        # backends accept-and-ignore the flag.
        _rr = self._config.get("rerank", False)
        self._rerank_default = (
            _rr.lower() in ("true", "1", "yes") if isinstance(_rr, str) else bool(_rr)
        )
        self._channel = kwargs.get("platform") or "cli"
        # On RE-initialization, close the previous backend FIRST. Overwriting it
        # leaked the old client, and a leaked path-mode Qdrant client keeps its
        # EXCLUSIVE storage lock for the life of the process — every later init
        # in this same process then failed with "already accessed by another
        # instance" (the desktop flapped between success and failure).
        prev = getattr(self, "_backend", None)
        if prev is not None:
            try:
                prev.close()
            except Exception:
                pass
            self._backend = None
        self._backend = self._create_backend()
        if self._backend and not self._atexit_registered:
            atexit.register(self._shutdown_backend)
            self._atexit_registered = True

    def _read_filters(self) -> Dict[str, Any]:
        # Scoped to user_id only — by design — so recall surfaces memories
        # written from any gateway/agent under this principal. Writes attach
        # agent_id (and metadata.channel) so per-agent / per-channel views are
        # still possible at query time when needed; reads default to the wider
        # cross-agent recall.
        return {"user_id": self._user_id}

    def _write_metadata(self) -> Dict[str, Any]:
        # Tag every write with the gateway channel so the dashboard can offer
        # per-channel filtered views without coupling identity to the channel.
        return {"channel": self._channel} if self._channel else {}

    def system_prompt_block(self) -> str:
        # Mirror the precedence in _create_backend (oss > host > platform) so
        # the label always names the backend that actually runs. Checking
        # ``host`` first here would mislabel an ``oss``+``host`` config as
        # self-hosted HTTP even though OSS wins the routing.
        if self._mode == "oss":
            mode_label = "OSS (self-hosted)"
        elif self._host:
            mode_label = "self-hosted (HTTP API)"
        else:
            mode_label = "platform (cloud API)"
        # Rerank is a Mem0 Platform feature only.
        rerank_note = " Rerank is available on search." if (self._mode == "platform" and not self._host) else ""
        return (
            "# Mem0 Memory\n"
            f"Active. Mode: {mode_label}. User: {self._user_id}.\n"
            "You have persistent memory of this user from past conversations. "
            "You should call mem0_search before answering anything that could depend "
            "on prior context (the user's preferences, facts, history, people, "
            "projects, or earlier decisions) — do not rely on the chat window "
            "alone, and do not assume you have no memory.\n"
            "For multi-part or multi-hop questions, run several searches with "
            "different wording/angles and follow-up searches on what the first "
            "results surface; one search is rarely enough. Keep searching until "
            "you have every fact the question needs before you answer.\n"
            "Tools: mem0_search to find memories, mem0_add to store facts, "
            f"mem0_update and mem0_delete to manage by ID.{rerank_note}"
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._start_prefetch(message)

    def _consume_prefetch_result(self, query: str) -> str | None:
        with self._prefetch_lock:
            if self._prefetch_query != query or not self._prefetch_done:
                return None
            result = self._prefetch_result
            self._prefetch_result = ""
            self._prefetch_done = False
            return result

    def _start_prefetch(self, query: str) -> None:
        if not query or self._backend is None or self._is_breaker_open():
            return
        backend = self._backend
        with self._prefetch_lock:
            if self._prefetch_query == query:
                if self._prefetch_done:
                    return
                if self._prefetch_thread and self._prefetch_thread.is_alive():
                    return
            self._prefetch_query = query
            self._prefetch_result = ""
            self._prefetch_done = False

        def _run():
            body = ""
            try:
                results = backend.search(
                    query, filters=self._read_filters(), top_k=10, rerank=False,
                )
                lines = [r.get("memory", "") for r in (results or []) if r.get("memory")]
                if lines:
                    body = "## Mem0 Memory\n" + "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._note_backend_error(e)
                logger.debug("Mem0 prefetch failed: %s", e)
            with self._prefetch_lock:
                if self._prefetch_query == query:
                    self._prefetch_result = body
                    self._prefetch_done = True

        t = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        with self._prefetch_lock:
            self._prefetch_thread = t
        t.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall memories for the CURRENT question with a short hot-path wait."""
        cached = self._consume_prefetch_result(query)
        if cached is not None:
            return cached
        self._start_prefetch(query)
        with self._prefetch_lock:
            thread = self._prefetch_thread if self._prefetch_query == query else None
        if thread:
            thread.join(timeout=_PREFETCH_WAIT_SECS)
        cached = self._consume_prefetch_result(query)
        if cached is not None:
            return cached
        # Slow backend: skip injection; mem0_search tool remains the backstop.
        return ""

    def _sync_max_chars(self) -> int:
        """Effective per-turn message cap: ``sync_max_chars`` from mem0.json, else the default."""
        raw = (self._config or {}).get("sync_max_chars", _SYNC_MSG_MAX_CHARS)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            return _SYNC_MSG_MAX_CHARS
        return cap if cap > 0 else _SYNC_MSG_MAX_CHARS

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for server-side fact extraction (non-blocking).

        Each message is capped at ``_sync_max_chars()`` first: a small-context
        embedding backend rejects the whole extraction otherwise (#106235).
        """
        if self._backend is None or self._is_breaker_open():
            return

        def _sync():
            backend = self._backend
            if backend is None:
                return
            # Signal that the backend is about to be used so shutdown() waits
            # for the extraction+insert to complete instead of closing the
            # Qdrant client out from under us.
            self._sync_in_flight.set()
            try:
                cap = self._sync_max_chars()
                messages = [
                    {"role": "user", "content": _truncate_for_sync(user_content, cap)},
                    {"role": "assistant", "content": _truncate_for_sync(assistant_content, cap)},
                ]
                backend.add(
                    messages,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=True,
                    metadata=self._write_metadata(),
                )
                self._record_success()
            except Exception as e:
                self._note_backend_error(e)
                logger.warning("Mem0 sync failed: %s", e)
            finally:
                self._sync_in_flight.clear()

        with self._sync_lock:
            if self._sync_thread and self._sync_thread.is_alive():
                self._sync_thread.join(timeout=5.0)
            # If still alive after timeout, skip to avoid duplicate ingestion.
            if self._sync_thread and self._sync_thread.is_alive():
                return
            self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
            self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, ADD_SCHEMA, UPDATE_SCHEMA, DELETE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._backend is None and not self._ensure_backend():
            err = getattr(self, "_init_error", "unknown error")
            hint = ""
            if self._mode == "oss":
                vs = self._config.get("oss", {}).get("vector_store", {})
                provider = vs.get("provider", "vector store")
                hint = f" Check that {provider} is running and reachable."
            return json.dumps({"error": f"Mem0 backend not initialized: {err}.{hint}"})

        if self._is_breaker_open():
            msg = "Mem0 temporarily unavailable (multiple consecutive failures). Will retry automatically."
            if self._mode == "oss":
                vs = self._config.get("oss", {}).get("vector_store", {})
                msg += f" Check that your {vs.get('provider', 'vector store')} is running."
            return json.dumps({"error": msg})

        if tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            try:
                top_k = max(1, min(int(args.get("top_k", 10)), 50))
                rerank_raw = args.get("rerank", getattr(self, "_rerank_default", False))
                if isinstance(rerank_raw, str):
                    rerank = rerank_raw.lower() not in ("false", "0", "no")
                else:
                    rerank = bool(rerank_raw)
                results = self._backend.search(query, filters=self._read_filters(), top_k=top_k, rerank=rerank)
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"id": r.get("id"), "memory": r.get("memory", ""),
                          "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._note_backend_error(e)
                return tool_error(self._format_error("Search failed", e))

        elif tool_name == "mem0_add":
            content = args.get("content", "")
            if not content:
                return tool_error("Missing required parameter: content")
            try:
                result = self._backend.add(
                    [{"role": "user", "content": content}],
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=False,
                    metadata=self._write_metadata(),
                )
                self._record_success()
                event_id = result.get("event_id") if isinstance(result, dict) else None
                # Cloud add is async (server-side extraction); OSS and self-hosted store synchronously.
                msg = "Fact stored." if (self._mode == "oss" or self._host) else "Fact queued for storage."
                return json.dumps({"result": msg, "event_id": event_id})
            except Exception as e:
                self._note_backend_error(e)
                return tool_error(self._format_error("Failed to store", e))

        elif tool_name == "mem0_update":
            memory_id = args.get("memory_id", "")
            text = args.get("text", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            if not text:
                return tool_error("Missing required parameter: text")
            try:
                result = self._backend.update(memory_id, text)
                self._record_success()
                return json.dumps(result)
            except Exception as e:
                if _is_client_error(e):
                    return tool_error(f"Memory not found: {memory_id}")
                self._note_backend_error(e)
                return tool_error(self._format_error("Update failed", e))

        elif tool_name == "mem0_delete":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            try:
                result = self._backend.delete(memory_id)
                self._record_success()
                return json.dumps(result)
            except Exception as e:
                if _is_client_error(e):
                    return tool_error(f"Memory not found: {memory_id}")
                self._note_backend_error(e)
                return tool_error(self._format_error("Delete failed", e))

        return tool_error(f"Unknown tool: {tool_name}")

    def _shutdown_backend(self):
        try:
            if self._backend:
                self._backend.close()
                self._backend = None
        except Exception:
            pass

    def shutdown(self) -> None:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5.0)
        sync = self._sync_thread
        if sync and sync.is_alive():
            # Absorb thread-startup latency first: `_sync` may not have set
            # `_sync_in_flight` yet even though it is about to call the backend.
            sync.join(timeout=0.2)
            if sync.is_alive():
                if self._sync_in_flight.is_set():
                    # Extraction is in flight (10-90 s). Wait for it so the
                    # backend (and its Qdrant client) is never closed
                    # mid-`backend.add`; that close is what drops the memory.
                    sync.join(timeout=_SYNC_SHUTDOWN_WAIT_SECS)
                else:
                    # Never entered the backend (or already idle) — short join.
                    sync.join(timeout=5.0)
        self._shutdown_backend()


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())

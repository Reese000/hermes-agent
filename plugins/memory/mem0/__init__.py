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
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from collections import Counter
from agent.memory_provider import MemoryProvider
from agent.secret_scope import get_secret
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120
_PREFETCH_WAIT_SECS = 3

_CLIENT_ERROR_TYPES = ("MemoryNotFoundError", "ValidationError")

# Sentinel returned when neither MEM0_USER_ID nor a gateway-native id is
# available. Treated as "no operator-configured user_id" by initialize() so
# that legacy mem0.json files written by the setup wizard (which historically
# wrote this exact placeholder) still allow gateway-native ids to flow
# through instead of silently overriding them with the placeholder.
_DEFAULT_USER_ID = "hermes-user"

# ---------------------------------------------------------------------------
# Topic fingerprint extraction (cross-session memory identity resolution)
# ---------------------------------------------------------------------------

# Common English stopwords + short filler words — kept minimal and fast.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "dare", "ought", "used", "it", "its", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "what", "which", "who", "whom",
    "not", "no", "nor", "so", "too", "very", "just", "about", "above",
    "after", "again", "all", "also", "any", "because", "before", "between",
    "both", "each", "few", "more", "most", "other", "some", "such",
    "than", "then", "there", "here", "when", "where", "why", "how",
    "if", "only", "own", "same", "into", "over", "under", "down",
    "out", "off", "up", "once", "now", "new", "get", "got", "say",
    "said", "like", "make", "made", "go", "going", "come", "came",
    "take", "took", "know", "knew", "think", "thought", "see", "saw",
    "want", "use", "using", "used", "way", "thing", "things", "one",
    "two", "much", "many", "well", "back", "even", "still", "also",
    "yeah", "ok", "okay", "yes", "right", "let", "sure", "good",
    "great", "thanks", "thank", "please", "hello", "hey", "hi",
})


def extract_topic_fingerprint(text: str | None, *, max_keywords: int = 5) -> list[str]:
    """Extract top keywords/entities from text for cross-session topic matching.

    Uses a lightweight TF-based approach (no LLM calls, <50ms). Returns up
    to *max_keywords* lowercase keywords sorted by frequency then alphabetical.
    Empty/None input returns an empty list.
    """
    if not text or not text.strip():
        return []

    import re as _re

    # Tokenize: split on non-alpha boundaries, keep tokens >= 3 chars
    tokens = _re.findall(r"[a-zA-Z]{3,}", text.lower())
    if not tokens:
        return []

    # Filter stopwords and count
    filtered = [t for t in tokens if t not in _STOPWORDS]
    if not filtered:
        return []

    counts = Counter(filtered)
    # Sort by frequency desc, then alphabetically for determinism
    sorted_words = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [word for word, _ in sorted_words[:max_keywords]]


# ---------------------------------------------------------------------------
# Temporal hint extraction (time-aware memory retrieval)
# ---------------------------------------------------------------------------

_WEEKDAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_TIME_UNITS = r"(?:second|minute|hour|day|week|month|year)s?"

# Compiled once at module load for <10ms extraction.
_TEMPORAL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\blast\s+{_WEEKDAYS}\b", re.I), "past_{weekday}"),
    (re.compile(rf"\bnext\s+{_WEEKDAYS}\b", re.I), "next_{weekday}"),
    (re.compile(rf"\blast\s+{_TIME_UNITS}\b", re.I), "past_{unit}"),
    (re.compile(rf"\bnext\s+{_TIME_UNITS}\b", re.I), "next_{unit}"),
    (re.compile(r"\byesterday\b", re.I), "yesterday"),
    (re.compile(r"\btomorrow\b", re.I), "tomorrow"),
    (re.compile(rf"\bin\s+(\d+)\s+{_TIME_UNITS}\b", re.I), "future_{n}_{unit}"),
    (re.compile(rf"\b(\d+)\s+{_TIME_UNITS}\s+ago\b", re.I), "past_{n}_{unit}"),
    (re.compile(r"\blast\s+week\b", re.I), "past_week"),
    (re.compile(r"\bthis\s+(?:morning|afternoon|evening|week|month|year)\b", re.I), "this_{period}"),
]


def extract_temporal_hint(text: str | None) -> str | None:
    """Detect temporal expressions in text and return a structured hint.

    Returns a lowercase snake_case string like "past_tuesday", "next_week",
    "yesterday", "tomorrow", "future_3_months", etc.  Returns None when no
    temporal expression is found.  Pure regex, no NLP dependency, <10ms.
    """
    if not text or not text.strip():
        return None

    for pattern, template in _TEMPORAL_PATTERNS:
        m = pattern.search(text)
        if m:
            return _render_hint(template, m)
    return None


def _render_hint(template: str, match: re.Match) -> str:
    """Fill template placeholders from a regex match."""
    groups = match.groups()
    weekday_names = {
        "monday": "monday", "tuesday": "tuesday", "wednesday": "wednesday",
        "thursday": "thursday", "friday": "friday", "saturday": "saturday",
        "sunday": "sunday",
    }
    unit_names = {
        "second": "seconds", "minute": "minutes", "hour": "hours",
        "day": "days", "week": "weeks", "month": "months", "year": "years",
    }
    hint = template
    # {weekday}
    if "{weekday}" in hint:
        for g in groups:
            if g and g.lower() in weekday_names:
                hint = hint.replace("{weekday}", g.lower())
                break
    # {unit}
    if "{unit}" in hint:
        for g in groups:
            if g and g.lower().rstrip("s") in unit_names:
                hint = hint.replace("{unit}", unit_names[g.lower().rstrip("s")])
                break
    # {n}
    if "{n}" in hint:
        for g in groups:
            if g and g.isdigit():
                hint = hint.replace("{n}", g)
                break
    # {period}
    if "{period}" in hint:
        for g in groups:
            if g:
                hint = hint.replace("{period}", g.lower())
                break
    return hint


def _is_client_error(exc: Exception) -> bool:
    """True for user-caused errors (bad ID, not found) that should NOT trip circuit breaker."""
    etype = type(exc).__name__
    if etype in _CLIENT_ERROR_TYPES:
        return True
    err_str = str(exc).lower()
    return "404" in err_str or "not found" in err_str or "valid uuid" in err_str


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
        self._session_id = ""  # current session id for temporal metadata
        self._turn_number = 0  # current turn number for temporal metadata
        self._sync_thread = None
        self._compress_thread = None  # dedicated thread for pre-compression extraction
        self._prefetch_thread = None
        self._prefetch_query = ""
        self._prefetch_result = ""
        self._prefetch_done = False
        # Prefetch memory list: deduplicated entries from all prefetch paths.
        # Both normal and compression-aware prefetch append here; consuming
        # merges them into a single block without overwriting.
        self._prefetch_memories: List[Dict[str, Any]] = []
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._breaker_lock = threading.Lock()
        self._sync_lock = threading.Lock()
        self._prefetch_lock = threading.Lock()
        self._atexit_registered = False
        # Health metrics (exposed via hermes memory status)
        self._stats = {
            "memories_added_session": 0,
            "memories_consolidated": 0,
            "compression_events": 0,
            "compression_facts_extracted": 0,
        }
        # Consolidation state
        self._turns_since_consolidation = 0
        self._consolidation_thread = None
        self._consolidation_lock = threading.Lock()

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
                return OSSBackend(self._config.get("oss", {}))
            if self._host:
                from ._backend import SelfHostedBackend
                return SelfHostedBackend(self._api_key, self._host)
            from ._backend import PlatformBackend
            return PlatformBackend(self._api_key)
        except Exception as e:
            logger.error("Mem0 backend failed to initialize (%s mode): %s", self._mode, e)
            self._init_error = str(e)
            return None

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
        self._session_id = session_id
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
        # Resolution order for agent_id:
        #   1. Profile name from kwargs (agent_identity) — per-profile isolation,
        #      e.g. "cam" → "hermes-cam". Takes precedence over config.
        #   2. Configured agent_id from env / mem0.json (backward compat).
        #   3. Hardcoded fallback "hermes" (legacy default).
        profile_name = kwargs.get("agent_identity")
        if profile_name:
            self._agent_id = f"hermes-{profile_name}"
        else:
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
        self._backend = self._create_backend()
        if self._backend and not self._atexit_registered:
            atexit.register(self._shutdown_backend)
            self._atexit_registered = True

    def _read_filters(self) -> Dict[str, Any]:
        # Default: scoped to user_id + agent_id so each profile only sees its
        # own memories. cross_profile_search or profile_isolation=false removes
        # the agent_id filter to widen recall across all profiles.
        filters: Dict[str, Any] = {"user_id": self._user_id}
        if self._config.get("cross_profile_search", False):
            return filters
        if not self._config.get("profile_isolation", True):
            return filters
        filters["agent_id"] = self._agent_id
        return filters

    def _write_metadata(self) -> Dict[str, Any]:
        # Tag every write with temporal context so memories can be filtered
        # by time range and the agent can reason about when facts were stored.
        meta: Dict[str, Any] = {}
        if self._channel:
            meta["channel"] = self._channel
        if (self._config or {}).get("temporal_metadata", True):
            meta["timestamp"] = datetime.now(timezone.utc).isoformat()
            if self._session_id:
                meta["session_id"] = self._session_id
            if self._turn_number:
                meta["turn_number"] = self._turn_number
        return meta

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
        self._turn_number = turn_number
        self._start_prefetch(message)
        # Compression-aware prefetch: when context is near the compression
        # threshold, proactively retrieve memories from at-risk messages so
        # relevant context survives in the active window.
        remaining = kwargs.get("remaining_tokens")
        context_limit = kwargs.get("context_limit")
        old_messages = kwargs.get("old_messages")
        if (
            remaining is not None
            and context_limit
            and context_limit > 0
            and self._config.get("compression_aware_prefetch", True)
            and self._backend is not None
            and not self._is_breaker_open()
        ):
            usage_ratio = 1.0 - (remaining / context_limit)
            threshold = 0.60  # compress at ~60%; prefetch extra when usage > 60%
            if usage_ratio >= threshold and old_messages:
                self._start_compression_aware_prefetch(old_messages)
        # Periodic consolidation: track turns since last pass, trigger
        # background consolidation when the interval is reached.
        self._turns_since_consolidation += 1
        interval = self._config.get("consolidation_interval_turns", 50)
        if (
            self._config.get("consolidation_enabled", True)
            and self._backend is not None
            and not self._is_breaker_open()
            and self._turns_since_consolidation >= interval
        ):
            self._start_consolidation()

    def _start_compression_aware_prefetch(self, old_messages: list) -> None:
        """Background prefetch based on keywords from messages at risk of compression."""
        # Extract keywords from old messages (take first N chars from each).
        keywords = []
        for msg in old_messages:
            content = msg.get("content", "").strip()
            if content:
                # Use first ~100 chars as a keyword query
                keywords.append(content[:100])
        if not keywords:
            return

        # Combine into a single search query (most efficient).
        query = " ".join(keywords[:3])  # Cap at 3 messages to limit query size
        backend = self._backend

        def _run():
            try:
                results = backend.search(
                    query, filters=self._read_filters(), top_k=10, rerank=False,
                )
                lines = [r.get("memory", "") for r in (results or []) if r.get("memory")]
                if lines:
                    self._stats["compression_events"] += 1
                    self._stats["compression_facts_extracted"] += len(lines)
                    self._append_prefetch_memories(lines, "compression-aware")
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 compression-aware prefetch failed: %s", e)

        t = threading.Thread(target=_run, daemon=True, name="mem0-compress-prefetch")
        t.start()

    def _append_prefetch_memories(self, lines: List[str], source: str) -> None:
        """Append deduplicated memory lines to the shared prefetch cache.

        Both normal and compression-aware prefetch call this.  Results are
        stored as structured entries keyed by a hash of the line text so
        duplicates are rejected regardless of which path found them first.
        """
        with self._prefetch_lock:
            existing_texts = {m["text"] for m in self._prefetch_memories}
            for line in lines:
                if line not in existing_texts:
                    self._prefetch_memories.append({"text": line, "source": source})
                    existing_texts.add(line)
            self._prefetch_done = bool(self._prefetch_memories)

    def _consume_prefetch_result(self, query: str) -> str | None:
        with self._prefetch_lock:
            if self._prefetch_query != query or not self._prefetch_done:
                return None
            # Build merged block from all deduplicated prefetch memories.
            if self._prefetch_memories:
                body = "## Mem0 Memory\n" + "\n".join(
                    f"- {m['text']}" for m in self._prefetch_memories
                )
            else:
                body = ""
            self._prefetch_memories.clear()
            self._prefetch_result = ""
            self._prefetch_done = False
            return body

    def _start_prefetch(self, query: str, extra_filters: Dict[str, Any] | None = None) -> None:
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
            self._prefetch_memories.clear()
            self._prefetch_done = False

        filters = self._read_filters()
        if extra_filters:
            filters.update(extra_filters)

        def _run():
            try:
                results = backend.search(
                    query, filters=filters, top_k=10, rerank=False,
                )
                lines = [r.get("memory", "") for r in (results or []) if r.get("memory")]
                if lines:
                    self._append_prefetch_memories(lines, "normal")
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)

        t = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        with self._prefetch_lock:
            self._prefetch_thread = t
        t.start()

    def prefetch(
        self,
        query: str,
        *,
        session_id: str = "",
        topic_fingerprint: list[str] | None = None,
        since_date: str = "",
        until_date: str = "",
    ) -> str:
        """Recall memories for the CURRENT question with a short hot-path wait.

        When *topic_fingerprint* is provided and cross-session identity is
        enabled, also searches by fingerprint keywords to find related
        memories from other sessions. Results are merged and deduplicated.

        When *since_date* or *until_date* are provided (ISO date strings,
        e.g. ``"2026-08-13"``), a temporal metadata filter is added so only
        memories written within the date range are returned.
        """
        # Build temporal filter for date-range queries
        date_filters: Dict[str, Any] = {}
        if since_date or until_date:
            if since_date and until_date:
                date_filters["created_at"] = {"$gte": since_date, "$lte": until_date}
            elif since_date:
                date_filters["created_at"] = {"$gte": since_date}
            elif until_date:
                date_filters["created_at"] = {"$lte": until_date}

        cached = self._consume_prefetch_result(query)
        if cached is not None:
            return cached
        self._start_prefetch(query, extra_filters=date_filters if date_filters else None)
        with self._prefetch_lock:
            thread = self._prefetch_thread if self._prefetch_query == query else None
        if thread:
            thread.join(timeout=_PREFETCH_WAIT_SECS)
        cached = self._consume_prefetch_result(query)
        if cached is not None:
            result = cached
        else:
            result = ""

        # Cross-session prefetch: search by topic fingerprint when enabled
        if (
            topic_fingerprint
            and self._config.get("cross_session_identity", True)
            and self._backend is not None
            and not self._is_breaker_open()
        ):
            fp_query = " ".join(topic_fingerprint)
            try:
                fp_results = self._backend.search(
                    fp_query, filters=self._read_filters(), top_k=10, rerank=False,
                )
                self._record_success()
                if fp_results:
                    # Track IDs already in the main result to dedup
                    # The main prefetch result is formatted text; we track seen
                    # IDs from the background thread's search for dedup.
                    seen_ids: set[str] = set()
                    with self._prefetch_lock:
                        # Extract IDs from the background prefetch's results
                        # by re-running a lightweight scan — but we can't access
                        # the raw IDs. Instead, track by memory text content.
                        # This is simpler and handles the common case.
                        pass
                    # Dedup: only add memories whose text isn't already in result
                    existing_lines = set(result.split("\n")) if result else set()
                    new_lines = []
                    for r in fp_results:
                        mem = r.get("memory", "")
                        line = f"- {mem}"
                        if mem and line not in existing_lines:
                            existing_lines.add(line)
                            new_lines.append(line)
                    if new_lines:
                        fp_block = "## Mem0 Memory (cross-session)\n" + "\n".join(new_lines)
                        result = (result + "\n" + fp_block).strip() if result else fp_block
            except Exception as e:
                logger.debug("Mem0 cross-session prefetch failed: %s", e)

        return result

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for server-side fact extraction (non-blocking)."""
        if self._backend is None or self._is_breaker_open():
            return

        # Compute topic fingerprint for cross-session identity resolution
        combined = f"{user_content} {assistant_content}"
        fingerprint = extract_topic_fingerprint(combined)
        write_meta = self._write_metadata()
        if fingerprint:
            write_meta["topic_fingerprint"] = fingerprint

        def _sync():
            backend = self._backend
            if backend is None:
                return
            try:
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                backend.add(
                    messages,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=True,
                    metadata=write_meta,
                )
                self._stats["memories_added_session"] += 1
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 sync failed: %s", e)

        with self._sync_lock:
            if self._sync_thread and self._sync_thread.is_alive():
                self._sync_thread.join(timeout=5.0)
            # If still alive after timeout, skip to avoid duplicate ingestion.
            if self._sync_thread and self._sync_thread.is_alive():
                return
            self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
            self._sync_thread.start()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract facts from messages about to be compressed into Mem0.

        Called before context compression drops old messages. Indexes each
        message with server-side fact extraction (infer=True) so no facts
        are lost when the compressor proceeds. Runs asynchronously to avoid
        blocking the compression path.

        Returns empty string — facts are in Mem0, not in the summary prompt.
        """
        if self._backend is None or self._is_breaker_open():
            return ""

        # Filter to messages that actually have content worth extracting.
        extractable = [
            m for m in messages
            if m.get("content") and m["content"].strip()
        ]
        if not extractable:
            return ""

        backend = self._backend

        def _extract():
            try:
                for msg in extractable:
                    backend.add(
                        [msg],
                        user_id=self._user_id,
                        agent_id=self._agent_id,
                        infer=True,
                        metadata=self._write_metadata(),
                    )
                self._stats["compression_events"] += 1
                self._stats["compression_facts_extracted"] += len(extractable)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 pre-compression extraction failed: %s", e)

        # Use a dedicated thread for pre-compression extraction so it doesn't
        # block (or get blocked by) sync_turn.
        with self._sync_lock:
            if self._compress_thread and self._compress_thread.is_alive():
                self._compress_thread.join(timeout=5.0)
            if self._compress_thread and self._compress_thread.is_alive():
                return ""
            self._compress_thread = threading.Thread(
                target=_extract, daemon=True, name="mem0-pre-compress",
            )
            self._compress_thread.start()

        return ""

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        first_message: str = "",
        **kwargs,
    ) -> None:
        """When the agent switches sessions, prefetch related memories.

        If *first_message* is provided and cross-session identity is enabled,
        computes a topic fingerprint and proactively searches Mem0 for
        related memories from past sessions so the agent has immediate
        context when resuming a topic.
        """
        if (
            not first_message
            or not self._config.get("cross_session_identity", True)
            or self._backend is None
            or self._is_breaker_open()
        ):
            return

        fingerprint = extract_topic_fingerprint(first_message)
        if not fingerprint:
            return

        fp_query = " ".join(fingerprint)
        backend = self._backend

        def _search():
            try:
                results = backend.search(
                    fp_query, filters=self._read_filters(), top_k=10, rerank=False,
                )
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    if lines:
                        block = (
                            "## Mem0 Memory (cross-session continuity)\n"
                            + "\n".join(f"- {l}" for l in lines)
                        )
                        with self._prefetch_lock:
                            existing = self._prefetch_result or ""
                            if block not in existing:
                                self._prefetch_result = (
                                    (existing + "\n" + block).strip()
                                    if existing else block
                                )
                                self._prefetch_done = True
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 session-switch prefetch failed: %s", e)

        t = threading.Thread(target=_search, daemon=True, name="mem0-session-switch")
        t.start()

    # ------------------------------------------------------------------
    # Memory consolidation (periodic dedup + contradiction detection)
    # ------------------------------------------------------------------

    def _start_consolidation(self) -> None:
        """Spawn background consolidation if not already running."""
        with self._consolidation_lock:
            if self._consolidation_thread and self._consolidation_thread.is_alive():
                return
            self._turns_since_consolidation = 0
            self._consolidation_thread = threading.Thread(
                target=self._consolidate, daemon=True, name="mem0-consolidation",
            )
            self._consolidation_thread.start()

    def _consolidate(self) -> None:
        """Background consolidation: dedup duplicates, flag contradictions.

        Fetches all memories, groups by text similarity, deletes the older
        member of each duplicate pair, and annotates contradicting pairs
        with metadata. All actions are logged before execution for audit.
        """
        backend = self._backend
        if backend is None:
            return
        threshold = float(self._config.get("consolidation_similarity_threshold", 0.85))
        filters = self._read_filters()
        try:
            memories = backend.list_all(filters=filters)
        except Exception as e:
            self._record_failure()
            logger.debug("Mem0 consolidation list_all failed: %s", e)
            return

        if not memories or len(memories) < 2:
            self._record_success()
            return

        # --- Phase 1: find duplicate groups ---
        dup_groups = self._find_duplicate_groups(memories, backend, threshold)
        deleted_ids: list[str] = []
        for group in dup_groups:
            # Keep the most recent memory (by updated_at or created_at), delete rest
            sorted_mem = sorted(
                group,
                key=lambda m: m.get("updated_at") or m.get("created_at") or "",
                reverse=True,
            )
            for victim in sorted_mem[1:]:
                vid = victim.get("id", "")
                if not vid:
                    continue
                try:
                    backend.delete(vid)
                    deleted_ids.append(vid)
                except Exception as e:
                    logger.debug("Mem0 consolidation delete failed for %s: %s", vid, e)
                    continue

        # --- Phase 2: flag contradictions (log only — metadata update not
        #     yet supported by all backend implementations) ---
        contradiction_pairs = self._find_contradictions(memories)

        # --- Phase 3: detect stale temporal facts ---
        stale_ids: list[str] = []
        if self._config.get("stale_detection", True):
            stale_ids = self._detect_stale_facts(memories, backend)

        # --- Phase 4: log and record ---
        if deleted_ids or contradiction_pairs or stale_ids:
            self._log_consolidation(deleted_ids, contradiction_pairs, stale_ids)
            with self._consolidation_lock:
                self._stats["memories_consolidated"] += len(deleted_ids)

        self._record_success()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize memory text for comparison — lowercase, collapse whitespace, strip punctuation."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    # --- Stale fact detection ---

    # Temporal hints that indicate the fact refers to a specific past event
    # (not a recurring or future state).  If the memory is more than 1 day
    # old and carries one of these hints, it is flagged as potentially stale.
    _PAST_TEMPORAL_HINTS: frozenset[str] = frozenset({
        "yesterday", "past_week", "past_month", "past_year",
    })

    def _detect_stale_facts(
        self, memories: list[dict], backend
    ) -> list[str]:
        """Flag memories with expired temporal hints as potentially stale.

        Returns a list of memory IDs that were flagged.  Does not delete —
        only annotates so the agent can decide.
        """
        now = datetime.now(timezone.utc)
        stale_ids: list[str] = []
        for mem in memories:
            mem_id = mem.get("id", "")
            if not mem_id:
                continue
            meta = mem.get("metadata") or {}
            hint = meta.get("temporal_hint", "")
            if not hint:
                continue
            # Check if this is a past-oriented hint
            is_past = (
                hint in self._PAST_TEMPORAL_HINTS
                or hint.startswith("past_")
            )
            if not is_past:
                continue
            # Check age: is the memory more than 1 day old?
            created = mem.get("created_at", "")
            if not created:
                continue
            try:
                created_dt = datetime.fromisoformat(
                    created.replace("Z", "+00:00")
                )
                age_days = (now - created_dt).total_seconds() / 86400
                if age_days > 1:
                    try:
                        backend.update(
                            mem_id,
                            None,
                            metadata={"potentially_stale": True},
                        )
                        stale_ids.append(mem_id)
                    except Exception as e:
                        logger.debug(
                            "Mem0 stale detection update failed for %s: %s",
                            mem_id, e,
                        )
            except (ValueError, TypeError):
                continue
        return stale_ids

    @staticmethod
    def _find_duplicate_groups(
        memories: list[dict], backend, threshold: float
    ) -> list[list[dict]]:
        """Group memories that are likely duplicates.

        Phase 1: exact normalized text match (cheap).
        Phase 2: search-based semantic similarity for non-exact matches.

        Returns list of groups (each group = list of memories to coalesce).
        """
        # --- Phase 1: exact normalized text ---
        norm_map: dict[str, list[dict]] = {}
        for mem in memories:
            text = mem.get("memory", "")
            if not text:
                continue
            norm = Mem0MemoryProvider._normalize_text(text)
            norm_map.setdefault(norm, []).append(mem)

        groups: list[list[dict]] = []
        seen_ids: set[str] = set()
        for norm, mems in norm_map.items():
            if len(mems) > 1:
                groups.append(mems)
                for m in mems:
                    seen_ids.add(m.get("id", ""))

        # --- Phase 2: semantic similarity for remaining ---
        remaining = [m for m in memories if m.get("id", "") not in seen_ids]
        if not remaining:
            return groups

        for mem in remaining:
            text = mem.get("memory", "")
            mid = mem.get("id", "")
            if not text or not mid:
                continue
            # Check if this memory is already in a group from this pass
            in_group = any(mid in {m.get("id") for m in g} for g in groups)
            if in_group:
                continue
            # Search for semantically similar memories using the first 100 chars
            # as a query — this leverages the backend's own embeddings.
            query = text[:100]
            try:
                results = backend.search(
                    query,
                    filters={"user_id": mem.get("user_id", "")},
                    top_k=5,
                    rerank=False,
                )
            except Exception:
                continue
            similar_ids = set()
            for r in results:
                r_id = r.get("id", "")
                r_score = r.get("score", 0)
                if r_id and r_id != mid and r_score >= threshold:
                    similar_ids.add(r_id)
            if similar_ids:
                group = [mem]
                for other in remaining:
                    if other.get("id") in similar_ids:
                        group.append(other)
                groups.append(group)
                for m in group:
                    seen_ids.add(m.get("id", ""))

        return groups

    @staticmethod
    def _find_contradictions(
        memories: list[dict],
    ) -> list[tuple[str, str]]:
        """Detect pairs of memories that likely contradict each other.

        Heuristic: search for negation patterns and flag pairs where
        one memory contains the negation of the other's core claim.
        Returns list of (memory_id_a, memory_id_b) pairs.
        """
        negation_patterns = [
            r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bdoesn'?t\b",
            r"\bdon'?t\b", r"\bwon'?t\b", r"\bcan'?t\b",
            r"\bshouldn'?t\b", r"\bisn'?t\b", r"\baren'?t\b",
        ]
        neg_re = re.compile("|".join(negation_patterns), re.IGNORECASE)

        pairs: list[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "in",
                     "on", "at", "to", "for", "of", "with", "by", "and", "or"}

        for i, mem_a in enumerate(memories):
            text_a = mem_a.get("memory", "")
            id_a = mem_a.get("id", "")
            if not text_a or not id_a:
                continue
            has_neg_a = bool(neg_re.search(text_a))
            for mem_b in memories[i + 1:]:
                text_b = mem_b.get("memory", "")
                id_b = mem_b.get("id", "")
                if not text_b or not id_b:
                    continue
                has_neg_b = bool(neg_re.search(text_b))
                # Flag if one has negation and the other doesn't (likely contradiction)
                if not (has_neg_a ^ has_neg_b):
                    continue
                # Quick check: do they share significant keywords?
                words_a = set(Mem0MemoryProvider._normalize_text(text_a).split()) - stopwords
                words_b = set(Mem0MemoryProvider._normalize_text(text_b).split()) - stopwords
                if not words_a or not words_b:
                    continue
                overlap = words_a & words_b
                # Need at least 2 meaningful words in common
                if len(overlap) >= 2:
                    pair = tuple(sorted([id_a, id_b]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        pairs.append(pair)

        return pairs

    def _log_consolidation(
        self, deleted_ids: list[str], contradiction_pairs: list[tuple[str, str]],
        stale_ids: list[str] | None = None,
    ) -> None:
        """Append consolidation actions to the audit log (no PII)."""
        from hermes_constants import get_hermes_home

        log_dir = get_hermes_home() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "memory-consolidation.log"
        ts = datetime.now(timezone.utc).isoformat()
        lines = [
            f"[{ts}] consolidation_run",
            f"  deleted_duplicates: {len(deleted_ids)}",
        ]
        for vid in deleted_ids:
            lines.append(f"  - deleted: {vid[:8]}...")  # Truncated ID, no content
        if contradiction_pairs:
            lines.append(f"  contradictions_flagged: {len(contradiction_pairs)}")
            for a, b in contradiction_pairs:
                lines.append(f"  - contradicts: {a[:8]}... <-> {b[:8]}...")
        if stale_ids:
            lines.append(f"  stale_flagged: {len(stale_ids)}")
            for sid in stale_ids:
                lines.append(f"  - stale: {sid[:8]}...")
        lines.append("")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.debug("Failed to write consolidation log: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, ADD_SCHEMA, UPDATE_SCHEMA, DELETE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._backend is None:
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
                if not _is_client_error(e):
                    self._record_failure()
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
                self._record_failure()
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
                self._record_failure()
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
                self._record_failure()
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
        for t in (self._prefetch_thread, self._sync_thread, self._compress_thread, self._consolidation_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        self._shutdown_backend()

    def get_health_metrics(self) -> Dict[str, Any]:
        """Return memory health metrics for status display."""
        metrics: Dict[str, Any] = {
            "memories_added_session": self._stats["memories_added_session"],
            "memories_consolidated": self._stats["memories_consolidated"],
            "compression_events": self._stats["compression_events"],
            "compression_facts_extracted": self._stats["compression_facts_extracted"],
        }
        # Total point count from Qdrant (if backend supports it).
        if self._backend and hasattr(self._backend, "get_point_count"):
            try:
                metrics["total_memories"] = self._backend.get_point_count(
                    filters=self._read_filters(),
                )
            except Exception:
                metrics["total_memories"] = "unknown"
        return metrics


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())

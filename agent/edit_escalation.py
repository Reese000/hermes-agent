"""Edit-format escalation on repeated failures (W4).

When an edit fails to apply, the least useful thing a model can do is resend
the same edit in the same format.  This module decides what to tell it
instead: on the first failure, how to fix the current attempt; on repeated
failures against the same file, switch to a different edit format entirely.

Why the escalation is keyed on the file
---------------------------------------
A model that cannot land an edit on one file is usually working from a stale
or mis-remembered copy of *that* file.  Counting failures globally would
escalate on unrelated files and punish a model having one bad edit in a long
session; counting per file targets the actual failure mode.

Why the scope is passed in
--------------------------
State is keyed on ``(scope, path)``, where the scope is the caller's agent
identity - ``task_id`` at the tool boundary.  One process can run several
agents at once (the Gateway, delegated child agents), and a module-level or
thread-local counter would let one agent's failures escalate another's
edits.  The caller knows who it is; this module does not guess.

Where the per-model chain lives
-------------------------------
:class:`~agent.harness_profiles.HarnessProfile` carries a per-family
``retry_format_chain``, and it is consumed in the *system prompt*
(:func:`escalation_brief_line`), not here.  The tool layer has no reliable
handle on the active profile - the same concurrency argument as above.  So
the split is:

  * **prompt** - the model is told its family's fallback order up front,
    once, in a block fixed for the session and therefore cache-safe;
  * **tool layer** - at failure time the hint is derived from *what actually
    failed*, which the tool layer genuinely knows.

The two agree because :func:`resolve_chain` derives the default chain from
the same ``edit_format`` that picks the prompt's edit-format line.

State is per-process and advisory.  Losing it degrades to first-failure
guidance, which is never wrong, only less pointed.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Optional, Tuple

#: Recognised edit formats.
REPLACE = "replace"
PATCH = "patch"
WRITE_FILE = "write_file"

#: What to try after *this* format fails repeatedly.  ``write_file`` is the
#: terminal fallback: it cannot fail to match, because it does not match.
_ESCALATION: Dict[str, str] = {
    REPLACE: PATCH,
    PATCH: REPLACE,
    WRITE_FILE: WRITE_FILE,
}

#: Failures against one file before advising a format switch.  The first
#: failure gets targeted advice (re-read, widen context); switching format
#: immediately would be premature, since most first failures are a stale
#: snippet rather than a wrong format.
ESCALATE_AFTER = 2

#: Distinct failing files tracked per scope.  A long session that fails
#: across many files must not grow this without bound; 64 is generous, and
#: eviction only costs an old counter its streak.
MAX_TRACKED_PATHS_PER_SCOPE = 64

_lock = threading.Lock()
_failures: Dict[str, Dict[str, int]] = {}


def _norm(path: str) -> str:
    """Normalise a path so two spellings of one file share a counter."""
    try:
        return os.path.normcase(os.path.abspath(path))
    except (OSError, ValueError, TypeError):
        return str(path)


def _scope_key(scope: Optional[str]) -> str:
    return str(scope or "default")


def record_failure(scope: Optional[str], path: str) -> int:
    """Record a failed edit against *path*; return the new consecutive count."""
    if not path:
        return 0
    key = _norm(path)
    with _lock:
        bucket = _failures.setdefault(_scope_key(scope), {})
        if len(bucket) >= MAX_TRACKED_PATHS_PER_SCOPE and key not in bucket:
            try:
                del bucket[next(iter(bucket))]
            except StopIteration:
                pass
        bucket[key] = bucket.get(key, 0) + 1
        return bucket[key]


def record_success(scope: Optional[str], path: str) -> None:
    """Clear the failure streak for *path* - the model got an edit to apply."""
    if not path:
        return
    with _lock:
        bucket = _failures.get(_scope_key(scope))
        if bucket:
            bucket.pop(_norm(path), None)


def failure_count(scope: Optional[str], path: str) -> int:
    """Current consecutive failure count for *path* under *scope*."""
    if not path:
        return 0
    with _lock:
        return _failures.get(_scope_key(scope), {}).get(_norm(path), 0)


def reset(scope: Optional[str] = None) -> None:
    """Drop tracked failures for *scope*, or for every scope when ``None``."""
    with _lock:
        if scope is None:
            _failures.clear()
        else:
            _failures.pop(_scope_key(scope), None)


def next_format(current: str) -> str:
    """The format to escalate to after *current* keeps failing."""
    return _ESCALATION.get((current or "").strip().lower(), WRITE_FILE)


def escalation_hint(
    scope: Optional[str],
    path: str,
    current_format: str,
    *,
    count: Optional[int] = None,
) -> str:
    """Guidance for a model whose edit just failed.

    Returns ``""`` when there is nothing useful to add.  The text is plain
    and imperative because it is read by a model mid-retry, not by a human.
    """
    if not path:
        return ""

    attempts = failure_count(scope, path) if count is None else count
    if attempts <= 0:
        return ""

    fmt = (current_format or "").strip().lower()

    if attempts < ESCALATE_AFTER:
        # First failure: the format is probably fine, the snippet is stale.
        # Name the tools explicitly - this text replaced an older hint that
        # did, and "re-read the file" alone left the model to guess how.
        return (
            "Before retrying: re-read the file with read_file to verify its "
            "current content, or search_files to locate the text. Do not "
            "resend the same edit - what you matched against is not what is "
            "on disk."
        )

    nxt = next_format(fmt)
    if nxt == WRITE_FILE or fmt == WRITE_FILE:
        return (
            f"This is failure {attempts} on this file with "
            f"'{fmt or 'this format'}'. Stop retrying it. Re-read the file, "
            "then rewrite the whole region with write_file."
        )
    return (
        f"This is failure {attempts} on this file with '{fmt}'. Stop retrying "
        f"'{fmt}' - switch edit format: re-read the file, then use '{nxt}' "
        "instead. If that also fails, rewrite the region with write_file."
    )


# ---------------------------------------------------------------------------
# Per-model chain (consumed in the system prompt, not in the tool layer)
# ---------------------------------------------------------------------------

#: Fallback order when a profile does not pin one, keyed by its
#: ``edit_format``.  Always terminates in ``write_file``.
_DEFAULT_CHAINS: Dict[str, Tuple[str, ...]] = {
    REPLACE: (REPLACE, PATCH, WRITE_FILE),
    PATCH: (PATCH, REPLACE, WRITE_FILE),
}


def resolve_chain(profile) -> Tuple[str, ...]:
    """The edit-format fallback order for *profile*.

    Honours an explicit ``retry_format_chain``; otherwise derives one from
    the profile's ``edit_format`` so the chain always opens with the format
    the prompt already told the model to prefer.  Returns ``()`` when the
    profile is unusable, which callers treat as "say nothing".
    """
    pinned = getattr(profile, "retry_format_chain", None)
    if pinned:
        cleaned = tuple(str(f).strip().lower() for f in pinned if str(f).strip())
        if cleaned:
            return cleaned
    fmt = str(getattr(profile, "edit_format", "") or "").strip().lower()
    return _DEFAULT_CHAINS.get(fmt, ())


def escalation_brief_line(profile) -> str:
    """One coding-brief line naming the family's edit-format fallback order.

    Fixed for the session (it depends only on the resolved profile), so it
    is safe to fold into the cached system prompt.
    """
    chain = resolve_chain(profile)
    if len(chain) < 2:
        return ""
    ordered = " -> ".join(chain)
    return (
        "If an edit fails to apply, do not resend it unchanged: re-read the "
        f"file, then fall back through {ordered}."
    )

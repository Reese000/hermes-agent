"""Edit-format escalation on repeated failures (W4).

When an edit fails to apply, the least useful thing a model can do is resend
the same edit in the same format.  This module decides what to tell it
instead: on the first failure, how to fix the current attempt; on repeated
failures against the same file, switch to a different edit format entirely.

Why the escalation is keyed on the file
---------------------------------------
A model that cannot land a ``replace`` on one file is usually working from a
stale or mis-remembered copy of *that* file.  Counting failures globally
would escalate on unrelated files and punish a model having one bad edit in
a long session; counting per file targets the actual failure mode.

Where the per-model chain lives
-------------------------------
:class:`~agent.harness_profiles.HarnessProfile` carries a per-family
``retry_format_chain``, and it is consumed in the *system prompt*
(:func:`escalation_brief_line`), not here.  The tool layer has no reliable
handle on the active profile: one process can run several agents on
different models at once (the Gateway, delegated child agents), so a
module-level "current profile" would hand one agent another's chain - the
same class of bug as a cache key missing the model.  So the split is:

  * **prompt** - the model is told its family's fallback order up front,
    once, in a block that is fixed for the session and therefore cache-safe;
  * **tool layer** - at failure time the hint is derived from *what actually
    failed*, which is information the tool layer genuinely has.

The two agree because :func:`resolve_chain` derives the default chain from
the same ``edit_format`` that picks the prompt's edit-format line.

State is per-process and advisory.  Losing it degrades to first-failure
guidance, which is never wrong, only less pointed.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Optional, Tuple

#: Recognised edit formats, in the order a model should fall back through.
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

_lock = threading.Lock()
_failures: Dict[Tuple[int, str], int] = {}


def _key(path: str) -> Tuple[int, str]:
    """Scope failure counts to the calling thread and normalised path.

    Delegated agents run on their own threads, so thread scoping keeps one
    agent's failures from escalating another's edits.
    """
    try:
        normalised = os.path.normcase(os.path.abspath(path))
    except (OSError, ValueError, TypeError):
        normalised = str(path)
    return (threading.get_ident(), normalised)


def record_failure(path: str) -> int:
    """Record a failed edit against *path*; return the new consecutive count."""
    if not path:
        return 0
    with _lock:
        k = _key(path)
        count = _failures.get(k, 0) + 1
        _failures[k] = count
        return count


def record_success(path: str) -> None:
    """Clear the failure streak for *path* - the model got it to apply."""
    if not path:
        return
    with _lock:
        _failures.pop(_key(path), None)


def failure_count(path: str) -> int:
    """Current consecutive failure count for *path*."""
    if not path:
        return 0
    with _lock:
        return _failures.get(_key(path), 0)


def reset() -> None:
    """Drop all tracked failures (call at turn or session boundary)."""
    with _lock:
        _failures.clear()


def next_format(current: str) -> str:
    """The format to escalate to after *current* keeps failing."""
    return _ESCALATION.get((current or "").strip().lower(), WRITE_FILE)


def escalation_hint(
    path: str,
    current_format: str,
    *,
    count: Optional[int] = None,
) -> str:
    """Guidance to append to a failed edit's error message.

    Returns ``""`` when there is nothing useful to add.  The text is plain
    and imperative because it is read by a model mid-retry, not by a human.
    """
    if not path:
        return ""

    attempts = failure_count(path) if count is None else count
    if attempts <= 0:
        return ""

    fmt = (current_format or "").strip().lower()

    if attempts < ESCALATE_AFTER:
        # First failure: the format is probably fine, the snippet is stale.
        return (
            "\n\nBefore retrying: re-read the file. Do not resend the same "
            "edit - the text you matched against is not what is on disk."
        )

    nxt = next_format(fmt)
    if nxt == WRITE_FILE or fmt == WRITE_FILE:
        return (
            f"\n\nThis is failure {attempts} on this file with "
            f"'{fmt or 'this format'}'. Stop retrying it. Re-read the file, "
            "then rewrite the whole region with write_file."
        )
    return (
        f"\n\nThis is failure {attempts} on this file with '{fmt}'. Stop "
        f"retrying '{fmt}' - switch edit format: re-read the file, then use "
        f"'{nxt}' instead. If '{nxt}' also fails, rewrite the region with "
        "write_file."
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
        cleaned = tuple(
            str(f).strip().lower() for f in pinned if str(f).strip()
        )
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

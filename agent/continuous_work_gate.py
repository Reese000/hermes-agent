"""Turn-end enforcement gate for continuous work mode.

Policy-only, like verification_stop.py: it never re-runs checks itself; it
turns the per-turn evidence counter into a bounded follow-up.

The gate makes override psychologically expensive for the model. AI models
will never unprompted admit they were blatantly wrong — so the override
declaration requires the model to fully write out its admission of personal
failure, listing every specific thing it failed to do and why. This is the
deterrent: a casual "I am overriding" is not accepted. The model must
actually reckon with its own incompleteness in front of the user.
"""

from __future__ import annotations

from typing import Any, Iterable

# Phrases that signal the model is claiming it finished. Matched case-
# insensitively against the ASSISTANT'S OWN RESPONSE ONLY (the note prefix
# is stripped before checking, so words like "certify" inside the protocol
# guidance text do not false-positive).
_COMPLETION_SIGNALS = (
    "i certify:",
    "certified:",
    "all done",
    "task complete",
    "job complete",
    "work is complete",
    "work is done",
    "everything is complete",
    "everything is done",
    "fully verified",
)

_MAX_DEFAULT_ATTEMPTS = 999999  # No artificial ceiling — loop detector handles stalls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_of(final_response: Any) -> str:
    """Flatten a final response (string or OpenAI-style content parts) to text."""
    if final_response is None:
        return ""
    if isinstance(final_response, str):
        return final_response
    if isinstance(final_response, list):
        chunks: list[str] = []
        for part in final_response:
            if isinstance(part, dict):
                c = part.get("text")
                if isinstance(c, str):
                    chunks.append(c)
            elif isinstance(part, str):
                chunks.append(part)
        return " ".join(chunks)
    try:
        return str(final_response)
    except Exception:
        return ""


def _strip_note_prefix(text: str) -> str:
    """Remove the CW note prefix from the model's input so completion-signal
    scanning only sees the MODEL'S OWN words, not protocol guidance text that
    was injected into the user message.

    The note is prepended to the user message by ``_prepend_note`` as:
        ``# Continuous Work Mode ...\\n\\n<actual user message>``

    The model's response does NOT include the note — but if the model echoes
    or quotes from its input (common with long protocol text), we strip the
    known prefix patterns to avoid false-positive matches on guidance text.
    """
    marker = "# Continuous Work Mode"
    idx = text.find(marker)
    if idx >= 0:
        after = text[idx + len(marker):]
        dbl = after.find("\n\n")
        if dbl >= 0:
            return text[idx + len(marker) + dbl + 2:].strip()
    return text


def _sounds_like_completion(final_response: Any) -> bool:
    """Check whether the model's own response claims completion.

    Strips the note prefix first so protocol guidance text (which contains
    words like "certify", "complete", "verified") doesn't false-positive.
    Only the MODEL'S OWN words are scanned.
    """
    raw = _text_of(final_response)
    clean = _strip_note_prefix(raw).lower()
    return any(sig in clean for sig in _COMPLETION_SIGNALS)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def build_continuous_work_nudge(
    *,
    final_response: Any,
    work_evidence_tools: int,
    attempts: int = 0,
    max_attempts: int = _MAX_DEFAULT_ATTEMPTS,
) -> str | None:
    """Return a synthetic follow-up when the agent stops without evidence.

    Fires when:
    - CW is ON for this session (caller checks agent._continuous_work)
    - The model produced a final answer that reads like completion
    - BUT performed NO work-evidence tool call this turn
    - AND the bounded budget is not exhausted

    Returns None when:
    - Real work was done this turn (work_evidence_tools > 0)
    - The response doesn't read like a completion claim
    - The budget is exhausted (prevents infinite loops)
    """
    if attempts >= max_attempts:
        return None
    if not final_response:
        return None
    if work_evidence_tools > 0:
        return None
    if not _sounds_like_completion(final_response):
        return None

    remaining = max_attempts - attempts - 1

    return (
        "[System: Continuous work mode is ON. You attempted to stop, but this "
        "turn performed no work tools (no terminal/execute_code/write_file/"
        "patch/navigate/delegate/send — only read-only lookups) before claiming "
        "completion. A read-only turn cannot certify real work.\n\n"
        "Keep working. Run verification commands, build the system, produce "
        "deliverables. The CW critic must approve before you can stop.\n"
        f"This is continuation {attempts + 1}. The user has not seen your "
        f"final answer yet.]"
    )


def mark_continuous_work_nudge_issued(agent: Any) -> int:
    """Increment and return the nudge counter for a turn."""
    current = getattr(agent, "_continuous_work_nudges", 0)
    agent._continuous_work_nudges = current + 1
    return agent._continuous_work_nudges


__all__ = [
    "build_continuous_work_nudge",
    "mark_continuous_work_nudge_issued",
    "_COMPLETION_SIGNALS",
    "_strip_note_prefix",
]
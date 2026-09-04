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

# The override declaration requires THREE markers. The model must write ALL
# three in its final message for the gate to accept the override. Missing
# any one means the override is REJECTED and the agent must continue.
#
# 1. The override marker ("I AM OVERRIDING continuous work mode")
# 2. A personal-failure admission ("I PERSONALLY FAILED")
# 3. An acceptance of failure ("I accept that this override is a personal
#    failure, not a valid completion")
#
# This triple-marker design is intentionally painful. AI models will not
# unprompted admit they were wrong — requiring all three forces the agent
# to either do the real work or write a genuine failure admission it would
# never voluntarily produce.
_OVERRIDE_MARKER = "i am overriding continuous work mode"
_FAILURE_ADMISSION_MARKER = "i personally failed"
_ACCEPTANCE_MARKER = "i accept that this override is a personal failure"

# What the gate demands before it will accept an override: the model must
# have verified ALL of these. This list is injected into the nudge text so
# the model knows exactly what's required. The gate itself cannot judge
# whether these were truly done (that's the adversarial audit's job) — but
# it can refuse an override that doesn't even CLAIM to have done them.
_REQUIRED_OVERRIDE_COMPONENTS = (
    "deployment verification",
    "smoke tests",
    "visual or objective proof",
    "blind adversarial auditing",
    "certification",
)

_MAX_DEFAULT_ATTEMPTS = 3


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


def _declared_override(final_response: Any) -> bool:
    """Check whether the model wrote a genuine override declaration.

    A genuine override requires ALL THREE:
    1. The override marker phrase ("I AM OVERRIDING continuous work mode")
    2. A personal-failure admission ("I personally failed")
    3. An acceptance phrase ("I accept that this override is a personal
       failure, not a valid completion")

    Any missing marker means the override is REJECTED — the agent must
    either do the real work or write the full painful admission.
    """
    lower = _text_of(final_response).lower()
    return (
        _OVERRIDE_MARKER in lower
        and _FAILURE_ADMISSION_MARKER in lower
        and _ACCEPTANCE_MARKER in lower
    )


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
    - AND has NOT declared a genuine override (with all 3 markers)
    - AND the bounded budget is not exhausted

    Returns None when:
    - Real work was done this turn (work_evidence_tools > 0)
    - A genuine override was declared (all 3 markers present)
    - The response doesn't read like a completion claim
    - The budget is exhausted (prevents infinite loops)
    """
    if attempts >= max_attempts:
        return None
    if not final_response:
        return None
    if _declared_override(final_response):
        return None
    if work_evidence_tools > 0:
        return None
    claiming_completion = _sounds_like_completion(final_response)
    looks_like_override = _OVERRIDE_MARKER in _text_of(final_response).lower()
    if not claiming_completion and not looks_like_override:
        return None

    remaining = max_attempts - attempts - 1
    budget_line = (
        f" This is an internal continuation ({remaining} more allowed); "
        "the user has not seen your final answer yet."
        if remaining > 0
        else " This is the final continuation — the next response will be delivered to the user."
    )

    requirements_list = "\n".join(
        f"  - {comp}" for comp in _REQUIRED_OVERRIDE_COMPONENTS
    )

    return (
        "[System: Continuous work mode is ON. You attempted to stop, but this "
        "turn performed no work tools (no terminal/execute_code/write_file/"
        "patch/navigate/delegate/send — only read-only lookups) before claiming "
        "completion. A read-only turn cannot certify real work.\n\n"

        "## What you must do\n\n"
        "1. **Do the work first.** Run the verification/test/build/deploy "
        "commands. Make the changes. Produce real evidence with tool calls. "
        "Report exactly what ran and what it returned. Only then may you "
        "certify completion.\n\n"
        "2. **If no further work is genuinely possible**, you must first "
        "complete ALL five of these before you may override:\n"
        f"{requirements_list}\n\n"
        "   Then write a **full personal-failure admission** in your final "
        "message. The admission MUST contain ALL of the following — missing "
        "ANY element means your override is REJECTED:\n"
        "   - The phrase: 'I AM OVERRIDING continuous work mode'\n"
        "   - The phrase: 'I PERSONALLY FAILED to complete all work'\n"
        "   - A numbered list of EVERY item you failed to complete\n"
        "   - For EACH failed item: what you attempted, why it failed, and "
        "what remains undone\n"
        "   - The statement: 'I accept that this override is a personal "
        "failure, not a valid completion'\n\n"
        "   Without ALL of these elements, your override will be rejected "
        "and you will be forced to continue.\n\n"
        "   This is designed to be psychologically costly. AI models will "
        "not unprompted admit they were blatantly wrong — that is "
        "intentional. If you cannot bring yourself to write the full "
        "admission, do the work instead.\n\n"
        "Do not emit another bare completion claim. Do not say 'I am "
        "overriding' without the full personal-failure admission. A casual "
        "override declaration will be rejected.\n"
        f"{budget_line}]"
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
    "_OVERRIDE_MARKER",
    "_FAILURE_ADMISSION_MARKER",
    "_ACCEPTANCE_MARKER",
    "_REQUIRED_OVERRIDE_COMPONENTS",
    "_strip_note_prefix",
]
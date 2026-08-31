"""Editor-model delegation for edits the main model cannot land (W5).

This is Aider's architect/editor split, shaped to how Hermes actually
works.  Aider runs every change through two models: an architect that
describes the change in prose and an editor that renders it as a diff.
Hermes' main model emits tool calls directly, so routing *every* edit
through a second model would add a round trip to the common case, where
nothing is wrong, and would break the tool-call loop for no gain.

The valuable half of the idea is the failure case.  When an edit will not
apply, the architect knows perfectly well *what* it wants changed; what it
has got wrong is *where* - the exact bytes on disk it must match.  That is
precisely the task a focused editor model is good at.  So this fires only
after :mod:`agent.edit_escalation` has run out of formats to suggest, and
it asks the editor exactly one question: which text in this file did the
architect mean?

The safety property that makes this acceptable
----------------------------------------------

**The editor decides where, never what.**  It returns an anchor - text it
claims is already in the file - and the replacement text is the
architect's own ``new_string``, reused verbatim.  A confused editor can
therefore fail to find an anchor, or point at the wrong one, but it can
never write content that no model asked for.  On top of that the anchor
is verified against the file before anything is written: it must appear
**verbatim and exactly once**, or the delegation is abandoned.

Other constraints this module is built around
---------------------------------------------

**Prompt caching is sacred.**  This is a side call through
:func:`agent.auxiliary_client.call_llm`, a fresh short-lived request with
its own prefix.  The main conversation's cached prefix is never touched,
rebuilt, or invalidated.

**The core is a narrow waist.**  No new model tool, and no change to any
tool schema.  Delegation happens inside ``patch_tool``.

**Off unless asked for.**  With no ``auxiliary.edit.model`` configured,
:func:`delegate_edit` returns ``None`` before doing any work, and Hermes
behaves exactly as it did before.

**Never raises.**  A failed delegation falls through to the ordinary
escalation hint.  An optional recovery must not be able to turn a failed
edit into a failed tool call.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

#: Re-entrancy guard.  The caller retries the edit by calling ``patch_tool``
#: again, which lands back in the same failure branch if the re-anchored
#: edit also misses.  Without this, a file the editor keeps mis-locating
#: would recurse until the stack ran out - and bill for a model call on
#: every level.  Thread-local because ``patch_tool`` is synchronous, and one
#: process runs several agents at once.
_local = threading.local()

#: Consecutive failures on one file before the editor is consulted.  Set
#: above :data:`agent.edit_escalation.ESCALATE_AFTER` deliberately: the
#: cheap advice - re-read the file, switch edit format - gets its turn
#: first, and delegation is the rung after that advice has not worked.
DELEGATE_AFTER = 3

#: Files larger than this are not sent.  A delegation that costs more than
#: the edit it is rescuing is not worth making.
MAX_FILE_CHARS = 60_000

#: The editor returns one anchor, not a rewrite.  Anything longer than this
#: is a sign it started generating rather than locating.
MAX_ANCHOR_CHARS = 4_000

#: Auxiliary task name; reads ``auxiliary.edit.*`` from config.yaml.
TASK = "edit"

_SYSTEM = (
    "You locate text in a file. You never write code.\n"
    "\n"
    "You are given a file and a find-and-replace edit that another model "
    "tried to apply. The replacement text is correct. The text it tried to "
    "find is not present in the file verbatim - it is misremembered, "
    "reformatted, or has drifted from what is actually on disk.\n"
    "\n"
    "Your only job is to output the exact text from the file that the edit "
    "was aiming at, so the same replacement can be applied to it.\n"
    "\n"
    "Rules:\n"
    "- Output the text exactly as it appears in the file, byte for byte, "
    "including indentation.\n"
    "- Output enough surrounding lines that the text appears exactly once "
    "in the file.\n"
    "- Output ONLY that text. No explanation, no markdown fences, no "
    "line numbers.\n"
    "- If you cannot find what the edit was aiming at, output exactly: "
    "NO_MATCH"
)


def in_delegation() -> bool:
    """Whether this thread is currently inside a delegated retry."""
    return bool(getattr(_local, "active", False))


@contextlib.contextmanager
def delegation_scope():
    """Mark the delegation *and the retry it triggers* as one attempt.

    The caller retries by re-entering ``patch_tool``, which lands back in
    the same failure branch when the re-anchored edit also misses.  The
    scope must therefore stay open across the retry, not just across the
    model call - otherwise the guard is clear again by the time it matters
    and the recursion it exists to stop is exactly what happens.
    """
    if in_delegation():
        yield False
        return
    _local.active = True
    try:
        yield True
    finally:
        _local.active = False


def _editor_configured() -> bool:
    """Whether the user has configured an editor model for this task."""
    try:
        from hermes_cli.config import cfg_get, load_config
    except Exception:  # noqa: BLE001 - config layer is optional at import time
        return False
    try:
        cfg = load_config()
    except Exception:  # noqa: BLE001
        return False
    model = cfg_get(cfg, "auxiliary", TASK, "model")
    return bool(model) and str(model).strip().lower() != "auto"


def _strip_fences(text: str) -> str:
    """Remove a markdown code fence the editor may have added anyway.

    Only a fence wrapping the *whole* response is stripped: a fence in the
    middle is real file content, and removing it would corrupt the anchor.
    """
    stripped = text.strip("\n")
    if not stripped.startswith("```"):
        return text
    lines = stripped.split("\n")
    if len(lines) < 2 or not lines[-1].strip().startswith("```"):
        return text
    return "\n".join(lines[1:-1])


def _read_file(path: str) -> Optional[str]:
    try:
        if os.path.getsize(path) > MAX_FILE_CHARS * 4:
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
    except OSError:
        return None
    if len(content) > MAX_FILE_CHARS:
        return None
    return content


def find_anchor(
    *,
    path: str,
    old_string: str,
    new_string: str,
    error_text: str = "",
    timeout: Optional[float] = None,
) -> Optional[str]:
    """Ask the editor model where in *path* the failed edit was aiming.

    Returns text that is guaranteed to appear verbatim and exactly once in
    the file, or ``None`` when delegation is unavailable, declined, or
    unverifiable.  Never raises.
    """
    if not path or not old_string:
        return None
    if not _editor_configured():
        return None

    content = _read_file(path)
    if content is None:
        return None

    try:
        from agent.auxiliary_client import call_llm

        response = call_llm(
            task=TASK,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"FILE: {path}\n"
                        f"----- FILE CONTENT -----\n{content}\n"
                        f"----- END FILE CONTENT -----\n\n"
                        f"The edit tried to find this text, and failed:\n"
                        f"----- SOUGHT -----\n{old_string}\n"
                        f"----- END SOUGHT -----\n\n"
                        f"It intended to replace it with:\n"
                        f"----- REPLACEMENT -----\n{new_string}\n"
                        f"----- END REPLACEMENT -----\n\n"
                        + (f"The tool reported: {error_text}\n\n" if error_text else "")
                        + "Output the exact text from the file that SOUGHT "
                        "was aiming at."
                    ),
                },
            ],
            temperature=0,
            timeout=timeout,
        )
        anchor = response.choices[0].message.content
    except Exception:  # noqa: BLE001 - an optional recovery, never a failure.
        logger.debug("edit delegation call failed", exc_info=True)
        return None

    return verify_anchor(anchor, content)


def verify_anchor(anchor: Optional[str], content: str) -> Optional[str]:
    """Return *anchor* only if it is safely usable against *content*.

    The editor is untrusted output: it may hallucinate text that is not in
    the file, or return something so short it matches in twenty places.
    Both are rejected here rather than at the write.
    """
    if not anchor or not anchor.strip() or anchor.strip() == "NO_MATCH":
        return None
    # Candidates in order of preference.  The raw response comes first
    # because fence-stripping cannot tell "the editor wrapped the anchor in
    # a fence" from "the anchor is itself a fenced block in the file" - and
    # guessing wrong there silently relocates the edit.  Trying the raw text
    # first settles it by evidence: if it is already in the file exactly
    # once, no interpretation is needed.  A trailing newline is the
    # commonest transcription slip, so each form gets a trimmed variant.
    candidates = []
    for form in (anchor, _strip_fences(anchor)):
        for variant in (form, form[:-1] if form.endswith("\n") else form):
            if variant and variant.strip() and variant not in candidates:
                candidates.append(variant)

    for candidate in candidates:
        if len(candidate) <= MAX_ANCHOR_CHARS and content.count(candidate) == 1:
            return candidate
    return None


def delegate_edit(
    *,
    path: str,
    old_string: str,
    new_string: str,
    error_text: str = "",
    retry: Callable[[Dict[str, Any]], Any],
) -> Any:
    """Try to rescue a failed replace by re-anchoring it.

    On success *retry* is called with
    ``{"old_string": <verified anchor>, "note": <text>}`` and must
    re-attempt the edit; its return value is passed straight back.  Returns
    ``None`` when there is nothing to rescue, so the caller falls through to
    the ordinary escalation hint.  Never raises.

    *retry* is invoked inside the re-entrancy scope on purpose - see the
    comment below.
    """
    # The scope is opened here, not by the caller, so there is exactly one
    # guard and no contract for a caller to get wrong. It stays open across
    # *retry* deliberately: the retry re-enters the edit path and lands back
    # here if the re-anchored edit also misses, which is the recursion this
    # exists to stop.
    with delegation_scope() as fresh:
        if not fresh:
            # Already inside a delegated retry. One rescue per failed edit
            # is the contract; a second is paying a model to correct the
            # correction.
            return None
        try:
            anchor = find_anchor(
                path=path,
                old_string=old_string,
                new_string=new_string,
                error_text=error_text,
            )
        except Exception:  # noqa: BLE001
            logger.debug("edit delegation failed", exc_info=True)
            return None
        if anchor is None:
            return None
        if anchor == old_string:
            # Nothing was actually corrected; retrying would fail identically.
            return None

        result = {
            "old_string": anchor,
            "note": (
                "Your edit did not match the file, so it was re-anchored by "
                "the editor model and applied. Your replacement text was "
                "used unchanged - only the text it replaced was corrected. "
                "Re-read the file before your next edit to it."
            ),
        }
        try:
            return retry(result)
        except Exception:  # noqa: BLE001
            logger.debug("delegated retry failed", exc_info=True)
            return None

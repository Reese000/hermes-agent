"""Harness profiles — per-model-family prompt customisation registry.

Each :class:`HarnessProfile` bundles every model-specific knob into one frozen
dataclass so the rest of the codebase never re-derives "which family is this?"
from scattered substring lists.  Resolution is via :func:`resolve_profile`,
which uses **longest-needle-wins** matching with an explicit tie-break order
(documented in the docstring).

Phase A (this module's initial release) is a behaviour-identical refactor of
the two existing dispatch sites:
  - ``agent/system_prompt.py`` — tool-use enforcement, Google/OpenAI guidance
  - ``agent/coding_context.py`` — edit-format guidance

Every pre-existing model family produces the exact same prompt output it did
before the refactor.  The ``mimo`` profile is the only new addition.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Literal, Optional

# ── Guidance constants (imported from prompt_builder for re-export) ──────────
# These remain importable from prompt_builder for any external consumer.
from agent.prompt_builder import (
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    OPENAI_MODEL_EXECUTION_GUIDANCE,
    TOOL_USE_ENFORCEMENT_GUIDANCE,
)

# ── Edit-format guidance lines (single source of truth) ─────────────────────

_EDIT_FORMAT_PATCH = (
    "- Edit format: author new files with `write_file`; for edits to "
    "existing code use `patch` with `mode='patch'` (V4A diff) — including "
    "single-file edits. It's the edit format you handle most reliably."
)

_EDIT_FORMAT_REPLACE = (
    "- Edit format: author new files with `write_file`; for edits to "
    "existing code prefer `patch` in `mode='replace'` — match a unique "
    "snippet and swap it. Reach for `mode='patch'` (V4A) only when an edit "
    "genuinely spans several files at once."
)

# ── MiMo execution guidance (hand-written for small/mid models) ─────────────

MIMO_EXECUTION_GUIDANCE = (
    "# Execution discipline\n"
    "- Call the tool. Do not describe the call.\n"
    "- If a patch fails, re-read the file before retrying.\n"
    "- State done only after a passing verify command.\n"
    "- Batch independent reads into one turn.\n"
    "- Never fabricate output — report blockers honestly."
)

# ── HarnessProfile dataclass ────────────────────────────────────────────────


@dataclass(frozen=True)
class HarnessProfile:
    """A model-family's prompt customisation knobs.  Pure data — no I/O.

    Fields consumed immediately:
      * ``needles``        — model-id substrings this profile claims
      * ``edit_format``    — ``"patch"`` or ``"replace"``
      * ``edit_format_line`` — the coding-brief line (verbatim)
      * ``execution_guidance`` — ``""`` for none; the OPENAI_/GOOGLE_/MIMO_ block
      * ``tool_use_enforcement`` — replaces TOOL_USE_ENFORCEMENT_MODELS membership
      * ``role_model``     — ``"system"`` or ``"developer"``

    Fields consumed by later workstreams (W2/W4):
      * ``tool_description_overrides`` — per-tool description replacement (W2)
      * ``tool_description_appends`` — per-tool description addendum (W2, preferred)
      * ``retry_format_chain`` — edit-format fallback order (W4).  Leave
        empty to derive it from ``edit_format``; pin it only when a family
        needs a different order.  Consumed by
        ``agent.edit_escalation.resolve_chain`` and surfaced in the coding
        brief, so it is fixed for the session and cache-safe.
    """

    name: str
    needles: tuple[str, ...]
    edit_format: Literal["patch", "replace"]
    edit_format_line: str
    execution_guidance: str = ""
    tool_use_enforcement: bool = False
    role_model: Literal["system", "developer"] = "system"
    verbosity_hint: str = ""
    tool_description_overrides: dict[str, str] = field(default_factory=dict)
    # Safer sibling of ``tool_description_overrides``: the text is APPENDED to
    # the tool's existing description instead of replacing it, so a profile
    # can add a model-specific nudge without any risk of dropping the tool's
    # real contract (parameters, semantics, safety notes).  Prefer this.
    tool_description_appends: dict[str, str] = field(default_factory=dict)
    retry_format_chain: tuple[str, ...] = ()


# ── Profile definitions ─────────────────────────────────────────────────────
# Order does NOT matter — resolve_profile uses longest-needle-wins, not
# first-match.  The profiles are listed in a logical grouping for readability.

ANTHROPIC_PROFILE = HarnessProfile(
    name="anthropic",
    needles=("claude", "sonnet", "opus", "haiku"),
    edit_format="replace",
    edit_format_line=_EDIT_FORMAT_REPLACE,
    # No execution_guidance — Claude doesn't need the OpenAI/Google blocks.
    # No tool_use_enforcement — Claude reliably calls tools on its own.
)

OPENAI_PROFILE = HarnessProfile(
    name="openai",
    needles=("gpt", "codex"),
    edit_format="patch",
    edit_format_line=_EDIT_FORMAT_PATCH,
    execution_guidance=OPENAI_MODEL_EXECUTION_GUIDANCE,
    tool_use_enforcement=True,
    role_model="developer",
)

GOOGLE_PROFILE = HarnessProfile(
    name="google",
    needles=("gemini", "gemma"),
    edit_format="replace",
    edit_format_line=_EDIT_FORMAT_REPLACE,
    execution_guidance=GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    tool_use_enforcement=True,
)

# Grok resolution: "grok" appears in coding_context's replace family AND in
# system_prompt's OpenAI-execution-guidance branch.  These are independent
# axes — edit_format="replace" (as today) and execution_guidance=OPENAI_ (as
# today).  This is precisely the argument for unifying them into one profile
# object rather than two disjoint substring lists.
XAI_PROFILE = HarnessProfile(
    name="xai",
    needles=("grok",),
    edit_format="replace",
    edit_format_line=_EDIT_FORMAT_REPLACE,
    execution_guidance=OPENAI_MODEL_EXECUTION_GUIDANCE,
    tool_use_enforcement=True,
)

DEEPSEEK_PROFILE = HarnessProfile(
    name="deepseek",
    needles=("deepseek",),
    edit_format="replace",
    edit_format_line=_EDIT_FORMAT_REPLACE,
    tool_use_enforcement=True,
)

QWEN_PROFILE = HarnessProfile(
    name="qwen",
    needles=("qwen",),
    edit_format="replace",
    edit_format_line=_EDIT_FORMAT_REPLACE,
    tool_use_enforcement=True,
)

# Zhipu (GLM) — replace family + tool-use enforcement (GLM is in
# TOOL_USE_ENFORCEMENT_MODELS).  Split from kimi/minimax because those
# are NOT in TOOL_USE_ENFORCEMENT_MODELS and must not get enforcement.
ZHIPU_PROFILE = HarnessProfile(
    name="zhipu",
    needles=("glm",),
    edit_format="replace",
    edit_format_line=_EDIT_FORMAT_REPLACE,
    tool_use_enforcement=True,
)

# Kimi and MiniMax — replace family, no tool-use enforcement (not in
# TOOL_USE_ENFORCEMENT_MODELS).  No execution guidance.
KIMI_MINIMAX_PROFILE = HarnessProfile(
    name="kimi_minimax",
    needles=("kimi", "minimax"),
    edit_format="replace",
    edit_format_line=_EDIT_FORMAT_REPLACE,
)

# Meta (Llama), Mistral, and Devstral share the replace family.
# No tool_use_enforcement — they're not in TOOL_USE_ENFORCEMENT_MODELS.
META_MISTRAL_PROFILE = HarnessProfile(
    name="meta_mistral",
    needles=("llama", "mistral", "devstral"),
    edit_format="replace",
    edit_format_line=_EDIT_FORMAT_REPLACE,
)

HERMES_PROFILE = HarnessProfile(
    name="hermes",
    needles=("hermes",),
    edit_format="replace",
    edit_format_line=_EDIT_FORMAT_REPLACE,
)

# MiMo — the user's primary worker model (xiaomi/mimo-v2.5 / mimo-v2.5-pro).
# Hand-written execution guidance tuned for a small/mid model.
MIMO_PROFILE = HarnessProfile(
    name="mimo",
    needles=("mimo", "xiaomi"),
    edit_format="replace",
    edit_format_line=_EDIT_FORMAT_REPLACE,
    execution_guidance=MIMO_EXECUTION_GUIDANCE,
    tool_use_enforcement=True,
    # Point-of-use reinforcement. This restates a rule already present in
    # MIMO_EXECUTION_GUIDANCE ("If a patch fails, re-read the file before
    # retrying") at the place the model actually decides to call the tool —
    # the Cursor/Aider pattern of putting guidance where it is acted on.
    # Appended, never replacing, so the tool contract is untouched.
    tool_description_appends={
        "patch": (
            "\n\nIf a patch fails to apply, re-read the file before "
            "retrying - do not resend the same patch."
        ),
    },
    # MIMO is the family that most often fails to land a patch (hence the
    # addendum above), so route its fallback around patch rather than
    # through it: the derived default would send it to the format it is
    # worst at.
    retry_format_chain=("replace", "write_file"),
)

GENERIC_PROFILE = HarnessProfile(
    name="generic",
    needles=(),  # fallback — never matched by needle
    edit_format="replace",
    edit_format_line="",
    # No guidance, no enforcement — the neutral posture.
)


# ── Registry ────────────────────────────────────────────────────────────────

_ALL_PROFILES: list[HarnessProfile] = [
    ANTHROPIC_PROFILE,
    OPENAI_PROFILE,
    GOOGLE_PROFILE,
    XAI_PROFILE,
    DEEPSEEK_PROFILE,
    QWEN_PROFILE,
    ZHIPU_PROFILE,
    KIMI_MINIMAX_PROFILE,
    META_MISTRAL_PROFILE,
    HERMES_PROFILE,
    MIMO_PROFILE,
    # GENERIC is not in the list — it's the fallback.
]

# Pre-build a needle→profile index for fast lookup.
_NEEDLE_INDEX: dict[str, HarnessProfile] = {}
for _p in _ALL_PROFILES:
    for _n in _p.needles:
        _NEEDLE_INDEX[_n] = _p


# ── Resolution ──────────────────────────────────────────────────────────────


def resolve_profile(
    model: str | None,
    provider: str | None = None,
) -> HarnessProfile:
    """Resolve a model id to its :class:`HarnessProfile`.

    **Longest-needle-wins** with explicit tie-break: when two needles of equal
    length match, the profile that appears first in ``_ALL_PROFILES`` wins
    (anthropic > openai > google > xai > deepseek > qwen > zhipu_kimi_minimax
    > meta_mistral > hermes > mimo).  This is deterministic and documented.

    ``provider`` disambiguates the Alibaba Coding Plan bug where the API
    returns ``"glm-4.7"`` regardless of the requested model.  When
    ``provider == "alibaba"``, the *original* model id (before the API
    overwrote it) is used for resolution.

    Unknown model → ``GENERIC_PROFILE``, never a crash, never None.

    Pure function — no I/O, no config reads — trivially testable and cannot
    become a startup cost.
    """
    if not model:
        return GENERIC_PROFILE

    lowered = model.lower()

    # Alibaba workaround: if provider is alibaba, the model id may have been
    # overwritten to "glm-4.7" by the API.  The caller should pass the
    # original model id; if they pass the overwritten one, GLM will still
    # match (which is correct — glm IS the zhipu family).
    # No special handling needed here — the workaround lives in the caller
    # (system_prompt.py line ~295) which injects model identity text.

    best_needle: str = ""
    best_profile: HarnessProfile = GENERIC_PROFILE

    for needle, profile in _NEEDLE_INDEX.items():
        if needle in lowered:
            # Longest-needle-wins
            if len(needle) > len(best_needle):
                best_needle = needle
                best_profile = profile
            elif len(needle) == len(best_needle):
                # Tie-break: earlier in _ALL_PROFILES wins.
                # We can compare by index in the list.
                try:
                    cur_idx = _ALL_PROFILES.index(best_profile)
                    new_idx = _ALL_PROFILES.index(profile)
                    if new_idx < cur_idx:
                        best_profile = profile
                except ValueError:
                    pass

    return best_profile


def get_profile_by_name(name: str) -> HarnessProfile:
    """Look up a profile by its ``name`` field.  Falls back to GENERIC."""
    for p in _ALL_PROFILES:
        if p.name == name:
            return p
    if name == "generic":
        return GENERIC_PROFILE
    return GENERIC_PROFILE

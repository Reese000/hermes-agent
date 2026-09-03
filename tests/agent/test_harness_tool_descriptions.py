"""Enforcement tests for per-model tool descriptions (W2).

The mechanism lets a harness profile adjust a tool's ``description`` — and
nothing else — for the model family it targets.  Two channels exist:

  * ``tool_description_appends``   — additive; cannot destroy a contract.
  * ``tool_description_overrides`` — full replacement; use sparingly.

These tests are written to FAIL when the mechanism is broken, not merely to
exercise it.  Each has been mutation-checked: neutralising the code it guards
makes it fail.  The properties under guard are:

  1. Default output is byte-identical to the pre-W2 behaviour (no profile,
     or a profile with no overrides -> unchanged schemas).
  2. Only ``description`` is ever mutated — name/parameters are untouched,
     so a bad override can never change a tool's contract.
  3. The memo key includes the profile, or a process serving two model
     families would hand one family the other's descriptions.
  4. ``agent_init`` actually applies the profile — otherwise the whole
     mechanism is dead code that never reaches the model.
"""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap

import pytest

from agent.harness_profiles import resolve_profile
from agent.harness_profiles.profiles import HarnessProfile
from tools.registry import registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _a_real_tool_name() -> str:
    """Pick a registered tool that actually passes its check_fn.

    Prefers a tool that is also present in the *default* toolset, so the
    cache-key tests exercise the real path instead of skipping.
    """
    # Ask model_tools FIRST: importing/calling it is what triggers builtin
    # tool discovery and registration.  Querying the bare registry before
    # that returns an empty list and skips every test in this module.
    default_names = []
    try:
        import model_tools

        default_names = [
            d["function"]["name"]
            for d in model_tools.get_tool_definitions(quiet_mode=True)
        ]
    except Exception:
        pass

    defs = registry.get_definitions(set(registry.get_all_tool_names()), quiet=True)
    if not defs:
        pytest.skip("no tools registered in this environment")

    registry_names = {d["function"]["name"] for d in defs}
    for n in default_names:
        if n in registry_names:
            return n

    return defs[0]["function"]["name"]


def _defs(names, **kwargs):
    return registry.get_definitions(set(names), quiet=True, **kwargs)


def _profile(**kwargs) -> HarnessProfile:
    base = dict(
        name="test-profile",
        needles=("never-matches-anything",),
        edit_format="replace",
        edit_format_line="- test",
    )
    base.update(kwargs)
    return HarnessProfile(**base)


# ---------------------------------------------------------------------------
# (1) Default behaviour is byte-identical
# ---------------------------------------------------------------------------

class TestDefaultUnchanged:
    def test_no_overrides_is_byte_identical(self):
        name = _a_real_tool_name()
        baseline = _defs([name])
        with_none = _defs([name], tool_description_overrides=None,
                          tool_description_appends=None)
        assert with_none == baseline

    def test_empty_dicts_are_byte_identical(self):
        name = _a_real_tool_name()
        baseline = _defs([name])
        assert _defs([name], tool_description_overrides={},
                     tool_description_appends={}) == baseline

    def test_override_for_a_different_tool_does_not_leak(self):
        name = _a_real_tool_name()
        baseline = _defs([name])
        got = _defs([name], tool_description_overrides={"some_other_tool": "X"})
        assert got == baseline

    def test_blank_override_is_ignored(self):
        """Whitespace-only text must not blank out a real description."""
        name = _a_real_tool_name()
        baseline = _defs([name])
        assert _defs([name], tool_description_overrides={name: "   "}) == baseline
        assert _defs([name], tool_description_appends={name: "   "}) == baseline

    def test_profiles_without_overrides_stay_empty(self):
        """Only mimo ships overrides today; the rest must remain untouched."""
        for model in ("claude-opus-4", "gpt-5", "gemini-2.5-pro",
                      "deepseek-v3", "grok-4", "qwen3-coder"):
            p = resolve_profile(model, None)
            assert not p.tool_description_overrides, model
            assert not p.tool_description_appends, model


# ---------------------------------------------------------------------------
# (2) The mechanism works, and touches ONLY the description
# ---------------------------------------------------------------------------

class TestOverrideApplied:
    def test_replacement_replaces_description(self):
        name = _a_real_tool_name()
        got = _defs([name], tool_description_overrides={name: "REPLACED"})
        assert got[0]["function"]["description"] == "REPLACED"

    def test_append_preserves_original_text(self):
        name = _a_real_tool_name()
        original = _defs([name])[0]["function"].get("description", "")
        got = _defs([name], tool_description_appends={name: " ADDENDUM"})
        new = got[0]["function"]["description"]
        assert new == original + " ADDENDUM"
        assert new.startswith(original), "append must never drop the original"

    def test_contract_fields_are_never_mutated(self):
        """A bad override must not be able to change what the tool accepts."""
        name = _a_real_tool_name()
        baseline = copy.deepcopy(_defs([name])[0]["function"])
        got = _defs([name], tool_description_overrides={name: "REPLACED"})[0]["function"]
        for field in ("name", "parameters"):
            if field in baseline:
                assert got.get(field) == baseline[field], field

    def test_append_applies_on_top_of_replacement(self):
        name = _a_real_tool_name()
        got = _defs(
            [name],
            tool_description_overrides={name: "BASE"},
            tool_description_appends={name: "+MORE"},
        )
        assert got[0]["function"]["description"] == "BASE+MORE"

    def test_mimo_ships_a_patch_addendum(self):
        p = resolve_profile("xiaomi/mimo-v2.5", None)
        assert "patch" in p.tool_description_appends
        assert "re-read" in p.tool_description_appends["patch"].lower()


# ---------------------------------------------------------------------------
# (3) Cache correctness — the memo key must include the profile
# ---------------------------------------------------------------------------

class TestMemoKeyIncludesProfile:
    def test_two_profiles_do_not_share_cached_descriptions(self):
        """Regression guard: a Gateway process serving two model families
        must not hand one family the other's tool descriptions."""
        import model_tools

        name = _a_real_tool_name()
        toolset = None

        p_a = _profile(name="prof-a", tool_description_appends={name: " AAA"})
        p_b = _profile(name="prof-b", tool_description_appends={name: " BBB"})

        defs_a = model_tools.get_tool_definitions(
            enabled_toolsets=toolset, quiet_mode=True, harness_profile=p_a
        )
        defs_b = model_tools.get_tool_definitions(
            enabled_toolsets=toolset, quiet_mode=True, harness_profile=p_b
        )

        def desc(defs):
            for d in defs:
                if d["function"]["name"] == name:
                    return d["function"].get("description", "")
            return ""

        da, db = desc(defs_a), desc(defs_b)
        if not da and not db:
            pytest.skip("tool not present in the default toolset here")
        assert da != db, (
            "profile-b received profile-a's cached descriptions - the "
            "get_tool_definitions memo key is missing the harness profile"
        )
        assert da.endswith(" AAA")
        assert db.endswith(" BBB")

    def test_none_profile_key_differs_from_named_profile(self):
        import model_tools

        name = _a_real_tool_name()
        p = _profile(name="prof-c", tool_description_appends={name: " CCC"})

        plain = model_tools.get_tool_definitions(quiet_mode=True)
        themed = model_tools.get_tool_definitions(quiet_mode=True, harness_profile=p)

        def desc(defs):
            for d in defs:
                if d["function"]["name"] == name:
                    return d["function"].get("description", "")
            return ""

        if not desc(plain) and not desc(themed):
            pytest.skip("tool not present in the default toolset here")
        assert desc(plain) != desc(themed)


# ---------------------------------------------------------------------------
# (4) Anti-drift — the wiring must actually exist in production
# ---------------------------------------------------------------------------

class TestProductionWiring:
    def test_agent_init_passes_the_profile_to_get_tool_definitions(self):
        """Static guard.  The behavioural tests above call the registry and
        model_tools directly, so they would all still pass if agent_init
        never applied the profile - leaving the feature dead code that never
        reaches the model.  This asserts the wiring exists."""
        import agent.agent_init as ai

        src = inspect.getsource(ai)
        tree = ast.parse(src)

        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            fname = getattr(func, "attr", None) or getattr(func, "id", None)
            if fname != "get_tool_definitions":
                continue
            if any(kw.arg == "harness_profile" for kw in node.keywords):
                found = True
                break

        assert found, (
            "agent_init never passes harness_profile to get_tool_definitions - "
            "per-model tool descriptions are dead code and will never reach "
            "the model."
        )

    def test_registry_accepts_both_channels(self):
        sig = inspect.signature(registry.get_definitions)
        assert "tool_description_overrides" in sig.parameters
        assert "tool_description_appends" in sig.parameters

    def test_get_tool_definitions_accepts_profile(self):
        import model_tools

        sig = inspect.signature(model_tools.get_tool_definitions)
        assert "harness_profile" in sig.parameters
        assert sig.parameters["harness_profile"].default is None, (
            "harness_profile must default to None so all existing callers "
            "keep byte-identical behaviour"
        )

    def test_memo_key_source_references_the_profile(self):
        """Guards the specific cache bug in TestMemoKeyIncludesProfile even
        if that test is skipped in a stripped-down tool environment."""
        import model_tools

        src = inspect.getsource(model_tools.get_tool_definitions)
        body = textwrap.dedent(src)
        # Target the tuple-assignment site (`cache_key = (`), not the earlier
        # `cache_key = None` sentinel — upstream's profile_scope logic sits
        # between them and would push the tuple past a naive window.
        after_key = body.split("cache_key = (")[1]
        # Window covers the tuple literal; generous enough to survive comment
        # edits but far short of the rest of the function.
        assert "harness_profile" in after_key[:1200], (
            "cache_key does not incorporate the harness profile"
        )

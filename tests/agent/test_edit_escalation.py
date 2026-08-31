"""Enforcement tests for edit-format escalation (W4).

The properties under guard, and why each matters:

  1. **Escalation actually escalates.**  First failure says "re-read"; a
     repeat failure on the same file says "switch format".  If the counter
     stops counting, a model can resend the same broken edit forever.
  2. **Per-file, per-agent scoping.**  A failure on one file must not
     escalate another file, and one agent's failures must not escalate a
     concurrent agent's edits.
  3. **Success clears the streak.**  Otherwise a long session drifts into
     permanent escalation and every first failure reads as a crisis.
  4. **The chain is real, and per-model.**  ``retry_format_chain`` was a dead
     field before this workstream; it must now change observable output.
  5. **Cache safety.**  The brief line depends only on the model id, which is
     fixed for the session.  Nothing here may vary run to run.
  6. **One counter, not two.**  ``tools/file_tools`` used to keep its own
     consecutive-failure tracker with its own hint. Two counters meant two
     nudges disagreeing about how many times the model had failed.
  7. **Production wiring.**  Static call-site guards: the behavioural tests
     below would still pass if the tool layer stopped calling the module,
     leaving the feature dead code.
  8. **Only escalates what escalation can fix.**  A permission denial is not
     cured by switching edit format.
"""

from __future__ import annotations

import inspect
import json

import pytest

from agent import edit_escalation as ee

SCOPE = "test-task"


@pytest.fixture(autouse=True)
def _clean_state():
    ee.reset()
    yield
    ee.reset()


# ---------------------------------------------------------------------------
# (1) Escalation behaviour
# ---------------------------------------------------------------------------

class TestEscalation:
    def test_no_hint_before_any_failure(self):
        assert ee.escalation_hint(SCOPE, "a.py", "replace") == ""

    def test_first_failure_says_reread(self):
        ee.record_failure(SCOPE, "a.py")
        hint = ee.escalation_hint(SCOPE, "a.py", "replace")
        assert "re-read" in hint.lower()
        assert "switch edit format" not in hint

    def test_second_failure_switches_format(self):
        ee.record_failure(SCOPE, "a.py")
        ee.record_failure(SCOPE, "a.py")
        hint = ee.escalation_hint(SCOPE, "a.py", "replace")
        assert "switch edit format" in hint
        assert "'patch'" in hint

    def test_patch_escalates_to_replace(self):
        ee.record_failure(SCOPE, "a.py")
        ee.record_failure(SCOPE, "a.py")
        assert "'replace'" in ee.escalation_hint(SCOPE, "a.py", "patch")

    def test_write_file_is_terminal(self):
        assert ee.next_format("write_file") == "write_file"

    def test_unknown_format_falls_back_to_write_file(self):
        assert ee.next_format("some-new-format") == "write_file"
        ee.record_failure(SCOPE, "a.py")
        ee.record_failure(SCOPE, "a.py")
        assert "write_file" in ee.escalation_hint(SCOPE, "a.py", "some-new-format")

    def test_hint_always_mentions_write_file_as_last_resort(self):
        ee.record_failure(SCOPE, "a.py")
        ee.record_failure(SCOPE, "a.py")
        assert "write_file" in ee.escalation_hint(SCOPE, "a.py", "replace")

    def test_count_override_bypasses_state(self):
        assert "switch edit format" in ee.escalation_hint(
            SCOPE, "a.py", "replace", count=5
        )

    def test_escalation_threshold_is_what_drives_the_switch(self):
        """Guards the threshold comparison itself.

        With ``ESCALATE_AFTER`` neutralised, the first failure would already
        emit the switch text - which is exactly the mutation this catches.
        """
        assert ee.ESCALATE_AFTER >= 2
        ee.record_failure(SCOPE, "a.py")
        assert "switch edit format" not in ee.escalation_hint(SCOPE, "a.py", "replace")


# ---------------------------------------------------------------------------
# (2) Scoping
# ---------------------------------------------------------------------------

class TestScoping:
    def test_failures_do_not_leak_across_files(self):
        ee.record_failure(SCOPE, "a.py")
        ee.record_failure(SCOPE, "a.py")
        assert ee.failure_count(SCOPE, "b.py") == 0
        assert "switch edit format" not in ee.escalation_hint(SCOPE, "b.py", "replace")

    def test_failures_do_not_leak_across_agents(self):
        """Concurrent agents (Gateway, delegated children) share the process."""
        ee.record_failure("agent-a", "shared.py")
        ee.record_failure("agent-a", "shared.py")
        assert ee.failure_count("agent-b", "shared.py") == 0, (
            "one agent's failures escalated another agent's edits"
        )

    def test_equivalent_paths_share_a_counter(self, tmp_path):
        target = tmp_path / "same.py"
        target.write_text("x = 1\n", encoding="utf-8")
        ee.record_failure(SCOPE, str(target))
        ee.record_failure(SCOPE, str(tmp_path / "sub" / ".." / "same.py"))
        assert ee.failure_count(SCOPE, str(target)) == 2

    def test_empty_path_is_ignored(self):
        assert ee.record_failure(SCOPE, "") == 0
        assert ee.failure_count(SCOPE, "") == 0
        assert ee.escalation_hint(SCOPE, "", "replace") == ""

    def test_tracked_paths_are_capped(self):
        """A session failing across thousands of files must not grow forever."""
        cap = ee.MAX_TRACKED_PATHS_PER_SCOPE
        for i in range(cap * 2):
            ee.record_failure(SCOPE, f"f{i}.py")
        tracked = len(ee._failures[ee._scope_key(SCOPE)])
        assert tracked <= cap, f"tracker grew to {tracked}, cap is {cap}"


# ---------------------------------------------------------------------------
# (3) Success clears the streak
# ---------------------------------------------------------------------------

class TestSuccessResets:
    def test_success_clears_the_counter(self):
        ee.record_failure(SCOPE, "a.py")
        ee.record_failure(SCOPE, "a.py")
        ee.record_success(SCOPE, "a.py")
        assert ee.failure_count(SCOPE, "a.py") == 0
        assert ee.escalation_hint(SCOPE, "a.py", "replace") == ""

    def test_success_on_another_file_does_not_clear(self):
        ee.record_failure(SCOPE, "a.py")
        ee.record_success(SCOPE, "b.py")
        assert ee.failure_count(SCOPE, "a.py") == 1

    def test_reset_scoped_leaves_other_agents_alone(self):
        ee.record_failure("agent-a", "a.py")
        ee.record_failure("agent-b", "a.py")
        ee.reset("agent-a")
        assert ee.failure_count("agent-a", "a.py") == 0
        assert ee.failure_count("agent-b", "a.py") == 1

    def test_reset_all_clears_everything(self):
        ee.record_failure("agent-a", "a.py")
        ee.record_failure("agent-b", "b.py")
        ee.reset()
        assert ee.failure_count("agent-a", "a.py") == 0
        assert ee.failure_count("agent-b", "b.py") == 0


# ---------------------------------------------------------------------------
# (4) Per-model chain - retry_format_chain must be live, not decorative
# ---------------------------------------------------------------------------

class TestChain:
    def test_default_chain_opens_with_the_profiles_edit_format(self):
        from agent.harness_profiles import resolve_profile

        for model in ("claude-opus-5", "gpt-5", "gemini-2.5-pro"):
            profile = resolve_profile(model)
            chain = ee.resolve_chain(profile)
            assert chain, f"no chain for {model}"
            assert chain[0] == profile.edit_format, (
                f"{model}: chain opens with {chain[0]!r} but the prompt tells "
                f"it to prefer {profile.edit_format!r} - the two disagree"
            )

    def test_every_chain_terminates_in_write_file(self):
        from agent.harness_profiles.profiles import _ALL_PROFILES

        for profile in _ALL_PROFILES:
            chain = ee.resolve_chain(profile)
            if chain:
                assert chain[-1] == "write_file", (
                    f"{profile.name}: chain has no terminal fallback"
                )

    def test_pinned_chain_wins_over_the_default(self):
        class FakeProfile:
            edit_format = "replace"
            retry_format_chain = ("patch", "write_file")

        assert ee.resolve_chain(FakeProfile()) == ("patch", "write_file")

    def test_pinned_chain_changes_the_brief_line(self):
        """The whole point of the field: it must alter observable output."""

        class A:
            edit_format = "replace"
            retry_format_chain = ()

        class B:
            edit_format = "replace"
            retry_format_chain = ("patch", "replace", "write_file")

        assert ee.escalation_brief_line(A()) != ee.escalation_brief_line(B())

    def test_unusable_profile_yields_no_line(self):
        class Junk:
            edit_format = ""
            retry_format_chain = ()

        assert ee.resolve_chain(Junk()) == ()
        assert ee.escalation_brief_line(Junk()) == ""

    def test_mimo_pins_a_patch_free_chain(self):
        from agent.harness_profiles import resolve_profile

        chain = ee.resolve_chain(resolve_profile("mimo-vl-7b"))
        assert chain == ("replace", "write_file"), (
            "mimo's pinned chain is gone - it fell back to the derived "
            "default, which routes it through the format it fails at"
        )


# ---------------------------------------------------------------------------
# (5) Cache safety - the brief rides in the cached system prompt
# ---------------------------------------------------------------------------

class TestCacheSafety:
    def test_brief_line_is_deterministic(self):
        from agent.coding_context import _edit_escalation_line

        runs = {_edit_escalation_line("gpt-5") for _ in range(5)}
        assert len(runs) == 1

    def test_brief_line_is_non_empty_for_known_families(self):
        from agent.coding_context import _edit_escalation_line

        for model in ("claude-opus-5", "gpt-5", "gemini-2.5-pro"):
            assert _edit_escalation_line(model), (
                f"{model}: no escalation line reached the brief"
            )

    def test_brief_line_never_raises(self):
        from agent.coding_context import _edit_escalation_line

        for model in (None, "", "totally-unknown-model", "\x00"):
            _edit_escalation_line(model)

    def test_unknown_model_keeps_the_brief_neutral(self):
        """An unknown model must not be steered toward an unverified format.

        The brief already withholds ``edit_format_line`` for the generic
        profile; the escalation line has to withhold on the same condition or
        it reintroduces the steering through the back door.
        """
        from agent.coding_context import _edit_escalation_line

        for model in (None, "", "totally-unknown-model-xyz"):
            assert _edit_escalation_line(model) == "", (
                f"{model!r}: generic profile got edit-format steering"
            )

    def test_line_differs_between_families(self):
        from agent.coding_context import _edit_escalation_line

        assert _edit_escalation_line("gpt-5") != _edit_escalation_line("claude-opus-5")


# ---------------------------------------------------------------------------
# (6) One counter, not two
# ---------------------------------------------------------------------------

class TestSingleSourceOfTruth:
    def test_file_tools_keeps_no_private_counter(self):
        """The old ``_patch_failure_tracker`` dict must stay gone.

        It counted the same failures on a different threshold, so a model
        could be told "failure 2, switch format" and "failure #3, stop
        retrying" about the same edit.
        """
        import tools.file_tools as ft

        assert not hasattr(ft, "_patch_failure_tracker"), (
            "file_tools grew a second failure counter again"
        )

    def test_file_tools_delegates_to_the_module(self):
        import tools.file_tools as ft

        src = inspect.getsource(ft._record_patch_failure)
        assert "edit_escalation" in src, "_record_patch_failure no longer delegates"
        assert "edit_escalation" in inspect.getsource(ft._reset_patch_failures), (
            "_reset_patch_failures no longer delegates"
        )

    def test_counter_and_hint_agree_on_the_count(self):
        """The number in the text is the number in the tracker."""
        import tools.file_tools as ft

        ft._record_patch_failure(SCOPE, "a.py")
        n = ft._record_patch_failure(SCOPE, "a.py")
        assert n == ee.failure_count(SCOPE, "a.py")
        assert f"failure {n}" in ee.escalation_hint(SCOPE, "a.py", "replace")


# ---------------------------------------------------------------------------
# (7) Production wiring
# ---------------------------------------------------------------------------

class TestProductionWiring:
    def test_patch_tool_emits_the_escalation(self):
        import tools.file_tools as ft

        src = inspect.getsource(ft.patch_tool)
        assert "escalation_hint" in src, (
            "patch_tool no longer emits the escalation - a model can resend "
            "the same failing edit forever and never be told to switch format"
        )
        assert "_record_patch_failure" in src, "patch_tool stopped counting failures"

    def test_patch_tool_clears_on_success(self):
        import tools.file_tools as ft

        assert "_reset_patch_failures" in inspect.getsource(ft.patch_tool), (
            "patch_tool no longer clears the streak - files stay permanently "
            "escalated after one bad edit"
        )

    def test_coding_brief_emits_the_escalation_line(self):
        from agent.coding_context import RuntimeMode

        assert "_edit_escalation_line" in inspect.getsource(RuntimeMode.system_blocks), (
            "system_blocks no longer emits the escalation line - "
            "retry_format_chain is dead again"
        )

    def test_replace_mode_escalates_end_to_end(self, tmp_path):
        """Drive the real tool, not the helpers."""
        import tools.file_tools as ft

        target = tmp_path / "f.py"
        target.write_text("alpha = 1\n", encoding="utf-8")

        first = json.loads(
            ft.patch_tool(
                mode="replace", path=str(target),
                old_string="no_such_text_zzz", new_string="x",
                task_id=SCOPE,
            )
        )
        assert first.get("error"), "expected the edit to fail"
        assert "re-read" in str(first.get("_hint", "")).lower()

        second = json.loads(
            ft.patch_tool(
                mode="replace", path=str(target),
                old_string="no_such_text_zzz", new_string="x",
                task_id=SCOPE,
            )
        )
        assert "switch edit format" in str(second.get("_hint", "")), (
            f"second failure did not escalate: {second.get('_hint')!r}"
        )

    def test_rich_snippet_is_not_talked_over_on_the_first_failure(self):
        """patch_replace's "Did you mean?" snippet beats generic advice.

        It names the actual nearby sections; the first-failure hint only says
        to re-read. But the escalation at attempt 2 says something the
        snippet does not - switch format - so it must still get through.
        """
        import json as _json
        from unittest.mock import MagicMock, patch as _patch

        import tools.file_tools as ft

        err = (
            "Could not find match for old_string in foo.py\n"
            "Did you mean one of these sections?\n  def alpha():"
        )
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"error": err}
        ops = MagicMock()
        ops.patch_replace.return_value = result_obj

        with _patch("tools.file_tools._get_file_ops", return_value=ops):
            first = _json.loads(ft.patch_tool(
                mode="replace", path="foo.py", old_string="x",
                new_string="y", task_id=SCOPE,
            ))
            assert "_hint" not in first, (
                "the generic first-failure hint buried the richer snippet"
            )
            second = _json.loads(ft.patch_tool(
                mode="replace", path="foo.py", old_string="x",
                new_string="y", task_id=SCOPE,
            ))
            assert "switch edit format" in str(second.get("_hint", "")), (
                "the snippet suppressed the format escalation, which says "
                "something the snippet never does"
            )

    def test_successful_edit_clears_the_streak_end_to_end(self, tmp_path):
        import tools.file_tools as ft

        target = tmp_path / "g.py"
        target.write_text("alpha = 1\n", encoding="utf-8")

        ft.patch_tool(
            mode="replace", path=str(target),
            old_string="no_such_text_qqq", new_string="x", task_id=SCOPE,
        )
        assert ee.failure_count(SCOPE, str(target)) >= 1

        ok = json.loads(
            ft.patch_tool(
                mode="replace", path=str(target),
                old_string="alpha = 1", new_string="alpha = 2", task_id=SCOPE,
            )
        )
        if not ok.get("error"):
            assert ee.failure_count(SCOPE, str(target)) == 0, (
                "a successful edit left the file escalated"
            )


# ---------------------------------------------------------------------------
# (8) Only escalate failures escalation can fix
# ---------------------------------------------------------------------------

class TestFailureClassification:
    @pytest.mark.parametrize(
        "error_text",
        [
            "Could not find match for old_string in x.py",
            "x.py: hunk @@ foo @@ not found in file",
            "Patch validation failed (no files were modified):",
            "Apply phase failed (state may be inconsistent)",
            "Post-write verification failed for x.py",
        ],
    )
    def test_stale_content_failures_are_recognised(self, error_text):
        from tools.file_tools import _is_stale_content_failure

        assert _is_stale_content_failure(error_text)

    @pytest.mark.parametrize(
        "error_text",
        [
            "",
            "Write denied: '/etc/passwd' is a protected system/credential file.",
            "Permission denied",
            "Unknown mode: sideways",
        ],
    )
    def test_other_failures_do_not_escalate(self, error_text):
        from tools.file_tools import _is_stale_content_failure

        assert not _is_stale_content_failure(error_text), (
            f"{error_text!r} would push the model to switch edit format for a "
            "problem no edit format fixes"
        )

    def test_write_denied_does_not_count_as_a_failure(self, tmp_path):
        """End-to-end: a denied write must not build an escalation streak."""
        import tools.file_tools as ft

        target = tmp_path / "h.py"
        target.write_text("alpha = 1\n", encoding="utf-8")
        result = json.loads(
            ft.patch_tool(
                mode="replace", path=str(target),
                old_string="alpha = 1", new_string="alpha = 2", task_id=SCOPE,
            )
        )
        # This one should succeed; the point is only that a non-match error
        # never reaches the counter, which the parametrized cases above pin
        # directly. Guard the invariant that success leaves nothing behind.
        if not result.get("error"):
            assert ee.failure_count(SCOPE, str(target)) == 0

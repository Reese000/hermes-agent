"""Enforcement tests for edit-format escalation (W4).

The properties under guard, and why each matters:

  1. **Escalation actually escalates.**  First failure says "re-read"; a
     repeat failure on the same file says "switch format".  If the counter
     stops counting, a model can resend the same broken edit forever.
  2. **Per-file scoping.**  A failure on one file must not escalate an
     unrelated file's first attempt.
  3. **Success clears the streak.**  Otherwise a long session drifts into
     permanent escalation and every first failure reads as a crisis.
  4. **The chain is real, and per-model.**  ``retry_format_chain`` was a dead
     field before this workstream; it must now change observable output.
  5. **Cache safety.**  The brief line depends only on the model id, which is
     fixed for the session.  Nothing here may vary run to run.
  6. **Production wiring.**  Static call-site guards: every behavioural test
     below drives the helpers directly and would still pass if the tool layer
     stopped calling them, leaving the feature dead code.
  7. **Never breaks an edit.**  Bookkeeping is advisory; it must not turn a
     patch error into an exception or swallow the real error text.
"""

from __future__ import annotations

import inspect
import threading

import pytest

from agent import edit_escalation as ee


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
        assert ee.escalation_hint("a.py", "replace") == ""

    def test_first_failure_says_reread(self):
        ee.record_failure("a.py")
        hint = ee.escalation_hint("a.py", "replace")
        assert "re-read" in hint.lower()
        assert "switch edit format" not in hint

    def test_second_failure_switches_format(self):
        ee.record_failure("a.py")
        ee.record_failure("a.py")
        hint = ee.escalation_hint("a.py", "replace")
        assert "switch edit format" in hint
        assert "'patch'" in hint

    def test_patch_escalates_to_replace(self):
        ee.record_failure("a.py")
        ee.record_failure("a.py")
        assert "'replace'" in ee.escalation_hint("a.py", "patch")

    def test_write_file_is_terminal(self):
        assert ee.next_format("write_file") == "write_file"

    def test_unknown_format_falls_back_to_write_file(self):
        assert ee.next_format("some-new-format") == "write_file"
        ee.record_failure("a.py")
        ee.record_failure("a.py")
        assert "write_file" in ee.escalation_hint("a.py", "some-new-format")

    def test_hint_always_mentions_write_file_as_last_resort(self):
        ee.record_failure("a.py")
        ee.record_failure("a.py")
        assert "write_file" in ee.escalation_hint("a.py", "replace")

    def test_count_override_bypasses_state(self):
        assert "switch edit format" in ee.escalation_hint("a.py", "replace", count=5)

    def test_escalation_threshold_is_what_drives_the_switch(self):
        """Guards the threshold comparison itself.

        With ``ESCALATE_AFTER`` neutralised, the first failure would already
        emit the switch text - which is exactly the mutation this catches.
        """
        assert ee.ESCALATE_AFTER >= 2
        ee.record_failure("a.py")
        assert "switch edit format" not in ee.escalation_hint("a.py", "replace")


# ---------------------------------------------------------------------------
# (2) Per-file scoping
# ---------------------------------------------------------------------------

class TestScoping:
    def test_failures_do_not_leak_across_files(self):
        ee.record_failure("a.py")
        ee.record_failure("a.py")
        assert ee.failure_count("b.py") == 0
        assert "switch edit format" not in ee.escalation_hint("b.py", "replace")

    def test_equivalent_paths_share_a_counter(self, tmp_path):
        target = tmp_path / "same.py"
        target.write_text("x = 1\n", encoding="utf-8")
        ee.record_failure(str(target))
        ee.record_failure(str(tmp_path / "sub" / ".." / "same.py"))
        assert ee.failure_count(str(target)) == 2

    def test_empty_path_is_ignored(self):
        assert ee.record_failure("") == 0
        assert ee.failure_count("") == 0
        assert ee.escalation_hint("", "replace") == ""

    def test_threads_do_not_escalate_each_other(self):
        """Delegated agents run concurrently on their own threads."""
        ee.record_failure("shared.py")
        ee.record_failure("shared.py")
        seen = []

        def worker():
            seen.append(ee.failure_count("shared.py"))

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert seen == [0], "one agent's failures escalated another's edits"


# ---------------------------------------------------------------------------
# (3) Success clears the streak
# ---------------------------------------------------------------------------

class TestSuccessResets:
    def test_success_clears_the_counter(self):
        ee.record_failure("a.py")
        ee.record_failure("a.py")
        ee.record_success("a.py")
        assert ee.failure_count("a.py") == 0
        assert ee.escalation_hint("a.py", "replace") == ""

    def test_success_on_another_file_does_not_clear(self):
        ee.record_failure("a.py")
        ee.record_success("b.py")
        assert ee.failure_count("a.py") == 1

    def test_reset_clears_everything(self):
        ee.record_failure("a.py")
        ee.record_failure("b.py")
        ee.reset()
        assert ee.failure_count("a.py") == 0
        assert ee.failure_count("b.py") == 0


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
# (6) Production wiring - static guards against dead code
# ---------------------------------------------------------------------------

class TestProductionWiring:
    def test_replace_mode_records_failures(self):
        from tools.file_operations import ShellFileOperations

        src = inspect.getsource(ShellFileOperations.patch_replace)
        assert "record_failure" in src and "escalation_hint" in src, (
            "patch_replace no longer escalates - a model can resend the same "
            "failing edit forever and never be told to switch format"
        )

    def test_replace_mode_clears_on_success(self):
        from tools.file_operations import ShellFileOperations

        assert "record_success" in inspect.getsource(ShellFileOperations.patch_replace), (
            "patch_replace no longer clears the streak - files stay "
            "permanently escalated after one bad edit"
        )

    def test_v4a_records_both_outcomes(self):
        from tools.file_operations import ShellFileOperations

        src = inspect.getsource(ShellFileOperations.patch_v4a)
        for symbol in ("record_failure", "record_success", "escalation_hint"):
            assert symbol in src, f"patch_v4a no longer calls {symbol}"

    def test_coding_brief_emits_the_escalation_line(self):
        from agent.coding_context import RuntimeMode

        assert "_edit_escalation_line" in inspect.getsource(RuntimeMode.system_blocks), (
            "system_blocks no longer emits the escalation line - "
            "retry_format_chain is dead again"
        )

    def test_turn_boundary_resets_tracking(self):
        import agent.turn_context as tc

        assert "edit_escalation" in inspect.getsource(tc), (
            "no turn-boundary reset - last turn's failures escalate this "
            "turn's first attempt"
        )


# ---------------------------------------------------------------------------
# (7) Advisory only - must never break an edit
# ---------------------------------------------------------------------------

class TestNeverBreaksAnEdit:
    def test_hint_is_an_appendable_suffix(self):
        ee.record_failure("a.py")
        hint = ee.escalation_hint("a.py", "replace")
        assert hint.startswith("\n"), "hint must append cleanly to an error"

    def test_real_error_text_survives_escalation(self, tmp_path):
        """The escalation must not displace the match error the model needs."""
        from tools.environments.local import LocalEnvironment
        from tools.file_operations import ShellFileOperations

        target = tmp_path / "f.py"
        target.write_text("alpha = 1\n", encoding="utf-8")
        ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))

        result = ops.patch_replace(str(target), "nonexistent_text_zzz", "x")
        assert result.error, "expected a failed match"
        assert "nonexistent_text_zzz" in result.error or "match" in result.error.lower()

        second = ops.patch_replace(str(target), "nonexistent_text_zzz", "x")
        assert second.error
        assert "switch edit format" in second.error, (
            "second failure on the same file did not escalate in production"
        )

    def test_successful_patch_clears_the_streak_in_production(self, tmp_path):
        from tools.environments.local import LocalEnvironment
        from tools.file_operations import ShellFileOperations

        target = tmp_path / "g.py"
        target.write_text("alpha = 1\n", encoding="utf-8")
        ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))

        ops.patch_replace(str(target), "no_such_text_qqq", "x")
        assert ee.failure_count(str(target)) >= 1

        ok = ops.patch_replace(str(target), "alpha = 1", "alpha = 2")
        if ok.success:
            assert ee.failure_count(str(target)) == 0, (
                "a successful edit left the file escalated"
            )

    def test_v4a_escalates_in_production(self, tmp_path):
        """Behavioural guard for the V4A path.

        The static guard alone cannot catch a deleted call: the imports keep
        the symbol names in the source. Drive a failing patch twice and
        require the second error to escalate.
        """
        from tools.environments.local import LocalEnvironment
        from tools.file_operations import ShellFileOperations

        target = tmp_path / "h.py"
        target.write_text("alpha = 1\n", encoding="utf-8")
        ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))

        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {target}\n"
            "@@\n"
            "-context_that_is_not_in_the_file_zzz\n"
            "+replacement\n"
            "*** End Patch\n"
        )
        first = ops.patch_v4a(patch)
        assert first.error, "expected the V4A patch to fail"
        assert ee.failure_count(str(target)) == 1, (
            "V4A failure was not recorded against the file"
        )

        second = ops.patch_v4a(patch)
        assert second.error
        assert "switch edit format" in second.error, (
            "second V4A failure on the same file did not escalate"
        )

    def test_v4a_primary_path_prefers_update(self):
        from tools.file_operations import _v4a_primary_path
        from tools.patch_parser import OperationType, PatchOperation

        ops = [
            PatchOperation(operation=OperationType.ADD, file_path="new.py"),
            PatchOperation(operation=OperationType.UPDATE, file_path="old.py"),
        ]
        assert _v4a_primary_path(ops) == "old.py"

    def test_v4a_primary_path_is_total(self):
        from tools.file_operations import _v4a_primary_path

        assert _v4a_primary_path([]) == ""
        assert _v4a_primary_path(None) == ""
        assert _v4a_primary_path([object()]) == ""

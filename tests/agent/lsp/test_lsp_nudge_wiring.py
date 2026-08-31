"""Tests for LSP error-nudge wiring into the live conversation loop.

Deliverable D4 — covers every wiring point:
  1. _maybe_lsp_diagnostics records structured diagnostics at the tool layer.
  2. A file with errors then re-edited clean is CLEARED from tracking.
  3. reset_lsp_error_tracking() is invoked on turn start.
  4. INTEGRATION: the LSP nudge reaches the model through the real turn-end
     path in conversation_loop.py (not merely that build_lsp_error_nudge
     returns a string in isolation).
  5. Shared cap: verify + LSP nudges together cannot exceed max_verify_nudges.
  6. D5: feedback_in_loop=False gate — _maybe_lsp_diagnostics returns "" and
     records nothing.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import textwrap

import pytest

from agent.verification_stop import (
    _introduced_lsp_errors,
    build_lsp_error_nudge,
    record_lsp_diagnostics,
    reset_lsp_error_tracking,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diag(line=0, col=0, msg="undefined variable 'x'", code="E001"):
    """Synthetic LSP diagnostic dict (severity 1 = ERROR)."""
    return {
        "severity": 1,
        "range": {"start": {"line": line, "character": col}},
        "message": msg,
        "code": code,
        "source": "pyright",
    }


def _make_file_ops():
    """Return a ShellFileOperations instance with a mock terminal env,
    bypassing __init__ to avoid side effects."""
    from tools.file_operations import ShellFileOperations
    ops = ShellFileOperations.__new__(ShellFileOperations)
    ops.env = SimpleNamespace()  # non-None, non-LocalEnvironment
    ops.cwd = "/repo"
    return ops


@pytest.fixture(autouse=True)
def _clean_tracking():
    """Reset module-level tracking before each test."""
    reset_lsp_error_tracking()
    yield
    reset_lsp_error_tracking()


@pytest.fixture
def clear_env(monkeypatch):
    """Clear env signals so _session_is_messaging_surface resolves to False."""
    for var in (
        "HERMES_VERIFY_ON_STOP",
        "HERMES_PLATFORM",
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_SOURCE",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def feedback_on(monkeypatch):
    """Ensure get_feedback_in_loop() returns True."""
    from agent.lsp import reporter
    reporter._LSP_CONFIG = {"feedback_in_loop": True}
    yield
    reporter._LSP_CONFIG = None


@pytest.fixture
def feedback_off(monkeypatch):
    """Ensure get_feedback_in_loop() returns False."""
    from agent.lsp import reporter
    reporter._LSP_CONFIG = {"feedback_in_loop": False}
    yield
    reporter._LSP_CONFIG = None


# ---------------------------------------------------------------------------
# (1) Tool-layer: _maybe_lsp_diagnostics records structured diagnostics
# ---------------------------------------------------------------------------

class TestMaybeLspDiagnosticsRecords:
    """_maybe_lsp_diagnostics must call record_lsp_diagnostics with the
    structured list returned by get_diagnostics_sync."""

    def test_records_errors_from_diagnostics(self, feedback_on):
        """When get_diagnostics_sync returns ERRORs, they are recorded."""
        mock_svc = MagicMock()
        mock_svc.enabled_for.return_value = True
        mock_svc.get_diagnostics_sync.return_value = [_diag(msg="bad import")]

        ops = _make_file_ops()
        # get_service is imported locally inside _maybe_lsp_diagnostics via
        # ``from agent.lsp import get_service`` — patch it at the source.
        with patch("agent.lsp.get_service", return_value=mock_svc), \
             patch.object(type(ops), "_lsp_local_only", return_value=True):
            result = ops._maybe_lsp_diagnostics("/repo/src/app.py")

        # The formatted string is returned (non-empty).
        assert result != ""
        # But more importantly, the structured diagnostics were recorded.
        assert "/repo/src/app.py" in _introduced_lsp_errors
        assert len(_introduced_lsp_errors["/repo/src/app.py"]) == 1
        assert _introduced_lsp_errors["/repo/src/app.py"][0]["message"] == "bad import"

    def test_records_empty_clears_tracking(self, feedback_on):
        """When get_diagnostics_sync returns [], the file is cleared."""
        # First, seed tracking with an error.
        record_lsp_diagnostics("/repo/src/app.py", [_diag(msg="stale")])
        assert "/repo/src/app.py" in _introduced_lsp_errors

        mock_svc = MagicMock()
        mock_svc.enabled_for.return_value = True
        mock_svc.get_diagnostics_sync.return_value = []

        ops = _make_file_ops()
        with patch("agent.lsp.get_service", return_value=mock_svc), \
             patch.object(type(ops), "_lsp_local_only", return_value=True):
            result = ops._maybe_lsp_diagnostics("/repo/src/app.py")

        # Empty diagnostics -> empty string returned.
        assert result == ""
        # And the file is CLEARED from tracking.
        assert "/repo/src/app.py" not in _introduced_lsp_errors


# ---------------------------------------------------------------------------
# (2) File with errors then re-edited clean is CLEARED from tracking
# ---------------------------------------------------------------------------

class TestErrorThenClean:
    """A file that had errors, then gets re-edited with no errors, must be
    cleared from the tracking dict so the nudge no longer fires."""

    def test_error_then_clean_no_nudge(self, clear_env, feedback_on):
        """Record error, then record empty -> nudge must not fire."""
        record_lsp_diagnostics("/repo/src/app.py", [_diag(msg="type error")])
        assert "/repo/src/app.py" in _introduced_lsp_errors

        # Model re-edits the file; diagnostics are now clean.
        record_lsp_diagnostics("/repo/src/app.py", [])
        assert "/repo/src/app.py" not in _introduced_lsp_errors

        nudge = build_lsp_error_nudge(attempts=0)
        assert nudge is None, "nudge must NOT fire after the model fixed the error"

    def test_error_then_clean_via_tool_layer(self, feedback_on):
        """Simulate the full tool-layer path: error on first edit, clean on second."""
        mock_svc = MagicMock()
        mock_svc.enabled_for.return_value = True

        ops = _make_file_ops()

        # First edit: error.
        mock_svc.get_diagnostics_sync.return_value = [_diag(msg="bad import")]
        with patch("agent.lsp.get_service", return_value=mock_svc), \
             patch.object(type(ops), "_lsp_local_only", return_value=True):
            ops._maybe_lsp_diagnostics("/repo/src/app.py")
        assert "/repo/src/app.py" in _introduced_lsp_errors

        # Second edit: clean.
        mock_svc.get_diagnostics_sync.return_value = []
        with patch("agent.lsp.get_service", return_value=mock_svc), \
             patch.object(type(ops), "_lsp_local_only", return_value=True):
            ops._maybe_lsp_diagnostics("/repo/src/app.py")
        assert "/repo/src/app.py" not in _introduced_lsp_errors


# ---------------------------------------------------------------------------
# (3) reset_lsp_error_tracking() is invoked on turn start
# ---------------------------------------------------------------------------

class TestResetOnTurnStart:
    """reset_lsp_error_tracking() must be called during turn-context setup."""

    def test_turn_context_calls_reset_lsp_error_tracking(self):
        """build_turn_context source must contain the reset_lsp_error_tracking call."""
        import inspect
        from agent.turn_context import build_turn_context
        source = inspect.getsource(build_turn_context)
        assert "reset_lsp_error_tracking" in source, (
            "build_turn_context must call reset_lsp_error_tracking"
        )

    def test_reset_clears_seeded_errors(self):
        """reset_lsp_error_tracking clears all tracked errors."""
        record_lsp_diagnostics("/a.py", [_diag(msg="err1")])
        record_lsp_diagnostics("/b.py", [_diag(msg="err2")])
        assert len(_introduced_lsp_errors) == 2

        reset_lsp_error_tracking()
        assert len(_introduced_lsp_errors) == 0


# ---------------------------------------------------------------------------
# (4) INTEGRATION: LSP nudge reaches the model through the real turn-end path
# ---------------------------------------------------------------------------

class TestLspNudgeIntegration:
    """Drive the REAL production code in agent/conversation_loop.py.

    These tests call ``apply_lsp_error_nudge`` — the exact function the
    turn-end loop invokes.  They contain NO copy of the production logic:
    neutralizing the real implementation must make them fail.
    """

    def test_real_helper_appends_synthetic_nudge(self, clear_env, feedback_on):
        from agent.conversation_loop import apply_lsp_error_nudge

        record_lsp_diagnostics("/repo/src/app.py", [_diag(msg="undefined 'foo'")])

        agent = SimpleNamespace(_verification_stop_nudges=0, _session_messages=[])
        messages = [{"role": "user", "content": "fix the bug"}]
        final_msg = {"role": "assistant", "content": "Done!", "finish_reason": "stop"}

        issued = apply_lsp_error_nudge(agent, messages, final_msg)

        assert issued is True, "helper must report that it issued a nudge"
        assert len(messages) == 3
        assert messages[1]["finish_reason"] == "lsp_error_required"
        assert messages[1]["_verification_stop_synthetic"] is True
        assert messages[2]["role"] == "user"
        assert messages[2]["_verification_stop_synthetic"] is True
        assert "undefined" in messages[2]["content"]
        assert agent._verification_stop_nudges == 1
        assert agent._session_messages is messages

    def test_real_helper_no_nudge_when_no_errors(self, clear_env, feedback_on):
        from agent.conversation_loop import apply_lsp_error_nudge

        agent = SimpleNamespace(_verification_stop_nudges=0, _session_messages=[])
        messages = [{"role": "user", "content": "hi"}]
        final_msg = {"role": "assistant", "content": "Done!", "finish_reason": "stop"}

        issued = apply_lsp_error_nudge(agent, messages, final_msg)

        assert issued is False
        assert len(messages) == 1, "no messages may be appended when clean"
        assert agent._verification_stop_nudges == 0

    def test_real_helper_respects_shared_cap(self, clear_env, feedback_on):
        from agent.conversation_loop import apply_lsp_error_nudge
        from agent.verify_hooks import max_verify_nudges

        record_lsp_diagnostics("/repo/src/app.py", [_diag(msg="undefined 'foo'")])

        agent = SimpleNamespace(
            _verification_stop_nudges=max_verify_nudges(), _session_messages=[]
        )
        messages = [{"role": "user", "content": "x"}]
        final_msg = {"role": "assistant", "content": "Done!", "finish_reason": "stop"}

        assert apply_lsp_error_nudge(agent, messages, final_msg) is False
        assert len(messages) == 1

    def test_production_turn_loop_calls_the_helper(self):
        """Static guard: the turn-end loop must actually CALL the helper.

        The behavioral tests above exercise the helper directly, so they
        would still pass if someone deleted the call site.  This asserts
        the call exists inside run_conversation, closing that gap.
        """
        import ast
        import inspect
        import agent.conversation_loop as cl

        src = inspect.getsource(cl.run_conversation)
        tree = ast.parse(textwrap.dedent(src))
        called = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "apply_lsp_error_nudge" in called, (
            "run_conversation no longer calls apply_lsp_error_nudge — the LSP "
            "error gate is dead code and will never fire in production."
        )


class TestSharedCap:
    """Both nudge kinds draw from the same _verification_stop_nudges counter."""

    def test_shared_cap_honored(self, clear_env, feedback_on):
        """After verify-on-stop uses some budget, LSP nudge respects the remainder."""
        from agent.verify_hooks import max_verify_nudges

        cap = max_verify_nudges()  # default 3

        record_lsp_diagnostics("/repo/src/app.py", [_diag(msg="type error")])

        # At cap, LSP nudge must not fire.
        nudge = build_lsp_error_nudge(attempts=cap, max_attempts=cap)
        assert nudge is None, (
            f"LSP nudge must NOT fire when attempts ({cap}) >= max_attempts ({cap})"
        )

        # One below cap, LSP nudge must fire.
        nudge = build_lsp_error_nudge(attempts=cap - 1, max_attempts=cap)
        assert nudge is not None, (
            f"LSP nudge must fire when attempts ({cap - 1}) < max_attempts ({cap})"
        )

    def test_verify_then_lsp_shares_counter(self, clear_env, feedback_on):
        """The _verification_stop_nudges counter is shared between both nudge kinds."""
        from agent.verify_hooks import max_verify_nudges

        cap = max_verify_nudges()

        record_lsp_diagnostics("/repo/src/app.py", [_diag(msg="err")])

        # LSP nudge at attempts=2 (verify used 2) -> should fire (2 < 3).
        nudge = build_lsp_error_nudge(attempts=2, max_attempts=cap)
        assert nudge is not None

        # LSP nudge at attempts=3 (cap reached) -> should NOT fire.
        nudge = build_lsp_error_nudge(attempts=3, max_attempts=cap)
        assert nudge is None


# ---------------------------------------------------------------------------
# (6) D5: feedback_in_loop=False gate — returns "" and records nothing
# ---------------------------------------------------------------------------

class TestFeedbackInLoopGate:
    """When lsp.feedback_in_loop is False, _maybe_lsp_diagnostics must return
    "" and must NOT record any diagnostics (the whole LSP feedback path is
    suppressed)."""

    def test_feedback_off_returns_empty_and_records_nothing(self, feedback_off):
        """With feedback_in_loop=False, _maybe_lsp_diagnostics returns "" and
        does NOT call record_lsp_diagnostics."""
        mock_svc = MagicMock()
        mock_svc.enabled_for.return_value = True
        mock_svc.get_diagnostics_sync.return_value = [_diag(msg="should not be recorded")]

        ops = _make_file_ops()
        with patch("agent.lsp.get_service", return_value=mock_svc), \
             patch.object(type(ops), "_lsp_local_only", return_value=True):
            result = ops._maybe_lsp_diagnostics("/repo/src/app.py")

        assert result == "", "must return empty when feedback_in_loop is False"
        assert "/repo/src/app.py" not in _introduced_lsp_errors, (
            "must NOT record diagnostics when feedback_in_loop is False"
        )

    def test_feedback_off_does_not_clear_existing_tracking(self, feedback_off):
        """When feedback_in_loop=False, existing tracking is NOT cleared
        (the recording path is never reached)."""
        # Seed tracking with an error.
        record_lsp_diagnostics("/repo/src/app.py", [_diag(msg="pre-existing")])
        assert "/repo/src/app.py" in _introduced_lsp_errors

        mock_svc = MagicMock()
        mock_svc.enabled_for.return_value = True
        mock_svc.get_diagnostics_sync.return_value = []  # clean edit

        ops = _make_file_ops()
        with patch("agent.lsp.get_service", return_value=mock_svc), \
             patch.object(type(ops), "_lsp_local_only", return_value=True):
            ops._maybe_lsp_diagnostics("/repo/src/app.py")

        # The pre-existing error is still tracked because the feedback gate
        # returned early before reaching record_lsp_diagnostics.
        assert "/repo/src/app.py" in _introduced_lsp_errors, (
            "feedback_in_loop=False means recording is skipped entirely"
        )

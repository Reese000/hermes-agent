"""Tests for LSP ERROR diagnostic tracking and turn-end nudge.

Three behavioral tests, each mutation-tested:
  (a) unresolved ERROR at turn end -> nudge fires, naming the file
  (b) model fixed it -> no nudge
  (c) cap respected -> after max_verify_nudges, no further nudge
"""
from __future__ import annotations

import pytest

from agent.verification_stop import (
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


# ---------------------------------------------------------------------------
# (a) Unresolved ERROR at turn end -> nudge fires, naming the file
# ---------------------------------------------------------------------------

def test_unresolved_error_fires_nudge(clear_env, feedback_on):
    """An unresolved ERROR diagnostic produces a nudge naming the file."""
    record_lsp_diagnostics("/repo/src/app.py", [_diag(msg="cannot resolve symbol 'foo'")])

    nudge = build_lsp_error_nudge(attempts=0)

    assert nudge is not None, "nudge must fire when ERROR diagnostics are unresolved"
    assert "/repo/src/app.py" in nudge, "nudge must name the offending file"
    assert "cannot resolve symbol" in nudge, "nudge must include the diagnostic message"
    assert "LSP ERROR" in nudge, "nudge must identify itself as an LSP ERROR nudge"


# ---------------------------------------------------------------------------
# (b) Model fixed it -> no nudge
# ---------------------------------------------------------------------------

def test_fixed_error_produces_no_nudge(clear_env, feedback_on):
    """When a subsequent edit clears the ERROR, no nudge fires."""
    record_lsp_diagnostics("/repo/src/app.py", [_diag(msg="cannot resolve symbol 'foo'")])
    # Model re-edits the file and the error is gone.
    record_lsp_diagnostics("/repo/src/app.py", [])

    nudge = build_lsp_error_nudge(attempts=0)

    assert nudge is None, "nudge must NOT fire after the model fixed the error"


# ---------------------------------------------------------------------------
# (c) Cap respected -> after max_verify_nudges, no further nudge
# ---------------------------------------------------------------------------

def test_cap_respected_no_nudge_after_max(clear_env, feedback_on):
    """After max_attempts is reached, no further LSP error nudge fires."""
    from agent.verify_hooks import max_verify_nudges

    cap = max_verify_nudges()  # default 3
    record_lsp_diagnostics("/repo/src/app.py", [_diag(msg="type mismatch")])

    # At the cap boundary, nudge must be suppressed.
    nudge = build_lsp_error_nudge(attempts=cap, max_attempts=cap)
    assert nudge is None, (
        f"nudge must NOT fire when attempts ({cap}) >= max_attempts ({cap})"
    )

    # One below the cap, nudge must fire (sanity check).
    nudge = build_lsp_error_nudge(attempts=cap - 1, max_attempts=cap)
    assert nudge is not None, (
        f"nudge must fire when attempts ({cap - 1}) < max_attempts ({cap})"
    )

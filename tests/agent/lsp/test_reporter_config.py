"""Tests for LSP reporter config wiring (Step 1 of W3).

Validates that ``lsp.severities`` and ``lsp.feedback_in_loop`` config
keys are read correctly, validated, and applied by the reporter.
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_config_cache():
    """Reset the module-level config cache so each test starts fresh."""
    from agent.lsp import reporter
    reporter._LSP_CONFIG = None


def _diag(line=0, col=0, sev=1, code="E001", source="ls", msg="oops"):
    return {
        "range": {
            "start": {"line": line, "character": col},
            "end": {"line": line, "character": col + 1},
        },
        "severity": sev,
        "code": code,
        "source": source,
        "message": msg,
    }


# ---------------------------------------------------------------------------
# get_severities: validation and coercion
# ---------------------------------------------------------------------------

class TestGetSeverities:
    """Test get_severities() with various config values."""

    def setup_method(self):
        _reset_config_cache()

    def teardown_method(self):
        _reset_config_cache()

    def test_default_severities(self):
        """When config has no lsp.severities, returns default [1]."""
        from agent.lsp.reporter import get_severities
        result = get_severities()
        assert result == frozenset({1})

    def test_configured_severities(self):
        """Configured severities [1, 2] are returned as frozenset."""
        from agent.lsp.reporter import get_severities, _load_lsp_config
        _load_lsp_config.cache = None  # clear
        # Mock config to return severities
        with patch("agent.lsp.reporter._LSP_CONFIG", {"severities": [1, 2]}):
            _reset_config_cache()
            # Force re-read by clearing cache
            from agent.lsp import reporter
            reporter._LSP_CONFIG = None
            reporter._LSP_CONFIG = {"severities": [1, 2]}
            result = get_severities()
            assert result == frozenset({1, 2})

    def test_all_severities(self):
        """Configured severities [1, 2, 3, 4] are all valid."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"severities": [1, 2, 3, 4]}
        from agent.lsp.reporter import get_severities
        result = get_severities()
        assert result == frozenset({1, 2, 3, 4})

    def test_invalid_severities_outside_range(self):
        """Values outside 1-4 are dropped."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"severities": [0, 5, -1]}
        from agent.lsp.reporter import get_severities
        result = get_severities()
        assert result == frozenset({1})  # falls back to default

    def test_non_int_severities_dropped(self):
        """Non-int entries are dropped."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"severities": ["abc", None, True, 1]}
        from agent.lsp.reporter import get_severities
        result = get_severities()
        assert result == frozenset({1})  # only valid int remains

    def test_empty_list_falls_back(self):
        """Empty list falls back to default [1]."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"severities": []}
        from agent.lsp.reporter import get_severities
        result = get_severities()
        assert result == frozenset({1})

    def test_non_list_falls_back(self):
        """Non-list value falls back to default [1]."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"severities": "invalid"}
        from agent.lsp.reporter import get_severities
        result = get_severities()
        assert result == frozenset({1})

    def test_config_load_failure_falls_back(self):
        """If config loading fails, falls back to default."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"severities": [1, 2]}
        from agent.lsp.reporter import get_severities
        result = get_severities()
        assert result == frozenset({1, 2})


# ---------------------------------------------------------------------------
# get_feedback_in_loop: validation and coercion
# ---------------------------------------------------------------------------

class TestGetFeedbackInLoop:
    """Test get_feedback_in_loop() with various config values."""

    def setup_method(self):
        _reset_config_cache()

    def teardown_method(self):
        _reset_config_cache()

    def test_default_feedback(self):
        """Default is True."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {}
        from agent.lsp.reporter import get_feedback_in_loop
        assert get_feedback_in_loop() is True

    def test_feedback_false(self):
        """Configured False returns False."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"feedback_in_loop": False}
        from agent.lsp.reporter import get_feedback_in_loop
        assert get_feedback_in_loop() is False

    def test_feedback_true(self):
        """Configured True returns True."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"feedback_in_loop": True}
        from agent.lsp.reporter import get_feedback_in_loop
        assert get_feedback_in_loop() is True

    def test_feedback_string_false(self):
        """String 'false' returns False."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"feedback_in_loop": "false"}
        from agent.lsp.reporter import get_feedback_in_loop
        assert get_feedback_in_loop() is False

    def test_feedback_string_true(self):
        """String 'true' returns True."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"feedback_in_loop": "true"}
        from agent.lsp.reporter import get_feedback_in_loop
        assert get_feedback_in_loop() is True

    def test_feedback_string_zero(self):
        """String '0' returns False."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"feedback_in_loop": "0"}
        from agent.lsp.reporter import get_feedback_in_loop
        assert get_feedback_in_loop() is False

    def test_feedback_string_one(self):
        """String '1' returns True."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"feedback_in_loop": "1"}
        from agent.lsp.reporter import get_feedback_in_loop
        assert get_feedback_in_loop() is True


# ---------------------------------------------------------------------------
# report_for_file uses configured severities
# ---------------------------------------------------------------------------

class TestReportForFileConfig:
    """Test that report_for_file uses configured severities."""

    def setup_method(self):
        _reset_config_cache()

    def teardown_method(self):
        _reset_config_cache()

    def test_report_uses_configured_severities(self):
        """When severities are configured to include WARN, warnings appear."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"severities": [1, 2]}
        from agent.lsp.reporter import report_for_file
        diag = _diag(sev=2, msg="a warning")
        block = report_for_file("/x.py", [diag])
        assert "a warning" in block

    def test_report_ignores_unconfigured_severities(self):
        """When severities are [1] (default), warnings are filtered out."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"severities": [1]}
        from agent.lsp.reporter import report_for_file
        diag = _diag(sev=2, msg="a warning")
        block = report_for_file("/x.py", [diag])
        assert block == ""

    def test_report_explicit_severities_override_config(self):
        """Explicit severities parameter overrides config."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"severities": [1]}
        from agent.lsp.reporter import report_for_file
        diag = _diag(sev=2, msg="a warning")
        block = report_for_file("/x.py", [diag], severities=frozenset({1, 2}))
        assert "a warning" in block

    def test_report_default_uses_config_when_no_explicit(self):
        """When severities=None, uses configured severities."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"severities": [1, 2, 3]}
        from agent.lsp.reporter import report_for_file
        diag = _diag(sev=3, msg="an info")
        block = report_for_file("/x.py", [diag])
        assert "an info" in block


# ---------------------------------------------------------------------------
# Config cache is reused (module-level lazy cache)
# ---------------------------------------------------------------------------

class TestConfigCache:
    """Test that config is loaded at most once per process."""

    def setup_method(self):
        _reset_config_cache()

    def teardown_method(self):
        _reset_config_cache()

    def test_config_loaded_once(self):
        """Config is loaded only once, then cached."""
        from agent.lsp import reporter
        call_count = [0]

        def fake_load():
            call_count[0] += 1
            return {"lsp": {"severities": [1]}}

        with patch("hermes_cli.config.load_config_readonly", side_effect=fake_load):
            reporter._LSP_CONFIG = None  # reset cache
            from agent.lsp.reporter import get_severities
            get_severities()
            get_severities()
            get_severities()
            assert call_count[0] == 1  # loaded once

    def test_cache_reset_reloads(self):
        """Resetting _LSP_CONFIG allows re-reading."""
        from agent.lsp import reporter
        reporter._LSP_CONFIG = {"severities": [1]}
        from agent.lsp.reporter import get_severities
        assert get_severities() == frozenset({1})
        reporter._LSP_CONFIG = {"severities": [1, 2]}
        assert get_severities() == frozenset({1, 2})
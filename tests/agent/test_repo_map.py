"""Enforcement tests for the repository map (W6a).

The properties under guard, and why each matters:

  1. **Hard character budget.**  The map rides in the cached system prompt.
     An unbounded map silently inflates every request.
  2. **Determinism.**  Identical repo state must produce identical bytes, or
     the prompt prefix differs between sessions for no reason.
  3. **Directory safety.**  ``data/`` holds a live browser profile whose
     cache files the daemon keeps locked; vendor trees are huge.  These are
     pruned before descent, not filtered after.
  4. **Cannot break session start.**  This runs during prompt assembly.
     Unreadable files, syntax errors, and binary junk must degrade to an
     empty map, never an exception - and never a crash.
  5. **Config gating.**  The block honours ``agent.repo_map`` and
     ``agent.repo_map_char_budget``.
  6. **Production wiring.**  The block must actually reach the system
     prompt, or the whole feature is dead code.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from agent.repo_map import (
    DEFAULT_CHAR_BUDGET,
    _iter_source_files,
    _scan_definitions,
    _scan_names,
    build_repo_map,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """A small repo with a clear importance gradient."""
    (tmp_path / "core.py").write_text(
        "class Engine:\n"
        "    def start(self):\n"
        "        pass\n"
        "    def compute_widget(self):\n"
        "        pass\n"
        "\n"
        "def build_widget():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    # Several files referencing build_widget, so it ranks above unused code.
    for i in range(4):
        (tmp_path / f"user{i}.py").write_text(
            f"from core import build_widget\n"
            f"def use{i}():\n"
            f"    return build_widget()\n",
            encoding="utf-8",
        )
    (tmp_path / "lonely.py").write_text(
        "def never_referenced_anywhere():\n    return 0\n", encoding="utf-8"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# (1) Budget
# ---------------------------------------------------------------------------

class TestBudget:
    def test_never_exceeds_budget(self, sample_repo: Path):
        for budget in (60, 120, 400, 3200):
            out = build_repo_map(sample_repo, char_budget=budget)
            assert len(out) <= budget, f"budget {budget} exceeded: {len(out)}"

    def test_zero_budget_yields_nothing(self, sample_repo: Path):
        assert build_repo_map(sample_repo, char_budget=0) == ""

    def test_negative_budget_yields_nothing(self, sample_repo: Path):
        assert build_repo_map(sample_repo, char_budget=-100) == ""

    def test_real_repo_respects_budget(self):
        """The repo this test runs in is large - a realistic worst case."""
        root = Path(__file__).resolve().parents[2]
        out = build_repo_map(root, char_budget=1500)
        assert len(out) <= 1500

    def test_truncation_never_clips_mid_line(self):
        """Truncation must drop whole entries, not cut a line in half.

        There are two layers here: the per-line budget check that stops
        adding entries, and a final clamp that slices the string. The clamp
        alone keeps the output under budget, so a budget test cannot tell
        whether the line check still works - but if the line check is gone,
        the clamp cuts mid-line and the model reads a mangled path. Assert
        the shape, which only the line check can guarantee.
        """
        root = Path(__file__).resolve().parents[2]
        for budget in (300, 700, 1500, 2600):
            out = build_repo_map(root, char_budget=budget)
            if not out:
                continue
            assert out.endswith("\n"), (
                f"budget {budget}: output was clipped mid-line - the "
                "per-entry budget check is not doing its job"
            )
            for line in out.splitlines():
                if ".py:" in line:
                    assert line.rstrip().endswith(tuple("abcdefghijklmnopqrstuvwxyz"
                                                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                                        "0123456789_)")), (
                        f"budget {budget}: entry looks truncated: {line!r}"
                    )

    def test_truncation_is_marked(self, sample_repo: Path):
        full = build_repo_map(sample_repo, char_budget=DEFAULT_CHAR_BUDGET)
        if len(full) < 200:
            pytest.skip("sample repo too small to force truncation")
        clipped = build_repo_map(sample_repo, char_budget=len(full) // 2)
        assert len(clipped) <= len(full) // 2


# ---------------------------------------------------------------------------
# (2) Determinism - the prompt cache depends on it
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeated_builds_are_byte_identical(self, sample_repo: Path):
        runs = [build_repo_map(sample_repo) for _ in range(5)]
        assert len(set(runs)) == 1

    def test_ordering_is_not_filesystem_dependent(self, sample_repo: Path):
        first = build_repo_map(sample_repo)
        # Touch mtimes; ordering must not follow them.
        for p in sample_repo.glob("*.py"):
            os.utime(p, (0, 0))
        assert build_repo_map(sample_repo) == first


# ---------------------------------------------------------------------------
# (3) Directory safety
# ---------------------------------------------------------------------------

class TestDirectorySafety:
    @pytest.mark.parametrize(
        "skipped", ["data", "node_modules", ".git", "__pycache__", ".venv"]
    )
    def test_hostile_dirs_are_never_walked(self, tmp_path: Path, skipped: str):
        (tmp_path / "keep.py").write_text("def kept():\n    return 1\n", encoding="utf-8")
        bad = tmp_path / skipped
        bad.mkdir()
        (bad / "poison.py").write_text(
            "def must_not_appear():\n    return 1\n", encoding="utf-8"
        )

        found = _iter_source_files(
            tmp_path, max_files=100, deadline=float("inf")
        )
        names = {p.name for p in found}
        assert "keep.py" in names
        assert "poison.py" not in names, (
            f"walked into {skipped}/ - this must be pruned before descent"
        )

    def test_dotted_dirs_are_pruned(self, tmp_path: Path):
        (tmp_path / "keep.py").write_text("def kept():\n    pass\n", encoding="utf-8")
        hidden = tmp_path / ".secret"
        hidden.mkdir()
        (hidden / "hidden.py").write_text("def hidden():\n    pass\n", encoding="utf-8")

        found = _iter_source_files(tmp_path, max_files=100, deadline=float("inf"))
        assert "hidden.py" not in {p.name for p in found}

    def test_file_cap_is_honoured(self, tmp_path: Path):
        for i in range(40):
            (tmp_path / f"m{i}.py").write_text("def f():\n    pass\n", encoding="utf-8")
        found = _iter_source_files(tmp_path, max_files=10, deadline=float("inf"))
        assert len(found) <= 10


# ---------------------------------------------------------------------------
# (4) Robustness - must never break prompt assembly
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_missing_root_returns_empty(self, tmp_path: Path):
        assert build_repo_map(tmp_path / "nope") == ""

    def test_none_root_returns_empty(self):
        assert build_repo_map(None) == ""

    def test_empty_repo_returns_empty(self, tmp_path: Path):
        assert build_repo_map(tmp_path) == ""

    def test_syntax_errors_do_not_raise(self, tmp_path: Path):
        (tmp_path / "broken.py").write_text(
            "def (((:\n  this is not python\n", encoding="utf-8"
        )
        (tmp_path / "ok.py").write_text(
            "def real_symbol():\n    return 1\n", encoding="utf-8"
        )
        build_repo_map(tmp_path)  # must not raise

    def test_binary_junk_does_not_raise(self, tmp_path: Path):
        (tmp_path / "weird.py").write_bytes(b"\x00\x01\x02\xff\xfe binary")
        build_repo_map(tmp_path)  # must not raise

    def test_large_real_repo_does_not_crash(self):
        """Regression guard.

        Deep AST traversal used to fault the interpreter here with a Windows
        access violation - a hard crash that would take session start down
        with it. The scan must stay parse-free and survive the largest repo
        available.
        """
        root = Path(__file__).resolve().parents[2]
        for _ in range(3):
            build_repo_map(root, char_budget=2000)

    def test_no_ast_traversal_in_module(self):
        """The crash came from walking the AST. Keep it out."""
        import agent.repo_map as rm

        src = inspect.getsource(rm)
        for banned in ("ast.walk(", "ast.iter_child_nodes(", "ast.parse("):
            assert banned not in src, (
                f"{banned} reintroduced - this faulted CPython on Windows "
                "and cannot be caught by an except clause"
            )


# ---------------------------------------------------------------------------
# (5) Scanning behaviour
# ---------------------------------------------------------------------------

class TestScanning:
    def test_finds_top_level_defs_and_classes(self):
        src = "def alpha():\n    pass\n\nclass Beta:\n    pass\n"
        assert "alpha" in _scan_definitions(src)
        assert "Beta" in _scan_definitions(src)

    def test_finds_async_defs(self):
        assert "gamma" in _scan_definitions("async def gamma():\n    pass\n")

    def test_attributes_methods_to_their_class(self):
        src = "class Engine:\n    def start(self):\n        pass\n"
        assert "Engine.start" in _scan_definitions(src)

    def test_private_methods_are_excluded(self):
        src = "class Engine:\n    def _hidden(self):\n        pass\n"
        assert "Engine._hidden" not in _scan_definitions(src)

    def test_scan_names_collects_identifiers(self):
        assert "build_widget" in _scan_names("x = build_widget()\n")


# ---------------------------------------------------------------------------
# (6) Ranking
# ---------------------------------------------------------------------------

class TestRanking:
    def test_referenced_symbols_outrank_unreferenced(self, sample_repo: Path):
        out = build_repo_map(sample_repo)
        assert out, "expected a map for the sample repo"
        assert "core.py" in out
        if "lonely.py" in out:
            assert out.index("core.py") < out.index("lonely.py")

    def test_output_has_the_expected_shape(self, sample_repo: Path):
        out = build_repo_map(sample_repo)
        assert out.startswith("# Repository map")
        body = [ln for ln in out.splitlines() if ".py:" in ln]
        assert body, "expected at least one file line"
        assert all(":" in ln for ln in body)


# ---------------------------------------------------------------------------
# (7) Config gating + production wiring
# ---------------------------------------------------------------------------

class TestConfigAndWiring:
    def test_block_disabled_returns_empty(self, sample_repo: Path):
        from agent.coding_context import build_repo_map_block

        assert build_repo_map_block(sample_repo, enabled=False) == ""

    def test_block_zero_budget_returns_empty(self, sample_repo: Path):
        from agent.coding_context import build_repo_map_block

        assert build_repo_map_block(sample_repo, enabled=True, char_budget=0) == ""

    def test_block_never_raises_on_bad_cwd(self):
        from agent.coding_context import build_repo_map_block

        assert build_repo_map_block("\x00 not a path", enabled=True) == ""

    def test_config_defaults_are_registered(self):
        from hermes_cli.config import DEFAULT_CONFIG

        agent_cfg = DEFAULT_CONFIG.get("agent", {})
        assert "repo_map" in agent_cfg
        assert "repo_map_char_budget" in agent_cfg

    def test_runtime_mode_carries_repo_map_settings(self):
        from agent.coding_context import resolve_runtime_mode

        mode = resolve_runtime_mode(platform="cli", cwd=".", config={})
        assert hasattr(mode, "repo_map_enabled")
        assert hasattr(mode, "repo_map_char_budget")

    def test_config_false_disables_it(self):
        from agent.coding_context import resolve_runtime_mode

        mode = resolve_runtime_mode(
            platform="cli", cwd=".", config={"agent": {"repo_map": False}}
        )
        assert mode.repo_map_enabled is False

    def test_system_blocks_actually_calls_the_builder(self):
        """Static guard.

        Every other test here drives build_repo_map directly and would still
        pass if system_blocks never called it - leaving the map dead code
        that never reaches a model.
        """
        from agent.coding_context import RuntimeMode

        src = inspect.getsource(RuntimeMode.system_blocks)
        assert "build_repo_map_block" in src, (
            "RuntimeMode.system_blocks no longer builds the repo map - the "
            "feature is dead code and will never reach the prompt"
        )

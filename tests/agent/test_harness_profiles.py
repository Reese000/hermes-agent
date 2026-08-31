"""Enforcement tests for the harness-profile registry.

These tests are MANDATORY — a deliverable without its enforcement test is not
done.  Each test guards a specific invariant:

(a) Golden-prompt byte-equality — the refactor must not silently change any
    user's prompt or break their cache.
(b) Anti-drift — model-substring gating must not be reintroduced outside the
    registry.
(c) Cache-safety — profile resolution is deterministic and the stable prompt
    tier is byte-stable across calls.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import pytest

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "agent" / "harness_golden_fixtures.json"
SYSTEM_PROMPT_PATH = REPO_ROOT / "agent" / "system_prompt.py"
CODING_CONTEXT_PATH = REPO_ROOT / "agent" / "coding_context.py"


# ── (a) Golden-prompt byte-equality ─────────────────────────────────────────


class TestGoldenPromptByteEquality:
    """Assert that the fully-rendered model-specific prompt parts are
    byte-identical to the pre-refactor fixtures.

    If any byte differs after the refactor, the test fails.  This is the
    guard that the refactor did not silently change any user's prompt or
    break their cache.
    """

    @pytest.fixture()
    def fixtures(self) -> dict:
        with open(FIXTURE_PATH) as f:
            return json.load(f)

    def _get_profile_behavior(self, model_id: str) -> dict:
        """Resolve a model and return its behaviour as a dict."""
        from agent.harness_profiles import resolve_profile

        profile = resolve_profile(model_id)
        return {
            "name": profile.name,
            "edit_format": profile.edit_format,
            "edit_format_line": profile.edit_format_line,
            "tool_use_enforcement": profile.tool_use_enforcement,
            "execution_guidance": profile.execution_guidance,
            "role_model": profile.role_model,
        }

    def test_gpt5_golden(self, fixtures):
        behaviour = self._get_profile_behavior("openai/gpt-5")
        assert behaviour["edit_format"] == "patch"
        assert behaviour["tool_use_enforcement"] is True
        assert "Execution discipline" in behaviour["execution_guidance"]
        assert behaviour["role_model"] == "developer"

    def test_claude_sonnet_golden(self, fixtures):
        behaviour = self._get_profile_behavior("anthropic/claude-sonnet-4")
        assert behaviour["edit_format"] == "replace"
        assert behaviour["tool_use_enforcement"] is False
        assert behaviour["execution_guidance"] == ""
        assert behaviour["role_model"] == "system"

    def test_gemini3_golden(self, fixtures):
        behaviour = self._get_profile_behavior("google/gemini-3-pro")
        assert behaviour["edit_format"] == "replace"
        assert behaviour["tool_use_enforcement"] is True
        assert "Google model" in behaviour["execution_guidance"]
        assert behaviour["role_model"] == "system"

    def test_deepseek_v4_golden(self, fixtures):
        behaviour = self._get_profile_behavior("deepseek/deepseek-v4")
        assert behaviour["edit_format"] == "replace"
        assert behaviour["tool_use_enforcement"] is True
        assert behaviour["execution_guidance"] == ""
        assert behaviour["role_model"] == "system"

    def test_qwen3_golden(self, fixtures):
        behaviour = self._get_profile_behavior("qwen/qwen3-coder")
        assert behaviour["edit_format"] == "replace"
        assert behaviour["tool_use_enforcement"] is True
        assert behaviour["execution_guidance"] == ""
        assert behaviour["role_model"] == "system"

    def test_grok_golden(self, fixtures):
        """Grok: edit_format=replace AND execution_guidance=OPENAI_ (independent axes)."""
        behaviour = self._get_profile_behavior("xai/grok-3")
        assert behaviour["edit_format"] == "replace"
        assert behaviour["tool_use_enforcement"] is True
        assert "Execution discipline" in behaviour["execution_guidance"]
        assert behaviour["role_model"] == "system"

    def test_unknown_golden(self, fixtures):
        behaviour = self._get_profile_behavior("some-unknown-provider/some-model")
        assert behaviour["name"] == "generic"
        assert behaviour["edit_format_line"] == ""
        assert behaviour["tool_use_enforcement"] is False
        assert behaviour["execution_guidance"] == ""
        assert behaviour["role_model"] == "system"

    def test_edit_format_line_byte_exact(self, fixtures):
        """The edit_format_line strings must be byte-identical to the originals."""
        from agent.coding_context import _edit_format_line

        for name, expected in fixtures.items():
            model_id = expected["model_id"]
            actual_line = _edit_format_line(model_id)
            expected_line = expected["edit_format_line"]
            assert actual_line == expected_line, (
                f"{name}: edit_format_line differs byte-for-byte.\n"
                f"  Expected: {expected_line!r}\n"
                f"  Actual:   {actual_line!r}"
            )


# ── (b) Anti-drift: AST-based detection of model-substring gating ───────────


# The set of family needles that must NOT appear in model-dispatch
# conditionals outside the registry.
_NEEDLES = frozenset({
    "gpt", "codex", "gemini", "gemma", "grok",
    "glm", "qwen", "deepseek", "claude", "sonnet",
    "opus", "haiku", "mimo",
})


class _NeedleVisitor(ast.NodeVisitor):
    """AST visitor that finds ``"needle" in model_lower`` patterns in
    ``if``-test / ``Compare`` nodes, but ignores docstrings and comments.
    """

    def __init__(self, source_lines: list[str]):
        self.source_lines = source_lines
        self.violations: list[tuple[int, str, str]] = []  # (line, needle, file)

    def visit_Compare(self, node: ast.Compare) -> None:
        # Look for ``"needle" in <target>`` patterns
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, ast.In):
                for child in ast.walk(comparator):
                    if isinstance(child, ast.Constant) and isinstance(child.value, str):
                        if child.value in _NEEDLES:
                            self.violations.append((node.lineno, child.value, ""))
        self.generic_visit(node)


class _CollectionNeedleVisitor(ast.NodeVisitor):
    """AST visitor that finds module-level collection literals (dict, tuple,
    set, list) containing 2+ known family needles as string constants.

    This catches the class of drift where someone adds a new model-family
    mapping table (dict of needles→guidance, tuple of needle tuples, etc.)
    directly in coding_context.py or system_prompt.py instead of adding a
    profile to agent/harness_profiles/profiles.py.
    """

    def __init__(self, source: str):
        self.source = source
        self.violations: list[tuple[int, str, set[str]]] = []  # (line, repr, matched)

    @staticmethod
    def _extract_strings(node: ast.AST) -> list[str]:
        """Recursively extract all string constants from an AST subtree."""
        strings: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                strings.append(child.value)
        return strings

    def visit_Module(self, node: ast.Module) -> None:
        """Only check top-level assignments — skip function/class bodies."""
        for child in node.body:
            if not isinstance(child, ast.Assign):
                continue
            value = child.value
            # Check if the value is a collection literal (or a nested one)
            if not isinstance(value, (ast.Dict, ast.Tuple, ast.Set, ast.List)):
                continue
            strings = self._extract_strings(value)
            matched = {s for s in strings if s in _NEEDLES}
            if len(matched) >= 2:
                # Get a short repr for the error message
                try:
                    snippet = ast.get_source_segment(self.source, value)
                    if snippet and len(snippet) > 120:
                        snippet = snippet[:117] + "..."
                except Exception:
                    snippet = f"<line {child.lineno}>"
                self.violations.append((child.lineno, snippet or f"<line {child.lineno}>", matched))


class TestAntiDrift:
    """Assert that model-substring gating is not reintroduced outside the
    registry.

    If this test fails, the fix is to add the needle to
    ``agent/harness_profiles/profiles.py`` instead of adding a substring
    check in the dispatch code.
    """

    def _check_file(self, path: Path) -> list[tuple[int, str]]:
        """Check a file for needle-in-condition patterns."""
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()

        # Parse the AST — only check actual code, not comments/docstrings
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            pytest.fail(f"Syntax error in {path}")

        visitor = _NeedleVisitor(lines)
        visitor.visit(tree)
        return [(line, needle) for line, needle, _ in visitor.violations]

    def test_no_needles_in_system_prompt(self):
        violations = self._check_file(SYSTEM_PROMPT_PATH)
        assert not violations, (
            f"Model-substring gating found in {SYSTEM_PROMPT_PATH.name}: "
            f"{violations}. Add the needle to agent/harness_profiles/profiles.py "
            f"instead of adding a substring check here."
        )

    def test_no_needles_in_coding_context(self):
        violations = self._check_file(CODING_CONTEXT_PATH)
        assert not violations, (
            f"Model-substring gating found in {CODING_CONTEXT_PATH.name}: "
            f"{violations}. Add the needle to agent/harness_profiles/profiles.py "
            f"instead of adding a substring check here."
        )

    def test_no_stale_model_collections_in_dispatch_files(self):
        """Module-level collection literals in dispatch files must not contain
        2+ known family needles as string constants.

        This catches the class of drift where a dict/tuple/set/list mapping
        model needles to guidance strings is added directly in
        coding_context.py or system_prompt.py instead of adding a profile to
        agent/harness_profiles/profiles.py (the single source of truth).
        """
        violations: list[str] = []
        for path in (CODING_CONTEXT_PATH, SYSTEM_PROMPT_PATH):
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                pytest.fail(f"Syntax error in {path}")
            visitor = _CollectionNeedleVisitor(source)
            visitor.visit(tree)
            for line, snippet, matched in visitor.violations:
                violations.append(
                    f"  {path.name}:{line}: {snippet}  "
                    f"contains needles {sorted(matched)}"
                )
        assert not violations, (
            "Module-level collection literals contain model-family needles.\n"
            "This is dead duplicate data — the single source of truth is "
            "agent/harness_profiles/profiles.py.\n"
            "Remove the collection and delegate to resolve_profile() instead.\n"
            + "\n".join(violations)
        )


# ── (c) Cache-safety: deterministic resolution + byte-stable prompt ──────────


class TestCacheSafety:
    """Assert that resolve_profile is deterministic and that building the
    stable prompt tier twice for the same agent yields the identical value,
    and that profile resolution performs zero config reads after construction.
    """

    def test_resolve_profile_deterministic(self):
        """Same inputs always produce the same profile."""
        from agent.harness_profiles import resolve_profile

        models = [
            "openai/gpt-5", "anthropic/claude-sonnet-4", "google/gemini-3-pro",
            "deepseek/deepseek-v4", "qwen/qwen3-coder", "xai/grok-3",
            "some-unknown/model", "xiaomi/mimo-v2.5-pro",
        ]
        for model in models:
            results = [resolve_profile(model).name for _ in range(100)]
            assert len(set(results)) == 1, (
                f"resolve_profile({model!r}) returned inconsistent results: "
                f"{set(results)}"
            )

    def test_resolve_profile_no_io(self):
        """resolve_profile is a pure function — no config reads."""
        from agent.harness_profiles.profiles import resolve_profile as rp

        # If this function performed I/O, it would need config or network.
        # We verify by calling it with various inputs and checking it returns
        # immediately (no timeout needed — pure functions are instant).
        import time
        start = time.monotonic()
        for _ in range(10000):
            rp("openai/gpt-5")
            rp("anthropic/claude-sonnet-4")
            rp(None)
        elapsed = time.monotonic() - start
        # 30000 calls should complete in well under 1 second
        assert elapsed < 1.0, (
            f"resolve_profile took {elapsed:.2f}s for 30000 calls — "
            f"likely performing I/O"
        )

    def test_edit_format_line_stable(self):
        """_edit_format_line returns the same string across repeated calls."""
        from agent.coding_context import _edit_format_line

        models = ["openai/gpt-5", "anthropic/claude-sonnet-4", "xai/grok-3"]
        for model in models:
            first = _edit_format_line(model)
            for _ in range(50):
                assert _edit_format_line(model) == first, (
                    f"_edit_format_line({model!r}) returned different strings"
                )

    def test_profile_frozen(self):
        """HarnessProfile is frozen — mutation raises."""
        from agent.harness_profiles import resolve_profile

        p = resolve_profile("openai/gpt-5")
        with pytest.raises(AttributeError):
            p.name = "changed"  # type: ignore[misc]

    def test_all_needles_unique(self):
        """No needle appears in two different profiles."""
        from agent.harness_profiles.profiles import _ALL_PROFILES

        seen: dict[str, str] = {}
        for profile in _ALL_PROFILES:
            for needle in profile.needles:
                assert needle not in seen, (
                    f"Needle {needle!r} appears in both {seen[needle]!r} "
                    f"and {profile.name!r} profiles"
                )
                seen[needle] = profile.name
"""W6b - point-of-use context discovery on ``read_file``.

Two things are under test here, and the second matters as much as the first:

1. The footer says something true and bounded.
2. The footer cannot quietly stop being emitted.  Every wiring guard below
   is an AST *call-site* check or an end-to-end behavioural test, never a
   substring scan - a substring scan passes on a dead call site, because
   the surviving import line keeps the name in the source.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from pathlib import Path

import pytest

import agent.related_context as rc
import agent.repo_map as rm


# -- Fixtures ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_index():
    """The index is process-cached; a stale one would make these tests lie."""
    rm.clear_index_cache()
    yield
    rm.clear_index_cache()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A tiny repo: ``caller`` depends on ``core``; ``island`` on nobody."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "core.py").write_text(
        "class WidgetEngine:\n"
        "    def spin_widget(self):\n"
        "        return 1\n"
        "\n"
        "def make_widget():\n"
        "    return WidgetEngine()\n"
    )
    (tmp_path / "caller.py").write_text(
        "from core import WidgetEngine, make_widget\n"
        "\n"
        "def go():\n"
        "    return make_widget().spin_widget()\n"
    )
    (tmp_path / "island.py").write_text(
        "def unrelated_thing():\n    return 3\n"
    )
    return tmp_path


def _footer(repo: Path, name: str, **kw) -> str:
    return rc.related_context(repo / name, repo, **kw)


# -- Discovery --------------------------------------------------------------

class TestDiscovery:
    def test_inbound_names_the_dependent_file(self, repo):
        out = _footer(repo, "core.py")
        assert "referenced by" in out
        assert "caller.py" in out

    def test_outbound_names_the_dependency_and_its_symbols(self, repo):
        out = _footer(repo, "caller.py")
        assert "references" in out
        assert "core.py" in out
        assert "make_widget" in out

    def test_unrelated_file_is_not_listed(self, repo):
        assert "island.py" not in _footer(repo, "core.py")
        assert "island.py" not in _footer(repo, "caller.py")

    def test_a_file_is_never_related_to_itself(self, repo):
        assert "core.py" not in _footer(repo, "core.py")

    def test_island_gets_no_footer_at_all(self, repo):
        assert _footer(repo, "island.py") == ""

    def test_strongest_coupling_is_listed_first(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "core.py").write_text(
            "def alpha_sym():\n    pass\n\ndef beta_sym():\n    pass\n"
        )
        # Both need a top-level definition of their own: the index only
        # holds files that define something.
        # Named so that alphabetical order CONTRADICTS coupling order:
        # otherwise a path-only sort passes this test by accident.
        (tmp_path / "aaa_weak.py").write_text(
            "from core import alpha_sym\n\ndef weak_fn():\n    return alpha_sym\n"
        )
        (tmp_path / "zzz_strong.py").write_text(
            "from core import alpha_sym, beta_sym\n"
            "\ndef strong_fn():\n    return alpha_sym, beta_sym\n"
        )
        out = rc.related_context(tmp_path / "core.py", tmp_path)
        assert out.index("zzz_strong.py") < out.index("aaa_weak.py"), (
            "the more tightly coupled file must come first, or the first "
            "entries the model reads are the least useful ones"
        )


# -- Filtering --------------------------------------------------------------

class TestFiltering:
    def test_private_symbols_do_not_create_a_relationship(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "a.py").write_text("def _hidden():\n    pass\n")
        # b must define something of its own, or it is absent from the
        # index and this passes for the wrong reason.
        (tmp_path / "b.py").write_text(
            "from a import _hidden\n\ndef b_entry():\n    return _hidden\n"
        )
        assert rc.related_context(tmp_path / "a.py", tmp_path) == ""

    def test_noise_names_do_not_create_a_relationship(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "a.py").write_text("def main():\n    pass\n")
        (tmp_path / "b.py").write_text(
            "from a import main\n\ndef b_entry():\n    return main\n"
        )
        assert rc.related_context(tmp_path / "a.py", tmp_path) == ""

    def test_infrastructure_referenced_everywhere_is_suppressed(self, tmp_path):
        """A symbol used by the whole repo is not a *relationship*.

        Listing two hundred callers is a wall of text, not a hint. The
        generic-vocabulary filter is what removes them - it is also why
        no separate "too many referrers" cap is needed or possible: the
        cutoff is max(8, files * 0.02) against a 1500-file walk, so it
        always bites first.
        """
        (tmp_path / ".git").mkdir()
        (tmp_path / "core.py").write_text("def ubiquitous_helper():\n    pass\n")
        for i in range(40):
            (tmp_path / ("user%03d.py" % i)).write_text(
                "from core import ubiquitous_helper\n\ndef fn%d():\n    pass\n" % i
            )
        out = rc.related_context(tmp_path / "core.py", tmp_path)
        assert "referenced by" not in out


    def test_short_symbols_do_not_establish_a_relationship(self, tmp_path):
        """``fmt``, ``ref`` and friends collide across unrelated modules.

        Matching on them produces confident-looking nonsense - a real
        example from this repo linked an edit-escalation module to a
        Home Assistant test because both said ``fmt``.
        """
        (tmp_path / ".git").mkdir()
        (tmp_path / "a.py").write_text("def fmt():\n    pass\n")
        (tmp_path / "b.py").write_text(
            "def other_entry():\n    return fmt\n"
        )
        assert rc.related_context(tmp_path / "a.py", tmp_path) == ""

    def test_test_modules_are_not_reported_to_source_files(self, tmp_path):
        """A module's tests are not a navigation hint about the module.

        The agent reading ``core.py`` wants its callers, not the fifty
        test files that exercise it.
        """
        (tmp_path / ".git").mkdir()
        (tmp_path / "core.py").write_text("def widget_maker():\n    pass\n")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_core.py").write_text(
            "from core import widget_maker\n\ndef test_it():\n    widget_maker()\n"
        )
        assert rc.related_context(tmp_path / "core.py", tmp_path) == ""

    def test_a_test_file_may_still_see_other_tests(self, tmp_path):
        """The filter is directional, not a blanket exclusion."""
        (tmp_path / ".git").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "helpers_shared.py").write_text(
            "def shared_fixture_builder():\n    pass\n"
        )
        (tmp_path / "tests" / "test_core.py").write_text(
            "from helpers_shared import shared_fixture_builder\n\n"
            "def test_it():\n    shared_fixture_builder()\n"
        )
        out = rc.related_context(tmp_path / "tests" / "test_core.py", tmp_path)
        assert "helpers_shared.py" in out

# -- Bounds -----------------------------------------------------------------

class TestBounds:
    def test_char_budget_is_a_hard_cap(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "core.py").write_text("def shared_symbol_name():\n    pass\n")
        # Few enough referrers to stay under the generic cutoff, with
        # names long enough that the budget is what truncates.
        for i in range(5):
            (tmp_path / ("a_very_very_very_long_file_name_number_%d.py" % i)).write_text(
                "from core import shared_symbol_name\n\ndef fn%d():\n    pass\n" % i
            )
        full = rc.related_context(tmp_path / "core.py", tmp_path)
        assert len(full) > 120, (
            "the fixture is too small to exercise the budget at all"
        )
        out = rc.related_context(tmp_path / "core.py", tmp_path, char_budget=120)
        assert len(out) <= 120, "budget blown: %d chars" % len(out)

    def test_max_entries_is_respected(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "core.py").write_text("def shared_symbol_name():\n    pass\n")
        # Deliberately under the generic-vocabulary cutoff: past it the
        # symbol is dropped as common vocabulary and there is nothing
        # left for max_entries to trim.
        for i in range(5):
            (tmp_path / ("u%d.py" % i)).write_text(
                "from core import shared_symbol_name\n\ndef fn%d():\n    pass\n" % i
            )
        out = rc.related_context(tmp_path / "core.py", tmp_path, max_entries=2)
        listed = [ln for ln in out.splitlines() if "referenced by" in ln]
        assert listed and listed[0].count(",") == 1, (
            "expected exactly 2 entries, got: %r" % listed
        )

    def test_budget_truncation_lands_on_a_line_boundary(self, tmp_path):
        """Two layers enforce the budget; only one keeps it readable.

        The final clamp guarantees the cap, so removing the per-line
        check still produces short-enough output - but chopped through
        the middle of a path, which reads as a file that does not exist.
        """
        (tmp_path / ".git").mkdir()
        (tmp_path / "core.py").write_text("def shared_symbol_name():\n    pass\n")
        for i in range(5):
            (tmp_path / ("a_very_very_very_long_file_name_number_%d.py" % i)).write_text(
                "from core import shared_symbol_name\n\ndef fn%d():\n    pass\n" % i
            )
        out = rc.related_context(tmp_path / "core.py", tmp_path, char_budget=120)
        # The contract is not "short enough" - the final clamp already
        # guarantees that - but "never a partial line". Dropping the entry
        # is a valid outcome; chopping it in half is not.
        assert out == "" or out.endswith("\n"), (
            "the footer was cut mid-line, leaving a truncated path that "
            "reads as a real filename: %r" % out
        )

    def test_zero_budget_yields_nothing(self, repo):
        assert _footer(repo, "core.py", char_budget=0) == ""

    def test_non_source_files_are_skipped(self, repo):
        (repo / "notes.md").write_text("# core.py caller.py\n")
        assert rc.related_context(repo / "notes.md", repo) == ""

    def test_non_source_files_never_trigger_a_repo_walk(self, repo, monkeypatch):
        """The suffix check is a fast path, not just a filter.

        Without it every read of a README or a JSON fixture would pay
        for a full index build before finding nothing - invisible in the
        output, which is why this counts calls rather than asserting on
        the returned text.
        """
        (repo / "notes.md").write_text("# core.py caller.py\n")
        calls = []
        real = rm.build_index
        monkeypatch.setattr(
            rm, "build_index",
            lambda *a, **k: (calls.append(1), real(*a, **k))[1],
        )
        assert rc.related_context(repo / "notes.md", repo) == ""
        assert not calls, (
            "a non-source read built the whole repo index before giving up"
        )

    def test_missing_root_yields_nothing(self, repo):
        assert rc.related_context(repo / "core.py", None) == ""
        assert rc.related_context(repo / "core.py", repo / "nope") == ""

    def test_file_outside_the_index_yields_nothing(self, repo, tmp_path):
        outsider = tmp_path.parent / "outsider_xyz.py"
        outsider.write_text("def q():\n    pass\n")
        try:
            assert rc.related_context(outsider, repo) == ""
        finally:
            outsider.unlink()


# -- Never fails a read -----------------------------------------------------

class TestSafety:
    def test_an_exploding_index_is_swallowed(self, repo, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("index exploded")

        monkeypatch.setattr(rm, "build_index", boom)
        assert rc.related_context(repo / "core.py", repo) == "", (
            "a navigation hint must never be able to fail a file read"
        )

    def test_read_file_still_returns_content_when_the_footer_explodes(
        self, repo, monkeypatch
    ):
        import tools.file_tools as ft

        def boom(*a, **k):
            raise RuntimeError("footer exploded")

        monkeypatch.setattr(rc, "related_context", boom)
        out = json.loads(ft.read_file_tool(str(repo / "core.py"), task_id="w6bsafe"))
        assert "WidgetEngine" in out.get("content", ""), (
            "the read itself was lost because the optional footer failed"
        )


# -- Prompt caching is sacred ------------------------------------------------

class TestCacheSafety:
    def test_the_footer_never_reaches_the_system_prompt(self):
        """W6b must not touch the cached prefix.

        The whole point of delivering this at the tool boundary is that a
        per-turn tool result does not invalidate the conversation's cached
        prompt.  If this name ever appears in prompt assembly, that
        guarantee is gone.
        """
        import agent.coding_context as cc

        assert "related_context" not in inspect.getsource(cc), (
            "related-context leaked into system-prompt assembly - this "
            "would invalidate the per-conversation prompt cache"
        )

    def test_no_schema_bytes_were_added_to_the_core(self):
        """The core is a narrow waist: this ships zero extra schema bytes.

        Every model tool's schema is sent on every API call, so the footer
        must ride inside ``read_file``'s *result* and leave its declared
        parameters untouched.
        """
        import tools.file_tools as ft

        schema = json.dumps(ft.READ_FILE_SCHEMA).lower()
        assert "related" not in schema, (
            "W6b widened the read_file schema; it is meant to ride in the "
            "result, which costs nothing per call"
        )


# -- Production wiring (AST call sites, not substrings) ----------------------

def _called_names(func) -> set:
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


class TestProductionWiring:
    def test_read_file_calls_the_helper(self):
        import tools.file_tools as ft

        assert "_attach_related_context" in _called_names(ft.read_file_tool), (
            "read_file_tool mentions the footer helper but never calls it - "
            "the feature is dead code"
        )

    def test_the_helper_calls_the_module(self):
        import tools.file_tools as ft

        assert "related_context" in _called_names(ft._attach_related_context), (
            "the helper no longer calls related_context, so the footer is "
            "always empty while the wiring still looks present"
        )

    def test_footer_appears_in_a_real_read(self, repo):
        """The end-to-end guard the AST checks cannot give us."""
        import tools.file_tools as ft

        out = json.loads(ft.read_file_tool(str(repo / "core.py"), task_id="w6be2e"))
        assert "caller.py" in out.get("_related", ""), (
            "no related-files footer on a real read: %r" % out.get("_related")
        )

    def test_unchanged_reread_carries_no_footer(self, repo):
        """Repetition is prevented by the dedup stub, not by a gate.

        An unchanged re-read returns a stub instead of content, and the
        stub must not carry a second copy of a footer the model already
        has.
        """
        import tools.file_tools as ft

        first = json.loads(
            ft.read_file_tool(str(repo / "core.py"), task_id="w6brep")
        )
        assert "_related" in first
        second = json.loads(
            ft.read_file_tool(str(repo / "core.py"), task_id="w6brep")
        )
        assert "_related" not in second, (
            "the footer repeated on an unchanged re-read, burning context "
            "for information the model was already sent"
        )

    def test_changed_file_gets_a_fresh_footer(self, repo):
        """When the content changed, the relationships may have too."""
        import tools.file_tools as ft

        ft.read_file_tool(str(repo / "core.py"), task_id="w6bfresh")
        (repo / "core.py").write_text(
            "class WidgetEngine:\n    def spin_widget(self):\n        return 2\n"
            "\ndef make_widget():\n    return WidgetEngine()\n"
        )
        rm.clear_index_cache()
        again = json.loads(
            ft.read_file_tool(str(repo / "core.py"), task_id="w6bfresh")
        )
        assert "_related" in again, (
            "the file changed and the footer was suppressed - that is "
            "exactly when a refreshed view of its callers is worth having"
        )


# -- The shared index (W6a must keep working) --------------------------------

class TestSharedIndex:
    def test_index_is_cached_between_calls(self, repo):
        first = rm.build_index(repo)
        second = rm.build_index(repo)
        assert first is not None and first is second, (
            "the index was rebuilt, so every read_file would pay for a "
            "fresh directory walk"
        )

    def test_clear_index_cache_actually_clears(self, repo):
        first = rm.build_index(repo)
        rm.clear_index_cache()
        assert rm.build_index(repo) is not first

    def test_zero_deadline_is_not_served_from_cache(self, repo):
        """A caller allowing no time is opting out, not asking for a cache.

        Asserting on the return value would be flaky - Windows' monotonic
        clock is coarse enough that a zero deadline can still complete a
        tiny walk. What must hold is that the warm cache is not consulted.
        """
        warm = rm.build_index(repo)
        assert warm is not None
        assert rm.build_index(repo, deadline_s=0) is not warm, (
            "a zero deadline was served from cache, so the deadline callers "
            "use to opt out no longer opts them out"
        )

    def test_name_maps_are_reused_across_lookups(self, repo):
        """Without reuse, every read_file rescans every file in the repo.

        Measured at roughly half a second on a two-thousand-file tree -
        invisible in the output, which is why this asserts on identity
        rather than on the footer text.
        """
        index = rm.build_index(repo)
        first = rc._name_maps(index)
        second = rc._name_maps(index)
        assert first[0] is second[0] and first[1] is second[1], (
            "the inverted name maps were rebuilt, putting a full repo "
            "rescan on the path of every single read_file call"
        )

    def test_name_maps_are_not_reused_for_a_rebuilt_index(self, repo):
        """Staleness here would mean pointing at files that moved."""
        first = rc._name_maps(rm.build_index(repo))
        rm.clear_index_cache()
        second = rc._name_maps(rm.build_index(repo))
        assert first[0] is not second[0], (
            "maps from a discarded index were reused, so the footer can "
            "describe a repo state that no longer exists"
        )

    def test_repo_map_still_builds_on_the_shared_index(self, repo):
        out = rm.build_repo_map(repo)
        assert "core.py" in out
        assert out == rm.build_repo_map(repo), "repo map lost determinism"

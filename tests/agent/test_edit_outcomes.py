"""Enforcement tests for the edit keep-rate ledger (W7).

The properties under guard, and why each matters:

  1. **Content never reaches disk.**  The ledger is a long-lived file and
     the text flowing through it is the user's source code.  Only hashes
     may be stored.
  2. **Classification is right.**  Superseded, reverted, and kept must mean
     what the docstring says, or the number is worse than no number.
  3. **Scoping.**  An edit cannot be undone by a change to a different file
     or by a different session.
  4. **Passive by contract.**  A broken ledger must never fail an edit, and
     nothing here may feed back into the agent loop.
  5. **Bounded.**  Age prune, row cap, and report cap all real.
  6. **Production wiring.**  Static call-site guards plus end-to-end runs:
     the unit tests would all pass with the tool layer unwired.
  7. **Reporting survives an empty ledger.**  A fresh install must not
     crash /insights or show a wall of zeroes.
"""

from __future__ import annotations

import inspect
import json
import sqlite3

import pytest

from agent import edit_outcomes as eo


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """Point the ledger at a throwaway database."""
    db = tmp_path / "edit_outcomes.db"
    monkeypatch.setattr(eo, "_db_path", lambda: db)
    return db


def _rec(session, path, fmt, status=eo.APPLIED, removed=None, added=None):
    return eo.record_edit(
        session_id=session, path=path, edit_format=fmt, status=status,
        removed_text=removed, added_text=added,
    )


# ---------------------------------------------------------------------------
# (1) Only hashes are persisted
# ---------------------------------------------------------------------------

class TestNoContentOnDisk:
    def test_edit_text_never_lands_in_the_database(self, ledger):
        secret_old = "API_KEY = 'sk-live-do-not-store-me'"
        secret_new = "API_KEY = os.environ['API_KEY']"
        _rec("s1", "a.py", "replace", removed=secret_old, added=secret_new)

        # WAL mode means a fresh row may still be sitting in the -wal
        # sidecar rather than the main file. Scan every file the ledger
        # owns, or a raw-text regression hides in the journal.
        raw = b""
        for suffix in ("", "-wal", "-shm"):
            sidecar = ledger.parent / (ledger.name + suffix)
            if sidecar.exists():
                raw += sidecar.read_bytes()
        assert raw, "ledger wrote nothing at all"
        assert b"sk-live-do-not-store-me" not in raw, (
            "edit content was written to the ledger - this file outlives the "
            "session and holds the user's source"
        )
        assert b"os.environ" not in raw

    def test_only_digest_touches_content(self):
        """Structural guard on the module's one content-handling function."""
        src = inspect.getsource(eo)
        # Anything that would put raw text in a row would have to reference
        # the parameters outside _digest.
        record_src = inspect.getsource(eo.record_edit)
        for param in ("removed_text", "added_text"):
            uses = record_src.count(param)
            digested = record_src.count(f"_digest({param})")
            length = record_src.count(f"len({param} or")
            assert uses <= digested + length + 2, (
                f"{param} is used in record_edit outside _digest/len - "
                "raw content may be reaching the row"
            )
        assert "hashlib" in src

    def test_digest_is_stable_and_short(self):
        assert eo._digest("x") == eo._digest("x")
        assert eo._digest("x") != eo._digest("y")
        assert len(eo._digest("x")) == 16
        assert eo._digest("") == ""
        assert eo._digest(None) == ""


# ---------------------------------------------------------------------------
# (2) Classification
# ---------------------------------------------------------------------------

class TestClassification:
    def test_a_lone_edit_is_kept(self, ledger):
        _rec("s1", "a.py", "replace", removed="old", added="new")
        r = eo.keep_rate_report()
        assert (r["kept"], r["superseded"], r["reverted"]) == (1, 0, 0)
        assert r["keep_rate"] == 1.0

    def test_rewriting_your_own_output_is_superseded(self, ledger):
        _rec("s1", "a.py", "replace", removed="old", added="v1")
        _rec("s1", "a.py", "replace", removed="v1", added="v2")
        r = eo.keep_rate_report()
        assert r["superseded"] == 1, "the churned first edit was not detected"
        assert r["kept"] == 1, "the surviving second edit should count as kept"
        assert r["keep_rate"] == 0.5

    def test_undoing_your_own_edit_is_reverted(self, ledger):
        _rec("s1", "a.py", "replace", removed="original", added="changed")
        _rec("s1", "a.py", "replace", removed="changed", added="original")
        r = eo.keep_rate_report()
        # The first edit was put back exactly: reverted, not merely churned.
        assert r["reverted"] >= 1, "an exact undo was not detected"

    def test_whole_file_write_supersedes_earlier_edits(self, ledger):
        _rec("s1", "a.py", "replace", removed="old", added="new")
        _rec("s1", "a.py", "write_file", added="a completely different file")
        r = eo.keep_rate_report()
        assert r["superseded"] == 1, (
            "a whole-file rewrite left an earlier edit scored as kept"
        )

    def test_v4a_is_counted_but_not_credited_as_churn(self, ledger):
        _rec("s1", "a.py", "patch")
        _rec("s1", "a.py", "patch")
        r = eo.keep_rate_report()
        assert r["applied"] == 2
        assert r["superseded"] == 0, (
            "V4A patches were scored as churning each other, but their "
            "per-file text is not recorded - that verdict is unfounded"
        )

    def test_failures_do_not_count_as_kept(self, ledger):
        _rec("s1", "a.py", "replace", status=eo.FAILED)
        _rec("s1", "a.py", "replace", removed="old", added="new")
        r = eo.keep_rate_report()
        assert r["failed"] == 1
        assert r["applied"] == 1
        assert r["apply_rate"] == 0.5
        assert r["keep_rate"] == 1.0

    def test_per_format_breakdown_adds_up(self, ledger):
        _rec("s1", "a.py", "replace", removed="o", added="n")
        _rec("s1", "b.py", "write_file", added="whole file")
        _rec("s1", "c.py", "replace", status=eo.FAILED)
        r = eo.keep_rate_report()
        assert set(r["by_format"]) == {"replace", "write_file"}
        assert r["by_format"]["replace"]["applied"] == 1
        assert r["by_format"]["replace"]["failed"] == 1
        assert r["by_format"]["write_file"]["applied"] == 1


# ---------------------------------------------------------------------------
# (3) Scoping
# ---------------------------------------------------------------------------

class TestScoping:
    def test_edits_to_different_files_do_not_interact(self, ledger):
        _rec("s1", "a.py", "replace", removed="old", added="shared")
        _rec("s1", "b.py", "replace", removed="shared", added="other")
        r = eo.keep_rate_report()
        assert r["superseded"] == 0, (
            "an edit to one file was scored as churned by an edit to another"
        )

    def test_sessions_do_not_interact(self, ledger):
        _rec("s1", "a.py", "replace", removed="old", added="shared")
        _rec("s2", "a.py", "replace", removed="shared", added="other")
        r = eo.keep_rate_report()
        assert r["superseded"] == 0, (
            "one session's edit was scored as churned by another session's"
        )

    def test_report_can_filter_to_one_session(self, ledger):
        _rec("s1", "a.py", "replace", removed="o", added="n")
        _rec("s2", "b.py", "replace", removed="o", added="n")
        assert eo.keep_rate_report(session_id="s1")["applied"] == 1
        assert eo.keep_rate_report()["applied"] == 2

    def test_equivalent_paths_are_one_file(self, ledger, tmp_path):
        target = tmp_path / "same.py"
        _rec("s1", str(target), "replace", removed="old", added="v1")
        _rec("s1", str(tmp_path / "x" / ".." / "same.py"), "replace",
             removed="v1", added="v2")
        assert eo.keep_rate_report()["superseded"] == 1, (
            "two spellings of one path were treated as different files"
        )


# ---------------------------------------------------------------------------
# (4) Passive by contract
# ---------------------------------------------------------------------------

class TestPassive:
    def test_unwritable_ledger_returns_false_not_raises(self, monkeypatch):
        def boom():
            raise OSError("disk on fire")

        monkeypatch.setattr(eo, "_db_path", boom)
        assert eo.record_edit(
            session_id="s", path="a.py", edit_format="replace", status="applied"
        ) is False

    def test_unreadable_ledger_reports_zeros_not_raises(self, monkeypatch):
        def boom():
            raise OSError("disk on fire")

        monkeypatch.setattr(eo, "_db_path", boom)
        r = eo.keep_rate_report()
        assert r["total"] == 0 and r["keep_rate"] is None

    def test_corrupt_database_does_not_raise(self, ledger):
        ledger.write_bytes(b"this is not a sqlite file at all")
        assert eo.keep_rate_report()["total"] == 0

    def test_bad_status_is_rejected(self, ledger):
        assert _rec("s1", "a.py", "replace", status="maybe") is False
        assert eo.keep_rate_report()["total"] == 0

    def test_empty_path_is_rejected(self, ledger):
        assert _rec("s1", "", "replace") is False

    def test_ledger_never_feeds_the_agent_loop(self):
        """Structural: this is a report, not a control signal.

        If something starts reading keep-rate back into prompt assembly or
        the turn loop, the metric stops measuring and starts steering - and
        a model can then be optimised against its own scoreboard.
        """
        import agent.coding_context as cc
        import agent.turn_context as tc

        for module in (cc, tc):
            assert "edit_outcomes" not in inspect.getsource(module), (
                f"{module.__name__} reads the keep-rate ledger - it must stay "
                "a passive report"
            )


# ---------------------------------------------------------------------------
# (5) Bounded growth
# ---------------------------------------------------------------------------

class TestBounds:
    def test_row_cap_is_enforced(self, ledger, monkeypatch):
        monkeypatch.setattr(eo, "MAX_EVENTS", 10)
        for i in range(30):
            _rec("s1", f"f{i}.py", "replace", removed="o", added="n")
        with sqlite3.connect(ledger) as conn:
            count = conn.execute("SELECT COUNT(*) FROM edit_events").fetchone()[0]
        assert count <= 10, f"ledger grew to {count} rows, cap is 10"

    def test_old_rows_are_pruned(self, ledger):
        """Assert on the table, not the report.

        keep_rate_report filters by date anyway, so a report-level check
        passes whether or not the prune runs - the rows just accumulate
        forever, unseen.
        """
        _rec("s1", "a.py", "replace", removed="o", added="n")
        with sqlite3.connect(ledger) as conn:
            conn.execute("UPDATE edit_events SET created_at = '2000-01-01T00:00:00+00:00'")
            conn.commit()
        # Any write triggers the prune.
        _rec("s1", "b.py", "replace", removed="o", added="n")
        with sqlite3.connect(ledger) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM edit_events").fetchone()[0]
        assert rows == 1, f"the stale row survived the prune ({rows} rows on disk)"

    def test_window_excludes_older_rows(self, ledger):
        _rec("s1", "a.py", "replace", removed="o", added="n")
        with sqlite3.connect(ledger) as conn:
            conn.execute("UPDATE edit_events SET created_at = '2020-01-01T00:00:00+00:00'")
            conn.commit()
        assert eo.keep_rate_report(days=1)["total"] == 0


# ---------------------------------------------------------------------------
# (6) Production wiring
# ---------------------------------------------------------------------------

class TestProductionWiring:
    def test_patch_tool_records_outcomes(self):
        import tools.file_tools as ft

        src = inspect.getsource(ft.patch_tool)
        assert src.count("_record_edit_outcome") >= 2, (
            "patch_tool no longer records both applied and failed edits - "
            "the keep-rate ledger is dead code"
        )

    def test_write_file_tool_records_outcomes(self):
        import tools.file_tools as ft

        # Two success paths (resolved and the legacy fallback); both must
        # record, or half the writes go unmeasured.
        assert inspect.getsource(ft.write_file_tool).count("_record_edit_outcome") >= 2, (
            "write_file_tool no longer records outcomes on every success "
            "path, so whole-file rewrites never supersede what they wipe out"
        )

    def test_write_file_is_recorded_end_to_end(self, ledger, tmp_path):
        """Behavioural guard: the static check above passes on a half-wired
        function, because the other path's call keeps the name in the source.
        """
        import tools.file_tools as ft

        target = tmp_path / "w.py"
        result = json.loads(ft.write_file_tool(
            str(target), "alpha = 1\n", task_id="w7w", session_id="w7w",
        ))
        if result.get("error"):
            pytest.skip(f"write_file could not run here: {result['error']}")

        r = eo.keep_rate_report(session_id="w7w")
        assert r["applied"] == 1, "a successful write_file was not recorded"
        assert "write_file" in r["by_format"]

    def test_recorder_reaches_the_ledger(self):
        import tools.file_tools as ft

        assert "edit_outcomes" in inspect.getsource(ft._record_edit_outcome)

    def test_insights_carries_the_section(self):
        from agent.insights import InsightsEngine

        src = inspect.getsource(InsightsEngine.generate)
        assert src.count('"edits"') >= 2, (
            "the edits section is missing from one of generate()'s returns - "
            "quiet accounts take the empty branch"
        )
        assert "_format_edits" in inspect.getsource(InsightsEngine.format_terminal), (
            "format_terminal never renders the keep-rate section"
        )

    def test_insights_calls_the_renderer_not_just_imports_it(self):
        """The substring check above passes on a dead call site.

        Deleting ``lines.extend(_format_edits(...))`` leaves the import
        line behind, so the name is still in the source while nothing
        renders. Parse for an actual call instead.
        """
        import ast
        import textwrap

        from agent.insights import InsightsEngine

        tree = ast.parse(textwrap.dedent(inspect.getsource(InsightsEngine.format_terminal)))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "_format_edits" in called, (
            "format_terminal imports the keep-rate renderer but never calls "
            "it - the section will never appear in a report"
        )

    def test_renderer_output_is_what_insights_appends(self, ledger):
        """And the thing it calls actually produces the section."""
        _rec("s1", "a.py", "replace", removed="o", added="n")
        rendered = "\n".join(eo.format_report(eo.keep_rate_report()))
        assert "EDIT OUTCOMES" in rendered
        assert "applied" in rendered

    def test_replace_edit_is_recorded_end_to_end(self, ledger, tmp_path):
        import tools.file_tools as ft

        target = tmp_path / "f.py"
        target.write_text("alpha = 1\n", encoding="utf-8")

        ok = json.loads(ft.patch_tool(
            mode="replace", path=str(target), old_string="alpha = 1",
            new_string="alpha = 2", task_id="w7", session_id="w7",
        ))
        if ok.get("error"):
            pytest.skip(f"patch could not run in this environment: {ok['error']}")

        r = eo.keep_rate_report(session_id="w7")
        assert r["applied"] == 1, "a successful edit was not recorded"
        assert r["by_format"]["replace"]["applied"] == 1

    def test_failed_edit_is_recorded_end_to_end(self, ledger, tmp_path):
        import tools.file_tools as ft

        target = tmp_path / "g.py"
        target.write_text("alpha = 1\n", encoding="utf-8")

        bad = json.loads(ft.patch_tool(
            mode="replace", path=str(target), old_string="no_such_text_zzz",
            new_string="x", task_id="w7f", session_id="w7f",
        ))
        assert bad.get("error")

        r = eo.keep_rate_report(session_id="w7f")
        assert r["failed"] == 1, "a failed edit was not recorded"
        assert r["apply_rate"] == 0.0

    def test_churn_is_visible_end_to_end(self, ledger, tmp_path):
        """The whole point: two edits, one of them wasted."""
        import tools.file_tools as ft

        target = tmp_path / "h.py"
        target.write_text("alpha = 1\n", encoding="utf-8")

        first = json.loads(ft.patch_tool(
            mode="replace", path=str(target), old_string="alpha = 1",
            new_string="alpha = 2", task_id="w7c", session_id="w7c",
        ))
        if first.get("error"):
            pytest.skip(f"patch could not run in this environment: {first['error']}")
        ft.patch_tool(
            mode="replace", path=str(target), old_string="alpha = 2",
            new_string="alpha = 3", task_id="w7c", session_id="w7c",
        )

        r = eo.keep_rate_report(session_id="w7c")
        assert r["superseded"] == 1, (
            f"the churned edit was not detected end to end: {r}"
        )


# ---------------------------------------------------------------------------
# (7) Reporting
# ---------------------------------------------------------------------------

class TestReporting:
    def test_empty_ledger_renders_nothing(self, ledger):
        assert eo.format_report(eo.keep_rate_report()) == []

    def test_empty_dict_renders_nothing(self):
        assert eo.format_report({}) == []
        assert eo.format_report(None) == []

    def test_populated_ledger_renders_lines(self, ledger):
        _rec("s1", "a.py", "replace", removed="o", added="n")
        lines = eo.format_report(eo.keep_rate_report())
        assert lines and any("EDIT OUTCOMES" in ln for ln in lines)
        assert any("replace" in ln for ln in lines)

    def test_report_shape_is_stable_when_empty(self, ledger):
        r = eo.keep_rate_report()
        for key in ("days", "total", "applied", "failed", "kept",
                    "superseded", "reverted", "keep_rate", "apply_rate",
                    "by_format"):
            assert key in r, f"empty report is missing {key!r}"

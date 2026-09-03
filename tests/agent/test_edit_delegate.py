"""W5 - architect/editor split for edits the main model cannot land.

This module writes to the user's files based on another model's output, so
the tests are weighted accordingly. The load-bearing property is not "the
editor is helpful" but **the editor decides where, never what**: the
replacement text is always the architect's own, and the anchor the editor
returns is verified against the file before anything is written.

As in W6b, every wiring guard is an AST call-site check or a behavioural
test. A substring scan passes on a dead call site whose import survived.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap

import pytest

import agent.edit_delegate as ed


# -- Fixtures ---------------------------------------------------------------

@pytest.fixture
def enabled(monkeypatch):
    """Pretend the user configured an editor model."""
    monkeypatch.setattr(ed, "_editor_configured", lambda: True)


@pytest.fixture
def sample(tmp_path):
    p = tmp_path / "core.py"
    p.write_text(
        "def alpha():\n"
        "    return 1\n"
        "\n"
        "def beta():\n"
        "    return 2\n"
    )
    return p


def _reply(text):
    """Build the minimal shape call_llm's result is read through."""
    class _M:
        content = text

    class _C:
        message = _M()

    class _R:
        choices = [_C()]

    return _R()


def _editor_says(monkeypatch, text):
    monkeypatch.setattr(ed, "_editor_configured", lambda: True)
    import agent.auxiliary_client as aux

    monkeypatch.setattr(aux, "call_llm", lambda **kw: _reply(text))
    return aux


# -- The safety property -----------------------------------------------------

class TestEditorDecidesWhereNeverWhat:
    def test_replacement_text_is_never_taken_from_the_editor(self):
        """The editor is never asked for, and never supplies, new content.

        This is the property that makes delegating an edit acceptable at
        all: a confused editor can fail to find an anchor or point at the
        wrong one, but it cannot write text no model asked for.
        """
        src = inspect.getsource(ed.delegate_edit)
        tree = ast.parse(textwrap.dedent(src))
        returned_keys = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                returned_keys.update(
                    k.value for k in node.keys if isinstance(k, ast.Constant)
                )
        assert "new_string" not in returned_keys, (
            "delegate_edit returns a replacement string - the editor must "
            "only ever correct WHERE an edit applies, never WHAT it writes"
        )

    def test_hallucinated_anchor_is_rejected(self, sample):
        """Text the editor invented is not in the file, so it cannot apply."""
        content = sample.read_text()
        assert ed.verify_anchor("def gamma():\n    return 3\n", content) is None

    def test_ambiguous_anchor_is_rejected(self):
        """An anchor matching twice would edit the wrong one half the time."""
        content = "    return 1\n\n    return 1\n"
        assert ed.verify_anchor("    return 1\n", content) is None

    def test_unique_anchor_is_accepted(self, sample):
        content = sample.read_text()
        assert ed.verify_anchor("def beta():\n", content) == "def beta():\n"

    def test_no_match_sentinel_is_honoured(self, sample):
        assert ed.verify_anchor("NO_MATCH", sample.read_text()) is None

    def test_no_match_is_honoured_even_when_the_file_contains_it(self):
        """The dangerous case for the sentinel.

        Treating NO_MATCH as ordinary text is harmless until a file
        happens to contain the word - then the editor's refusal is read
        as an anchor and the replacement lands on it.
        """
        content = "STATUS = \"NO_MATCH\"\nother = 1\n"
        assert ed.verify_anchor("NO_MATCH", content) is None, (
            "the editor declined, but its refusal was matched against "
            "the file and used as an edit target"
        )

    def test_empty_and_oversized_anchors_are_rejected(self, sample):
        content = sample.read_text()
        assert ed.verify_anchor("", content) is None
        assert ed.verify_anchor("   \n  ", content) is None
        assert ed.verify_anchor("x" * (ed.MAX_ANCHOR_CHARS + 1), content) is None

    def test_an_oversized_anchor_is_refused_even_when_it_matches(self):
        """The cap is about intent, not about lookup failure.

        An anchor this long means the editor started rewriting the file
        instead of locating a line in it. Rejecting it only when it
        happens not to be found would miss exactly the case that
        matters.
        """
        body = "".join("line_%04d = %d\n" % (i, i) for i in range(400))
        assert len(body) > ed.MAX_ANCHOR_CHARS
        assert body.count(body) == 1
        assert ed.verify_anchor(body, body) is None, (
            "the editor returned a whole-file rewrite and it was "
            "accepted as an anchor"
        )

    def test_a_trailing_newline_slip_is_forgiven_but_still_verified(self):
        """The commonest transcription slip, and it is still checked."""
        content = "alpha = 1"
        assert ed.verify_anchor("alpha = 1\n", content) == "alpha = 1"
        # ...but only when the trimmed form is itself unique.
        assert ed.verify_anchor("x\n", "x x") is None

    def test_whole_response_fences_are_stripped(self, sample):
        content = sample.read_text()
        fenced = "```python\ndef beta():\n```"
        assert ed.verify_anchor(fenced, content) == "def beta():"

    def test_a_fence_inside_the_content_is_not_stripped(self):
        """Stripping a mid-text fence would corrupt the anchor."""
        content = 'DOC = """\n```\nsample\n```\n"""\n'
        anchor = "```\nsample\n```"
        assert ed.verify_anchor(anchor, content) == anchor


# -- Off unless configured ---------------------------------------------------

class TestDisabledByDefault:
    def test_no_configured_model_means_no_call(self, monkeypatch, sample):
        import agent.auxiliary_client as aux

        monkeypatch.setattr(ed, "_editor_configured", lambda: False)
        called = []
        monkeypatch.setattr(
            aux, "call_llm", lambda **kw: called.append(1) or _reply("x")
        )
        assert ed.find_anchor(
            path=str(sample), old_string="def alpha():", new_string="def a():"
        ) is None
        assert not called, "an unconfigured editor still billed a model call"

    def test_auto_is_not_a_configured_editor(self, monkeypatch):
        """``auto`` means "pick something for me", which is not a choice here.

        Delegation spends money on every failed edit; it must be opted into
        explicitly, not inherited from a generic fallback chain.
        """
        import hermes_cli.config as hc

        monkeypatch.setattr(hc, "load_config", lambda: {
            "auxiliary": {"edit": {"model": "auto"}}
        })
        assert ed._editor_configured() is False

    def test_an_explicit_model_enables_it(self, monkeypatch):
        import hermes_cli.config as hc

        monkeypatch.setattr(hc, "load_config", lambda: {
            "auxiliary": {"edit": {"model": "some/editor-model"}}
        })
        assert ed._editor_configured() is True


# -- Bounds and failure handling ---------------------------------------------

class TestBounds:
    def test_oversized_files_are_not_sent(self, tmp_path, enabled, monkeypatch):
        import agent.auxiliary_client as aux

        big = tmp_path / "big.py"
        big.write_text("x = 1\n" * (ed.MAX_FILE_CHARS // 2))
        called = []
        monkeypatch.setattr(
            aux, "call_llm", lambda **kw: called.append(1) or _reply("x")
        )
        assert ed.find_anchor(
            path=str(big), old_string="x = 1", new_string="x = 2"
        ) is None
        assert not called, (
            "a delegation was billed for a file too large to send"
        )

    def test_a_missing_file_is_not_an_error(self, tmp_path, enabled):
        assert ed.find_anchor(
            path=str(tmp_path / "nope.py"), old_string="a", new_string="b"
        ) is None

    def test_a_failing_model_call_is_swallowed(self, monkeypatch, sample, enabled):
        import agent.auxiliary_client as aux

        def boom(**kw):
            raise RuntimeError("provider down")

        monkeypatch.setattr(aux, "call_llm", boom)
        assert ed.find_anchor(
            path=str(sample), old_string="def alpha():", new_string="def a():"
        ) is None

    def test_an_unchanged_anchor_is_not_worth_retrying(
        self, monkeypatch, sample
    ):
        """Returning the same text would fail identically, for money."""
        _editor_says(monkeypatch, "def beta():\n")
        called = []
        assert ed.delegate_edit(
            path=str(sample),
            old_string="def beta():\n",
            new_string="def gamma():\n",
            retry=lambda d: called.append(1) or d,
        ) is None
        assert not called, "an unchanged anchor was retried anyway"


# -- Re-entrancy -------------------------------------------------------------

class TestReentrancy:
    def test_scope_blocks_a_nested_delegation(self):
        with ed.delegation_scope() as outer:
            assert outer is True
            with ed.delegation_scope() as inner:
                assert inner is False, (
                    "a nested delegation scope was granted, so a retry that "
                    "also fails would delegate again - and recurse"
                )

    def test_scope_is_released_even_on_an_exception(self):
        with pytest.raises(RuntimeError):
            with ed.delegation_scope():
                raise RuntimeError("boom")
        assert ed.in_delegation() is False, (
            "the guard leaked, so no further edit in this thread can ever "
            "be delegated"
        )

    def test_delegate_edit_declines_inside_a_scope(self, monkeypatch, sample):
        """A retry that misses again must not delegate a second time."""
        _editor_says(monkeypatch, "def alpha():\n")
        with ed.delegation_scope():
            assert ed.delegate_edit(
                path=str(sample),
                old_string="def alfa():\n",
                new_string="def a():\n",
                retry=lambda d: d,
            ) is None

    def test_the_retry_runs_inside_the_scope(self, monkeypatch, sample):
        """The scope must still be held when the retry runs.

        If it were released first, a retry that also missed would
        delegate again and recurse - which is the whole point of the
        guard.
        """
        _editor_says(monkeypatch, "def alpha():\n    return 1\n")
        seen = []
        ed.delegate_edit(
            path=str(sample),
            old_string="class Absent:\n    X = 1\n",
            new_string="def a():\n",
            retry=lambda d: seen.append(ed.in_delegation()) or d,
        )
        assert seen == [True], (
            "the retry ran outside the re-entrancy scope: %r" % seen
        )


# -- Delegation end to end ---------------------------------------------------

class TestDelegation:
    def test_a_misremembered_edit_is_re_anchored(self, monkeypatch, sample):
        _editor_says(monkeypatch, "def beta():\n    return 2\n")
        out = ed.delegate_edit(
            path=str(sample),
            old_string="class Absent:\n    X = 1\n",
            new_string="def beta():\n    return 22\n",
            retry=lambda d: d,
        )
        assert out is not None
        assert out["old_string"] == "def beta():\n    return 2\n"
        assert "re-anchored" in out["note"]

    def test_the_note_tells_the_model_what_happened(self, monkeypatch, sample):
        """Silent correction would leave the model's mental model wrong."""
        _editor_says(monkeypatch, "def beta():\n    return 2\n")
        out = ed.delegate_edit(
            path=str(sample),
            old_string="class Absent:\n    X = 1\n",
            new_string="def beta():\n    return 22\n",
            retry=lambda d: d,
        )
        note = out["note"]
        assert "unchanged" in note and "Re-read" in note


# -- Prompt caching is sacred ------------------------------------------------

class TestCacheSafety:
    def test_delegation_never_reaches_the_system_prompt(self):
        """A side call must not touch the main conversation's prefix."""
        import agent.coding_context as cc

        assert "edit_delegate" not in inspect.getsource(cc), (
            "edit delegation leaked into system-prompt assembly - this "
            "would invalidate the per-conversation prompt cache"
        )

    def test_it_uses_the_auxiliary_side_channel(self):
        """The side channel is what keeps the main prefix untouched."""
        src = inspect.getsource(ed.find_anchor)
        assert "auxiliary_client" in src and "call_llm" in src, (
            "delegation stopped using the auxiliary side-call channel; a "
            "call on the main conversation would invalidate its cache"
        )

    def test_no_schema_bytes_were_added_to_the_core(self):
        """The narrow waist: no new tool, no widened schema."""
        import tools.file_tools as ft

        schema = json.dumps(ft.PATCH_SCHEMA).lower()
        for leaked in ("delegate", "editor", "architect"):
            assert leaked not in schema, (
                "W5 widened the patch schema; it is meant to happen inside "
                "the tool, which costs nothing per call"
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
    def test_patch_tool_calls_the_delegation_helper(self):
        import tools.file_tools as ft

        assert "_try_edit_delegation" in _called_names(ft.patch_tool), (
            "patch_tool mentions the delegation helper but never calls it - "
            "the feature is dead code"
        )

    def test_the_helper_calls_delegate_edit(self):
        import tools.file_tools as ft

        assert "delegate_edit" in _called_names(ft._try_edit_delegation), (
            "the helper no longer calls delegate_edit, so no edit is ever "
            "rescued while the wiring still looks present"
        )

    def test_the_helper_hands_the_retry_to_delegate_edit(self):
        """delegate_edit owns the scope, so it must own the retry too."""
        import tools.file_tools as ft

        src = inspect.getsource(ft._try_edit_delegation)
        assert "retry=retry" in src, (
            "the retry no longer runs inside delegate_edit's re-entrancy "
            "scope, so a repeatedly mis-anchored edit recurses and bills "
            "a model call per level"
        )

    def test_delegation_waits_for_the_format_chain(self):
        """The cheap advice gets its turn first."""
        from agent.edit_escalation import ESCALATE_AFTER

        assert ed.DELEGATE_AFTER > ESCALATE_AFTER, (
            "delegation fires no later than the free format-switch hint, so "
            "the user pays for a model call before the cheap fix was tried"
        )

    def test_delegation_is_gated_on_the_failure_count(self):
        import tools.file_tools as ft

        src = inspect.getsource(ft.patch_tool)
        assert "_attempts >= DELEGATE_AFTER" in src, (
            "delegation is no longer gated on repeated failure, so every "
            "first miss bills a second model"
        )

    def test_a_failed_edit_is_rescued_end_to_end(
        self, monkeypatch, tmp_path
    ):
        """The guard the AST checks cannot give us.

        Drives the real patch_tool until the failure count reaches
        DELEGATE_AFTER, then asserts the file on disk actually changed
        and that the replacement written was the architect's own.
        """
        import tools.file_tools as ft
        from agent.edit_escalation import reset

        target = tmp_path / "core.py"
        target.write_text("def beta():\n    return 2\n")
        reset("w5e2e")
        _editor_says(monkeypatch, "def beta():\n    return 2\n")

        # Text from a stale read - the file has been rewritten since, so
        # this is genuinely absent. It has to be this far off: patch_replace
        # fuzzy-matches, and an old_string that merely drifted in whitespace
        # or in one literal still applies on the first try and never reaches
        # delegation. That narrowness is the honest scope of this feature -
        # it rescues the edits the fuzzy matcher gives up on.
        wrong = "class TotallyUnrelated:\n    QUUX = 42\n"
        right = "def beta():\n    return 222\n"

        result = None
        for _ in range(ed.DELEGATE_AFTER):
            result = json.loads(ft.patch_tool(
                mode="replace", path=str(target),
                old_string=wrong, new_string=right,
                task_id="w5e2e", session_id="w5e2e",
            ))

        assert not result.get("error"), (
            "the edit was never rescued: %r" % result.get("error")
        )
        assert "re-anchored" in result.get("_hint", "")
        assert target.read_text() == right, (
            "the file does not hold the architect's replacement text: %r"
            % target.read_text()
        )

    def test_delegation_does_not_fire_before_the_threshold(
        self, monkeypatch, tmp_path
    ):
        """Every first miss must not bill a second model."""
        import agent.auxiliary_client as aux
        import tools.file_tools as ft
        from agent.edit_escalation import reset

        target = tmp_path / "core.py"
        target.write_text("def beta():\n    return 2\n")
        reset("w5early")
        monkeypatch.setattr(ed, "_editor_configured", lambda: True)
        calls = []
        monkeypatch.setattr(
            aux, "call_llm",
            lambda **kw: calls.append(1) or _reply("def beta():\n"),
        )

        out = json.loads(ft.patch_tool(
            mode="replace", path=str(target),
            old_string="class TotallyUnrelated:\n    QUUX = 42\n",
            new_string="x\n",
            task_id="w5early", session_id="w5early",
        ))
        assert out.get("error"), "the bad edit somehow applied"
        assert not calls, (
            "a single failed edit already billed an editor model call"
        )

    def test_only_replace_mode_is_delegated(self, monkeypatch, tmp_path):
        """V4A patches carry no unambiguous single intent to re-anchor.

        Behavioural, not a substring scan: 'mode == "replace"' appears
        elsewhere in patch_tool, so a source check passes even with the
        delegation guard removed.

        The call below is deliberately malformed - a V4A patch that also
        carries ``path`` and ``old_string``. A well-formed patch-mode call
        leaves ``old_string`` empty, so the truthiness check alone would
        stop it and the mode check would never be exercised. Here every
        other condition is satisfied, which leaves the mode check as the
        only thing standing between a V4A patch and being silently
        re-applied as a replace against text the caller never named.
        """
        import agent.auxiliary_client as aux
        import tools.file_tools as ft
        from agent.edit_escalation import reset

        target = tmp_path / "core.py"
        target.write_text("def beta():\n    return 2\n")
        reset("w5v4a")
        monkeypatch.setattr(ed, "_editor_configured", lambda: True)
        calls = []
        monkeypatch.setattr(
            aux, "call_llm",
            lambda **kw: calls.append(1) or _reply("def beta():\n"),
        )

        v4a = (
            "*** Begin Patch\n"
            "*** Update File: %s\n" % target
            + "@@\n"
            "-class Absent:\n"
            "+class Present:\n"
            "*** End Patch\n"
        )
        for _ in range(ed.DELEGATE_AFTER + 1):
            ft.patch_tool(
                mode="patch", patch=v4a, path=str(target),
                old_string="class Absent:\n",
                new_string="class Present:\n",
                task_id="w5v4a", session_id="w5v4a",
            )
        assert not calls, (
            "a V4A patch was delegated - there is no single unambiguous "
            "old_string to re-anchor, so the editor would be guessing, "
            "and the rescue would re-enter patch_tool in replace mode"
        )

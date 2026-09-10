"""Continuous work reaches the model as a per-conversation, per-turn note.

Continuous work is toggled per conversation and can be flipped mid-session, so
it cannot live in the byte-stable system prompt. It rides the model-bound
message instead — the same channel as the HUD note — computed fresh every turn
from the session's own flag.

The flag is recorded on every ``prompt.submit`` (the desktop sends it on each
submit so a mid-session enable takes effect on the next turn), so a submit
without the flag clears it.
"""

import threading
import types

import pytest

from agent.prompt_builder import CONTINUOUS_WORK_GUIDANCE
from tui_gateway import server


def _session(**extra):
    return {
        "agent": types.SimpleNamespace(valid_tool_names=set()),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": None,
        "attached_images": [],
        **extra,
    }


class TestNoteContents:
    def test_enabled_session_gets_the_full_guidance(self):
        assert server._continuous_work_note(_session(continuous_work=True)) == CONTINUOUS_WORK_GUIDANCE

    def test_disabled_session_gets_nothing(self):
        assert server._continuous_work_note(_session(continuous_work=False)) == ""

    def test_session_that_never_reported_the_flag_gets_nothing(self):
        """Other clients (CLI, TUI, dashboard) omit the field entirely."""
        assert server._continuous_work_note(_session()) == ""

    def test_guidance_carries_the_adversarial_termination_protocol(self):
        note = server._continuous_work_note(_session(continuous_work=True))

        assert "TRIPLE CERTIFICATION" in note
        assert "Gate 1" in note
        assert "Gate 2" in note
        assert "Gate 3" in note


class TestTurnRouting:
    def test_continuous_work_turn_gets_the_note_prepended(self):
        session = _session(continuous_work=True)
        note = server._continuous_work_note(session)

        assert server._prepend_note("go do the work", note) == f"{note}\n\ngo do the work"

    def test_plain_turn_gets_no_continuous_work_note(self):
        session = _session(continuous_work=False)

        assert server._continuous_work_note(session) == ""
        assert server._prepend_note("go do the work", "") == "go do the work"


class TestSubmitRecording:
    """``prompt.submit`` stamps the flag this conversation carries."""

    @pytest.fixture
    def busy_session(self):
        session = _session(running=True)
        server._sessions["sid"] = session
        yield session
        server._sessions.pop("sid", None)

    def _submit(self, **params):
        return server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "what is this?", "queued": True, **params}
        )

    def test_continuous_work_submit_is_recorded(self, busy_session):
        self._submit(continuous_work=True)

        assert busy_session["continuous_work"] is True

    def test_a_plain_submit_clears_it(self):
        """The flag rides every submit; omitting it means the toggle is off."""
        session = _session(running=True, continuous_work=True)
        server._sessions["sid"] = session
        try:
            self._submit()

            assert session["continuous_work"] is False
        finally:
            server._sessions.pop("sid", None)

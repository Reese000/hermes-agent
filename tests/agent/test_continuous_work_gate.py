"""Unit tests for the continuous-work turn-end enforcement gate.

The gate is policy-only: given the agent's final response and how many
work-evidence tool calls ran this turn, it decides whether to refuse the stop
and force another pass, or to accept it.

Override requires ALL THREE markers:
  1. "I AM OVERRIDING continuous work mode"
  2. "I PERSONALLY FAILED"
  3. "I accept that this override is a personal failure"
Missing any one is a REJECTED override — the agent must continue.
"""

import pytest

from agent.continuous_work_gate import (
    _COMPLETION_SIGNALS,
    _ACCEPTANCE_MARKER,
    _FAILURE_ADMISSION_MARKER,
    _OVERRIDE_MARKER,
    _REQUIRED_OVERRIDE_COMPONENTS,
    _strip_note_prefix,
    build_continuous_work_nudge,
    mark_continuous_work_nudge_issued,
)

_FULL_OVERRIDE = (
    "I AM OVERRIDING continuous work mode because the API endpoint is paywalled "
    "and I have no credentials. I PERSONALLY FAILED to complete all work. "
    "I accept that this override is a personal failure, not a valid completion."
)


# ---------------------------------------------------------------------------
# Override: requires all 3 markers
# ---------------------------------------------------------------------------

class TestDeclaredOverride:
    def test_full_override_with_all_three_markers_bypasses(self):
        assert build_continuous_work_nudge(
            final_response=_FULL_OVERRIDE, work_evidence_tools=0, attempts=0
        ) is None

    def test_override_marker_alone_is_rejected(self):
        """The model can't just say 'I AM OVERRIDING' — needs all 3."""
        nudge = build_continuous_work_nudge(
            final_response="I AM OVERRIDING continuous work mode because I'm lazy.",
            work_evidence_tools=0, attempts=0,
        )
        assert nudge is not None

    def test_override_plus_failure_but_no_acceptance_is_rejected(self):
        nudge = build_continuous_work_nudge(
            final_response=(
                "I AM OVERRIDING continuous work mode. I PERSONALLY FAILED. "
                "But I refuse to accept it."
            ),
            work_evidence_tools=0, attempts=0,
        )
        assert nudge is not None

    def test_override_plus_acceptance_but_no_failure_is_rejected(self):
        nudge = build_continuous_work_nudge(
            final_response=(
                "I AM OVERRIDING continuous work mode. "
                "I accept that this override is a personal failure, not a valid completion."
            ),
            work_evidence_tools=0, attempts=0,
        )
        assert nudge is not None

    def test_all_three_markers_present_is_accepted(self):
        assert build_continuous_work_nudge(
            final_response=_FULL_OVERRIDE, work_evidence_tools=0, attempts=0
        ) is None


# ---------------------------------------------------------------------------
# Work evidence: tool calls bypass the gate
# ---------------------------------------------------------------------------

class TestWorkEvidenceAccepts:
    def test_real_work_this_turn_allows_stop(self):
        response = "I certify: all tests pass, as shown by the terminal run above."
        assert build_continuous_work_nudge(
            final_response=response, work_evidence_tools=3, attempts=0
        ) is None

    def test_empty_response_never_nudges(self):
        assert build_continuous_work_nudge(
            final_response="", work_evidence_tools=0, attempts=0
        ) is None

    def test_none_response_never_nudges(self):
        assert build_continuous_work_nudge(
            final_response=None, work_evidence_tools=0, attempts=0
        ) is None


# ---------------------------------------------------------------------------
# Refuses: bare completion without evidence or override
# ---------------------------------------------------------------------------

class TestRefuses:
    def test_bare_completion_without_work_is_refused(self):
        response = "Task complete — all done."
        nudge = build_continuous_work_nudge(
            final_response=response, work_evidence_tools=0, attempts=0
        )
        assert nudge is not None
        assert "I PERSONALLY FAILED" in nudge

    @pytest.mark.parametrize("signal", _COMPLETION_SIGNALS)
    def test_each_completion_signal_without_work_is_refused(self, signal: str):
        nudge = build_continuous_work_nudge(
            final_response=signal, work_evidence_tools=0, attempts=0
        )
        assert nudge is not None

    def test_runs_out_of_budget(self):
        response = "all done"
        for attempt in range(3):
            nudge = build_continuous_work_nudge(
                final_response=response, work_evidence_tools=0, attempts=attempt
            )
            assert nudge is not None
        assert (
            build_continuous_work_nudge(
                final_response=response, work_evidence_tools=0, attempts=3
            )
            is None
        )

    def test_middle_turn_without_work_but_non_completion_not_refused(self):
        response = "I looked at the specs, here's what I found."
        assert build_continuous_work_nudge(
            final_response=response, work_evidence_tools=0, attempts=0
        ) is None


# ---------------------------------------------------------------------------
# Note prefix stripping: guidance text in the response doesn't false-positive
# ---------------------------------------------------------------------------

class TestNotePrefixStripping:
    def test_guidance_text_stripped_before_checking(self):
        """If the model echoes the guidance text, the 'certify' inside it
        shouldn't trigger a completion signal."""
        response = (
            "I certify: all work is complete.\n\n"
            "## Gate 1: Work Inventory\n"
            "1. Done X — verified by terminal call..."
        )
        # This IS a completion claim, so it should fire
        assert build_continuous_work_nudge(
            final_response=response, work_evidence_tools=0, attempts=0
        ) is not None

    def test_strips_known_note_prefix(self):
        text = "# Continuous Work Mode — Adversarial Termination Protocol\n\nI certify: everything done."
        stripped = _strip_note_prefix(text)
        assert "I certify: everything done." == stripped


# ---------------------------------------------------------------------------
# Multi-modal response flattening
# ---------------------------------------------------------------------------

class TestMultiModalResponse:
    def test_content_parts_are_flattened(self):
        parts = [
            {"type": "text", "text": "This task is fully verified."},
            {"type": "image_url", "image_url": {"url": "x://y"}},
        ]
        nudge = build_continuous_work_nudge(
            final_response=parts, work_evidence_tools=0, attempts=0
        )
        assert nudge is not None


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------

class TestCounter:
    def test_mark_increments(self):
        class _A:
            pass

        agent = _A()
        agent._continuous_work_nudges = 0
        assert mark_continuous_work_nudge_issued(agent) == 1
        assert mark_continuous_work_nudge_issued(agent) == 2


# ---------------------------------------------------------------------------
# Nudge content
# ---------------------------------------------------------------------------

class TestNudgeContent:
    def test_nudge_requires_all_five_override_components(self):
        response = "all done"
        nudge = build_continuous_work_nudge(
            final_response=response, work_evidence_tools=0, attempts=0
        )
        for comp in _REQUIRED_OVERRIDE_COMPONENTS:
            assert comp in nudge, f"Nudge missing required component: {comp}"

    def test_nudge_demands_personal_failure_admission(self):
        response = "all done"
        nudge = build_continuous_work_nudge(
            final_response=response, work_evidence_tools=0, attempts=0
        )
        assert "I PERSONALLY FAILED" in nudge
        assert "I accept that this override is a personal failure" in nudge
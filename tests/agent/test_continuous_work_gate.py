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


# ─── CW v2: Critic Gate Tests ─────────────────────────────────────────────────

from agent.continuous_work_critic import (
    CircuitBreaker,
    CriticVerdict,
    TurnEvidence,
    gather_turn_evidence,
    parse_critic_response,
)


class TestCircuitBreaker:
    def test_trips_after_max_strikes(self):
        cb = CircuitBreaker(max_strikes=3)
        assert cb.record_rejection("r1") is None
        assert cb.record_rejection("r2") is None
        msg = cb.record_rejection("r3")
        assert msg is not None
        assert "Circuit breaker tripped" in msg
        assert cb.tripped is True

    def test_does_not_fire_after_trip(self):
        cb = CircuitBreaker(max_strikes=3)
        cb.record_rejection("r1")
        cb.record_rejection("r2")
        cb.record_rejection("r3")  # trips
        assert cb.record_rejection("r4") is None
        assert cb.record_rejection("r5") is None

    def test_resets_on_approval(self):
        cb = CircuitBreaker(max_strikes=3)
        cb.record_rejection("r1")
        cb.record_rejection("r2")
        cb.record_approval()
        assert cb.strike_count == 0
        assert cb.tripped is False
        # Can trip again after reset
        cb.record_rejection("r1")
        cb.record_rejection("r2")
        msg = cb.record_rejection("r3")
        assert msg is not None

    def test_strikes_remaining(self):
        cb = CircuitBreaker(max_strikes=3)
        assert cb.strikes_remaining == 3
        cb.record_rejection("r1")
        assert cb.strikes_remaining == 2
        cb.record_rejection("r2")
        assert cb.strikes_remaining == 1
        cb.record_rejection("r3")
        assert cb.strikes_remaining == 0


class TestParseCriticResponse:
    def test_explicit_approved(self):
        r = parse_critic_response("[STATUS]\nAPPROVED\n\n[VIOLATIONS]\nNone")
        assert r.passed is True
        assert r.status == "APPROVED"

    def test_explicit_rejected(self):
        r = parse_critic_response("[STATUS]\nREJECTED\n\n[VIOLATIONS]\n1, 3")
        assert r.passed is False
        assert r.status == "REJECTED"
        assert "1" in r.violations
        assert "3" in r.violations

    def test_no_status_positive_critique_approved(self):
        """When LLM omits [STATUS], positive critique with no violations = APPROVED."""
        r = parse_critic_response(
            "The work is exceptional and meets all criteria.\n\n"
            "[VIOLATIONS]\nNone\n\n"
            "[CRITIQUE]\nThe work is solid and well-structured.\n\n"
            "[REQUIRED_ACTION]\nNone"
        )
        assert r.passed is True
        assert r.status == "APPROVED"

    def test_no_status_negative_critique_rejected(self):
        """When LLM omits [STATUS], negative critique with violations = REJECTED."""
        r = parse_critic_response(
            "The agent did not complete the work.\n\n"
            "[VIOLATIONS]\n1, 3, 5\n\n"
            "[CRITIQUE]\nMultiple criteria failed.\n\n"
            "[REQUIRED_ACTION]\nFix the issues."
        )
        assert r.passed is False
        assert r.status == "REJECTED"

    def test_no_status_with_none_dash_text_approved(self):
        """'None — the work meets the bar...' should be treated as no action."""
        r = parse_critic_response(
            "The work is exceptional.\n\n"
            "[VIOLATIONS]\nNone\n\n"
            "[CRITIQUE]\nThe work meets the bar and should be considered complete.\n\n"
            "[REQUIRED_ACTION]\nNone — the work meets the bar and should be considered complete."
        )
        assert r.passed is True

    def test_empty_response_rejected(self):
        r = parse_critic_response("")
        assert r.passed is False

    def test_positive_signals_detected(self):
        """All positive signals should trigger approval when no [STATUS]."""
        for signal in ["substantial", "verified", "solid", "exceptional",
                       "meets the bar", "polished", "thorough", "strong"]:
            r = parse_critic_response(
                f"The work review.\n\n[VIOLATIONS]\nNone\n\n"
                f"[CRITIQUE]\nThe work is {signal} and well-structured.\n\n"
                "[REQUIRED_ACTION]\nNone"
            )
            assert r.passed is True, f"Signal '{signal}' not detected as positive"


class TestGatherTurnEvidence:
    def test_only_walks_current_turn(self):
        """Historical commands from previous turns should be excluded."""
        messages = [
            # Turn 1: exploration
            {"role": "user", "content": "explore"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "ls -la"}'}},
            ]},
            {"role": "tool", "content": "file listing"},
            {"role": "assistant", "content": "done"},
            # Turn 2: verification
            {"role": "user", "content": "verify"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "pytest"}'}},
            ]},
            {"role": "tool", "content": "48 passed"},
            {"role": "assistant", "content": "verified"},
        ]
        evidence = gather_turn_evidence(messages)
        assert "ls -la" not in evidence.terminal_commands
        assert "pytest" in evidence.terminal_commands
        assert evidence.total_tool_calls == 1

    def test_skips_synthetic_cw_nudges(self):
        """Synthetic CW nudge messages should not be treated as user messages."""
        messages = [
            {"role": "user", "content": "do work"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "pytest"}'}},
            ]},
            {"role": "tool", "content": "48 passed"},
            {"role": "assistant", "content": "done"},
            # Synthetic CW nudge
            {"role": "user", "content": "[CW CRITIC: rejected]", "_continuous_work_synthetic": True},
            # Agent's response to nudge
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "pytest -v"}'}},
            ]},
            {"role": "tool", "content": "48 passed"},
            {"role": "assistant", "content": "verified"},
        ]
        evidence = gather_turn_evidence(messages)
        # Should include the latest pytest, not the old one
        assert "pytest -v" in evidence.terminal_commands

    def test_empty_messages(self):
        evidence = gather_turn_evidence([])
        assert evidence.total_tool_calls == 0
        assert evidence.terminal_commands == []

    def test_work_vs_readonly_classification(self):
        messages = [
            {"role": "user", "content": "work"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "write_file", "arguments": '{"path": "f.py", "content": "x"}'}},
                {"function": {"name": "read_file", "arguments": '{"path": "f.py"}'}},
                {"function": {"name": "terminal", "arguments": '{"command": "pytest"}'}},
                {"function": {"name": "search_files", "arguments": '{"pattern": "x"}'}},
            ]},
            {"role": "tool", "content": "written"},
            {"role": "tool", "content": "content"},
            {"role": "tool", "content": "48 passed"},
            {"role": "tool", "content": "found"},
            {"role": "assistant", "content": "done"},
        ]
        evidence = gather_turn_evidence(messages)
        assert evidence.work_tool_calls == 2  # write_file + terminal
        assert evidence.read_only_tool_calls == 2  # read_file + search_files
        assert "f.py" in evidence.files_written
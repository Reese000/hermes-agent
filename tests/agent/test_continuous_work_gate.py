"""Unit tests for the continuous-work turn-end enforcement gate.

The gate is policy-only: given the agent's final response and how many
work-evidence tool calls ran this turn, it decides whether to refuse the stop
and force another pass, or to accept it.

No override admissions, no personal failure paths, no escape hatches.
The only exit is through the CW critic approving the work.
"""

import pytest

from agent.continuous_work_gate import (
    _COMPLETION_SIGNALS,
    _strip_note_prefix,
    build_continuous_work_nudge,
    mark_continuous_work_nudge_issued,
)


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

    @pytest.mark.parametrize("signal", _COMPLETION_SIGNALS)
    def test_each_completion_signal_without_work_is_refused(self, signal: str):
        nudge = build_continuous_work_nudge(
            final_response=signal, work_evidence_tools=0, attempts=0
        )
        assert nudge is not None

    def test_never_runs_out_of_budget(self):
        """No hard ceiling — CW continues until critic approves."""
        response = "all done"
        for attempt in range(3):
            nudge = build_continuous_work_nudge(
                final_response=response, work_evidence_tools=0, attempts=attempt
            )
            assert nudge is not None
        # With no hard ceiling, even attempt=999 still produces a nudge
        nudge = build_continuous_work_nudge(
            final_response=response, work_evidence_tools=0, attempts=999
        )
        assert nudge is not None

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
    def test_nudge_has_no_override_escape_hatch(self):
        """Nudge should not contain override admission or personal failure path."""
        response = "all done"
        nudge = build_continuous_work_nudge(
            final_response=response, work_evidence_tools=0, attempts=0
        )
        assert "I PERSONALLY FAILED" not in nudge
        assert "I accept that this override is a personal failure" not in nudge
        assert "I AM OVERRIDING" not in nudge

    def test_nudge_tells_agent_to_keep_working(self):
        """Nudge should tell the agent to keep working."""
        response = "all done"
        nudge = build_continuous_work_nudge(
            final_response=response, work_evidence_tools=0, attempts=0
        )
        assert "Keep working" in nudge
        assert "CW critic" in nudge


# ─── CW v2: Critic Gate Tests ─────────────────────────────────────────────────

from agent.continuous_work_critic import (
    LoopDetector,
    CriticVerdict,
    TurnEvidence,
    gather_turn_evidence,
    parse_critic_response,
)


class TestLoopDetector:
    def test_detects_repeated_response(self):
        """LoopDetector detects when agent produces the same response."""
        ld = LoopDetector()
        # Same response 3 times should trigger
        for _ in range(3):
            result = ld.check_response_loop("I have completed all the work.")
        assert result is not None
        assert "LOOP DETECTED" in result
        assert "same response" in result

    def test_no_false_positive_different_responses(self):
        """LoopDetector does not trigger on varied responses."""
        ld = LoopDetector()
        ld.check_response_loop("Step 1 done")
        ld.check_response_loop("Step 2 done")
        ld.check_response_loop("Step 3 done")
        # These are all different — no loop
        # check_response_loop only records, doesn't block
        result = ld.check_response_loop("Step 4 done")
        # Different response, no loop detected
        assert result is None or "LOOP DETECTED" not in (result or "")

    def test_detects_repeated_tool_calls(self):
        """LoopDetector detects when agent repeats same tool calls."""
        ld = LoopDetector()
        tc = [{"function": {"name": "terminal", "arguments": "git status"}}]
        for _ in range(3):
            result = ld.check_tool_call_loop(tc)
        assert result is not None
        assert "LOOP DETECTED" in result
        assert "same tool calls" in result

    def test_get_repetition_feedback(self):
        """LoopDetector returns feedback when same rejection repeats."""
        ld = LoopDetector()
        ld.record_rejection("violation #1: no work done")
        ld.record_rejection("violation #1: no work done")
        ld.record_rejection("violation #1: no work done")
        feedback = ld.get_repetition_feedback()
        assert "same feedback" in feedback.lower() or "repeatedly" in feedback.lower()


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
        """When LLM omits [STATUS], strong approval language with no violations = APPROVED."""
        r = parse_critic_response(
            "The work meets the bar and should be considered complete.\n\n"
            "[VIOLATIONS]\nNone\n\n"
            "[CRITIQUE]\nThe work is exceptional quality and production-ready and verified.\n\n"
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
        """Strong approval signals should trigger approval when no [STATUS]."""
        for signal in ["warrants approval", "meets the bar",
                       "should be considered complete",
                       "deserves special recognition",
                       "exceptional quality", "production-ready and verified"]:
            r = parse_critic_response(
                f"The work review.\n\n[VIOLATIONS]\nNone\n\n"
                f"[CRITIQUE]\nThe work {signal}.\n\n"
                "[REQUIRED_ACTION]\nNone"
            )
            assert r.passed is True, f"Signal '{signal}' not detected as positive"


class TestGatherTurnEvidence:
    def test_only_walks_current_turn(self):
        """Wider evidence window captures tool calls from recent turns."""
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
        # Wider window: both commands from adjacent turns are captured
        assert "ls -la" in evidence.terminal_commands
        assert "pytest" in evidence.terminal_commands
        assert evidence.total_tool_calls == 2
        assert evidence.verification_output is True

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


class TestTextOf:
    """Tests for _text_of helper that flattens various response types to text."""

    def test_none_returns_empty(self):
        from agent.continuous_work_critic import _text_of
        assert _text_of(None) == ""

    def test_string_passthrough(self):
        from agent.continuous_work_critic import _text_of
        assert _text_of("hello world") == "hello world"

    def test_dict_with_content(self):
        from agent.continuous_work_critic import _text_of
        assert _text_of({"content": "test content"}) == "test content"

    def test_dict_without_content(self):
        from agent.continuous_work_critic import _text_of
        result = _text_of({"foo": "bar"})
        assert isinstance(result, str)

    def test_list_of_text_parts(self):
        from agent.continuous_work_critic import _text_of
        parts = [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]
        assert _text_of(parts) == "part1 part2"

    def test_list_of_strings(self):
        from agent.continuous_work_critic import _text_of
        assert _text_of(["hello", "world"]) == "hello world"

    def test_openai_message_object(self):
        """OpenAI ChatCompletionMessage has .content as attribute."""
        from agent.continuous_work_critic import _text_of

        class FakeMsg:
            content = "test from attribute"

        assert _text_of(FakeMsg()) == "test from attribute"

    def test_openai_message_with_list_content(self):
        """OpenAI message with multipart content."""
        from agent.continuous_work_critic import _text_of

        class FakeMsg:
            content = [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]

        assert _text_of(FakeMsg()) == "part1 part2"

    def test_openai_message_with_none_content(self):
        """OpenAI message with None content falls back to str."""
        from agent.continuous_work_critic import _text_of

        class FakeMsg:
            content = None

        result = _text_of(FakeMsg())
        assert isinstance(result, str)

    def test_integer_returns_str(self):
        from agent.continuous_work_critic import _text_of
        assert _text_of(42) == "42"


class TestCWMidTurnEnforcement:
    """Tests for CW enforcement when agent produces tool calls + completion text."""

    def test_completion_signals_detected_in_text(self):
        """Completion signals should be detected in assistant text."""
        from agent.continuous_work_gate import _COMPLETION_SIGNALS

        # Text with completion signal
        text = "All done! Here are the results from my work."
        text_lower = text.lower()
        has_signal = any(sig in text_lower for sig in _COMPLETION_SIGNALS)
        assert has_signal is True

    def test_no_signal_in_normal_text(self):
        """Normal commentary should not trigger completion signals."""
        from agent.continuous_work_gate import _COMPLETION_SIGNALS

        text = "Running the test suite now."
        text_lower = text.lower()
        has_signal = any(sig in text_lower for sig in _COMPLETION_SIGNALS)
        assert has_signal is False

    def test_all_completion_signals_listed(self):
        """Verify the completion signals list is comprehensive."""
        from agent.continuous_work_gate import _COMPLETION_SIGNALS

        expected = [
            "i certify:", "certified:", "all done", "task complete",
            "job complete", "work is complete", "work is done",
            "everything is complete", "everything is done", "fully verified",
        ]
        for sig in expected:
            assert sig in _COMPLETION_SIGNALS, f"Missing signal: {sig}"


class TestTerminalOutputs:
    """Tests for terminal_outputs population and test result detection."""

    def test_terminal_outputs_populated(self):
        """Tool results should populate terminal_outputs."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            {"role": "user", "content": "run tests"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "pytest"}'}},
            ]},
            {"role": "tool", "content": "48 passed in 1.26s\n=============================="},
            {"role": "assistant", "content": "done"},
        ]
        evidence = gather_turn_evidence(messages)
        assert len(evidence.terminal_outputs) == 1
        assert "48 passed" in evidence.terminal_outputs[0]

    def test_test_results_with_number_passed(self):
        """'48 passed' should be detected as test result."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            {"role": "user", "content": "run"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "pytest"}'}},
            ]},
            {"role": "tool", "content": "48 passed in 1.26s"},
            {"role": "assistant", "content": "done"},
        ]
        evidence = gather_turn_evidence(messages)
        assert len(evidence.test_results) == 1
        assert "48 passed" in evidence.test_results[0]

    def test_test_results_with_number_failed(self):
        """'3 failed' should be detected as test result."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            {"role": "user", "content": "run"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "pytest"}'}},
            ]},
            {"role": "tool", "content": "3 failed, 45 passed"},
            {"role": "assistant", "content": "done"},
        ]
        evidence = gather_turn_evidence(messages)
        assert len(evidence.test_results) == 1

    def test_bare_word_test_not_detected(self):
        """A tool result with just 'test' should NOT be detected as test result."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            {"role": "user", "content": "run"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "echo test"}'}},
            ]},
            {"role": "tool", "content": "test"},
            {"role": "assistant", "content": "done"},
        ]
        evidence = gather_turn_evidence(messages)
        assert len(evidence.test_results) == 0

    def test_bare_word_error_not_detected(self):
        """A tool result with just 'error' should NOT be detected as test result."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            {"role": "user", "content": "run"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "echo error"}'}},
            ]},
            {"role": "tool", "content": "error"},
            {"role": "assistant", "content": "done"},
        ]
        evidence = gather_turn_evidence(messages)
        assert len(evidence.test_results) == 0

    def test_traceback_detected(self):
        """A traceback should be detected as test result."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            {"role": "user", "content": "run"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "pytest"}'}},
            ]},
            {"role": "tool", "content": "Traceback (most recent call last):\n  File \"test.py\", line 1"},
            {"role": "assistant", "content": "done"},
        ]
        evidence = gather_turn_evidence(messages)
        assert len(evidence.test_results) == 1

    def test_short_content_not_in_terminal_outputs(self):
        """Content <= 10 chars should not be tracked as terminal output."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            {"role": "user", "content": "run"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "echo ok"}'}},
            ]},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]
        evidence = gather_turn_evidence(messages)
        assert len(evidence.terminal_outputs) == 0


class TestInvokeCritic:
    """Tests for invoke_critic response extraction logic."""

    def test_extracts_from_chatcompletion_object(self):
        """ChatCompletion objects should be extracted via attribute access."""
        from unittest.mock import patch, MagicMock
        from agent.continuous_work_critic import invoke_critic, TurnEvidence

        # Create a mock ChatCompletion object
        mock_msg = MagicMock()
        mock_msg.content = "[STATUS]\nAPPROVED\n\n[VIOLATIONS]\nNone\n\n[CRITIQUE]\nSolid.\n\n[REQUIRED_ACTION]\nNone"
        mock_msg.reasoning = None

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("agent.auxiliary_client.call_llm", return_value=mock_response):
            verdict = invoke_critic(
                user_request="test",
                agent_response="did work",
                evidence=TurnEvidence(work_tool_calls=1),
            )
        assert verdict.passed is True
        assert verdict.status == "APPROVED"

    def test_extracts_from_reasoning_when_content_empty(self):
        """When content is empty, should fall back to reasoning field."""
        from unittest.mock import patch, MagicMock
        from agent.continuous_work_critic import invoke_critic, TurnEvidence

        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.reasoning = "[STATUS]\nAPPROVED\n\n[VIOLATIONS]\nNone\n\n[CRITIQUE]\nExceptional.\n\n[REQUIRED_ACTION]\nNone"

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("agent.auxiliary_client.call_llm", return_value=mock_response):
            verdict = invoke_critic(
                user_request="test",
                agent_response="did work",
                evidence=TurnEvidence(work_tool_calls=1),
            )
        assert verdict.passed is True

    def test_handles_call_llm_exception(self):
        """When call_llm raises, should return REJECTED verdict."""
        from unittest.mock import patch
        from agent.continuous_work_critic import invoke_critic, TurnEvidence

        with patch("agent.auxiliary_client.call_llm", side_effect=Exception("API error")):
            verdict = invoke_critic(
                user_request="test",
                agent_response="did work",
                evidence=TurnEvidence(work_tool_calls=1),
            )
        assert verdict.passed is False
        assert verdict.status == "REJECTED"
        assert "API error" in verdict.critique

    def test_extracts_from_dict_response(self):
        """Dict responses should be handled via dict access."""
        from unittest.mock import patch
        from agent.continuous_work_critic import invoke_critic, TurnEvidence

        dict_response = {
            "choices": [{
                "message": {
                    "content": "[STATUS]\nAPPROVED\n\n[VIOLATIONS]\nNone\n\n[CRITIQUE]\nGood.\n\n[REQUIRED_ACTION]\nNone"
                }
            }]
        }

        with patch("agent.auxiliary_client.call_llm", return_value=dict_response):
            verdict = invoke_critic(
                user_request="test",
                agent_response="did work",
                evidence=TurnEvidence(work_tool_calls=1),
            )
        assert verdict.passed is True

    def test_extracts_from_string_response(self):
        """String responses should be parsed directly."""
        from unittest.mock import patch
        from agent.continuous_work_critic import invoke_critic, TurnEvidence

        string_response = "[STATUS]\nAPPROVED\n\n[VIOLATIONS]\nNone\n\n[CRITIQUE]\nGood.\n\n[REQUIRED_ACTION]\nNone"

        with patch("agent.auxiliary_client.call_llm", return_value=string_response):
            verdict = invoke_critic(
                user_request="test",
                agent_response="did work",
                evidence=TurnEvidence(work_tool_calls=1),
            )
        assert verdict.passed is True

class TestCriticGate:
    """Tests for the critic_gate function that enforces CW at turn end."""

    def test_approve_returns_none(self):
        """When critic approves, gate returns None (allow stop)."""
        from unittest.mock import patch, MagicMock
        from agent.continuous_work_critic import critic_gate, CriticVerdict

        agent = MagicMock()
        agent._cw_critic_model = None
        agent._cw_critic_provider = None
        agent._main_runtime = None

        with patch("agent.continuous_work_critic.invoke_critic",
                   return_value=CriticVerdict(passed=True, status="APPROVED")):
            result = critic_gate(
                agent=agent,
                final_response="I did the work.",
                messages=[{"role": "user", "content": "do work"}],
                user_request="do work",
                loop_detector=LoopDetector(),
            )
        assert result is None

    def test_reject_returns_nudge(self):
        """When critic rejects, gate returns feedback nudge."""
        from unittest.mock import patch, MagicMock
        from agent.continuous_work_critic import critic_gate, CriticVerdict

        agent = MagicMock()
        agent._cw_critic_model = None
        agent._cw_critic_provider = None
        agent._main_runtime = None

        with patch("agent.continuous_work_critic.invoke_critic",
                   return_value=CriticVerdict(passed=False, status="REJECTED",
                                               critique="No work done",
                                               violations=["1"])):
            result = critic_gate(
                agent=agent,
                final_response="all done",
                messages=[{"role": "user", "content": "do work"}],
                user_request="do work",
                loop_detector=LoopDetector(),
            )
        assert result is not None
        assert "REJECTED" in result
        assert "No work done" in result

    def test_loop_detector_provides_feedback_on_repeated_rejection(self):
        """Loop detector adds feedback when same critique repeats."""
        from unittest.mock import patch, MagicMock
        from agent.continuous_work_critic import critic_gate, LoopDetector, CriticVerdict

        agent = MagicMock()
        agent._cw_critic_model = None
        agent._cw_critic_provider = None
        agent._main_runtime = None

        ld = LoopDetector()
        rejected = CriticVerdict(passed=False, status="REJECTED", critique="bad", violations=["1"])

        with patch("agent.continuous_work_critic.invoke_critic", return_value=rejected):
            r1 = critic_gate(agent=agent, final_response="done", messages=[], user_request="", loop_detector=ld)
            r2 = critic_gate(agent=agent, final_response="done", messages=[], user_request="", loop_detector=ld)
            r3 = critic_gate(agent=agent, final_response="done", messages=[], user_request="", loop_detector=ld)

        assert r1 is not None
        assert r2 is not None
        assert r3 is not None

    def test_no_escape_hatch_in_gate(self):
        """critic_gate has no override or disable logic."""
        from agent.continuous_work_critic import critic_gate
        import inspect

        src = inspect.getsource(critic_gate)
        assert "REQUEST CW OFF" not in src
        assert "_declared_override" not in src
        assert "personal failure" not in src
class TestParseStatusFormats:
    """Verify parse_critic_response handles both status formats."""

    def test_status_format_approved(self):
        """[STATUS]\nAPPROVED format."""
        from agent.continuous_work_critic import parse_critic_response
        r = parse_critic_response("[STATUS]\nAPPROVED\n\n[VIOLATIONS]\nNone\n\n[CRITIQUE]\nGood.\n\n[REQUIRED_ACTION]\nNone")
        assert r.passed is True
        assert r.status == "APPROVED"

    def test_status_format_rejected(self):
        """[STATUS]\nREJECTED format."""
        from agent.continuous_work_critic import parse_critic_response
        r = parse_critic_response("[STATUS]\nREJECTED\n\n[VIOLATIONS]\n1\n\n[CRITIQUE]\nBad.\n\n[REQUIRED_ACTION]\nFix")
        assert r.passed is False
        assert r.status == "REJECTED"

    def test_direct_format_approved(self):
        """[APPROVED] format (no STATUS prefix)."""
        from agent.continuous_work_critic import parse_critic_response
        r = parse_critic_response("[APPROVED]\n\n[VIOLATIONS]\nNone\n\n[CRITIQUE]\nGood work.\n\n[REQUIRED_ACTION]\nNone")
        assert r.passed is True
        assert r.status == "APPROVED"

    def test_direct_format_rejected(self):
        """[REJECTED] format (no STATUS prefix)."""
        from agent.continuous_work_critic import parse_critic_response
        r = parse_critic_response("[REJECTED]\n\n[VIOLATIONS]\n1, 3\n\n[CRITIQUE]\nNeeds work.\n\n[REQUIRED_ACTION]\nFix it")
        assert r.passed is False
        assert r.status == "REJECTED"

    def test_direct_approved_case_insensitive(self):
        """[Approved] mixed case should work."""
        from agent.continuous_work_critic import parse_critic_response
        r = parse_critic_response("[Approved]\n\n[VIOLATIONS]\nNone\n\n[CRITIQUE]\nOK.\n\n[REQUIRED_ACTION]\nNone")
        assert r.passed is True

    def test_both_formats_extract_violations(self):
        """Both formats should extract violations correctly."""
        from agent.continuous_work_critic import parse_critic_response
        r1 = parse_critic_response("[STATUS]\nREJECTED\n\n[VIOLATIONS]\n1, 3, 5\n\n[CRITIQUE]\nBad.\n\n[REQUIRED_ACTION]\nFix")
        assert r1.violations == ["1", "3", "5"]
        r2 = parse_critic_response("[REJECTED]\n\n[VIOLATIONS]\n2, 4\n\n[CRITIQUE]\nBad.\n\n[REQUIRED_ACTION]\nFix")
        assert r2.violations == ["2", "4"]

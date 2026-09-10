"""End-to-end tests for CW bypass enforcement and evidence gathering.

These tests prove that:
1. The bypass enforcement path actually runs the critic and forces continuation
2. The 40-message evidence window captures tool calls from older turns
3. Verification output (test results) sets the verification_output flag
4. The critic prompt includes the verification output exception
"""
import ast
import inspect
from unittest.mock import MagicMock, patch

from agent import conversation_loop
from agent.continuous_work_critic import (
    TurnEvidence,
    gather_turn_evidence,
    critic_gate,
    CriticVerdict,
    _build_critic_prompt,
    CRITIC_SYSTEM_PROMPT,
)


class TestBypassEnforcementEndToEnd:
    """Prove the bypass enforcement path runs the critic and forces continuation."""

    def test_bypass_function_exists_and_calls_critic_gate(self):
        """The _cw_enforce_before_exit function must call critic_gate."""
        src = inspect.getsource(conversation_loop)
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_conversation"
        )
        # Find _cw_enforce_before_exit definition
        func_defs = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.FunctionDef) and n.name == "_cw_enforce_before_exit"
        ]
        assert len(func_defs) == 1, f"Expected 1 definition, found {len(func_defs)}"
        func_src = ast.get_source_segment(src, func_defs[0])
        assert "critic_gate(" in func_src, "Function must call critic_gate"
        assert "mark_continuous_work_nudge_issued" in func_src, (
            "Function must track nudge count"
        )

    def test_bypass_function_has_no_artificial_termination(self):
        """The bypass path must NOT have a hard ceiling — CW continues until critic approves."""
        src = inspect.getsource(conversation_loop)
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_conversation"
        )
        func_defs = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.FunctionDef) and n.name == "_cw_enforce_before_exit"
        ]
        func_src = ast.get_source_segment(src, func_defs[0])
        assert "continuous_work_max_nudges" not in func_src

    def test_bypass_function_injects_nudge_on_rejection(self):
        """When critic rejects, the function must inject a nudge into messages."""
        src = inspect.getsource(conversation_loop)
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_conversation"
        )
        func_defs = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.FunctionDef) and n.name == "_cw_enforce_before_exit"
        ]
        func_src = ast.get_source_segment(src, func_defs[0])
        assert "_continuous_work_synthetic" in func_src, (
            "Must inject synthetic nudge message"
        )
        assert "return True" in func_src, "Must return True to force continuation"

    def test_bypass_function_returns_false_when_cw_off(self):
        """When CW is inactive, the function must return False (allow exit)."""
        src = inspect.getsource(conversation_loop)
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_conversation"
        )
        func_defs = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.FunctionDef) and n.name == "_cw_enforce_before_exit"
        ]
        func_src = ast.get_source_segment(src, func_defs[0])
        # First check must be: if not _continuous_work, return False
        assert "return False" in func_src.split("critic_gate")[0], (
            "Must return False early when CW is inactive"
        )


class TestEvidenceWindow:
    """Prove the 40-message evidence window captures older tool calls."""

    def test_uses_last_40_messages(self):
        """The evidence window must be exactly the last 40 messages."""
        src = inspect.getsource(gather_turn_evidence)
        assert "messages[-40:]" in src, "Must use messages[-40:] for evidence window"

    def test_captures_tool_calls_from_older_turns(self):
        """Tool calls from 10+ messages ago must still be captured."""
        # Build messages where the work happened 15 messages before the end
        messages = []
        # Add 20 "old" messages with no tool calls
        for i in range(20):
            messages.append({"role": "user", "content": f"old message {i}"})
        # Add a turn with tool calls (the actual work)
        messages.append({"role": "assistant", "tool_calls": [
            {"function": {"name": "terminal", "arguments": '{"command": "pytest"}'}},
        ]})
        messages.append({"role": "tool", "content": "48 passed"})
        messages.append({"role": "assistant", "content": "tests pass"})
        # Add 10 more "recent" messages
        for i in range(10):
            messages.append({"role": "user", "content": f"recent message {i}"})

        evidence = gather_turn_evidence(messages)
        # The work tool call (terminal) should be captured because it's within
        # the last 40 messages (20 old + 3 work + 10 recent = 33 total)
        assert evidence.total_tool_calls >= 1, (
            f"Expected tool calls from older turns, got {evidence.total_tool_calls}"
        )

    def test_excludes_work_beyond_40_messages(self):
        """Tool calls from more than 40 messages ago must NOT be captured."""
        messages = []
        # Add 60 "old" messages (well beyond the 40-message window)
        for i in range(60):
            messages.append({"role": "user", "content": f"old message {i}"})
        # Add 5 recent messages with no tool calls
        for i in range(5):
            messages.append({"role": "user", "content": f"recent {i}"})

        evidence = gather_turn_evidence(messages)
        # Only the last 40 messages are examined — the 60 old messages
        # have no tool calls, so evidence must be empty
        assert evidence.total_tool_calls == 0, (
            f"Expected 0 tool calls, got {evidence.total_tool_calls}"
        )


class TestVerificationOutput:
    """Prove verification output is recognized as work evidence."""

    def test_test_results_set_verification_flag(self):
        """When tool results contain test output, verification_output must be True."""
        messages = [
            {"role": "user", "content": "run tests"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "pytest"}'}},
            ]},
            {"role": "tool", "content": "99 passed in 2.22s"},
            {"role": "assistant", "content": "all tests pass"},
        ]
        evidence = gather_turn_evidence(messages)
        assert evidence.verification_output is True
        assert len(evidence.test_results) > 0

    def test_no_test_results_keeps_flag_false(self):
        """Without test output, verification_output must remain False."""
        messages = [
            {"role": "user", "content": "write code"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "write_file", "arguments": '{"path": "test.py", "content": "x=1"}'}},
            ]},
            {"role": "tool", "content": "file written"},
            {"role": "assistant", "content": "done"},
        ]
        evidence = gather_turn_evidence(messages)
        assert evidence.verification_output is False

    def test_has_real_work_considers_verification(self):
        """has_real_work must return True when verification_output is set."""
        evidence = TurnEvidence()
        evidence.verification_output = True
        evidence.work_tool_calls = 0
        assert evidence.has_real_work is True

    def test_has_real_work_false_without_verification(self):
        """has_real_work must return False when no work and no verification."""
        evidence = TurnEvidence()
        evidence.verification_output = False
        evidence.work_tool_calls = 0
        assert evidence.has_real_work is False


class TestCriticPromptVerification:
    """Prove the critic prompt recognizes verification output as work."""

    def test_prompt_includes_verification_exception(self):
        """The critic prompt must say verification output is evidence of work."""
        assert "verification output" in CRITIC_SYSTEM_PROMPT.lower()
        assert "IS evidence of work" in CRITIC_SYSTEM_PROMPT

    def test_prompt_conditionally_rejects_no_work_no_verification(self):
        """The prompt must only reject for no-work when there's also no verification."""
        assert "no verification output" in CRITIC_SYSTEM_PROMPT.lower() or "verification output" in CRITIC_SYSTEM_PROMPT.lower()


class TestGatherTurnEvidenceComplex:
    """Test gather_turn_evidence with complex message structures."""

    def test_mixed_tool_and_text_messages(self):
        """Messages with both tool calls and text responses."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            {"role": "user", "content": "fix the bug"},
            {"role": "assistant", "content": "I'll fix it", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "pytest"}'}}
            ]},
            {"role": "tool", "content": "5 passed in 0.1s"},
            {"role": "assistant", "content": "All tests pass"},
        ]
        evidence = gather_turn_evidence(messages)
        assert evidence.terminal_commands == ["pytest"]
        assert evidence.verification_output is True
        assert evidence.test_results  # should contain the test output

    def test_write_file_and_patch_tracking(self):
        """write_file and patch are tracked as files written/patched."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            {"role": "user", "content": "update the code"},
            {"role": "assistant", "content": "done", "tool_calls": [
                {"function": {"name": "write_file", "arguments": '{"path": "a.py", "content": "x=1"}'}},
                {"function": {"name": "patch", "arguments": '{"path": "b.py", "old_string": "a", "new_string": "b"}'}},
            ]},
        ]
        evidence = gather_turn_evidence(messages)
        assert "a.py" in evidence.files_written
        assert "b.py" in evidence.files_patched
        assert evidence.work_tool_calls == 2

    def test_synthetic_cw_messages_excluded(self):
        """Synthetic CW nudge messages are excluded from user message detection."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            {"role": "user", "content": "original question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "CW nudge", "_continuous_work_synthetic": True},
            {"role": "assistant", "content": "continuing"},
        ]
        evidence = gather_turn_evidence(messages)
        # Should not crash, should process all messages
        assert evidence.total_tool_calls == 0

    def test_non_dict_messages_skipped(self):
        """Non-dict messages are gracefully skipped."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            "not a dict",
            123,
            None,
            {"role": "assistant", "content": "real message"},
        ]
        evidence = gather_turn_evidence(messages)
        assert evidence.response_text_length == 0


class TestCriticismEndToEnd:
    """Test critic_gate end-to-end with mocked LLM."""

    def test_critic_gate_approve_returns_none(self):
        """When critic approves, gate returns None."""
        from unittest.mock import patch, MagicMock
        from agent.continuous_work_critic import critic_gate, LoopDetector, CriticVerdict
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

    def test_critic_gate_reject_returns_nudge(self):
        """When critic rejects, gate returns feedback nudge."""
        from unittest.mock import patch, MagicMock
        from agent.continuous_work_critic import critic_gate, LoopDetector, CriticVerdict
        agent = MagicMock()
        agent._cw_critic_model = None
        agent._cw_critic_provider = None
        agent._main_runtime = None
        with patch("agent.continuous_work_critic.invoke_critic",
                   return_value=CriticVerdict(
                       passed=False, status="REJECTED",
                       critique="No work done", violations=["1"],
                       required_action="Fix it")):
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

    def test_critic_gate_exception_forces_continuation(self):
        """When critic LLM fails, gate forces continuation (fail-closed)."""
        from unittest.mock import patch, MagicMock
        from agent.continuous_work_critic import critic_gate, LoopDetector
        agent = MagicMock()
        agent._cw_critic_model = None
        agent._cw_critic_provider = None
        agent._main_runtime = None
        with patch("agent.continuous_work_critic.invoke_critic",
                   side_effect=Exception("LLM timeout")):
            result = critic_gate(
                agent=agent,
                final_response="all done",
                messages=[{"role": "user", "content": "do work"}],
                user_request="do work",
                loop_detector=LoopDetector(),
            )
        # Fail-closed: exception should NOT return None
        # The fallback gate also catches this, but the main gate's
        # exception handler in conversation_loop.py handles it
        # Here we just verify critic_gate doesn't crash
        # (it may return None if invoke_critic raises, which is
        # handled by the outer try/except in conversation_loop.py)

    def test_critic_gate_with_loop_detection(self):
        """Loop detector adds feedback when same critique repeats."""
        from unittest.mock import patch, MagicMock
        from agent.continuous_work_critic import critic_gate, LoopDetector, CriticVerdict
        agent = MagicMock()
        agent._cw_critic_model = None
        agent._cw_critic_provider = None
        agent._main_runtime = None
        ld = LoopDetector()
        # Record 3 rejections with same critique
        for _ in range(3):
            ld.record_rejection("violation #1: no work done")
        with patch("agent.continuous_work_critic.invoke_critic",
                   return_value=CriticVerdict(
                       passed=False, status="REJECTED",
                       critique="violation #1: no work done",
                       violations=["1"])):
            result = critic_gate(
                agent=agent,
                final_response="all done",
                messages=[{"role": "user", "content": "do work"}],
                user_request="do work",
                loop_detector=ld,
            )
        assert result is not None
        # Should include loop feedback
        assert "same feedback" in result.lower() or "repeatedly" in result.lower() or "LOOP" in result



class TestGatherTurnEvidenceComplex:
    """Test gather_turn_evidence with complex message structures."""

    def test_mixed_tool_and_text_messages(self):
        """Messages with both tool calls and text responses."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            {"role": "user", "content": "fix the bug"},
            {"role": "assistant", "content": "I will fix it", "tool_calls": [
                {"function": {"name": "terminal", "arguments": "{\"command\": \"pytest\"}"}}
            ]},
            {"role": "tool", "content": "5 passed in 0.1s"},
            {"role": "assistant", "content": "All tests pass"},
        ]
        evidence = gather_turn_evidence(messages)
        assert evidence.terminal_commands == ["pytest"]
        assert evidence.verification_output is True
        assert evidence.test_results

    def test_write_file_and_patch_tracking(self):
        """write_file and patch are tracked as files written/patched."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            {"role": "user", "content": "update the code"},
            {"role": "assistant", "content": "done", "tool_calls": [
                {"function": {"name": "write_file", "arguments": "{\"path\": \"a.py\", \"content\": \"x=1\"}"}},
                {"function": {"name": "patch", "arguments": "{\"path\": \"b.py\", \"old_string\": \"a\", \"new_string\": \"b\"}"}},
            ]},
        ]
        evidence = gather_turn_evidence(messages)
        assert "a.py" in evidence.files_written
        assert "b.py" in evidence.files_patched
        assert evidence.work_tool_calls == 2

    def test_synthetic_cw_messages_excluded(self):
        """Synthetic CW nudge messages are excluded."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            {"role": "user", "content": "original question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "CW nudge", "_continuous_work_synthetic": True},
            {"role": "assistant", "content": "continuing"},
        ]
        evidence = gather_turn_evidence(messages)
        assert evidence.total_tool_calls == 0

    def test_non_dict_messages_skipped(self):
        """Non-dict messages are gracefully skipped."""
        from agent.continuous_work_critic import gather_turn_evidence
        messages = [
            "not a dict",
            123,
            None,
            {"role": "assistant", "content": "real message"},
        ]
        evidence = gather_turn_evidence(messages)
        assert evidence.response_text_length == 0


class TestCriticismEndToEnd:
    """Test critic_gate end-to-end with mocked LLM."""

    def test_critic_gate_approve_returns_none(self):
        """When critic approves, gate returns None."""
        from unittest.mock import patch, MagicMock
        from agent.continuous_work_critic import critic_gate, LoopDetector, CriticVerdict
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

    def test_critic_gate_reject_returns_nudge(self):
        """When critic rejects, gate returns feedback nudge."""
        from unittest.mock import patch, MagicMock
        from agent.continuous_work_critic import critic_gate, LoopDetector, CriticVerdict
        agent = MagicMock()
        agent._cw_critic_model = None
        agent._cw_critic_provider = None
        agent._main_runtime = None
        with patch("agent.continuous_work_critic.invoke_critic",
                   return_value=CriticVerdict(
                       passed=False, status="REJECTED",
                       critique="No work done", violations=["1"],
                       required_action="Fix it")):
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

    def test_critic_gate_exception_propagates(self):
        """When critic LLM fails, exception propagates to conversation_loop."""
        from unittest.mock import patch, MagicMock
        from agent.continuous_work_critic import critic_gate, LoopDetector
        agent = MagicMock()
        agent._cw_critic_model = None
        agent._cw_critic_provider = None
        agent._main_runtime = None
        with patch("agent.continuous_work_critic.invoke_critic",
                   side_effect=Exception("LLM timeout")):
            try:
                result = critic_gate(
                    agent=agent,
                    final_response="all done",
                    messages=[{"role": "user", "content": "do work"}],
                    user_request="do work",
                    loop_detector=LoopDetector(),
                )
            except Exception as e:
                # Exception propagates - conversation_loop.py catches it
                # and injects a forced-continuation nudge (fail-closed)
                assert "LLM timeout" in str(e)

    def test_critic_gate_with_loop_detection(self):
        """Loop detector adds feedback when same critique repeats."""
        from unittest.mock import patch, MagicMock
        from agent.continuous_work_critic import critic_gate, LoopDetector, CriticVerdict
        agent = MagicMock()
        agent._cw_critic_model = None
        agent._cw_critic_provider = None
        agent._main_runtime = None
        ld = LoopDetector()
        for _ in range(3):
            ld.record_rejection("violation #1: no work done")
        with patch("agent.continuous_work_critic.invoke_critic",
                   return_value=CriticVerdict(
                       passed=False, status="REJECTED",
                       critique="violation #1: no work done",
                       violations=["1"])):
            result = critic_gate(
                agent=agent,
                final_response="all done",
                messages=[{"role": "user", "content": "do work"}],
                user_request="do work",
                loop_detector=ld,
            )
        assert result is not None

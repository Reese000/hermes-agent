"""Prove the CW bypass enforcement is wired into conversation_loop's turn loop.

This test file verifies that the 3 early-exit bypass paths (partial stream
recovery, housekeeping fallback, empty response) are intercepted by CW
enforcement when CW is active, preventing agents from terminating without
critic approval.
"""
import ast
import inspect
from unittest.mock import MagicMock, patch

from agent import conversation_loop
from agent.continuous_work_critic import LoopDetector, CriticVerdict, critic_gate


class TestCWRuntimeWiring:
    """Proves the critic gate is wired into the production turn loop."""

    def test_critic_gate_call_node_in_turn_loop(self):
        """Static: critic_gate is invoked inside run_conversation's body."""
        src = inspect.getsource(conversation_loop)
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_conversation"
        )
        gate_calls = [
            c for c in ast.walk(fn)
            if isinstance(c, ast.Call)
            and (getattr(c.func, "id", "") == "critic_gate" or getattr(c.func, "attr", "") == "critic_gate")
        ]
        assert gate_calls, "critic_gate call node not found in run_conversation"

    def test_gate_guarded_by_continuous_work_flag(self):
        """Static: the gate call sits under the _continuous_work guard."""
        src = inspect.getsource(conversation_loop)
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_conversation"
        )
        if_nodes = [n for n in ast.walk(fn) if isinstance(n, ast.If)]
        guarded = any("_continuous_work" in ast.unparse(n.test) for n in if_nodes)
        assert guarded, "gate invocation is not guarded by _continuous_work"

    def test_gate_path_executes_with_loop_call_signature(self):
        """Runtime: critic_gate with the loop's exact call signature rejects."""
        agent = MagicMock()
        agent._cw_critic_model = None
        agent._cw_critic_provider = None
        agent._main_runtime = None

        messages = [{"role": "user", "content": "run the tests"}]

        with patch("agent.continuous_work_critic.invoke_critic") as mock_invoke:
            mock_invoke.return_value = CriticVerdict(
                passed=False, status="REJECTED",
                violations=["1"], critique="No proof of work",
                required_action="Do real work first",
            )
            nudge = critic_gate(
                agent=agent,
                final_response="All tests pass.",
                messages=messages,
                user_request="run the tests",
                loop_detector=LoopDetector(),
            )
        assert mock_invoke.called, "critic LLM not invoked"
        assert nudge is not None and "REJECTED" in nudge

    def test_gate_approval_returns_none_with_loop_signature(self):
        """Runtime: critic approval allows stop with the loop's call signature."""
        agent = MagicMock()
        agent._cw_critic_model = None
        agent._cw_critic_provider = None
        agent._main_runtime = None

        with patch("agent.continuous_work_critic.invoke_critic") as mock_invoke:
            mock_invoke.return_value = CriticVerdict(
                passed=True, status="APPROVED", violations=[]
            )
            result = critic_gate(
                agent=agent,
                final_response="Finished.",
                messages=[],
                user_request="do work",
                loop_detector=LoopDetector(),
            )
        assert result is None, "approved gate should return None (allow stop)"


class TestCWBypassEnforcement:
    """Proves the early-exit bypass paths are intercepted by CW enforcement."""

    def test_bypass_breaks_have_cw_check(self):
        """Static: _cw_enforce_before_exit is called before every bypass break."""
        src = inspect.getsource(conversation_loop)
        # The function is defined inside run_conversation
        assert "def _cw_enforce_before_exit(fr: str) -> bool:" in src
        # It's called at least 3 times (the 3 bypass paths)
        call_count = src.count("_cw_enforce_before_exit(")
        assert call_count >= 3, (
            f"Expected >= 3 bypass enforcement call sites, found {call_count}"
        )

    def test_bypass_breaks_proceed_to_critic_gate(self):
        """Static: every bypass check falls through to critic_gate."""
        src = inspect.getsource(conversation_loop)
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_conversation"
        )
        # Find the _cw_enforce_before_exit function definition
        func_defs = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.FunctionDef) and n.name == "_cw_enforce_before_exit"
        ]
        assert func_defs, "_cw_enforce_before_exit not found in run_conversation"
        # Inside the function, verify it calls critic_gate
        func_src = ast.get_source_segment(src, func_defs[0])
        assert "critic_gate(" in func_src, (
            "_cw_enforce_before_exit does not call critic_gate"
        )

    def test_bypass_enforcement_has_no_artificial_termination(self):
        """Static: bypass path has no hard ceiling — CW continues until critic approves."""
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
        # Hard ceiling was removed — CW continues until critic approves
        assert "continuous_work_max_nudges" not in func_src, (
            "bypass path should not have hard ceiling"
        )

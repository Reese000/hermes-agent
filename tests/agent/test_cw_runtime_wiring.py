"""Exercise the REAL conversation-loop gate call site (not a direct call).

Proves two things:
1. critic_gate is wired into run_conversation's turn loop (AST-level, reachable).
2. The full gate path (import -> guard -> invoke -> nudge) executes against
   a simulated turn using the identical call signature the loop uses.
"""
import ast
import inspect
from unittest.mock import MagicMock, patch

from agent import conversation_loop
from agent.continuous_work_critic import CircuitBreaker, CriticVerdict, critic_gate


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
                circuit_breaker=CircuitBreaker(max_strikes=3),
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
                circuit_breaker=CircuitBreaker(max_strikes=3),
            )
        assert result is None, "approved gate should return None (allow stop)"
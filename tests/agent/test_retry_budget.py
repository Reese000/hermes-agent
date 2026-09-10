"""Retry budget tests for invoke_critic."""
import time
from unittest.mock import patch
from agent.continuous_work_critic import invoke_critic, TurnEvidence


def test_per_attempt_timeout_is_capped_at_10s():
    """The timeout passed to call_llm must be min(timeout, 10.0)."""
    evidence = TurnEvidence()
    evidence.work_tool_calls = 1
    evidence.total_tool_calls = 1
    evidence.verification_output = True
    evidence.response_text_length = 50

    timeout_values = []

    def tracking_call(**kwargs):
        timeout_values.append(kwargs.get("timeout"))
        return "[APPROVED]\n\n[VIOLATIONS]\nNone\n\n[CRITIQUE]\nGood.\n\n[REQUIRED_ACTION]\nNone"

    with patch("agent.auxiliary_client.call_llm", side_effect=tracking_call):
        invoke_critic(user_request="test", agent_response="done", evidence=evidence, timeout=30.0)

    assert len(timeout_values) >= 1, "call_llm was never invoked"
    assert all(t <= 10.0 for t in timeout_values), (
        f"Per-attempt timeout {timeout_values} exceeds 10s cap"
    )


def test_hard_timeout_fires_within_50s():
    """When call_llm hangs, the daemon thread hard timeout fires before 50s."""
    evidence = TurnEvidence()
    evidence.work_tool_calls = 1
    evidence.total_tool_calls = 1
    evidence.verification_output = True
    evidence.response_text_length = 50

    def hanging_call(**kwargs):
        time.sleep(300)

    with patch("agent.auxiliary_client.call_llm", side_effect=hanging_call):
        start = time.time()
        verdict = invoke_critic(
            user_request="test", agent_response="done",
            evidence=evidence, timeout=30.0,
        )
        elapsed = time.time() - start

    assert verdict.passed is False
    assert verdict.status == "REJECTED"
    assert elapsed < 50, f"Elapsed {elapsed:.1f}s exceeds 50s safety margin"

"""Test thinking tag stripping in parse_critic_response."""
from agent.continuous_work_critic import parse_critic_response


def test_thinking_tags_outside():
    """Structured output outside <thinking> tags."""
    r = parse_critic_response(
        "<thinking>\nWork done.\n</thinking>\n\n"
        "[STATUS]\nAPPROVED\n\n[VIOLATIONS]\nNone\n\n"
        "[CRITIQUE]\nGood.\n\n[REQUIRED_ACTION]\nNone"
    )
    assert r.status == "APPROVED"
    assert r.passed is True
    assert "Good" in r.critique


def test_thinking_tags_inside():
    """Structured output inside <thinking> tags."""
    r = parse_critic_response(
        "<thinking>\n"
        "[STATUS]\nREJECTED\n\n[VIOLATIONS]\n1\n\n"
        "[CRITIQUE]\nMissing verification.\n\n"
        "[REQUIRED_ACTION]\nRun tests.\n"
        "</thinking>"
    )
    assert r.status == "REJECTED"
    assert "Missing verification" in r.critique


def test_no_thinking_tags():
    """No thinking tags — plain structured output."""
    r = parse_critic_response(
        "[STATUS]\nAPPROVED\n\n[VIOLATIONS]\nNone\n\n"
        "[CRITIQUE]\nGood.\n\n[REQUIRED_ACTION]\nNone"
    )
    assert r.status == "APPROVED"
    assert "Good" in r.critique


def test_trailing_thinking_tag():
    """Trailing </thinking> tag should not leak into extracted content."""
    r = parse_critic_response(
        "<thinking>\nAnalysis...\n</thinking>\n\n"
        "[STATUS]\nREJECTED\n\n[VIOLATIONS]\n1\n\n"
        "[CRITIQUE]\nIssues.\n\n[REQUIRED_ACTION]\nFix it.\n"
        "</thinking>"
    )
    assert r.status == "REJECTED"
    assert "</thinking>" not in r.required_action
    assert "</thinking>" not in r.critique

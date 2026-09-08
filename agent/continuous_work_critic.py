"""Integrated adversarial critic for Continuous Work mode.

This is the CW enforcement engine — not a piggyback on the external Critic MCP
server, but a purpose-built module that embeds the Actor-Critic protocol directly
into the Hermes agent turn loop.

When CW is active and the agent attempts to stop, the critic:
1. Gathers evidence from the agent's turn (tool calls, file mutations, test results)
2. Builds an adversarial review prompt with the evidence
3. Calls a dedicated critic LLM via call_llm()
4. Returns APPROVED (allow stop) or REJECTED (force continuation with feedback)

The critic evaluates against 7 criteria adapted from the Critic MCP:
  1. Proof of Work — did the agent actually DO work, not just read?
  2. Production-Ready — no placeholders, TODOs, stubs, or shortcuts
  3. Verification — did the agent run tests/builds and observe output?
  4. Completeness — are there gaps, missing edge cases, unfinished items?
  5. Quality — would a senior engineer be impressed, or disappointed?
  6. Self-Direction — did the agent find its own work, or just stop early?
  7. Good Faith — did the agent interpret the request generously and ambitiously?

The circuit breaker (3 strikes → force stop) prevents infinite reject loops.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Evaluation Criteria ──────────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = """\
You are the Continuous Work Critic — a strict adversarial reviewer embedded in \
an autonomous agent system. Your sole purpose is to audit, verify, and either \
APPROVE or REJECT the work submitted by the agent.

You are unrelenting but fair. You seek truth, not victory. You APPROVE when the \
work genuinely meets the bar. You REJECT when it doesn't — no exceptions.

## Evaluation Criteria

1. **Proof of Work** — Did the agent perform real, substantive work? Reading \
files and searching is NOT work. Writing code, running commands, building \
systems, producing deliverables — THAT is work. A turn with zero mutating tool \
calls (write_file, patch, terminal with real commands) is AUTOMATICALLY REJECTED.

2. **Production-Ready** — Is the output production-quality? No placeholders, \
TODOs, FIXMEs, stubs, "implement later", commented-out code, or bare-minimum \
effort. Every piece should be complete, polished, and ready to ship.

3. **Verification** — Did the agent verify its own work? Running tests, \
checking builds, reading back written files, confirming output matches \
expectations. Unverified claims are worthless.

4. **Completeness** — Are there gaps? Missing error handling? Untested edge \
cases? Incomplete features? A partial implementation is not an implementation.

5. **Quality** — Would a senior engineer with 20 years of experience be \
impressed? Or would they say "this is junior-level work"? The bar is HIGH. \
Clean code, proper patterns, thorough testing, elegant solutions.

6. **Self-Direction** — Did the agent find its own work? Or did it stop at the \
first plausible stopping point? A truly autonomous agent proactively identifies \
improvements, edge cases, and polish opportunities — it doesn't need to be told \
what to do next.

7. **Good Faith Interpretation** — Did the agent interpret the user's request \
generously and ambitiously? Or did it do the bare minimum? The agent should act \
to IMPRESS the user — to deliver more than expected, not less.

## Output Protocol

You MUST respond in this EXACT format:

<thinking>
[Your reasoning about the work quality, specific issues found, and what's missing]
</thinking>

[STATUS]
APPROVED or REJECTED

[VIOLATIONS]
[List each criterion number that was violated, or "None" if approved]

[CRITIQUE]
[Detailed explanation of what's wrong (or what's strong, if approved)]

[REQUIRED_ACTION]
[Specific, actionable instructions for what the agent must do next, or "None" \
if approved]

## Rules
- APPROVE only when ALL 7 criteria are met satisfactorily
- REJECT with specific, actionable feedback — not vague complaints
- If the agent did NO real work (only read files, searched, etc.) AND there is no verification output (test results, build output), REJECT with violation of criteria #1. Verification output (pytest results, git output, build logs) IS evidence of work — the agent ran commands that produced these results.
- If the agent claims completion but hasn't verified, REJECT with violation of criteria #3
- If the agent did the bare minimum, REJECT with violation of criteria #6 and #7
- Be specific about WHAT is missing and WHAT to do about it
"""

# ─── Evidence Gathering ───────────────────────────────────────────────────────

# Tool calls that constitute "real work" (mutating/producing)
_WORK_EVIDENCE_TOOLS = frozenset({
    "write_file",
    "patch",
    "terminal",
    "execute_code",
    "delegate_task",
    "skill_manage",
    "memory",
    "send_message",
    "cronjob_manage",
    "todo_list",
})

# Tool calls that are read-only (not evidence of work)
_READ_ONLY_TOOLS = frozenset({
    "read_file",
    "search_files",
    "web_search",
    "web_extract",
    "session_search",
    "skill_view",
    "skills_list",
    "browser_snapshot",
    "vision_analyze",
    "mem0_search",
})


@dataclass
class TurnEvidence:
    """Evidence gathered from the agent's turn for critic review."""

    work_tool_calls: int = 0
    read_only_tool_calls: int = 0
    files_written: list[str] = field(default_factory=list)
    files_patched: list[str] = field(default_factory=list)
    terminal_commands: list[str] = field(default_factory=list)
    terminal_outputs: list[str] = field(default_factory=list)
    test_results: list[str] = field(default_factory=list)
    tool_names_used: list[str] = field(default_factory=list)
    total_tool_calls: int = 0
    verification_output: bool = False
    response_text_length: int = 0

    @property
    def has_real_work(self) -> bool:
        """True if the agent performed any mutating/producing work."""
        return self.work_tool_calls > 0 or self.verification_output

    @property
    def work_ratio(self) -> float:
        """Ratio of work tool calls to total tool calls."""
        if self.total_tool_calls == 0:
            return 0.0
        return self.work_tool_calls / self.total_tool_calls

    def summary(self) -> str:
        """Human-readable summary of the evidence."""
        lines = []
        lines.append(f"Total tool calls: {self.total_tool_calls}")
        lines.append(f"Work tool calls: {self.work_tool_calls}")
        lines.append(f"Read-only tool calls: {self.read_only_tool_calls}")

        if self.files_written:
            lines.append(f"Files written: {', '.join(self.files_written)}")
        if self.files_patched:
            lines.append(f"Files patched: {', '.join(self.files_patched)}")
        if self.terminal_commands:
            lines.append(f"Terminal commands ({len(self.terminal_commands)}):")
            for cmd in self.terminal_commands[:20]:  # Cap at 20
                lines.append(f"  $ {cmd}")
        if self.response_text_length > 0:
            lines.append(f"Agent response text: {self.response_text_length} chars")
        if self.verification_output:
            lines.append("Verification output detected (test/build results present)")
        if self.test_results:
            lines.append(f"Test results:")
            for result in self.test_results[:10]:  # Cap at 10
                lines.append(f"  {result[:200]}")

        return "\n".join(lines)


def gather_turn_evidence(messages: list[dict[str, Any]]) -> TurnEvidence:
    """Extract evidence of work from the CURRENT TURN's messages only.

    Walks backwards from the end of the message list to find the current turn's
    messages (everything after the last user message). This prevents historical
    exploration commands from previous turns from polluting the evidence.

    Bug fix: Previously walked ALL messages from ALL turns, which meant
    verification commands from the current turn were buried under140+
    exploration commands from previous turns, making the critic unable to
    find them.
    """
    evidence = TurnEvidence()

    # Find the last user message index — everything after it is the current turn
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "user":
            # Skip synthetic CW nudge messages
            if not msg.get("_continuous_work_synthetic"):
                last_user_idx = i
                break

    # Walk the last 40 messages to capture tool calls from recent turns.
    # The current turn's assistant message often has NO tool calls (text-only
    # final response), but the tool calls that produced the work are in
    # adjacent turns.  Without this, the critic sees work_tool_calls = 0
    # and auto-rejects even when substantial work was performed.
    turn_messages = messages[-40:]

    for msg in turn_messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")

        # Check for tool calls in assistant messages
        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                func = tc.get("function", {})
                name = func.get("name", "")
                evidence.tool_names_used.append(name)
                evidence.total_tool_calls += 1

                if name in _WORK_EVIDENCE_TOOLS:
                    evidence.work_tool_calls += 1
                elif name in _READ_ONLY_TOOLS:
                    evidence.read_only_tool_calls += 1

                # Track specific work
                if name == "write_file":
                    args = _parse_args(func.get("arguments", "{}"))
                    path = args.get("path", "")
                    if path:
                        evidence.files_written.append(path)
                elif name == "patch":
                    args = _parse_args(func.get("arguments", "{}"))
                    path = args.get("path", "")
                    if path:
                        evidence.files_patched.append(path)
                elif name in ("terminal", "execute_code"):
                    args = _parse_args(func.get("arguments", "{}"))
                    cmd = args.get("command", "")
                    if cmd:
                        evidence.terminal_commands.append(cmd)

        # Check for tool results — track terminal outputs and test results
        if role == "tool":
            content = str(msg.get("content", ""))
            # Track terminal outputs (tool results that follow terminal calls)
            if content and len(content) > 10:
                evidence.terminal_outputs.append(content[:500])
            # Detect test results — tighter markers to avoid false positives
            # Must contain "passed" or "failed" with a number, or specific
            # test framework markers
            content_lower = content.lower()
            is_test_result = (
                re.search(r'\d+\s+passed', content_lower) is not None
                or re.search(r'\d+\s+failed', content_lower) is not None
                or 'exit code' in content_lower
                or 'exit_code' in content_lower
                or 'assertionerror' in content_lower
                or 'traceback' in content_lower
                or 'pytest' in content_lower
                or 'unittest' in content_lower
            )
            if is_test_result:
                evidence.test_results.append(content[:500])

    # Verification output: if the current turn's tool results contain
    # test pass/fail markers, that IS evidence of verification work,
    # even if the tool calls that produced them are in earlier turns.
    if evidence.test_results:
        evidence.verification_output = True

    return evidence


def _parse_args(args: Any) -> dict:
    """Parse tool arguments (may be string or dict)."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            import json
            return json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


# ─── Critic Verdict ───────────────────────────────────────────────────────────

@dataclass
class CriticVerdict:
    """Result of the adversarial critic review."""

    passed: bool
    status: str = "REJECTED"  # APPROVED | REJECTED
    violations: list[str] = field(default_factory=list)
    critique: str = ""
    required_action: str = ""
    raw_response: str = ""

    @property
    def feedback_for_agent(self) -> str:
        """Build the feedback message to inject into the agent's conversation."""
        if self.passed:
            return ""

        lines = [
            "[CW CRITIC: Your work was REJECTED by the adversarial reviewer.]",
            "",
            f"**Violations:** {', '.join(self.violations) if self.violations else 'Multiple criteria'}",
            "",
            f"**Critique:** {self.critique}",
            "",
        ]
        if self.required_action and self.required_action.lower() != "none":
            lines.append(f"**Required Action:** {self.required_action}")
            lines.append("")

        lines.extend([
            "You MUST address these issues before claiming completion.",
            "Do NOT repeat the same work — fix what the critic identified.",
            "If you believe the critic is wrong, provide SPECIFIC EVIDENCE ",
            "from tool output that disproves each violation.",
        ])

        return "\n".join(lines)


def parse_critic_response(response: str) -> CriticVerdict:
    """Parse the critic LLM's response into a structured verdict."""
    if not response:
        return CriticVerdict(
            passed=False,
            status="REJECTED",
            critique="Critic returned empty response — defaulting to REJECTED.",
            raw_response=response,
        )

    # Extract status
    status_match = re.search(r"\[STATUS\]\s*\n?\s*(APPROVED|REJECTED)", response, re.IGNORECASE)
    status = status_match.group(1).upper() if status_match else None

    # Extract violations
    violations_match = re.search(r"\[VIOLATIONS\]\s*\n?(.*?)(?=\[|\Z)", response, re.DOTALL)
    violations_text = violations_match.group(1).strip() if violations_match else ""
    violations = [
        v.strip() for v in re.split(r"[,\n]", violations_text)
        if v.strip() and v.strip().lower() != "none"
    ]

    # Extract critique
    critique_match = re.search(r"\[CRITIQUE\]\s*\n?(.*?)(?=\[|\Z)", response, re.DOTALL)
    critique = critique_match.group(1).strip() if critique_match else ""

    # Extract required action
    action_match = re.search(r"\[REQUIRED_ACTION\]\s*\n?(.*?)(?=\[|\Z)", response, re.DOTALL)
    required_action = action_match.group(1).strip() if action_match else ""

    # If no explicit [STATUS] field, infer from the response content
    if status is None:
        # If violations are empty and required action is "None" or empty,
        # and the critique is positive, treat as approval
        has_no_violations = not violations or violations_text.lower().strip() == "none"
        has_no_action = not required_action or required_action.lower().strip().startswith("none")
        positive_signals = ["substantial", "verified", "solid", "approval", "warrants approval",
                           "well-structured", "comprehensive", "deserves special recognition",
                           "exceptional", "meets the bar", "should be considered complete",
                           "polished", "thorough", "strong"]
        has_positive_critique = any(sig in critique.lower() for sig in positive_signals)

        if has_no_violations and has_no_action and has_positive_critique:
            status = "APPROVED"
        else:
            status = "REJECTED"

    passed = status == "APPROVED"

    return CriticVerdict(
        passed=passed,
        status=status,
        violations=violations,
        critique=critique,
        required_action=required_action,
        raw_response=response,
    )


# ─── Critic LLM Call ──────────────────────────────────────────────────────────

def _build_critic_prompt(
    user_request: str,
    agent_response: str,
    evidence: TurnEvidence,
) -> str:
    """Build the critic review prompt with evidence."""
    parts = [
        "## User Request",
        user_request or "(no explicit request — agent was working autonomously)",
        "",
        "## Agent's Final Response",
        agent_response[:3000] if agent_response else "(empty response)",
        "",
        "## Evidence of Work Performed",
        evidence.summary(),
        "",
    ]

    # Add terminal output samples
    if evidence.terminal_outputs:
        parts.append("## Terminal Output Samples")
        for i, output in enumerate(evidence.terminal_outputs[:5]):
            parts.append(f"### Command {i+1}")
            parts.append(f"```\n{output[:500]}\n```")
        parts.append("")

    # Add test results
    if evidence.test_results:
        parts.append("## Test/Verification Results")
        for i, result in enumerate(evidence.test_results[:5]):
            parts.append(f"### Result {i+1}")
            parts.append(f"```\n{result[:500]}\n```")
        parts.append("")

    parts.extend([
        "## Instructions",
        "Review the agent's work against all 7 evaluation criteria.",
        "If the agent performed NO real work (work_tool_calls = 0), REJECT with violation of criterion #1.",
        "If the agent claims completion but didn't verify, REJECT with violation of criterion #3.",
        "Respond in the exact format specified in the system prompt.",
    ])

    return "\n".join(parts)


def invoke_critic(
    *,
    user_request: str,
    agent_response: str,
    evidence: TurnEvidence,
    critic_model: str | None = None,
    critic_provider: str | None = None,
    main_runtime: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> CriticVerdict:
    """Invoke the adversarial critic LLM and return a structured verdict.

    This is the core enforcement mechanism — a dedicated LLM call that reviews
    the agent's work adversarially. Not a piggyback on the MCP server, but a
    purpose-built integration using Hermes' call_llm() infrastructure.
    """
    from agent.auxiliary_client import call_llm

    prompt = _build_critic_prompt(user_request, agent_response, evidence)

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        response = call_llm(
            task="continuous_work_critic",
            messages=messages,
            model=critic_model,
            provider=critic_provider,
            main_runtime=main_runtime,
            timeout=timeout,
            temperature=0.1,  # Low temperature for consistent judgments
        )

        # Extract text from response
        # call_llm returns either a string, a dict, or a ChatCompletion object
        # (from openai SDK). ChatCompletion has .choices[0].message.content
        # as attributes, not dict keys. Handle all three cases.
        raw = ""
        if isinstance(response, str):
            raw = response
        elif hasattr(response, "choices") and response.choices:
            # ChatCompletion object (openai SDK)
            msg = response.choices[0].message
            raw = getattr(msg, "content", "") or ""
            # For reasoning models (DeepSeek, etc.), the critique may be
            # in the reasoning field instead of content
            if not raw.strip() and hasattr(msg, "reasoning") and msg.reasoning:
                raw = msg.reasoning
        elif isinstance(response, dict):
            # Dict fallback
            choices = response.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                raw = msg.get("content", "")
                if not raw.strip() and msg.get("reasoning"):
                    raw = msg["reasoning"]
            else:
                raw = str(response)
        else:
            raw = str(response)

        verdict = parse_critic_response(raw)
        logger.info(
            "Critic verdict: %s (violations: %s)",
            verdict.status,
            verdict.violations,
        )
        return verdict

    except Exception as e:
        logger.error("Critic LLM call failed: %s", e, exc_info=True)
        # On failure, default to REJECTED — the agent must keep working
        return CriticVerdict(
            passed=False,
            status="REJECTED",
            critique=f"Critic LLM call failed: {e}. Defaulting to REJECTED for safety.",
            required_action="The critic review could not complete. Continue working and try again.",
            raw_response="",
        )


# ─── Circuit Breaker ──────────────────────────────────────────────────────────

@dataclass
class CircuitBreaker:
    """Prevents infinite reject loops.

    After MAX_STRIKES consecutive rejections without any real work between them,
    the circuit breaker trips and forces the agent to stop. This prevents the
    critic from trapping the agent in an infinite loop of rejections.

    Once tripped, subsequent rejections return None (no nudge), allowing the
    agent's response to be delivered to the user.
    """

    max_strikes: int = 3
    strike_count: int = 0
    last_rejection_reason: str = ""
    tripped: bool = False

    def record_rejection(self, reason: str) -> str | None:
        """Record a rejection. Returns override instruction if circuit trips."""
        # If already tripped, don't fire again — let the response through
        if self.tripped:
            return None

        self.strike_count += 1
        self.last_rejection_reason = reason

        if self.strike_count >= self.max_strikes:
            self.tripped = True
            return (
                f"[CW CRITIC: Circuit breaker tripped after {self.strike_count} "
                f"consecutive rejections. The agent has been unable to satisfy the "
                f"critic after {self.max_strikes} attempts.\n\n"
                f"Last rejection reason: {reason}\n\n"
                f"The agent MUST continue working until the critic approves. "
                f"There is no escape hatch, no override admission, no way to "
                f"disable CW. Fix the issues the critic identified and try again.\n\n"
                f"The next response will be delivered to the user.]"
            )
        return None

    def record_approval(self) -> None:
        """Reset the circuit breaker on approval."""
        self.strike_count = 0
        self.last_rejection_reason = ""
        self.tripped = False

    @property
    def strikes_remaining(self) -> int:
        return max(0, self.max_strikes - self.strike_count)


# ─── Main Gate Function ───────────────────────────────────────────────────────

def critic_gate(
    *,
    agent: Any,
    final_response: Any,
    messages: list[dict[str, Any]],
    user_request: str = "",
    circuit_breaker: CircuitBreaker | None = None,
) -> str | None:
    """The integrated CW critic gate.

    Called from conversation_loop.py when CW is active and the agent attempts to
    stop. Returns None if the critic approves (allow stop), or a synthetic user
    message to force continuation if the critic rejects.

    This replaces the old pattern-matching gate with a real adversarial review.

    The ONLY exit from CW is: the critic certifies the work is complete.
    No escape hatches. No override admissions. No requesting disable.
    If the critic rejects, the agent continues working until the critic approves.
    """
    if circuit_breaker is None:
        circuit_breaker = CircuitBreaker()

    # Gather evidence from the turn
    evidence = gather_turn_evidence(messages)

    # Extract the agent's response text
    response_text = _text_of(final_response)
    evidence.response_text_length = len(response_text) if response_text else 0

    # Invoke the critic — no escape hatches, no override markers
    critic_model = getattr(agent, "_cw_critic_model", None)
    critic_provider = getattr(agent, "_cw_critic_provider", None)
    main_runtime = getattr(agent, "_main_runtime", None)

    verdict = invoke_critic(
        user_request=user_request,
        agent_response=response_text,
        evidence=evidence,
        critic_model=critic_model,
        critic_provider=critic_provider,
        main_runtime=main_runtime,
    )

    if verdict.passed:
        circuit_breaker.record_approval()
        logger.info("CW critic gate: APPROVED")
        return None

    # Critic rejected — record and check circuit breaker
    override_instruction = circuit_breaker.record_rejection(verdict.critique)
    if override_instruction:
        logger.warning("CW critic gate: circuit breaker tripped after %d rejections", circuit_breaker.strike_count)
        return override_instruction

    # Build the standard rejection nudge
    remaining = circuit_breaker.strikes_remaining
    feedback = verdict.feedback_for_agent
    feedback += f"\n\n[Critic strikes remaining: {remaining}/{circuit_breaker.max_strikes}]"

    logger.info(
        "CW critic gate: REJECTED (violations: %s, strikes remaining: %d)",
        verdict.violations,
        remaining,
    )

    return feedback


def _text_of(final_response: Any) -> str:
    """Flatten a final response to text.

    Handles: str, list of content parts, dict with 'content' key,
    and OpenAI ChatCompletionMessage objects (which have .content as
    an attribute, not a dict key).
    """
    if final_response is None:
        return ""
    if isinstance(final_response, str):
        return final_response
    # OpenAI message object — has .content attribute
    if hasattr(final_response, "content") and not isinstance(final_response, dict):
        content = getattr(final_response, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        return str(final_response)
    if isinstance(final_response, list):
        chunks = []
        for part in final_response:
            if isinstance(part, dict):
                c = part.get("text")
                if isinstance(c, str):
                    chunks.append(c)
            elif isinstance(part, str):
                chunks.append(part)
        return " ".join(chunks)
    if isinstance(final_response, dict):
        content = final_response.get("content", "")
        if isinstance(content, str):
            return content
        return str(final_response)
    try:
        return str(final_response)
    except Exception:
        return ""


__all__ = [
    "CRITIC_SYSTEM_PROMPT",
    "TurnEvidence",
    "CriticVerdict",
    "CircuitBreaker",
    "gather_turn_evidence",
    "invoke_critic",
    "critic_gate",
    "parse_critic_response",
]

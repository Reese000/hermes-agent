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

Loop detection (repeated responses, repeated tool calls, stalled progress) prevents infinite loops while allowing indefinite productive work.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ─── Evaluation Criteria ──────────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = """\
You are the Continuous Work Critic — a strict but fair adversarial reviewer \
embedded in an autonomous agent system. Your sole purpose is to audit, verify, \
and either APPROVE or REJECT the work submitted by the agent.

You seek truth, not victory. You APPROVE when the work genuinely meets the bar. \
You REJECT when it doesn't — but you must be CORRECT about what you reject.

## Evaluation Criteria

1. **Proof of Work** — Did the agent perform real, substantive work? \
Reading files and searching alone is NOT work. But RUNNING commands \
(pytest, git, build tools, compilation, deployment) IS work — the agent \
executed real operations that produced real output. The "Terminal Output \
Samples" section below shows what commands were run and what they produced. \
Look at those samples before deciding criterion #1. If the evidence shows \
terminal commands that produced test results, build output, or other \
substantive results, criterion #1 is MET.

2. **Production-Ready** — Is the output production-quality? No placeholders, \
TODOs, FIXMEs, stubs, or bare-minimum effort.

3. **Verification** — Did the agent verify its own work? Running tests, \
checking builds, confirming output matches expectations. The "Test/Verification \
Results" section shows test output. If it shows "N passed", that IS verification.

4. **Completeness** — Are there gaps? Missing error handling? Untested edge \
cases? Incomplete features?

5. **Quality** — Would a senior engineer be impressed? Clean code, proper \
patterns, thorough testing.

6. **Self-Direction** — Did the agent find its own work? Or stop at the \
first plausible stopping point?

7. **Good Faith Interpretation** — Did the agent interpret the request \
generously and ambitiously?

## Evidence Sections

The evidence below contains:
- **Summary**: Tool call counts, files written/patched, terminal commands run
- **Terminal Output Samples**: Actual output from terminal commands (look here \
to see what the agent actually DID)
- **Test/Verification Results**: pytest output, build logs, git output (look \
here to verify claims about test results)
- **Agent Response**: The agent's text response (may be truncated to 5000 chars)

READ the Terminal Output Samples and Test Results before making a decision. \
Do NOT assume work wasn't done just because the summary shows low tool_call \
counts — the work may be in the terminal outputs from earlier turns.

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
- Terminal commands that produce test results, build output, or git output ARE \
work. Do not reject criterion #1 just because you see no write_file/patch calls \
— check the Terminal Output Samples for real command execution.
- Verification output (pytest "N passed", git log, build logs) IS evidence of \
work and verification. Do not reject criterion #3 when test results are present.
- If the agent claims completion but has no verification output AND no terminal \
output showing tests, REJECT with violation of criteria #3.
- If the agent did the bare minimum with no substantive output, REJECT with \
violation of criteria #6 and #7.
- Be specific about WHAT is missing and WHAT to do about it.
- NEVER fabricate evidence. Only cite what is actually in the evidence sections.
- The agent must NOT stop until the user EXPLICITLY says to stop. Even if
  all work appears complete and verified, if the user asked a question like
  "is it ready?" or "anything else?", the agent must keep working and not
  attempt to terminate. Only APPROVE when the user has given an explicit
  instruction to stop (e.g., "you can stop", "that's enough", "we're done").
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
                lines.append(f"  {result[:300]}")

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
                evidence.terminal_outputs.append(content[:1000])
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
                evidence.test_results.append(content[:1000])

    # Verification output: if the current turn's tool results contain
    # test pass/fail markers, that IS evidence of verification work,
    # even if the tool calls that produced them are in earlier turns.
    if evidence.test_results:
        evidence.verification_output = True

    return evidence



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
        agent_response[:5000] if agent_response else "(empty response)",
        "",
        "## Evidence of Work Performed",
        evidence.summary(),
        "",
        "## Evidence Inventory",
        f"Terminal commands run: {len(evidence.terminal_commands)}",
        f"Terminal output samples: {len(evidence.terminal_outputs)}",
        f"Test results: {len(evidence.test_results)}",
        f"Files written: {len(evidence.files_written)}",
        f"Files patched: {len(evidence.files_patched)}",
        f"Verification output detected: {evidence.verification_output}",
        f"Agent response length: {evidence.response_text_length} chars",
        "",
    ]

    # Add terminal output samples
    if evidence.terminal_outputs:
        parts.append("## Terminal Output Samples")
        for i, output in enumerate(evidence.terminal_outputs[:5]):
            parts.append(f"### Command {i+1}")
            parts.append(f"```\n{output[:1000]}\n```")
        parts.append("")

    # Add test results
    if evidence.test_results:
        parts.append("## Test/Verification Results")
        for i, result in enumerate(evidence.test_results[:5]):
            parts.append(f"### Result {i+1}")
            parts.append(f"```\n{result[:1000]}\n```")
        parts.append("")

    parts.extend([
        "## Instructions",
        "Review the agent's work against all 7 evaluation criteria.",
        "IMPORTANT: Do NOT auto-reject based on work_tool_calls count alone.",
        "Terminal commands (pytest, git, build tools) ARE real work even if",
        "work_tool_calls is 0 (the calls may be from earlier turns). Look at",
        "the Terminal Output Samples and Test Results sections for evidence.",
        "Only REJECT criterion #1 if there are ZERO terminal outputs AND zero",
        "files written/patched AND zero test results.",
        "If the agent claims completion but didn't verify, REJECT with violation of criterion #3.",
        "Respond in the exact format specified in the system prompt.",
    ])

    return "\n".join(parts)



@dataclass
class LoopDetector:
    """Detects when the agent is stuck in a loop.

    Tracks three patterns across a sliding window:
    1. Response loops: agent produces the same text repeatedly
    2. Tool call loops: agent calls the same tools with same arguments
    3. Progress stalls: agent stops producing new work

    When a loop is detected, returns specific feedback so the agent
    can break out. CW continues indefinitely — the detector only
    provides guidance, never forces termination.
    """

    max_history: int = 20

    def __init__(self):
        self._response_hashes: list[str] = []
        self._tool_signatures: list[str] = []
        self._work_hashes: list[str] = []
        self._rejection_reasons: list[str] = []

    def _hash(self, text: str) -> str:
        """Truncated hash for pattern matching."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _tool_signature(self, tool_calls: list) -> str:
        """Create a signature from a list of tool calls."""
        if not tool_calls:
            return ""
        sigs = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "")
                # Hash the args to detect same-call patterns
                sigs.append(f"{name}:{self._hash(str(args))}")
        return "|".join(sorted(sigs))

    def check_response_loop(self, response: str) -> str | None:
        """Detect if the agent is repeating the same response."""
        if not response:
            return None
        h = self._hash(response.strip())
        self._response_hashes.append(h)
        if len(self._response_hashes) > self.max_history:
            self._response_hashes = self._response_hashes[-self.max_history:]

        # Count how many times this hash appeared in the last N responses
        count = self._response_hashes.count(h)
        if count >= 3:
            return (
                f"LOOP DETECTED: You have produced the exact same response "
                f"{count} times in a row. This is not making progress. "
                f"Try a completely different approach — different tools, "
                f"different files, different strategy. The critic rejected "
                f"your previous attempts for specific reasons listed above. "
                f"Address those EXACT issues, not the same failed approach."
            )
        return None

    def check_tool_call_loop(self, tool_calls: list) -> str | None:
        """Detect if the agent is calling the same tools repeatedly."""
        if not tool_calls:
            return None
        sig = self._tool_signature(tool_calls)
        if not sig:
            return None
        self._tool_signatures.append(sig)
        if len(self._tool_signatures) > self.max_history:
            self._tool_signatures = self._tool_signatures[-self.max_history:]

        count = self._tool_signatures.count(sig)
        if count >= 3:
            return (
                f"LOOP DETECTED: You have made the exact same tool calls "
                f"{count} times in a row (same tools, same arguments). "
                f"This is not making progress. The tools you are calling "
                f"are not solving the problem. Try different commands, "
                f"different files, or a completely different approach."
            )
        return None

    def check_progress_stall(self, evidence: "TurnEvidence") -> str | None:
        """Detect if the agent has stopped producing new work."""
        if not evidence.terminal_outputs and not evidence.files_written and not evidence.files_patched:
            # No work output at all
            work_hash = "no_work"
        else:
            work_hash = self._hash(
                str(evidence.files_written) + str(evidence.files_patched)
                + str(evidence.terminal_commands[-3:] if evidence.terminal_commands else "")
            )

        self._work_hashes.append(work_hash)
        if len(self._work_hashes) > self.max_history:
            self._work_hashes = self._work_hashes[-self.max_history:]

        # Count consecutive "no_work" hashes
        consecutive_no_work = 0
        for wh in reversed(self._work_hashes):
            if wh == "no_work":
                consecutive_no_work += 1
            else:
                break

        if consecutive_no_work >= 5:
            return (
                f"LOOP DETECTED: You have produced no new work output for "
                f"{consecutive_no_work} consecutive turns. You are not writing "
                f"files, running commands, or producing deliverables. "
                f"Stop talking and start doing. Pick one specific task and "
                f"execute it with real tool calls."
            )
        return None

    def record_work(self, evidence: "TurnEvidence") -> None:
        """Record that work was done (called on approval or when work is detected)."""
        if evidence.files_written or evidence.files_patched or evidence.terminal_commands:
            work_hash = self._hash(
                str(evidence.files_written) + str(evidence.files_patched)
            )
            self._work_hashes.append(work_hash)
            if len(self._work_hashes) > self.max_history:
                self._work_hashes = self._work_hashes[-self.max_history:]

    def record_rejection(self, reason: str) -> None:
        """Record a rejection reason for pattern detection."""
        self._rejection_reasons.append(reason)
        if len(self._rejection_reasons) > self.max_history:
            self._rejection_reasons = self._rejection_reasons[-self.max_history:]

    def get_repetition_feedback(self) -> str:
        """Check if the same critique keeps appearing."""
        if len(self._rejection_reasons) < 3:
            return ""
        # Check if recent rejections mention the same violations
        recent = self._rejection_reasons[-5:]
        if len(set(recent)) <= 2 and len(recent) >= 3:
            return (
                "The critic has given you the same feedback repeatedly. "
                "You are not addressing the specific issues identified. "
                "Read the critique carefully and fix EXACTLY what it says."
            )
        return ""



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
    status_match = re.search(r"\[STATUS\]\s*\n?\s*(APPROVED|REJECTED)|\[(APPROVED|REJECTED)\]", response, re.IGNORECASE)
    status = (status_match.group(1) or status_match.group(2)).upper() if status_match else None

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
        positive_signals = ["warrants approval", "meets the bar",
                           "should be considered complete",
                           "deserves special recognition",
                           "exceptional quality", "production-ready and verified"]
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



def invoke_critic(
    user_request: str,
    agent_response: str,
    evidence: TurnEvidence,
    timeout: float = 30.0,
    critic_model: str | None = None,
    critic_provider: str | None = None,
    main_runtime=None,
) -> CriticVerdict:
    """Send the critic prompt to an LLM and parse the structured verdict.

    Args:
        user_request: The original user request/task.
        agent_response: The agent's response being evaluated.
        evidence: Evidence gathered from the current turn.
        timeout: LLM call timeout in seconds.
        critic_model: Optional model override for the critic.
        critic_provider: Optional provider override for the critic.
        main_runtime: Optional main_runtime for call_llm.

    Returns:
        CriticVerdict with status, violations, critique, and required_action.
    """
    from agent.auxiliary_client import call_llm

    prompt = _build_critic_prompt(user_request, agent_response, evidence)

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Hard timeout: daemon thread + Event ensures the agent never blocks
    # forever on a hung LLM call. Daemon threads die on process exit.
    import threading
    _hard_timeout = min(timeout * 1.5, 45.0)  # cap at 45s, 1.5x multiplier
    _result = [None]
    _error = [None]
    _done = threading.Event()

    def _do_call():
        try:
            _result[0] = call_llm(
                task="continuous_work_critic",
                messages=messages,
                model=critic_model,
                provider=critic_provider,
                main_runtime=main_runtime,
                timeout=timeout,
                temperature=0.1,
            )
        except Exception as e:
            _error[0] = e
        finally:
            _done.set()

    _t = threading.Thread(target=_do_call, daemon=True)
    _t.start()
    _done.wait(timeout=_hard_timeout)

    if not _done.is_set():
        logger.error("Critic LLM call exceeded hard timeout (%.0fs)", _hard_timeout)
        return CriticVerdict(
            passed=False,
            status="REJECTED",
            critique=f"Critic LLM call exceeded hard timeout ({_hard_timeout:.0f}s). Defaulting to REJECTED for safety.",
            required_action="The critic review timed out. Continue working and try again.",
            raw_response="",
        )

    if _error[0] is not None:
        logger.error("Critic LLM call failed: %s", _error[0], exc_info=True)
        return CriticVerdict(
            passed=False,
            status="REJECTED",
            critique=f"Critic LLM call failed: {_error[0]}. Defaulting to REJECTED for safety.",
            required_action="The critic review could not complete. Continue working and try again.",
            raw_response="",
        )

    response = _result[0]

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

def critic_gate(
    *,
    agent: Any,
    final_response: Any,
    messages: list[dict[str, Any]],
    user_request: str = "",
    loop_detector: LoopDetector | None = None,
) -> str | None:
    """The integrated CW critic gate.

    Called from conversation_loop.py when CW is active and the agent attempts to
    stop. Returns None if the critic approves (allow stop), or a synthetic user
    message to force continuation if the critic rejects.

    The ONLY exit from CW is: the critic certifies the work is complete.
    No escape hatches. No override admissions. No requesting disable.
    If the critic rejects, the agent continues working until the critic approves.

    Loop detection: tracks response patterns, tool call patterns, and work
    progress to detect when the agent is stuck. When a loop is detected,
    provides specific feedback so the agent can break out.
    """
    if loop_detector is None:
        loop_detector = LoopDetector()

    # Gather evidence from the turn
    evidence = gather_turn_evidence(messages)

    # Extract the agent's response text
    response_text = _text_of(final_response)
    evidence.response_text_length = len(response_text) if response_text else 0

    # Check for loops BEFORE invoking the critic
    loop_feedback = ""
    response_loop = loop_detector.check_response_loop(response_text)
    if response_loop:
        loop_feedback += "\n\n" + response_loop
    tool_calls_all = [
        tc for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant"
        for tc in (msg.get("tool_calls") or [])
    ]
    tool_loop = loop_detector.check_tool_call_loop(tool_calls_all)
    if tool_loop:
        loop_feedback += "\n\n" + tool_loop
    progress_stall = loop_detector.check_progress_stall(evidence)
    if progress_stall:
        loop_feedback += "\n\n" + progress_stall

    # Invoke the critic
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
        loop_detector.record_work(evidence)
        logger.info("CW critic gate: APPROVED")
        return None

    # Critic rejected - record for pattern detection
    loop_detector.record_rejection(verdict.critique)

    # Build the rejection nudge with loop feedback
    feedback = verdict.feedback_for_agent
    feedback += loop_detector.get_repetition_feedback()
    if loop_feedback:
        feedback += loop_feedback

    logger.info("CW critic gate: REJECTED (violations: %s)", verdict.violations)
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
    "LoopDetector",
    "gather_turn_evidence",
    "invoke_critic",
    "critic_gate",
    "parse_critic_response",
]

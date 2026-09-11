#!/usr/bin/env python3
"""CW System Diagnostic Script.
Run this to verify the CW system is configured correctly.
Usage: python scripts/cw-diagnose.py
"""
import sys
import os
import time

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if detail:
        print(f"         {detail}")
    return condition

print("=" * 60)
print("CW System Diagnostic")
print("=" * 60)

# 1. Check config defaults
print("\n1. Config Defaults")
try:
    sys.path.insert(0, os.getcwd())
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    agent = DEFAULT_CONFIG.get("agent", {})
    cw = agent.get("continuous_work", {})
    
    check("continuous_work_default exists", "continuous_work_default" in agent,
          f"value={agent.get('continuous_work_default')}")
    check("critic model configured", "continuous_work_critic_model" in agent,
          f"value={agent.get('continuous_work_critic_model')}")
    check("critic provider configured", "continuous_work_critic_provider" in agent,
          f"value={agent.get('continuous_work_critic_provider')}")
except Exception as e:
    check("config defaults load", False, str(e))

# 2. Check module imports
print("\n2. Module Imports")
try:
    from agent.continuous_work_critic import (
        invoke_critic, critic_gate, parse_critic_response,
        gather_turn_evidence, LoopDetector, TurnEvidence,
        CRITIC_SYSTEM_PROMPT
    )
    check("continuous_work_critic imports", True)
except ImportError as e:
    check("continuous_work_critic imports", False, str(e))

try:
    from agent.continuous_work_gate import build_continuous_work_nudge
    check("continuous_work_gate imports", True)
except ImportError as e:
    check("continuous_work_gate imports", False, str(e))

# 3. Check parser handles thinking tags
print("\n3. Parser: Thinking Tag Support")
from agent.continuous_work_critic import parse_critic_response

r = parse_critic_response("<thinking>\n[STATUS]\nAPPROVED\n\n[VIOLATIONS]\nNone\n\n[CRITIQUE]\nGood.\n\n[REQUIRED_ACTION]\nNone\n</thinking>")
check("Strips thinking tags (inside)", r.status == "APPROVED" and "</thinking>" not in r.critique)

r2 = parse_critic_response("<thinking>\nAnalysis\n</thinking>\n\n[STATUS]\nAPPROVED\n\n[VIOLATIONS]\nNone\n\n[CRITIQUE]\nGood.\n\n[REQUIRED_ACTION]\nNone")
check("Strips thinking tags (outside)", r2.status == "APPROVED")

r3 = parse_critic_response("[STATUS]\nAPPROVED\n\n[VIOLATIONS]\nNone\n\n[CRITIQUE]\nGood.\n\n[REQUIRED_ACTION]\nNone")
check("Works without thinking tags", r3.status == "APPROVED")

# 4. Check parser handles both status formats
print("\n4. Parser: Status Format Support")
r4 = parse_critic_response("[STATUS]\nAPPROVED\n\n[VIOLATIONS]\nNone\n\n[CRITIQUE]\nGood.\n\n[REQUIRED_ACTION]\nNone")
check("[STATUS] format", r4.status == "APPROVED")

r5 = parse_critic_response("[APPROVED]\n\n[VIOLATIONS]\nNone\n\n[CRITIQUE]\nGood.\n\n[REQUIRED_ACTION]\nNone")
check("[APPROVED] format", r5.status == "APPROVED")

r6 = parse_critic_response("[REJECTED]\n\n[VIOLATIONS]\n1\n\n[CRITIQUE]\nBad.\n\n[REQUIRED_ACTION]\nFix")
check("[REJECTED] format", r6.status == "REJECTED")

# 5. Check timeout configuration
print("\n5. Timeout Configuration")
import inspect
sig = inspect.signature(invoke_critic)
check("Default timeout is 30s", sig.parameters['timeout'].default == 30.0,
      f"actual={sig.parameters['timeout'].default}")

src = inspect.getsource(invoke_critic)
check("Hard timeout formula: max(1.0, min(timeout * 2, 90.0))",
      "min(timeout * 2, 90.0)" in src or "min(timeout*2, 90.0)" in src)
check("Per-attempt timeout: min(timeout, 10.0)",
      "min(timeout, 10.0)" in src or "min(timeout,10.0)" in src)
check("max_tokens=1024", "max_tokens=1024" in src)

# 6. Check LoopDetector
print("\n6. LoopDetector")
ld = LoopDetector()
check("LoopDetector instantiates", True)
check("LoopDetector has check_response_loop", hasattr(ld, 'check_response_loop'))
check("LoopDetector has record_work", hasattr(ld, 'record_work'))

# 7. Check fallback gate
print("\n7. Fallback Gate")
nudge = build_continuous_work_nudge(final_response="all done", work_evidence_tools=0)
check("Fallback gate returns nudge for bare completion", nudge is not None)
nudge2 = build_continuous_work_nudge(final_response="all done", work_evidence_tools=1)
check("Fallback gate approves when work was done", nudge2 is None)

# 8. Check timeout counter fallback
print("\n8. Timeout Counter Fallback (conversation_loop.py)")
with open("agent/conversation_loop.py", "r") as f:
    cl_content = f.read()
check("_cw_critic_timeout_count in conversation_loop.py",
      "_cw_critic_timeout_count" in cl_content)
check("Timeout counter threshold (>= 3)", "timeout_count >= 3" in cl_content)

# Summary
print("\n" + "=" * 60)
print("Diagnostic complete.")
print("=" * 60)

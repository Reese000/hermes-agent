# Hermes Agent — Configuration & System Prompt Repository

**Purpose:** This repo holds the personality definition (`SOUL.md`), system prompt (`system_prompt.md`), operational launchers, and accumulated operational wisdom for a deployed **Hermes Agent** instance (by [Nous Research](https://hermes-agent.nousresearch.com)). It is a **configuration and documentation project** — no runtime Python code, no build step.

The deployed Hermes backend lives at `AppData\Local\hermes\hermes-agent\venv`. This source repo defines *how that agent behaves*, not the agent binary itself.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Agent runtime** | [Hermes Agent](https://hermes-agent.nousresearch.com) (by Nous Research) |
| **LLM backend** | DeepSeek via [OpenRouter](https://openrouter.ai) (provider-locked: `deepseek/deepseek-v4-flash:DeepSeek`) |
| **Host OS** | Windows 11 — WSL2 runs the agent; Windows is the target platform |
| **Launcher** | `run-hermes.bat` — batch script for CLI / web dashboard / Discord gateway |
| **Sub-agents / MCP** | Critic MCP server (independent repository), Foundry Manager MCP (Orchestration) |
| **Versioning** | Git → GitHub (`Reese000/Hermes-Agent`) |

---

## Key Files

| File | Purpose |
|------|---------|
| `SOUL.md` | Agent personality definition — Three-Gate Workflow (Debate → Execute → Critique), visual oracle requirements, autonomy rules |
| `system_prompt.md` | Full system prompt (~37 KB) — instruction set the agent follows on every turn |
| `system_prompt_dump.md` | Exported/dumped snapshot of the system prompt (~40 KB) |
| `.hermes.md` | Hard provider-routing rules — enforces DeepSeek-only backend, blocks costlier routes |
| `run-hermes.bat` | Windows batch launcher — CLI mode, web dashboard (port 8080), Discord gateway |
| `lessons_learned.md` | Append-only bug journal — root causes, fixes, prevention for every operational issue |
| `CLAUDE.md` | Guidance for Claude Code when editing this repo |
| `.githooks/pre-commit` | Shared pre-commit hook — blocks `.env` files, `__pycache__`, secrets, large files |

---

## Setup & Run

### Prerequisites

- Hermes Agent installed (see [official docs](https://hermes-agent.nousresearch.com/docs))
- WSL2 with the `hermes` CLI on PATH (the agent runs inside WSL)
- OpenRouter API key configured in the deployed Hermes config (`~/.hermes/config.yaml`)

### Launch

```batch
run-hermes.bat cli            # Interactive CLI session
run-hermes.bat web            # Web dashboard at localhost:8080
run-hermes.bat discord        # Start Discord gateway
run-hermes.bat discord-stop   # Stop Discord gateway
run-hermes.bat status         # Check gateway status
run-hermes.bat setup          # Gateway setup wizard
run-hermes.bat help           # Show all commands
```

The batch script delegates to `wsl hermes <command>`. The agent runs on DeepSeek V4 Flash via OpenRouter with hard provider routing.

---

## Architecture

### Three-Gate Workflow

Every non-trivial task passes through three mandatory gates:

1. **Gate 1 — Strategic Debate** (`mcp_critic_agent_debate`): debate approaches before coding
2. **Gate 2 — Parallel Execution** (`delegate_task` / MACO Swarm): dispatch through an orchestration pyramid (managers → coders)
3. **Gate 3 — Quality Critique** (`mcp_critic_get_critique`): independent audit with deterministic oracle verification for visual/geometric work

### Visual & Geometric Oracle Requirement

This is a defining architectural constraint for this agent: **LLM judgment is ~0% precise** on sub-pixel rendering, overlap, toolpath gouging, nesting density, and similar visual/geometric properties. Any deliverable involving UI, SVG, CAM toolpaths, 3D scenes, or nesting layouts requires a **deterministic oracle** (pixel-diff, CSG boolean, CP-SAT solver) plus an independent auditor agent. Self-reports of visual correctness are treated as violations.

### Provider Lock-In

Cost discipline — the agent exclusively routes through DeepSeek's backend on OpenRouter (`provider_routing.only: [DeepSeek]`). Higher-cost backends (GMICloud, StreamLake, DigitalOcean) are blocked at the config level. Auxiliary services (web extraction, compression, session search, etc.) also have their own `provider.only` enforcement.

### Execution Pyramid

- **CEO** (this agent) receives requests, plans, and delegates
- **Orchestrators** (sub-agents with `role='orchestrator'`) decompose segments
- **Coders** (leaf sub-agents) produce work with `reasoning.md` before and `done.md` after
- **Auditor agents** independently verify visual/geometric output
- **Deterministic oracles** provide the only trusted verdict on rendering correctness

---

## Current Status

| Aspect | Status |
|--------|--------|
| Agent runtime | ✅ **Deployed and running** (live python.exe processes) |
| Provider routing | ✅ DeepSeek-only enforced at config + auxiliary levels |
| Three-Gate workflow | ✅ Operational — debate, execute, critique all working |
| Discord gateway | ✅ Operational with systemd-based management |
| Web dashboard | ✅ Operational on port 8080 |
| Visual oracle verification | 🔄 Partial — CAM CSG and Nesting CP-SAT oracles exist; pixel-diff oracle is future |
| Self-evolution experiment | ⚪ Experimental — DSPy-based skill evolution, limited results |
| Proxy streaming | ✅ Fixed — full SSE tool_calls-to-Anthropic content block conversion |
| Pre-commit hook | ✅ Installed but needs `git config core.hooksPath .githooks` |

---

## Contributing / Editing

- Edit `SOUL.md` for personality/behavior changes
- Edit `system_prompt.md` for instruction changes
- Append to `lessons_learned.md` after every bug fix or failed approach (dated entry with Problem/Cause/Solution/Prevention)
- Never commit `.env` files or API keys
- Run `git config core.hooksPath .githooks` to enable the pre-commit hook locally

---

## License

Private — Reese CNC Machine Shop LLC. All rights reserved.

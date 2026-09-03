## 2026-05-31 — Auxiliary OpenRouter Calls Bypass provider_routing

- **Problem**: Auxiliary services (web_extract, compression, session_search, etc.) occasionally routed through Baidu Qianfan instead of DeepSeek, despite `provider_routing.only: [DeepSeek]` being set in config.yaml.
- **Cause**: The main agent and delegation subagents pass `provider: { only: [DeepSeek] }` via `extra_body` in API calls, but auxiliary services use a separate OpenRouter client (`auxiliary_client.py:_try_openrouter()`) that does NOT include any provider routing. Auxiliary `extra_body` was `{}` for all tasks.
- **Solution**: Added `extra_body: { provider: { only: [DeepSeek] } }` to all 9 OpenRouter auxiliary services (web_extract, compression, session_search, skills_hub, approval, mcp, title_generation, triage_specifier, curator) in `~/.hermes/config.yaml`. Vision auxiliary was left unchanged because it uses `google/gemini-3-flash-preview` which can't route through DeepSeek.
- **Prevention**: When configuring provider routing, check ALL code paths that make OpenRouter API calls — not just the main agent. Auxiliary services, MCP servers (if they call OpenRouter independently), and any background processes each need their own `provider.only` enforcement. The `provider_routing.only` config only covers the main agent and its delegation subagents.

## 2026-05-31 — Discord Photo Delivery Fails for Cache Files

- **Problem**: Discord photos only worked ~25% of the time. Failed files showed "Skipping unsafe MEDIA directive path" in gateway logs, followed by "Cannot send an empty message" (400) when the stripped text was empty.
- **Cause**: `MEDIA_DELIVERY_SAFE_ROOTS` included `/root/.hermes/cache/screenshots/`, `/root/.hermes/cache/images/`, etc., but NOT `/root/.hermes/cache/` itself. When an agent saved a file directly to `/root/.hermes/cache/` (not under a subdirectory) and output `MEDIA:/root/.hermes/cache/file.png`, `validate_media_delivery_path` rejected it because `/root` is in `_MEDIA_DELIVERY_DENIED_PREFIXES` and the path didn't match any safe root.
- **Solution**: Added `_HERMES_HOME / "cache"` to `MEDIA_DELIVERY_SAFE_ROOTS` in `/usr/local/lib/hermes-agent/gateway/platforms/base.py`. This allows any file under `/root/.hermes/cache/` to be delivered as a MEDIA attachment.
- **Prevention**: When adding MEDIA: delivery support, ensure all plausible agent-writable paths are in `MEDIA_DELIVERY_SAFE_ROOTS`. The safe roots check runs BEFORE the denylist check, so paths in safe roots bypass the `/root` denylist.

## 2026-06-28 — Proxy Streaming Drops Tool Calls; --bare Strips Sub-Agent Tools

- **Problem**: Claude Code's native sub-agent tools (Read, Search, Explore, sub-agent spawning panel) stopped working after proxy configuration changes. The model returned tool_calls but Claude Code never received them, and sub-agents only had Read/Edit/Bash.
- **Cause (1) — Streaming tool_cals silently dropped**: The proxy's streaming SSE handler only read `c.delta?.content` (text) from the OpenAI response. When the model returned `c.delta?.tool_calls` (which MiMo 2.5 does for Read/Search/Explore), those deltas were completely ignored. No Anthropic-format `content_block_start`/`content_block_delta`/`content_block_stop` events were emitted for tool_use blocks. Claude Code never saw the tool call responses.
- **Cause (2) — --bare flag disables sub-agent system**: The `--bare` flag sets `CLAUDE_CODE_SIMPLE=1`, which strips Claude Code to Read/Edit/Bash only — no sub-agent spawning tool, no advisor system, no sub-agent UI panel. It was used to prevent OAuth credentials from bypassing the proxy, but `--bare` is a sledgehammer that removes sub-agent tools entirely.
- **Cause (3) — advisorModel removal**: Removing `"advisorModel": "opus"` from settings.json tells Claude Code to disable the advisor/sub-agent system entirely. This setting controls WHETHER sub-agents exist, not just which model runs them.
- **Solution**:
  1. **Fixed streaming tool_calls conversion**: Added a state machine in the proxy's streaming section that tracks OpenAI SSE `delta.tool_calls` by `tc.index`, emits Anthropic `content_block_start` (tool_use), `content_block_delta` (input_json_delta), and `content_block_stop` events. Added a `finished` flag to deduplicate the two finish_reason chunks OpenRouter sends (first with `usage:null`, second with real usage).
  2. **Removed --bare, deleted OAuth credentials**: Instead of using `--bare` to force proxy routing, simply removed `~/.claude/.credentials.json` (backed up). Without credentials, Claude Code falls through to `ANTHROPIC_API_KEY` from env → routes through the proxy naturally. Launch command changed to `claude --dangerously-skip-permissions` (no `--bare`).
  3. **Restored advisorModel**: Re-added `"advisorModel": "opus"` to settings.json. Added `claude-opus-4-8` to the proxy's MODEL_MAP so sub-agent requests (which Claude Code hardcodes as `claude-opus-4-8`) are explicitly mapped to the target model.
  4. **Unique tool_call_id fallback**: Changed from hardcoded `'call_missing'` to `id = () => 'call_' + Math.random().toString(36).slice(2, 10)` — prevents ID mismatch rejections.
  5. **Pass all tool types**: Changed tool filter from `body.tools.filter(t => t.type === 'custom' || !t.type)` to `body.tools.map(...)` — passes all tools regardless of type.
- **Prevention**:
  - Never use `--bare` as a proxy-routing workaround. Delete OAuth credentials instead.
  - Never remove `advisorModel` from settings.json — it disables the sub-agent system.
  - The proxy's streaming path must implement full SSE tool_calls → Anthropic content block conversion. The non-streaming path alone is insufficient because Claude Code uses streaming by default.
  - Always add `claude-opus-4-8` to the MODEL_MAP when routing through a proxy — Claude Code hardcodes this for sub-agents regardless of env vars.
  - OpenRouter can send two finish_reason chunks per response. Any proxy must deduplicate with a `finished` flag.
  - After changing the proxy, verify with a streaming test: `STRM OK text=false tools=1` confirms tool calls are correctly streamed. If `tools=0` on a tool-using request, the streaming path is broken.
- **Proxy log signature of working state**: `STRM OK text=false tools=1` (pure tool call, no text). Broken state: `STRM OK text=true tools=0` (model returned tool calls but they were dropped).
- **Skill updated**: claude-code-proxy-setup v1.4.0

- **Problem**: Running `run-hermes.bat` (or executing `wsl hermes`) crashed with `ModuleNotFoundError: No module named 'dotenv'`.
- **Cause**: A global installation step had placed wrapper scripts in `/usr/local/bin/` (`hermes`, `hermes-acp`, `hermes-agent`) using the system python interpreter (`#!/usr/bin/python3`). The system python lacked the project's dependencies, which are isolated inside the virtual environment `/usr/local/lib/hermes-agent/venv`. Since `/usr/local/bin` takes precedence in the `PATH` over `$HOME/.local/bin`, the system ran the broken global files.
- **Solution**: Replaced `/usr/local/bin/hermes`, `/usr/local/bin/hermes-acp`, and `/usr/local/bin/hermes-agent` with symbolic links pointing to their working counterparts in the virtual environment (`/usr/local/lib/hermes-agent/venv/bin/...`).
- **Prevention**: Avoid installing wrapper scripts directly using the system Python interpreter when a project manages dependencies inside a virtual environment. Always symlink the venv-specific executables to global bin paths or configure shell aliases/paths properly.

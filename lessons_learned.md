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

## 2026-06-22 — Global hermes commands crash with ModuleNotFoundError: No module named 'dotenv'

- **Problem**: Running `run-hermes.bat` (or executing `wsl hermes`) crashed with `ModuleNotFoundError: No module named 'dotenv'`.
- **Cause**: A global installation step had placed wrapper scripts in `/usr/local/bin/` (`hermes`, `hermes-acp`, `hermes-agent`) using the system python interpreter (`#!/usr/bin/python3`). The system python lacked the project's dependencies, which are isolated inside the virtual environment `/usr/local/lib/hermes-agent/venv`. Since `/usr/local/bin` takes precedence in the `PATH` over `$HOME/.local/bin`, the system ran the broken global files.
- **Solution**: Replaced `/usr/local/bin/hermes`, `/usr/local/bin/hermes-acp`, and `/usr/local/bin/hermes-agent` with symbolic links pointing to their working counterparts in the virtual environment (`/usr/local/lib/hermes-agent/venv/bin/...`).
- **Prevention**: Avoid installing wrapper scripts directly using the system Python interpreter when a project manages dependencies inside a virtual environment. Always symlink the venv-specific executables to global bin paths or configure shell aliases/paths properly.

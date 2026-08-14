# Enhance Feature — Implementation Record
**Date:** 2026-08-14
**Branch:** main (pushed to fork)

---

## Current State Summary

### ✅ Completed & Working
1. **Enhance model selector** — Uses same `ModelCatalogMenu` as main composer (shared code, not duplicate)
2. **Context menu** — Right-click enhance button: Hide toggle, Model selector (hover-to-open), Profile selector
3. **Tooltip** — Two-line: "Enhance prompt — Model · Reasoning" / "right-click to configure" (small italic faded)
4. **Enhance settings store** — Persists model, reasoning level, profile across sessions
5. **Cancel/restore** — Cancel button during generation, restore original text on cancel
6. **Loading indicator** — Sparkles → spinner during generation
7. **IPC timeout** — Set to 5 minutes (was 15s default)
8. **Token rate fix** — Uses output-only tokens (was inflated by cache tokens)
9. **Usage indicator toggle** — Configurable enable/disable in Edit Models dialog
10. **i18n** — Enhance strings in en, ja, zh, zh-hant, ar

### ⚠️ Backend Changes (require gateway restart)
1. **`prompt_enhance` and `subagent` added to `_AUX_TASK_SLOTS`** — Was causing 400 errors when ModelCatalogMenu tried to register models
2. **Streaming enhance endpoint** — `POST /api/model/enhance-prompt-stream` with SSE
3. **Context gathering in threadpool** — Non-blocking so "thinking" event reaches client immediately
4. **Compact system prompt** — Reduced from ~600 tokens to ~100 tokens for faster TTFT

### 🔧 Streaming Architecture
- **Backend:** SSE endpoint → `call_llm(stream=True)` → chunks via `run_in_threadpool`
- **IPC:** `hermes:api-stream` handler → streams HTTP response → sends chunks via `webContents.send()`
- **Preload:** `apiStream` method → registers chunk/done listeners → returns dispose function
- **Frontend:** `enhancePromptStream()` async generator → yields text chunks
- **Composer:** `requestAnimationFrame` throttle → updates DOM at ~30fps for visible streaming

### 📋 Remaining Issues
- Backend needs restart for Python changes (subagent slot, streaming endpoint, compact prompt)
- Streaming latency still depends on LLM time-to-first-token (model-dependent)

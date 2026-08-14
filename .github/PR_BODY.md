## What does this PR do?

Four features for the Hermes desktop app model system, built incrementally and QA'd in production:

### 1. Live OpenRouter model search
- **`model-picker.tsx`**: Typing in the model search box queries OpenRouter's live catalog (debounced 300ms), deduplicates against the static curated list, and shows "Live Search · openrouter · N results" below curated models.
- **`model-catalog-menu.tsx`** (chat-bar picker): Same live search wired into the Radix DropdownMenu picker. Includes `useLayoutEffect` focus preservation so the search input doesn't lose focus when Radix items mount.
- Error surface: loading skeleton → live results OR "Live search failed" error box with the actual error message.

### 2. Subagent model selection
- Backend: `subagent` is exposed as a virtual aux task in `web_server.py`, handled as a special case in `GET /api/model/auxiliary` / `POST /api/model/set` — not added to `_AUX_TASK_SLOTS`; it reads/writes the pre-existing `cfg["delegation"]` namespace (the same one `delegate_task`'s credential resolution already consumed before this PR), not `cfg["auxiliary"]`.
- Frontend: `model-settings.tsx` adds a `subagent` row to its generic aux-task list; `searchable-select.tsx` has no subagent-specific code — it's the generic live-search dropdown from feature 1, reused here.
- Confirmed wired end-to-end: `tools/delegate_tool.py` actually reads the saved model at delegation time (sync and background dispatch both), so this isn't settings-UI-only. Reuses the pre-existing generic `AuxiliaryTaskAssignment` type — no dedicated `SubagentAuxConfig` type exists.

### 3. Session usage / cost indicator
- Backend: `GET /api/sessions/{session_id}/usage` endpoint in `web_routers/sessions.py` — returns aggregated token counts, cost estimates, and TPS.
- Frontend: `UsageIndicator` component in the composer footer. Glance line shows a rolling cost/hour and tokens/sec, both derived from a time-adaptive EWMA over consecutive `/usage` polls (`session-usage.ts`) rather than cumulative-cost ÷ session-elapsed-time — the latter dilutes toward zero for any session with real idle time (a session active ~20 minutes over a 61-hour span reported a rate ~180x below its real cost while running). tokens/sec includes cache read/write tokens so it stays reconcilable with cost/hour, which bills them too; on a heavily-cached session this can read in the thousands during a burst of context replay, shown with compact notation (`16.5K`). Hardened against reopening old/dormant chats: a poll arriving more than 60s after the last one (`MAX_SAMPLE_GAP_S`) reseeds instead of diffing against a stale baseline, and non-finite inputs are rejected at both the store and the display layer. Tooltip adds cache hit rate, reasoning tokens, cost/hour, and tokens/sec, and formats elapsed time as a duration (`6h 51m`) instead of raw seconds. Fixed a `box-decoration-clone` layout bug in the shared `TooltipContent` that corrupted the 10-row detail panel; the tooltip now builds directly on Radix primitives.

### 4. Context-aware prompt enhancement
- Backend: `POST /api/model/enhance-prompt` endpoint with:
  - Deep conversation history (last 25 messages via `get_messages_as_conversation`)
  - Partial session system prompt injection (first 4000 chars — not the full multi-page prompt — of memory/project rules/context)
  - Project guidelines auto-detection (AGENTS.md/CLAUDE.md/.cursorrules), searched in order: the session's git_root/cwd, then a hardcoded `~/Projects`, then the web server process's own cwd — the latter two aren't scoped to the session and can pull in an unrelated project's guidelines file
  - Style mirroring, correction enforcement, and anti-drift system prompt
  - Empty-prompt generation (no text → synthesizes next step from history)
  - Rate limiting (3s min interval), input size validation, `ok`/`error` error contract
  - Respects user's configured `auxiliary.prompt_enhance` model from settings; defaults to `mimo-v2.5` when unconfigured to avoid falling back to slow reasoning models
- Frontend: Enhance button (Sparkles icon) in composer controls. Works with empty composer (generates from context). Error notification on failure. Ctrl+Z undo support (does not cover the empty-composer/generate-from-context path — no pre-enhance snapshot is recorded when the composer starts empty).
- Eval harness: `scripts/eval_enhance.py` — creates real session contexts, calls the enhance endpoint, scores output on 5 axes (correction memory, precision, style fidelity, specificity, context awareness) via rule-based/regex scoring, not LLM grading. Honest ~19/25 average with the corrected scorer (cache forced off via `HERMES_OPENROUTER_CACHE=0`; scorer fixes from the audit pass) — a narrative figure from manual runs; the script only prints results, nothing is checked in or CI-wired to reproduce this number. Iterated the system prompt to a refined-in-the-user's-voice model: terse imperative output, no filler, no clarifying questions, conversation-first grounding. Realistic scenarios score 18-24/25; the one low case is an adversarial 2-message sparse-context stress test, not a real-feature regression.

### 5. i18n
- All 5 locales (ar/en/ja/zh/zh-hant) have the `enhance`/`enhancing`/`enhanceFailed` composer keys and `prompt_enhance` aux task label.

## Verification

- [x] 52/52 pytest (`bash scripts/run_tests.sh tests/test_model_search_and_subagent.py -q`)
- [x] ESLint 0 errors, 0 warnings (all modified .tsx/.ts files)
- [x] `npx tsc --noEmit` 0 errors
- [x] i18n keys present in all 5 locales (including `usage{}`, previously missing from `ar.ts`)
- [x] No build artifacts committed (`.gitignore` updated)
- [x] Backend uses `env -u PYTHONPATH` for all test invocations
- [x] Live-tested in production asar: enhance button, empty-prompt generation, model respect, focus preservation
- [x] Live-tested in production asar: session usage / cost indicator — confirmed working end-to-end after fixing
      a runtime-vs-stored session id mismatch (`composer/index.tsx` polled `/usage` with the RUNTIME session id;
      the endpoint and `$sessionUsageBySession` are keyed by STORED id, so any resumed/persisted session 404'd
      silently). Metrics were subsequently redesigned again (rolling cost/hour + tokens/sec EWMA, see above) and
      hardened against a reopened-old-chat stale-baseline bug; verified via user-provided screenshots across
      three deploy iterations, not an automated visual test.

## Files changed
```
apps/desktop/src/app/chat/composer/controls.tsx         # enhance button
apps/desktop/src/app/chat/composer/index.tsx            # handleEnhance + UsageIndicator
apps/desktop/src/app/chat/composer/usage-indicator.tsx  # cost/TPS display
apps/desktop/src/app/settings/model-settings.tsx        # subagent aux row
apps/desktop/src/app/settings/searchable-select.tsx     # live search in select
apps/desktop/src/app/shell/model-catalog-menu.tsx       # chat-bar live search + focus fix
apps/desktop/src/components/model-picker.tsx            # model picker live search + error
apps/desktop/src/hermes.ts                              # enhancePrompt + searchProviderModels
apps/desktop/src/i18n/{ar,en,ja,zh,zh-hant}.ts         # enhance i18n keys
apps/desktop/src/i18n/types.ts                          # composer type additions
apps/desktop/src/lib/icons.ts                           # Sparkles icon
apps/desktop/src/store/session-usage.ts                 # session usage store
apps/desktop/src/store/session-usage.test.ts            # usage store tests
apps/desktop/src/types/hermes.ts                        # EnhancePromptResponse + subagent types
hermes_cli/web_routers/sessions.py                      # usage endpoint
hermes_cli/web_server.py                                # enhance endpoint + search endpoint
scripts/eval_enhance.py                                 # enhance eval harness
tests/test_model_search_and_subagent.py                 # 52 backend tests
.gitignore                                              # build artifact exclusion
```

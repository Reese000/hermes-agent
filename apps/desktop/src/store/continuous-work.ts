import { atom } from 'nanostores'

import { setContinuousWork } from '@/api/config'
import { persistBoolean, storedBoolean } from '@/lib/storage'

// ── Continuous Work settings store ─────────────────────────────────────────
// Persists the continuous-work feature's on/off state to localStorage for
// instant UI feedback, and mirrors the setting to the backend config
// (agent.continuous_work) so the next agent build applies it.
//
// When enabled, the agent is instructed via system prompt to keep working
// until all tasks are genuinely complete or perfection is certified. The
// agent must explicitly state when overriding the user's instructions to
// terminate. Anti-loop/anti-time-waste guards use the existing
// tool_loop_guardrails system (thresholds tightened when active).
//
// Consumed by:
//   - controls.tsx: toggle button in composer row
//   - web_server.py: GET/POST /api/agent/continuous-work
//   - system_prompt.py: CONTINUOUS_WORK_GUIDANCE suffix when active
// ───────────────────────────────────────────────────────────────────────────
const ENABLED_KEY = 'hermes.desktop.continuous-work'

/** Whether continuous work mode is active. */
export const $continuousWork = atom(storedBoolean(ENABLED_KEY, false))

$continuousWork.subscribe(enabled => persistBoolean(ENABLED_KEY, enabled))

/** Get the current continuous work state. */
export function isContinuousWorkEnabled(): boolean {
  return $continuousWork.get()
}

/**
 * Flip the toggle. The local atom updates immediately (instant UI) and the
 * backend config is written best-effort — a failed write (backend down, older
 * gateway without the endpoint) leaves the local state authoritative for this
 * session, matching the wake-word toggle's local-first posture.
 */
export async function toggleContinuousWork(): Promise<void> {
  const next = !$continuousWork.get()
  $continuousWork.set(next)

  try {
    await setContinuousWork(next)
  } catch {
    // Best-effort: keep the local toggle as the source of truth this session.
  }
}

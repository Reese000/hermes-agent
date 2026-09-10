import { atom } from 'nanostores'

// ── Continuous Work settings store ─────────────────────────────────────────
// Per-CONVERSATION continuous-work state. Each chat has its own flag, so
// enabling it in one conversation never leaks into another.
//
// The flag is keyed by the runtime session id (the same id the submit path
// uses). Runtime ids are ephemeral (they change across app restarts), so this
// is an in-memory map rather than localStorage — a session's continuous-work
// state lives exactly as long as the session does.
//
// Consumed by:
//   - continuous-work-statusbar.tsx: toggle reads/writes the ACTIVE session
//   - submit.ts: sends `continuous_work` with each prompt.submit so the
//     backend can inject the guidance for that turn
//   - tui_gateway/server.py: `_continuous_work_note(session)` reads the flag
//     the submit carried and prepends the guidance to that turn's model input
// ───────────────────────────────────────────────────────────────────────────

/** Per-session continuous-work flags, keyed by runtime session id. */
export const $continuousWorkBySession = atom<Record<string, boolean>>({})

/** Read the continuous-work flag for a session (absent = off). */
export function continuousWorkForSession(sessionId: string | null | undefined): boolean {
  if (!sessionId) {
    return false
  }
  return $continuousWorkBySession.get()[sessionId] ?? false
}

/** Set the continuous-work flag for a session. */
export function setContinuousWorkForSession(sessionId: string | null | undefined, enabled: boolean): void {
  if (!sessionId) {
    return
  }
  const next = { ...$continuousWorkBySession.get() }
  if (enabled) {
    next[sessionId] = true
  } else {
    delete next[sessionId]
  }
  $continuousWorkBySession.set(next)
}

/** Toggle the continuous-work flag for a session, returning the new value. */
export function toggleContinuousWorkForSession(sessionId: string | null | undefined): boolean {
  const next = !continuousWorkForSession(sessionId)
  setContinuousWorkForSession(sessionId, next)
  return next
}

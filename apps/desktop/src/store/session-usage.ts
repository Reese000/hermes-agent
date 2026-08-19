import { atom } from 'nanostores'

import { getSessionUsage } from '@/hermes'
import type { SessionUsageResponse } from '@/types/hermes'

export interface SessionUsageData {
  actual_cost_usd: number
  api_call_count: number
  cache_read_tokens: number
  cache_write_tokens: number
  elapsed_seconds: number
  estimated_cost_usd: number
  input_tokens: number
  output_tokens: number
  reasoning_tokens: number
  tokens_per_second: number
}

/** A session's "current pace" — cost/token throughput derived from recent
 *  activity only. See `updateSmoothedRate` for why this exists instead of
 *  `estimated_cost_usd / elapsed_seconds`: elapsed_seconds is wall-clock
 *  since session START, so a session that ran for 61 hours but was only
 *  actively worked for 20 minutes reports a rate diluted ~180x toward zero
 *  — technically an average, but not an answer to "what does this cost me
 *  to run." */
export interface SmoothedRate {
  tokensPerSecond: number
  costPerHour: number
}

/** Per-session usage data, keyed by stored session id. */
export const $sessionUsageBySession = atom<Record<string, SessionUsageData>>({})

/** Per-session current-pace rate, keyed by stored session id. Absent until a
 *  session has produced two polls with real elapsed time between them. */
export const $smoothedRateBySession = atom<Record<string, SmoothedRate>>({})

const EMPTY_USAGE: SessionUsageData = {
  actual_cost_usd: 0,
  api_call_count: 0,
  cache_read_tokens: 0,
  cache_write_tokens: 0,
  elapsed_seconds: 0,
  estimated_cost_usd: 0,
  input_tokens: 0,
  output_tokens: 0,
  reasoning_tokens: 0,
  tokens_per_second: 0
}

/** Time constant (seconds) for the rolling-pace EWMA — roughly how long a
 *  real pace change takes to fully register in the displayed number. The
 *  weight given to each new sample is adaptive (`1 - e^(-dt/TAU)`), not
 *  fixed: back-to-back 15s polls (the busy-polling cadence in composer's
 *  index.tsx) blend smoothly, while a slightly-delayed poll counts more
 *  heavily instead of being dragged down by a now-stale reading. Gaps beyond
 *  MAX_SAMPLE_GAP_S skip this blending entirely — see updateSmoothedRate. */
const RATE_SMOOTHING_TAU_S = 30

/** Ceiling (seconds) on the gap between two polls that's still trusted as
 *  one continuous observation window — a few multiples of the 15s
 *  busy-polling cadence, generous enough to absorb a throttled/delayed poll
 *  without false-reseeding during genuinely continuous use. Reopening a chat
 *  that was last polled minutes, hours, or days ago blows well past this: the
 *  diff would otherwise run against whatever sample happened to be sitting
 *  in `lastSampleBySession` from that last visit, which is exactly what
 *  produced wildly wrong numbers on reopening old chats. See
 *  updateSmoothedRate for how the cap is applied. */
const MAX_SAMPLE_GAP_S = 60

/** Last raw (tokens, cost, time) sample per session. Bookkeeping for the
 *  EWMA above — deliberately not store state, since nothing renders it
 *  directly. */
const lastSampleBySession = new Map<string, { t: number; tokens: number; cost: number }>()

/** Folds one backend poll into the rolling-pace EWMA. Only advances the
 *  visible rate when it has a real, recent (sessionId, prior sample) pair to
 *  diff against — the first poll for a session, one right after
 *  `clearSessionUsage`, or one arriving more than MAX_SAMPLE_GAP_S after the
 *  last observed poll all just reseed the baseline silently instead of
 *  computing a window. */
function updateSmoothedRate(sessionId: string, data: SessionUsageResponse): void {
  const now = Date.now()
  // Use output_tokens ONLY for the generation-speed rate — the user sees
  // "t/s" and interprets it as "how fast is the model generating text?"
  // Cache read/write tokens are input-side overhead (replayed context) that
  // can dwarf actual generation by 100x on a long agentic session with a
  // large system prompt, inflating the displayed rate to thousands of t/s
  // that bear no relation to generation speed. Cost is driven by
  // estimated_cost_usd (which already bills cache tokens), so cost/hr
  // stays accurate regardless of what tokens we count for the rate.
  const tokens = data.output_tokens
  const cost = data.estimated_cost_usd

  // A malformed/partial response (a missing field reads as undefined, which
  // turns the sum above into NaN) must never reach the baseline or the
  // displayed rate — one bad poll would otherwise corrupt every blended
  // sample after it until the session is cleared.
  if (!Number.isFinite(tokens) || !Number.isFinite(cost)) {return}

  const prev = lastSampleBySession.get(sessionId)

  lastSampleBySession.set(sessionId, { t: now, tokens, cost })

  if (!prev) {return}

  const dt = (now - prev.t) / 1000

  // Gap too large to trust as one continuous window (see MAX_SAMPLE_GAP_S) —
  // the baseline above already advanced, so treat this exactly like the
  // session's first poll: reseed silently, no rate yet. Also clear any
  // previously-displayed rate rather than leaving a now-stale reading on
  // screen with nothing to indicate it's minutes, hours, or days old — the
  // caller falls back to the plain cumulative total when no rate is present.
  if (dt > MAX_SAMPLE_GAP_S) {
    const currentRates = $smoothedRateBySession.get()

    if (sessionId in currentRates) {
      const nextRates = { ...currentRates }

      delete nextRates[sessionId]
      $smoothedRateBySession.set(nextRates)
    }

    return
  }

  const dTokens = tokens - prev.tokens
  const dCost = cost - prev.cost

  // Cumulative counters never shrink and two polls never land on the same
  // millisecond — either means a stale/racing response. Drop the sample;
  // the baseline above already advanced, so the next real poll recovers.
  if (dt < 1 || dTokens < 0 || dCost < 0) {return}

  const windowTps = dTokens / dt
  const windowCostPerHour = (dCost / dt) * 3600
  const alpha = 1 - Math.exp(-dt / RATE_SMOOTHING_TAU_S)
  const prevRate = $smoothedRateBySession.get()[sessionId]

  const nextRate: SmoothedRate = {
    tokensPerSecond: Math.min(
      prevRate ? alpha * windowTps + (1 - alpha) * prevRate.tokensPerSecond : windowTps,
      200  // cap: no model sustains >200 t/s; spikes above this are measurement noise
    ),
    costPerHour: prevRate ? alpha * windowCostPerHour + (1 - alpha) * prevRate.costPerHour : windowCostPerHour
  }

  // Belt-and-suspenders: finite, gap-bounded, non-negative inputs can't
  // actually produce a non-finite result here, but a display-facing number
  // should never be allowed to surface NaN/Infinity even if a future change
  // to this function breaks that invariant.
  if (!Number.isFinite(nextRate.tokensPerSecond) || !Number.isFinite(nextRate.costPerHour)) {return}

  $smoothedRateBySession.set({
    ...$smoothedRateBySession.get(),
    [sessionId]: nextRate
  })
}

/** Get the usage data for a session (returns zeros if absent). */
export function getSessionUsageData(sessionId: string): SessionUsageData {
  return $sessionUsageBySession.get()[sessionId] ?? EMPTY_USAGE
}

/** Get the current smoothed pace for a session, or null if it hasn't been
 *  established yet (fewer than two polls) — callers should show a
 *  placeholder rather than deriving a rate from cumulative totals, see
 *  `SmoothedRate`. */
export function getSmoothedRate(sessionId: string): SmoothedRate | null {
  return $smoothedRateBySession.get()[sessionId] ?? null
}

/** Store usage data from the backend response. */
export function setSessionUsage(sessionId: string, data: SessionUsageResponse) {
  updateSmoothedRate(sessionId, data)

  const current = $sessionUsageBySession.get()
  $sessionUsageBySession.set({
    ...current,
    [sessionId]: data
  })
}

/** Clear usage data for a session. */
export function clearSessionUsage(sessionId: string) {
  const current = $sessionUsageBySession.get()

  lastSampleBySession.delete(sessionId)

  const currentRates = $smoothedRateBySession.get()

  if (sessionId in currentRates) {
    const nextRates = { ...currentRates }
    delete nextRates[sessionId]
    $smoothedRateBySession.set(nextRates)
  }

  if (!(sessionId in current)) {
    return
  }

  const next = { ...current }
  delete next[sessionId]
  $sessionUsageBySession.set(next)
}

/** Fetch usage data from the backend for a session. */
export async function fetchSessionUsage(sessionId: string): Promise<void> {
  try {
    const data = await getSessionUsage(sessionId)
    setSessionUsage(sessionId, data)
  } catch {
    // Transient network or backend error — silently ignore.
  }
}

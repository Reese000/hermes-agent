import { atom } from 'nanostores'

import { persistBoolean, persistString, storedBoolean, storedString } from '@/lib/storage'

const ENABLED_KEY = 'hermes.desktop.usage-indicator.enabled'
const INTERVAL_KEY = 'hermes.desktop.usage-indicator.interval-ms'

/** Whether the usage indicator is visible above the composer. */
export const $usageIndicatorEnabled = atom(storedBoolean(ENABLED_KEY, true))

$usageIndicatorEnabled.subscribe(enabled => persistBoolean(ENABLED_KEY, enabled))

/** Polling interval in milliseconds while the session is busy.
 *  Faster than the original 15 s — the EWMA in session-usage.ts smooths
 *  aggressive cadences without instability, and 5 s gives the user a much
 *  more responsive cost/pace readout. */
export const USAGE_INTERVAL_OPTIONS = [3_000, 5_000, 10_000, 15_000] as const

export type UsageIntervalMs = (typeof USAGE_INTERVAL_OPTIONS)[number]

function parseInterval(raw: string | null): UsageIntervalMs {
  const n = Number(raw)

  return USAGE_INTERVAL_OPTIONS.includes(n as UsageIntervalMs) ? (n as UsageIntervalMs) : 5_000
}

export const $usageIntervalMs = atom<UsageIntervalMs>(parseInterval(storedString(INTERVAL_KEY)))

$usageIntervalMs.subscribe(ms => persistString(INTERVAL_KEY, String(ms)))

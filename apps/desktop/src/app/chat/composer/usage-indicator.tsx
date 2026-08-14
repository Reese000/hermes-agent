import { useStore } from '@nanostores/react'
import { Tooltip as TooltipPrimitive } from 'radix-ui'
import type { ReactNode } from 'react'

import { formatDurationSeconds } from '@/components/assistant-ui/tool/fallback-model/format'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import { Tooltip, TooltipTrigger } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Cpu } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $sessionUsageBySession, $smoothedRateBySession } from '@/store/session-usage'
import {
  $usageIndicatorEnabled,
  $usageIntervalMs,
  USAGE_INTERVAL_OPTIONS,
  type UsageIntervalMs
} from '@/store/usage-indicator'

interface UsageIndicatorProps {
  busy: boolean
  sessionId: string | null
}

// tokensPerSecond is output-only (session-usage.ts), so values are
// generation-speed (typically 10-80 tok/s) — compact notation is still
// useful as a safety net for unusually fast providers.
function formatRate(value: number): string {
  if (value >= 1000) {
    return new Intl.NumberFormat(undefined, { maximumFractionDigits: 1, notation: 'compact' }).format(value)
  }

  if (value >= 10) {
    return Math.round(value).toString()
  }

  return value.toFixed(1)
}

const INTERVAL_LABELS: Record<UsageIntervalMs, string> = {
  [3_000]: '3s',
  [5_000]: '5s',
  [10_000]: '10s',
  [15_000]: '15s'
}

export function UsageIndicator({ busy, sessionId }: UsageIndicatorProps) {
  const { t } = useI18n()
  const u = t.usage
  const enabled = useStore($usageIndicatorEnabled)
  const intervalMs = useStore($usageIntervalMs)
  const bySession = useStore($sessionUsageBySession)
  const rateBySession = useStore($smoothedRateBySession)
  const data = sessionId ? bySession[sessionId] : null
  const rawRate = sessionId ? rateBySession[sessionId] : null

  if (!enabled) {
    return null
  }

  // Defense in depth: session-usage.ts already guards against non-finite
  // inputs and stale-gap baselines, but a display component should never
  // trust an upstream value enough to risk rendering "NaN/hr" or
  // "Infinity t/s" to the user.
  const rate =
    rawRate &&
    Number.isFinite(rawRate.costPerHour) &&
    Number.isFinite(rawRate.tokensPerSecond) &&
    rawRate.costPerHour >= 0 &&
    rawRate.tokensPerSecond >= 0
      ? rawRate
      : null

  const hasData =
    data &&
    (data.input_tokens > 0 ||
      data.output_tokens > 0 ||
      data.estimated_cost_usd > 0)

  // Auto-hide when no usage data
  if (!hasData) {
    return null
  }

  const d = data

  // Guard against NaN/Infinity/negative values
  let cost = d.estimated_cost_usd

  if (!isFinite(cost) || cost < 0) { cost = 0 }

  const formattedCost =
    cost >= 0.01
      ? `$${cost.toFixed(2)}`
      : cost > 0
        ? `$${cost.toFixed(4)}`
        : '$0'

  // Cache hit rate — tooltip detail only now (see the row list below). It's
  // a meaningful "is my context getting reused" signal but not a pace
  // readout, so it no longer competes with the two rate metrics for the
  // glance line.
  const cacheableTokens = d.cache_read_tokens + d.input_tokens
  const cacheHitRate = cacheableTokens > 0 ? d.cache_read_tokens / cacheableTokens : null
  const formattedCacheHitRate = cacheHitRate === null ? '—' : `${Math.round(cacheHitRate * 100)}%`

  // Cost/hour and tokens/sec — both driven by `rate` (a rolling EWMA over
  // recent /usage polls, computed in session-usage.ts), never by dividing a
  // cumulative total by `elapsed_seconds`. `elapsed_seconds` is wall-clock
  // time since the session STARTED, so any session with real idle time in
  // it (afk, thinking, a long-running tool left overnight) dilutes that
  // ratio toward zero — a session active for 20 minutes over a 61-hour span
  // reports a rate ~180x lower than what it actually costs to run while
  // you're using it. The rolling rate sidesteps this structurally: it only
  // advances on wall-clock time between actual polls, and polls only happen
  // while busy, so idle time never enters the computation at all. Until it
  // exists (needs two polls — roughly the first 15s of activity), show a
  // plain total instead of a rate that would carry the same distortion.
  const formattedCostPerHour = rate ? `$${rate.costPerHour.toFixed(2)}/hr` : null

  const formattedTps = rate ? formatRate(rate.tokensPerSecond) : null

  const formattedElapsed = formatDurationSeconds(d.elapsed_seconds) || '0s'

  // Detailed tooltip content.
  //
  // This bypasses `Tip`/`TooltipContent` entirely and builds the popover
  // directly on the Radix primitives. `TooltipContent`'s decoration span uses
  // `box-decoration-clone` on an `inline` box (see tooltip.tsx's #62022 note)
  // — a model built for short one-line label chips that wrap across text
  // lines. A tall 9-row data panel forced into that inline/decoration-clone
  // context fragments unpredictably (values rendering detached from labels,
  // rows losing their box entirely) regardless of whether the rows inside are
  // flex or grid — the corruption is in the wrapper, not our content's
  // layout model. A plain block-level `TooltipPrimitive.Content` sidesteps it.
  // opacity-60, not text-muted-foreground: this panel's `bg-foreground` /
  // `text-background` are already inverted relative to the page, so a
  // page-background-relative muted color lands dark-on-dark and disappears.
  // Dimming the inherited (correctly contrasting) text color keeps it legible.
  const row = (label: string, value: ReactNode) => (
    <div className="flex items-baseline justify-between gap-3" key={label}>
      <span className="opacity-60">{label}:</span>
      <span>{value}</span>
    </div>
  )

  const detail = (
    <div className="flex w-40 flex-col gap-y-0.5 text-xs tabular-nums">
      {row(u.inputTokens, d.input_tokens.toLocaleString())}
      {row(u.outputTokens, d.output_tokens.toLocaleString())}
      {d.cache_read_tokens > 0 && row(u.cacheReadTokens, d.cache_read_tokens.toLocaleString())}
      {d.cache_read_tokens > 0 && row(u.cacheHitRate, formattedCacheHitRate)}
      {d.reasoning_tokens > 0 && row(u.reasoningTokens, d.reasoning_tokens.toLocaleString())}
      {row(u.estimatedCost, cost >= 0.0001 ? `$${cost.toFixed(4)}` : '<$0.0001')}
      {row(u.costPerHour, formattedCostPerHour ?? '—')}
      {row(u.tokensPerSecond, formattedTps ?? '—')}
      {row(u.elapsed, formattedElapsed)}
      {row(u.calls, d.api_call_count)}
    </div>
  )

  const indicator = (
    <Tooltip delayDuration={200} disableHoverableContent>
      <TooltipTrigger asChild>
        <div
          className={cn(
            'flex items-center gap-1.5 px-2 py-0.5 text-xs tabular-nums',
            'text-(--ui-text-tertiary)',
            busy && 'animate-pulse'
          )}
        >
          <Cpu className="size-3 shrink-0 opacity-50" />
          <span>{formattedCostPerHour ?? formattedCost}</span>
          <span aria-hidden="true" className="opacity-30">
            ·
          </span>
          <span>{formattedTps ?? '—'} t/s</span>
        </div>
      </TooltipTrigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          className="pointer-events-none z-(--z-over-modal) select-none rounded-md bg-foreground px-2 py-1.5 text-background"
          data-slot="tooltip-content"
          sideOffset={6}
        >
          {detail}
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </Tooltip>
  )

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <div className="contents">{indicator}</div>
      </ContextMenuTrigger>
      <ContextMenuContent className="w-48">
        <ContextMenuItem onSelect={() => $usageIndicatorEnabled.set(false)}>
          Hide usage indicator
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuSub>
          <ContextMenuSubTrigger>Refresh interval</ContextMenuSubTrigger>
          <ContextMenuSubContent className="w-32">
            {USAGE_INTERVAL_OPTIONS.map(ms => (
              <ContextMenuItem
                key={ms}
                onSelect={() => $usageIntervalMs.set(ms)}
              >
                <span className={cn('mr-2', ms === intervalMs && 'font-bold')}>
                  {ms === intervalMs ? '✓' : '\u2003'}
                </span>
                {INTERVAL_LABELS[ms]}
              </ContextMenuItem>
            ))}
          </ContextMenuSubContent>
        </ContextMenuSub>
      </ContextMenuContent>
    </ContextMenu>
  )
}

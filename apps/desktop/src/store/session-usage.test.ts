import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionUsageResponse } from '@/types/hermes'

import {
  $sessionUsageBySession,
  clearSessionUsage,
  fetchSessionUsage,
  getSessionUsageData,
  getSmoothedRate,
  setSessionUsage
} from './session-usage'

// Mock the hermes module so fetchSessionUsage can call the API
vi.mock('@/hermes', () => ({
  getSessionUsage: vi.fn()
}))

import { getSessionUsage } from '@/hermes'

const mockUsageResponse = (overrides: Partial<SessionUsageResponse> = {}): SessionUsageResponse => ({
  actual_cost_usd: 0.0042,
  api_call_count: 5,
  cache_read_tokens: 12000,
  cache_write_tokens: 3000,
  elapsed_seconds: 12.5,
  estimated_cost_usd: 0.005,
  input_tokens: 8000,
  output_tokens: 1500,
  reasoning_tokens: 500,
  tokens_per_second: 760,
  ...overrides
})

afterEach(() => {
  // Reset the atom between tests
  $sessionUsageBySession.set({})
  vi.clearAllMocks()
})

describe('$sessionUsageBySession atom', () => {
  it('starts empty', () => {
    expect($sessionUsageBySession.get()).toEqual({})
  })
})

describe('setSessionUsage', () => {
  it('stores usage data for a session', () => {
    const data = mockUsageResponse()
    setSessionUsage('s1', data)

    const stored = $sessionUsageBySession.get()['s1']
    expect(stored.input_tokens).toBe(8000)
    expect(stored.output_tokens).toBe(1500)
    expect(stored.estimated_cost_usd).toBe(0.005)
  })

  it('preserves existing sessions when adding a new one', () => {
    setSessionUsage('s1', mockUsageResponse({ input_tokens: 100 }))
    setSessionUsage('s2', mockUsageResponse({ input_tokens: 200 }))

    const bySession = $sessionUsageBySession.get()
    expect(bySession['s1'].input_tokens).toBe(100)
    expect(bySession['s2'].input_tokens).toBe(200)
  })

  it('overwrites data for the same session', () => {
    setSessionUsage('s1', mockUsageResponse({ input_tokens: 100 }))
    setSessionUsage('s1', mockUsageResponse({ input_tokens: 999 }))

    expect($sessionUsageBySession.get()['s1'].input_tokens).toBe(999)
  })
})

describe('getSessionUsageData', () => {
  it('returns zeros for an unknown session', () => {
    const data = getSessionUsageData('unknown')
    expect(data.input_tokens).toBe(0)
    expect(data.output_tokens).toBe(0)
    expect(data.estimated_cost_usd).toBe(0)
    expect(data.actual_cost_usd).toBe(0)
    expect(data.api_call_count).toBe(0)
    expect(data.elapsed_seconds).toBe(0)
    expect(data.tokens_per_second).toBe(0)
  })

  it('returns stored data for a known session', () => {
    setSessionUsage('s1', mockUsageResponse({ input_tokens: 777 }))
    expect(getSessionUsageData('s1').input_tokens).toBe(777)
  })

  it('returns the canonical empty object for unknown sessions', () => {
    const a = getSessionUsageData('unknown')
    const b = getSessionUsageData('unknown')
    // Same reference is intentional — a shared immutable zero-value constant.
    expect(a).toBe(b)
    expect(a.input_tokens).toBe(0)
  })
})

describe('clearSessionUsage', () => {
  it('removes the session key', () => {
    setSessionUsage('s1', mockUsageResponse())
    expect('s1' in $sessionUsageBySession.get()).toBe(true)

    clearSessionUsage('s1')
    expect('s1' in $sessionUsageBySession.get()).toBe(false)
  })

  it('is a no-op for an unknown session', () => {
    const before = { ...$sessionUsageBySession.get() }
    clearSessionUsage('never-added')
    expect($sessionUsageBySession.get()).toEqual(before)
  })

  it('leaves other sessions intact', () => {
    setSessionUsage('s1', mockUsageResponse({ input_tokens: 1 }))
    setSessionUsage('s2', mockUsageResponse({ input_tokens: 2 }))

    clearSessionUsage('s1')

    const bySession = $sessionUsageBySession.get()
    expect('s1' in bySession).toBe(false)
    expect('s2' in bySession).toBe(true)
    expect(bySession['s2'].input_tokens).toBe(2)
  })

  it('resets rolling-pace bookkeeping so a later poll starts fresh, not mid-average', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)

    try {
      setSessionUsage('s1', mockUsageResponse({ estimated_cost_usd: 0.01, input_tokens: 0, output_tokens: 1000 }))
      vi.setSystemTime(15_000)
      setSessionUsage('s1', mockUsageResponse({ estimated_cost_usd: 0.013, input_tokens: 0, output_tokens: 1150 }))
      expect(getSmoothedRate('s1')).not.toBeNull()

      clearSessionUsage('s1')
      expect(getSmoothedRate('s1')).toBeNull()

      // A poll right after clearing must seed the baseline, not diff against
      // the pre-clear sample (which would show as a bogus multi-thousand
      // token/hour spike or, worse, a negative delta since the response is
      // free to start from a smaller cumulative total on a fresh session).
      vi.setSystemTime(16_000)
      setSessionUsage('s1', mockUsageResponse({ estimated_cost_usd: 0, input_tokens: 0, output_tokens: 5 }))
      expect(getSmoothedRate('s1')).toBeNull()
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('getSmoothedRate', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('is null before a session has produced two polls', () => {
    expect(getSmoothedRate('rate-fresh')).toBeNull()

    setSessionUsage('rate-fresh', mockUsageResponse())
    expect(getSmoothedRate('rate-fresh')).toBeNull()
  })

  it('seeds exactly from the first real interval between two polls', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)

    setSessionUsage('rate-seed', mockUsageResponse({ estimated_cost_usd: 0.01, input_tokens: 0, output_tokens: 1000 }))

    vi.setSystemTime(15_000)
    setSessionUsage('rate-seed', mockUsageResponse({ estimated_cost_usd: 0.013, input_tokens: 0, output_tokens: 1150 }))

    // 150 output tokens / 15s = 10 tok/s; $0.003 / 15s * 3600 = $0.72/hr. No prior
    // smoothed value exists yet, so the first window IS the rate verbatim.
    const rate = getSmoothedRate('rate-seed')
    expect(rate?.tokensPerSecond).toBeCloseTo(10, 5)
    expect(rate?.costPerHour).toBeCloseTo(0.72, 5)
  })

  it('blends a second interval toward the new pace without jumping straight to it', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    setSessionUsage('rate-blend', mockUsageResponse({ estimated_cost_usd: 0.01, input_tokens: 0, output_tokens: 1000 }))
    vi.setSystemTime(15_000)
    setSessionUsage('rate-blend', mockUsageResponse({ estimated_cost_usd: 0.013, input_tokens: 0, output_tokens: 1150 }))
    const first = getSmoothedRate('rate-blend')!

    // Pace doubles: 300 output tokens in the next 15s (20 tok/s window) vs the
    // first window's 10 tok/s.
    vi.setSystemTime(30_000)
    setSessionUsage('rate-blend', mockUsageResponse({ estimated_cost_usd: 0.019, input_tokens: 0, output_tokens: 1450 }))
    const second = getSmoothedRate('rate-blend')!

    expect(second.tokensPerSecond).toBeGreaterThan(first.tokensPerSecond)
    expect(second.tokensPerSecond).toBeLessThan(20)
  })

  it('clears the displayed rate after a long gap instead of computing a misleading one', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    setSessionUsage('rate-idle', mockUsageResponse({ estimated_cost_usd: 0.01, input_tokens: 0, output_tokens: 1000 }))
    vi.setSystemTime(15_000)
    setSessionUsage('rate-idle', mockUsageResponse({ estimated_cost_usd: 0.013, input_tokens: 0, output_tokens: 1150 }))
    expect(getSmoothedRate('rate-idle')).not.toBeNull()
    // Established pace: 10 tok/s, $0.72/hr (see the seeding test above).

    // Reopening a chat last polled hours ago: diffing against whatever
    // sample happened to still be sitting in memory from that last visit is
    // exactly what produced wildly wrong numbers on reopening old chats, so
    // this should clear the stale reading rather than keep showing it or
    // compute a new one from an untrusted gap.
    vi.setSystemTime(15_000 + 2 * 60 * 60 * 1000)
    setSessionUsage('rate-idle', mockUsageResponse({ estimated_cost_usd: 0.013, input_tokens: 0, output_tokens: 1152 }))

    expect(getSmoothedRate('rate-idle')).toBeNull()
  })

  it('still computes normally for a gap just under the trust ceiling', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    setSessionUsage('rate-gap-ok', mockUsageResponse({ estimated_cost_usd: 0.01, input_tokens: 0, output_tokens: 1000 }))
    vi.setSystemTime(59_000)
    setSessionUsage('rate-gap-ok', mockUsageResponse({ estimated_cost_usd: 0.013, input_tokens: 0, output_tokens: 1150 }))

    expect(getSmoothedRate('rate-gap-ok')).not.toBeNull()
  })

  it('clears instead of computing for a gap just over the trust ceiling', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    setSessionUsage('rate-gap-bad', mockUsageResponse({ estimated_cost_usd: 0.01, input_tokens: 0, output_tokens: 1000 }))
    vi.setSystemTime(61_000)
    setSessionUsage('rate-gap-bad', mockUsageResponse({ estimated_cost_usd: 0.013, input_tokens: 0, output_tokens: 1150 }))

    expect(getSmoothedRate('rate-gap-bad')).toBeNull()
  })

  it('ignores a non-finite field instead of corrupting the baseline or the rate', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    setSessionUsage('rate-nan', mockUsageResponse({ estimated_cost_usd: 0.01, input_tokens: 0, output_tokens: 1000 }))
    vi.setSystemTime(15_000)
    // A malformed poll — e.g. a field missing from the response — reads as
    // NaN. It must not become the new baseline.
    setSessionUsage('rate-nan', mockUsageResponse({ estimated_cost_usd: 0.013, input_tokens: 0, output_tokens: NaN }))
    expect(getSmoothedRate('rate-nan')).toBeNull()

    // A subsequent well-formed poll recovers normally, diffing against the
    // last GOOD sample (t=0), not the rejected NaN one.
    vi.setSystemTime(30_000)
    setSessionUsage('rate-nan', mockUsageResponse({ estimated_cost_usd: 0.016, input_tokens: 0, output_tokens: 1300 }))
    expect(getSmoothedRate('rate-nan')).not.toBeNull()
  })

  it('ignores a sample with no meaningful time gap (dt < 1s)', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    setSessionUsage('rate-dup', mockUsageResponse({ estimated_cost_usd: 0.01, input_tokens: 0, output_tokens: 1000 }))
    vi.setSystemTime(15_000)
    setSessionUsage('rate-dup', mockUsageResponse({ estimated_cost_usd: 0.013, input_tokens: 0, output_tokens: 1150 }))
    const established = getSmoothedRate('rate-dup')

    // Same instant, absurd token jump — if this were folded in, the rate
    // would spike toward infinity.
    setSessionUsage('rate-dup', mockUsageResponse({ estimated_cost_usd: 5, input_tokens: 0, output_tokens: 999_999 }))

    expect(getSmoothedRate('rate-dup')).toEqual(established)
  })

  it('excludes cache read/write tokens from the rate so cached-context replay does not inflate t/s', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    setSessionUsage(
      'rate-cache',
      mockUsageResponse({
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        estimated_cost_usd: 0.01,
        input_tokens: 100,
        output_tokens: 100
      })
    )

    vi.setSystemTime(15_000)
    // A burst of cached-context replay dwarfing fresh output — the
    // scenario from a long agentic session replaying a large system prompt
    // every turn. The rate should reflect ONLY the 50 output tokens, not
    // the 150k cache tokens.
    setSessionUsage(
      'rate-cache',
      mockUsageResponse({
        cache_read_tokens: 100_000,
        cache_write_tokens: 50_000,
        estimated_cost_usd: 0.013,
        input_tokens: 110,
        output_tokens: 150
      })
    )

    const rate = getSmoothedRate('rate-cache')!
    // (150 - 100) output tokens / 15s = ~3.33 tok/s. Cache tokens are excluded.
    expect(rate.tokensPerSecond).toBeCloseTo(50 / 15, 1)
    // Cost/hr still reflects the full estimated_cost_usd (which includes cache billing).
    expect(rate.costPerHour).toBeCloseTo(0.72, 1)
  })

  it('drops a sample where cumulative totals decreased instead of corrupting the rate', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    setSessionUsage('rate-regress', mockUsageResponse({ estimated_cost_usd: 0.01, input_tokens: 0, output_tokens: 1000 }))
    vi.setSystemTime(15_000)
    setSessionUsage('rate-regress', mockUsageResponse({ estimated_cost_usd: 0.013, input_tokens: 0, output_tokens: 1150 }))
    const established = getSmoothedRate('rate-regress')

    // A stale/racing response reporting fewer cumulative tokens than we've
    // already observed — must not read as a negative rate.
    vi.setSystemTime(30_000)
    setSessionUsage('rate-regress', mockUsageResponse({ estimated_cost_usd: 0.012, input_tokens: 0, output_tokens: 1100 }))

    expect(getSmoothedRate('rate-regress')).toEqual(established)
  })
})

describe('fetchSessionUsage', () => {
  it('calls getSessionUsage and stores the result', async () => {
    const mockData = mockUsageResponse({ input_tokens: 42 })
    vi.mocked(getSessionUsage).mockResolvedValueOnce(mockData)

    await fetchSessionUsage('s1')

    expect(getSessionUsage).toHaveBeenCalledWith('s1')
    expect($sessionUsageBySession.get()['s1'].input_tokens).toBe(42)
  })

  it('silently ignores API errors', async () => {
    vi.mocked(getSessionUsage).mockRejectedValueOnce(new Error('Network error'))

    // Should not throw
    await fetchSessionUsage('s1')

    // Store should remain unchanged
    expect($sessionUsageBySession.get()['s1']).toBeUndefined()
  })

  it('does not overwrite existing data on error', async () => {
    setSessionUsage('s1', mockUsageResponse({ input_tokens: 99 }))
    vi.mocked(getSessionUsage).mockRejectedValueOnce(new Error('timeout'))

    await fetchSessionUsage('s1')

    expect($sessionUsageBySession.get()['s1'].input_tokens).toBe(99)
  })
})

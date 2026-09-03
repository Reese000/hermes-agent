import type {
  AnalyticsResponse,
  AuxiliaryModelsResponse,
  MoaConfigResponse,
  ModelAssignmentRequest,
  ModelAssignmentResponse,
  ModelInfoResponse,
  ModelOptionsResponse
} from '@/types/hermes'

import { capabilityScoped, hermesApi, type ProfileScope, profileScoped, STARTUP_REQUEST_TIMEOUT_MS } from './client'

export function getGlobalModelInfo(profile?: null | string): Promise<ModelInfoResponse> {
  return hermesApi<ModelInfoResponse>({
    ...profileScoped(profile),
    path: '/api/model/info',
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}

export function getUsageAnalytics(days = 30, profile?: ProfileScope): Promise<AnalyticsResponse> {
  return window.hermesDesktop.api<AnalyticsResponse>({
    ...capabilityScoped(profile),
    path: `/api/analytics/usage?days=${Math.max(1, Math.floor(days))}`
  })
}

export function getGlobalModelOptions(
  opts?: {
    refresh?: boolean
    includeUnconfigured?: boolean
    explicitOnly?: boolean
  },
  profile?: null | string
): Promise<ModelOptionsResponse> {
  const params = new URLSearchParams()

  if (opts?.refresh) {
    params.set('refresh', '1')
  }

  if (opts?.includeUnconfigured) {
    params.set('include_unconfigured', '1')
  }

  if (opts?.explicitOnly !== false) {
    params.set('explicit_only', '1')
  }

  return hermesApi<ModelOptionsResponse>({
    ...profileScoped(profile),
    path: params.size > 0 ? `/api/model/options?${params.toString()}` : '/api/model/options',
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}

export interface RecommendedDefaultModel {
  provider: string
  model: string
  /** True/false for Nous (free vs paid tier); null for other providers. */
  free_tier: boolean | null
}

// Recommended default model for a freshly-authenticated provider. Mirrors the
// curation `hermes model` does — for Nous it honors the free/paid tier so a
// free user gets a free model instead of a paid default.
export function getRecommendedDefaultModel(
  provider: string,
  profile?: null | string
): Promise<RecommendedDefaultModel> {
  return hermesApi<RecommendedDefaultModel>({
    ...profileScoped(profile),
    path: `/api/model/recommended-default?provider=${encodeURIComponent(provider)}`
  })
}

export function setGlobalModel(
  provider: string,
  model: string
): Promise<{ ok: boolean; provider: string; model: string }> {
  return hermesApi<{ ok: boolean; provider: string; model: string }>({
    ...profileScoped(),
    path: '/api/model/set',
    method: 'POST',
    body: {
      scope: 'main',
      provider,
      model
    }
  })
}

export function getAuxiliaryModels(profile?: null | string): Promise<AuxiliaryModelsResponse> {
  return hermesApi<AuxiliaryModelsResponse>({
    ...profileScoped(profile),
    path: '/api/model/auxiliary'
  })
}

export function getMoaModels(profile?: null | string): Promise<MoaConfigResponse> {
  return hermesApi<MoaConfigResponse>({
    ...profileScoped(profile),
    path: '/api/model/moa'
  })
}

export function saveMoaModels(
  body: MoaConfigResponse,
  profile?: null | string
): Promise<MoaConfigResponse & { ok: boolean }> {
  return hermesApi<MoaConfigResponse & { ok: boolean }>({
    ...profileScoped(profile),
    path: '/api/model/moa',
    method: 'PUT',
    body
  })
}

export function setModelAssignment(
  body: ModelAssignmentRequest,
  profile?: null | string
): Promise<ModelAssignmentResponse> {
  return hermesApi<ModelAssignmentResponse>({
    ...profileScoped(profile),
    path: '/api/model/set',
    method: 'POST',
    body
  })
}

export function searchProviderModels(provider: string, query: string): Promise<{ models: string[] }> {
  return window.hermesDesktop.api<{ models: string[] }>({
    ...profileScoped(),
    path: `/api/model/search?${new URLSearchParams({ q: query, provider }).toString()}`
  })
}

export interface EnhancePromptResponse {
  enhanced: string
  ok: boolean
  error?: string
}

export function enhancePrompt(
  text: string,
  sessionId?: string | null,
  profile?: string | null
): Promise<EnhancePromptResponse> {
  return window.hermesDesktop.api<EnhancePromptResponse>({
    ...profileScoped(),
    path: '/api/model/enhance-prompt',
    method: 'POST',
    body: { prompt: text, session_id: sessionId || undefined, profile: profile || undefined },
    timeoutMs: 300_000
  })
}

/** Streaming version of enhancePrompt — yields text chunks as the LLM generates. */
export async function* enhancePromptStream(
  text: string,
  sessionId?: string | null,
  profile?: string | null,
  signal?: AbortSignal
): AsyncGenerator<string, void, unknown> {
  if (!window.hermesDesktop?.apiStream) {
    const res = await enhancePrompt(text, sessionId, profile)
    if (res.ok && res.enhanced) {
      yield res.enhanced
    }
    return
  }

  const chunks: string[] = []
  let resolve: (() => void) | null = null
  let done = false
  let donePayload: { ok: boolean; error?: string } | null = null

  const { dispose } = window.hermesDesktop.apiStream(
    {
      ...profileScoped(),
      path: '/api/model/enhance-prompt-stream',
      method: 'POST',
      body: { prompt: text, session_id: sessionId || undefined, profile: profile || undefined },
      timeoutMs: 300_000
    },
    {
      onChunk: (payload: { data: string }) => {
        if (payload.data === '[DONE]') {
          done = true
          donePayload = { ok: true }
          resolve?.()
          return
        }
        try {
          const parsed = JSON.parse(payload.data)
          if (parsed.error) {
            done = true
            donePayload = { ok: false, error: parsed.error }
            resolve?.()
            return
          }
          if (parsed.text) {
            chunks.push(parsed.text)
            resolve?.()
          }
        } catch {
          // Non-JSON data line — skip
        }
      },
      onDone: (payload: { ok: boolean; error?: string }) => {
        done = true
        donePayload = payload
        resolve?.()
      }
    }
  )

  const abortHandler = () => {
    done = true
    resolve?.()
  }
  signal?.addEventListener('abort', abortHandler)

  try {
    while (!done) {
      await new Promise<void>(r => { resolve = r })
      while (chunks.length > 0) {
        yield chunks.shift()!
      }
    }
    if (donePayload && !(donePayload as { ok: boolean }).ok) {
      throw new Error((donePayload as { error?: string }).error || 'Enhance failed')
    }
  } finally {
    signal?.removeEventListener('abort', abortHandler)
    dispose()
  }
}

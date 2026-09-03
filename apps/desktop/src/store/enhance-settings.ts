import { atom } from 'nanostores'

import { persistBoolean, persistString, storedBoolean, storedString } from '@/lib/storage'
import type { ReasoningEffort } from '@/lib/reasoning-effort'

// ── Enhance settings store ───────────────────────────────────────────────
// Persists the prompt-enhancement feature's configuration to localStorage.
// Each atom is independently persisted so changes write through immediately.
//
// Architecture:
//   - $enhanceEnabled:  whether the enhance button is visible at all
//   - $enhanceModel:    model id sent to the backend enhance endpoint
//   - $enhanceProvider: provider slug for the enhance model
//   - $enhanceReasoning: reasoning effort level (global, not per-model)
//   - $enhanceProfile:  active enhance profile key (syntax|minimal|balanced|maximum)
//
// These are consumed by:
//   - controls.tsx: context menu, tooltip, model catalog controller
//   - composer/index.tsx: handleEnhance() reads config to call enhancePrompt()
//   - hermes.ts: enhancePrompt() sends model/provider to backend
//
// The backend endpoint (web_server.py enhance_prompt) uses these to select
// the LLM model and applies the profile's system prompt overrides.
// ─────────────────────────────────────────────────────────────────────────
const ENABLED_KEY = 'hermes.desktop.enhance.enabled'
const MODEL_KEY = 'hermes.desktop.enhance.model'
const PROVIDER_KEY = 'hermes.desktop.enhance.provider'
const REASONING_KEY = 'hermes.desktop.enhance.reasoning'
const PROFILE_KEY = 'hermes.desktop.enhance.profile'

export interface EnhanceProfile {
  description: string
  label: string
  reasoning: ReasoningEffort
  systemPrompt?: string
}

export const ENHANCE_PROFILES: Record<string, EnhanceProfile> = {
  syntax: {
    description: 'Fix code syntax errors only — never reword or restructure',
    label: 'Syntax Only',
    reasoning: 'low',
  },
  minimal: {
    description: 'Quick, minimal changes — preserve original intent',
    label: 'Minimal',
    reasoning: 'minimal',
  },
  balanced: {
    description: 'Balance clarity and detail',
    label: 'Balanced',
    reasoning: 'medium',
  },
  maximum: {
    description: 'Maximum enhancement — add specificity, constraints, context',
    label: 'Maximum',
    reasoning: 'high',
  },
}

/** Whether the enhance button is visible. */
export const $enhanceEnabled = atom(storedBoolean(ENABLED_KEY, true))

$enhanceEnabled.subscribe(enabled => persistBoolean(ENABLED_KEY, enabled))

/** The model used for prompt enhancement. */
export const $enhanceModel = atom(storedString(MODEL_KEY) || 'xiaomi/mimo-v2.5')

$enhanceModel.subscribe(model => persistString(MODEL_KEY, model))

/** The provider for the enhance model. */
export const $enhanceProvider = atom(storedString(PROVIDER_KEY) || 'openrouter')

$enhanceProvider.subscribe(provider => persistString(PROVIDER_KEY, provider))

/** Reasoning level for the enhance model. */
export const $enhanceReasoning = atom<ReasoningEffort>(
  (storedString(REASONING_KEY) as ReasoningEffort) || 'medium'
)

$enhanceReasoning.subscribe(level => persistString(REASONING_KEY, level))

/** Active enhance profile key. */
export const $enhanceProfile = atom(storedString(PROFILE_KEY) || 'balanced')

$enhanceProfile.subscribe(profile => persistString(PROFILE_KEY, profile))

/** Get the current enhance configuration. */
export function getEnhanceConfig() {
  return {
    enabled: $enhanceEnabled.get(),
    model: $enhanceModel.get(),
    profile: $enhanceProfile.get(),
    provider: $enhanceProvider.get(),
    reasoning: $enhanceReasoning.get(),
  }
}

/** Get the active profile's settings. */
export function getActiveProfile(): EnhanceProfile {
  const profileKey = $enhanceProfile.get()
  return ENHANCE_PROFILES[profileKey] || ENHANCE_PROFILES.balanced
}

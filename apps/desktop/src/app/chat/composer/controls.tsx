import { useStore } from '@nanostores/react'
import { useCallback, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Codicon } from '@/components/ui/codicon'
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
// ── Enhance model catalog: separate DropdownMenu ──────────────────────────
// ModelCatalogMenu renders DropdownMenu* primitives (DropdownMenuItem,
// DropdownMenuSub, etc.) which REQUIRE a DropdownMenu context. It cannot be
// nested inside a ContextMenuSubContent — Radix throws
// "MenuItem must be used within Menu" because ContextMenu and DropdownMenu
// have separate, incompatible provider trees. The solution: a standalone
// DropdownMenu with its own hidden trigger (<span className="sr-only" />),
// controlled via `modelMenuOpen` state. The context menu's "Model:" row
// opens it via onSelect + onPointerEnter. The sr-only trigger gives Radix
// a DOM element to anchor the dropdown to (without it, the dropdown defaults
// to the screen corner). Positioning is side="top" align="start" so it
// appears above the composer controls, near the enhance button.
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import {
  AudioLines,
  Ear,
  EarOff,
  iconSize,
  Layers3,
  Loader2,
  Sparkles,
  Square,
  SteeringWheel,
  Volume2,
  VolumeX,
  Zap
} from '@/lib/icons'
import { displayModelName } from '@/lib/model-status-label'
import { reasoningEffortLabel, type ReasoningEffort } from '@/lib/reasoning-effort'
import { cn } from '@/lib/utils'
import {
  $enhanceEnabled,
  $enhanceModel,
  $enhanceProfile,
  $enhanceProvider,
  $enhanceReasoning,
  ENHANCE_PROFILES
} from '@/store/enhance-settings'
import { $hudMode, closeHud, resetHudLayout } from '@/store/hud'
import { $continuousWork } from '@/store/continuous-work'
import { setContinuousWork } from '@/api/config'
import { $wakeWord, toggleWakeWord } from '@/store/wake-word'

import { ModelCatalogMenu, ModelMenuCloseContext, type ModelMenuController } from '@/app/shell/model-catalog-menu'
import { ACTIVE_ICON_BTN, GHOST_ICON_BTN, PRIMARY_ICON_BTN } from './control-classes'
import type { ConversationStatus } from './hooks/use-voice-conversation'
import { ModelPill } from './model-pill'
import type { ChatBarState, VoiceStatus } from './types'
import { VoiceMenu } from './voice-menu'

// Re-exported: `context-menu.tsx` and other row neighbours have always reached
// for these here, and the row is where they read as belonging.
export { ACTIVE_ICON_BTN, GHOST_ICON_BTN, ICON_BTN, PRIMARY_ICON_BTN } from './control-classes'

interface ConversationProps {
  active: boolean
  level: number
  muted: boolean
  status: ConversationStatus
  onEnd: () => void
  onStart: () => void
  onStopTurn: () => void
  onToggleMute: () => void
}

export function ComposerControls({
  autoSpeak,
  busy,
  busyAction,
  canSubmit,
  compactModelPill = false,
  conversation,
  disabled,
  enhancing = false,
  foldVoice = false,
  hasComposerPayload,
  minimal = false,
  onCancelEnhance,
  onEnhance,
  state,
  voiceStatus,
  onDictate,
  onQueue,
  onToggleAutoSpeak
}: {
  autoSpeak: boolean
  busy: boolean
  busyAction: 'steer' | 'queue' | 'stop'
  canSubmit: boolean
  compactModelPill?: boolean
  conversation: ConversationProps
  disabled: boolean
  enhancing?: boolean
  foldVoice?: boolean
  hasComposerPayload: boolean
  minimal?: boolean
  onCancelEnhance?: () => void
  onEnhance?: () => void
  state: ChatBarState
  voiceStatus: VoiceStatus
  onDictate: () => void
  onQueue: () => void
  onToggleAutoSpeak: () => void
}) {
  const { t } = useI18n()
  const c = t.composer
  const hudMode = useStore($hudMode)

  const enhanceEnabled = useStore($enhanceEnabled)
  const enhanceModel = useStore($enhanceModel)
  const enhanceProvider = useStore($enhanceProvider)
  const enhanceReasoning = useStore($enhanceReasoning)
  const enhanceProfile = useStore($enhanceProfile)
  const [modelMenuOpen, setModelMenuOpen] = useState(false)

  // ── Enhance model controller ───────────────────────────────────────────
  // Adapts ModelMenuController (designed for session-state model selection)
  // to the enhance settings atoms. ModelCatalogMenu reads `current` for the
  // active checkmark, calls `select()` on row click, and `setOptions()` for
  // per-row reasoning/fast edits via hover submenus. `applyPreset` and
  // `presetFor` are no-ops because enhance doesn't persist per-model presets
  // — it stores a single global reasoning level in $enhanceReasoning.
  const enhanceController: ModelMenuController = useMemo(() => ({
    applyPreset: () => {},
    current: {
      effort: enhanceReasoning,
      fast: false,
      model: enhanceModel,
      provider: enhanceProvider,
    },
    presetFor: () => ({ effort: enhanceReasoning }),
    select: (model: string, provider: string) => {
      $enhanceModel.set(model)
      $enhanceProvider.set(provider)
      setModelMenuOpen(false)
    },
    setOptions: (patch) => {
      if (patch.effort !== undefined) {
        $enhanceReasoning.set(patch.effort as ReasoningEffort)
      }
    },
  }), [enhanceModel, enhanceProvider, enhanceReasoning])

  if (conversation.active) {
    return <ConversationPill {...conversation} disabled={disabled} />
  }

  const showVoicePrimary = !busy && !hasComposerPayload
  // Steer is just send: a payload keeps the Send affordance mid-turn. Stop
  // only when the composer is empty and a turn is running.
  const showStop = busy && !hasComposerPayload
  const showQueueButton = busyAction !== 'stop' && hasComposerPayload
  const busyLabel = busyAction === 'queue' ? c.queueMessage : busyAction === 'steer' ? c.steer : c.stop
  const foldedVoice = hudMode || foldVoice

  const voiceControls = foldedVoice ? (
    <VoiceMenu
      autoSpeak={autoSpeak}
      disabled={disabled}
      onDictate={onDictate}
      onStartConversation={conversation.onStart}
      onToggleAutoSpeak={onToggleAutoSpeak}
      state={state}
      voiceStatus={voiceStatus}
    />
  ) : (
    <>
      <DictationButton disabled={disabled} onToggle={onDictate} state={state.voice} status={voiceStatus} />
      <AutoSpeakButton active={autoSpeak} disabled={disabled} onToggle={onToggleAutoSpeak} />
      <WakeWordButton disabled={disabled} />
    </>
  )

  return (
    <div className="ml-auto flex min-w-0 shrink items-center gap-(--composer-control-gap)">
      {onEnhance && enhanceEnabled ? (
        <ContextMenu>
          <ContextMenuTrigger asChild>
            <div className="contents">
              <Tip label={enhancing ? c.enhancing : (
                <>
                  <span className="!block font-medium">{c.enhance}</span>
                  <span className="!block text-[10px] opacity-70">{ENHANCE_PROFILES[enhanceProfile]?.label || enhanceProfile} · {displayModelName(enhanceModel)} · {reasoningEffortLabel(enhanceReasoning)}</span>
                  <span className="!block mt-0.5 text-[8px] italic opacity-40">right-click to configure</span>
                </>
              )} className="max-w-52">
                <Button
                  aria-label={enhancing ? c.enhancing : c.enhance}
                  className={cn(GHOST_ICON_BTN, 'p-0')}
                  disabled={disabled}
                  onClick={enhancing ? onCancelEnhance : onEnhance}
                  size="icon"
                  type="button"
                  variant="ghost"
                >
                  {enhancing ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className={iconSize.sm} />}
                </Button>
              </Tip>
            </div>
          </ContextMenuTrigger>
          <ContextMenuContent className="w-64">
            <ContextMenuItem onSelect={() => $enhanceEnabled.set(false)}>
              <Codicon name="eye-closed" size="0.875rem" className="mr-2" />
              Hide enhance button
            </ContextMenuItem>
            <ContextMenuSeparator />
            <ContextMenuSub>
              <ContextMenuSubTrigger>
                <Codicon name="list-filter" size="0.875rem" className="mr-2" />
                Profile: {ENHANCE_PROFILES[enhanceProfile]?.label || enhanceProfile}
              </ContextMenuSubTrigger>
              <ContextMenuSubContent className="w-48">
                {Object.entries(ENHANCE_PROFILES).map(([key, profile]) => (
                  <ContextMenuItem key={key} onSelect={() => $enhanceProfile.set(key)}>
                    <Checkbox checked={key === enhanceProfile} className="mr-2 size-3" />
                    <span className="min-w-0 flex-1">
                      {profile.label}
                      <span className="ml-1 text-[0.625rem] text-muted-foreground">{profile.description}</span>
                    </span>
                  </ContextMenuItem>
                ))}
              </ContextMenuSubContent>
            </ContextMenuSub>
            <ContextMenuItem
              onSelect={() => setModelMenuOpen(true)}
              onPointerEnter={() => setModelMenuOpen(true)}
            >
              <Codicon name="settings-gear" size="0.875rem" className="mr-2" />
              <span className="min-w-0 flex-1 truncate">Model: {displayModelName(enhanceModel)}</span>
              <Codicon name="chevron-right" size="0.75rem" className="ml-auto opacity-50" />
            </ContextMenuItem>
          </ContextMenuContent>
        </ContextMenu>
      ) : null}
      <DropdownMenu open={modelMenuOpen} onOpenChange={setModelMenuOpen}>
        <DropdownMenuTrigger asChild>
          <span className="sr-only" />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-64 p-0" side="top" sideOffset={8}>
          <ModelMenuCloseContext.Provider value={() => setModelMenuOpen(false)}>
            <ModelCatalogMenu
              controller={enhanceController}
              includeMoa={false}
            />
          </ModelMenuCloseContext.Provider>
        </DropdownMenuContent>
      </DropdownMenu>
      <ContinuousWorkButton disabled={disabled} />
      {minimal ? null : (
        <>
          <ModelPill compact={compactModelPill} disabled={disabled} model={state.model} />
          {voiceControls}
        </>
      )}
      {showQueueButton ? (
        <Tip label={<TipKeybindLabel actionId="composer.queue" text={c.queueMessage} />}>
          <Button
            aria-label={c.queueMessage}
            className={GHOST_ICON_BTN}
            disabled={disabled}
            onClick={onQueue}
            size="icon"
            type="button"
            variant="ghost"
          >
            <Layers3 className={iconSize.sm} />
          </Button>
        </Tip>
      ) : null}
      {showVoicePrimary ? (
        <Tip label={c.startVoice}>
          <Button
            aria-label={c.startVoice}
            className={PRIMARY_ICON_BTN}
            disabled={disabled}
            onClick={() => {
              triggerHaptic('open')
              conversation.onStart()
            }}
            size="icon"
            type="button"
          >
            <AudioLines className={iconSize.sm} />
          </Button>
        </Tip>
      ) : (
        <Tip
          label={
            busy ? (
              <TipKeybindLabel
                actionId={
                  busyAction === 'steer'
                    ? 'composer.steer'
                    : busyAction === 'queue'
                      ? 'composer.queue'
                      : 'composer.send'
                }
                text={busyLabel}
              />
            ) : (
              <TipKeybindLabel actionId="composer.send" text={c.send} />
            )
          }
        >
          <Button
            aria-label={busy ? busyLabel : c.send}
            className={PRIMARY_ICON_BTN}
            disabled={disabled || !canSubmit}
            type="submit"
          >
            {busy ? (
              busyAction === 'queue' ? (
                <Layers3 className={iconSize.sm} />
              ) : busyAction === 'steer' ? (
                <SteeringWheel className={iconSize.sm} />
              ) : (
                <span className="block size-2.5 rounded-[0.1875rem] bg-current" />
              )
            ) : (
              <Codicon name="arrow-up" size="0.875rem" />
            )}
          </Button>
        </Tip>
      )}
      {/* The way out of HUD mode, riding the controls row rather than floating
          above the bar. The old chip lived in a 26px transparent strip reserved
          over the composer (--hud-chip-strip), which under glass is bare
          untinted material with a hidden button in it — a band of chrome above
          the surface, paid for in every state, for a control that is invisible
          until hovered. Here it costs no reserved space and sits with the other
          things you can press. */}
      {hudMode ? <HudWindowButtons /> : null}
    </div>
  )
}

function HudWindowButtons() {
  const { t } = useI18n()

  return (
    <>
      <Tip label={t.titlebar.resetHudLayout}>
        <Button
          aria-label={t.titlebar.resetHudLayout}
          className={cn(GHOST_ICON_BTN, 'p-0')}
          onClick={resetHudLayout}
          size="icon"
          type="button"
          variant="ghost"
        >
          <Codicon name="discard" size="0.875rem" />
        </Button>
      </Tip>
      <Tip label={t.titlebar.exitHud}>
        <Button
          aria-label={t.titlebar.exitHud}
          className={cn(GHOST_ICON_BTN, 'p-0')}
          onClick={closeHud}
          size="icon"
          type="button"
          variant="ghost"
        >
          <Codicon name="screen-normal" size="0.875rem" />
        </Button>
      </Tip>
    </>
  )
}

function ConversationPill({
  disabled,
  level,
  muted,
  onEnd,
  onStopTurn,
  onToggleMute,
  status
}: ConversationProps & { disabled: boolean }) {
  const { t } = useI18n()
  const c = t.composer
  const speaking = status === 'speaking'
  const listening = status === 'listening' && !muted

  const label =
    status === 'speaking'
      ? c.speaking
      : status === 'transcribing'
        ? c.transcribing
        : status === 'thinking'
          ? c.thinking
          : muted
            ? c.muted
            : c.listening

  return (
    <div className="ml-auto flex shrink-0 items-center gap-(--composer-control-gap)">
      {/* Keep the ear visible during voice chat — shown paused, since the
          conversation holds the mic (the one time wake must not listen). */}
      <WakeWordButton disabled={disabled} pausedForVoice />
      <Tip label={muted ? c.unmuteMic : c.muteMic}>
        <Button
          aria-label={muted ? c.unmuteMic : c.muteMic}
          aria-pressed={muted}
          className={cn(GHOST_ICON_BTN, 'p-0', muted && 'bg-muted text-muted-foreground')}
          disabled={disabled}
          onClick={() => {
            triggerHaptic('selection')
            onToggleMute()
          }}
          size="icon"
          type="button"
          variant="ghost"
        >
          <Codicon name={muted ? 'mic-off' : 'mic'} size="1rem" />
        </Button>
      </Tip>
      {listening && (
        <Button
          aria-label={c.stopListening}
          className="h-(--composer-control-size) shrink-0 gap-1.5 rounded-full px-2.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
          disabled={disabled}
          onClick={() => {
            triggerHaptic('submit')
            onStopTurn()
          }}
          type="button"
          variant="ghost"
        >
          <Square className={cn('fill-current', iconSize.xs)} />
          <span>{c.stopShort}</span>
        </Button>
      )}
      <Button
        aria-label={c.endConversation}
        className="h-(--composer-control-size) gap-1.5 rounded-full bg-primary px-3 text-xs font-medium text-primary-foreground hover:bg-primary/90"
        disabled={disabled}
        onClick={() => {
          triggerHaptic('close')
          onEnd()
        }}
        type="button"
      >
        <ConversationIndicator level={level} listening={listening} speaking={speaking} />
        <span>{c.endShort}</span>
      </Button>
      <span className="sr-only" role="status">
        {label}
      </span>
    </div>
  )
}

function ConversationIndicator({
  level,
  listening,
  speaking
}: {
  level: number
  listening: boolean
  speaking: boolean
}) {
  if (speaking) {
    return <Loader2 className={cn('animate-spin', iconSize.xs)} />
  }

  const bars = [0.55, 0.85, 1, 0.85, 0.55]
  const normalized = Math.max(0, Math.min(level, 1))

  return (
    <span aria-hidden="true" className="flex h-3 items-center gap-0.5">
      {bars.map((weight, index) => {
        const height = listening ? 0.3 + Math.min(0.7, normalized * weight) : 0.3

        return <span className="w-0.5 rounded-full bg-current" key={index} style={{ height: `${height * 100}%` }} />
      })}
    </span>
  )
}

// Pure-TTS toggle: type normally, but have every assistant reply read aloud —
// no dictation, no full conversation loop. Filled/accent when on, mirroring the
// muted-mic pressed state above. Driven by (and persisted to) `voice.auto_tts`.
function AutoSpeakButton({ active, disabled, onToggle }: { active: boolean; disabled: boolean; onToggle: () => void }) {
  const { t } = useI18n()
  const c = t.composer
  const label = active ? c.stopSpeakingReplies : c.speakReplies

  return (
    <Tip label={label}>
      <Button
        aria-label={label}
        aria-pressed={active}
        className={cn(GHOST_ICON_BTN, 'p-0', active && ACTIVE_ICON_BTN)}
        disabled={disabled}
        onClick={() => {
          triggerHaptic(active ? 'close' : 'open')
          onToggle()
        }}
        size="icon"
        type="button"
        variant="ghost"
      >
        {active ? <Volume2 className={iconSize.sm} /> : <VolumeX className={iconSize.sm} />}
      </Button>
    </Tip>
  )
}

// "Hey Hermes" wake-word toggle. ALWAYS rendered — the ear never hides. A
// user must always be able to click it to turn passive listening on; if the
// backend can't start (missing STT/TTS, deps still installing, no mic
// permission, etc.) the click surfaces the reason in the tooltip and the
// toggle stays off. States: listening (accent-highlighted), off (muted
// ear-off), and paused-for-voice (disabled while a voice conversation holds
// the mic — the one time wake genuinely must not listen). Backend refusals
// ({started:false, reason}) keep the toggle off and put the reason/hint in
// the tooltip.
function WakeWordButton({ disabled, pausedForVoice = false }: { disabled: boolean; pausedForVoice?: boolean }) {
  const { t } = useI18n()
  const c = t.composer
  const wake = useStore($wakeWord)

  const phrase = wake.phrase || 'hey hermes'

  const label = pausedForVoice
    ? c.wakeWordPausedVoice(phrase)
    : wake.listening
      ? c.wakeWordListening(phrase)
      : c.wakeWordOff(phrase)

  const tooltip = !pausedForVoice && wake.notice ? `${label} — ${wake.notice}` : label

  return (
    <Tip label={tooltip}>
      <Button
        aria-label={label}
        aria-pressed={wake.listening && !pausedForVoice}
        className={cn(GHOST_ICON_BTN, 'p-0', wake.listening && !pausedForVoice && ACTIVE_ICON_BTN)}
        disabled={disabled || pausedForVoice || wake.pending}
        onClick={() => {
          triggerHaptic(wake.listening ? 'close' : 'open')
          void toggleWakeWord()
        }}
        size="icon"
        type="button"
        variant="ghost"
      >
        {wake.listening && !pausedForVoice ? <Ear className={iconSize.sm} /> : <EarOff className={iconSize.sm} />}
      </Button>
    </Tip>
  )
}

// Continuous Work toggle: when active, the agent is instructed to keep working
// until all tasks are done or perfection is certified. Uses the Zap icon
// (bolt) with accent highlight when active, mirroring the wake-word pattern.
// Syncs with backend config via POST /api/agent/continuous-work.
function ContinuousWorkButton({ disabled }: { disabled: boolean }) {
  const { t } = useI18n()
  const c = t.composer
  const active = useStore($continuousWork)

  const handleClick = useCallback(() => {
    triggerHaptic(active ? 'close' : 'open')
    const next = !active
    $continuousWork.set(next)
    // Sync with backend config so the next agent build picks it up
    void setContinuousWork(next).catch(() => {
      // Backend sync failed — local atom is still set, will retry on next toggle
    })
  }, [active])

  return (
    <Tip label={active ? c.continuousWorkActive : c.continuousWorkOff}>
      <Button
        aria-label={active ? c.continuousWorkActive : c.continuousWorkOff}
        aria-pressed={active}
        className={cn(
          'h-(--composer-control-size) shrink-0 gap-1.5 rounded-full px-2.5 text-xs font-medium',
          active
            ? 'bg-primary/15 text-primary hover:bg-primary/25'
            : 'bg-muted text-muted-foreground hover:bg-muted/80 hover:text-foreground'
        )}
        disabled={disabled}
        onClick={handleClick}
        type="button"
        variant="ghost"
      >
        <RefreshCw className={cn(iconSize.sm, active && 'animate-spin')} />
        <span className="hidden sm:inline">{active ? 'On' : 'Off'}</span>
      </Button>
    </Tip>
  )
}

function DictationButton({
  disabled,
  state,
  status,
  onToggle
}: {
  disabled: boolean
  state: ChatBarState['voice']
  status: VoiceStatus
  onToggle: () => void
}) {
  const { t } = useI18n()
  const c = t.composer
  const active = state.active || status !== 'idle'

  const aria =
    status === 'recording' ? c.stopDictation : status === 'transcribing' ? c.transcribingDictation : c.voiceDictation

  return (
    <Tip label={aria}>
      <Button
        aria-label={aria}
        aria-pressed={active}
        className={cn(
          GHOST_ICON_BTN,
          'p-0',
          'data-[active=true]:bg-accent data-[active=true]:text-foreground',
          status === 'recording' && ACTIVE_ICON_BTN,
          status === 'transcribing' && 'bg-primary/10 text-primary'
        )}
        data-active={active}
        disabled={disabled || !state.enabled || status === 'transcribing'}
        onClick={() => {
          triggerHaptic(active ? 'close' : 'open')
          onToggle()
        }}
        size="icon"
        type="button"
        variant="ghost"
      >
        {status === 'recording' ? (
          <Square className={cn('fill-current', iconSize.xs)} />
        ) : status === 'transcribing' ? (
          <Loader2 className={cn('animate-spin', iconSize.sm)} />
        ) : (
          <Codicon name="mic" size="0.875rem" />
        )}
      </Button>
    </Tip>
  )
}

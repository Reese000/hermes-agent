import { useStore } from '@nanostores/react'
import { useCallback, useMemo } from 'react'

import type { StatusbarItem } from '@/app/shell/statusbar-controls'
import {
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { Zap, ZapFilled } from '@/lib/icons'
import { $continuousWorkBySession, setContinuousWorkForSession } from '@/store/continuous-work'

/**
 * Per-conversation continuous-work statusbar item. Reads and writes the
 * ACTIVE session's flag (keyed by runtime session id), so one chat's toggle
 * never leaks into another.
 */
export function useContinuousWorkStatusbarItem(sessionId: string | null): StatusbarItem {
  const { t } = useI18n()
  const copy = t.composer
  const active = useStore($continuousWorkBySession)[sessionId ?? ''] ?? false

  const setEnabled = useCallback(
    (enabled: boolean) => {
      setContinuousWorkForSession(sessionId, enabled)
    },
    [sessionId]
  )

  const toggle = useMemo(() => () => setEnabled(!active), [active, setEnabled])

  return {
    className: active ? 'bg-(--chrome-action-hover) text-foreground' : undefined,
    icon: active ? <ZapFilled className="size-3.5" /> : <Zap className="size-3.5 opacity-70" />,
    id: 'continuous-work',
    label: active ? 'CW On' : 'CW Off',
    menuAlign: 'end',
    menuClassName: 'w-64 p-1',
    menuContent: (
      <>
        <DropdownMenuLabel>{copy.continuousWork}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuRadioGroup
          onValueChange={value => setEnabled(value === 'on')}
          value={active ? 'on' : 'off'}
        >
          <DropdownMenuRadioItem value="on">
            <span className="min-w-0 flex-1">{copy.continuousWorkActive}</span>
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="off">
            <span className="min-w-0 flex-1">{copy.continuousWorkOff}</span>
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </>
    ),
    onSelect: toggle,
    title: active ? copy.continuousWorkActive : copy.continuousWorkOff,
    variant: 'menu'
  }
}

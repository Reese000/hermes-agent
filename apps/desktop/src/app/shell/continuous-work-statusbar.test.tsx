import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it } from 'vitest'

import { StatusbarControls } from '@/app/shell/statusbar-controls'
import {
  $continuousWorkBySession,
  continuousWorkForSession,
  setContinuousWorkForSession
} from '@/store/continuous-work'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { useContinuousWorkStatusbarItem } from './continuous-work-statusbar'

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
})

afterEach(() => {
  cleanup()
  $continuousWorkBySession.set({})
})

function Harness({ sessionId }: { sessionId: string | null }) {
  const item = useContinuousWorkStatusbarItem(sessionId)

  return (
    <MemoryRouter>
      <StatusbarControls items={[item]} />
    </MemoryRouter>
  )
}

describe('continuous work statusbar item (per-conversation)', () => {
  it('renders a visible trigger with CW Off when the session flag is absent', () => {
    render(<Harness sessionId="session-a" />)

    const statusbar = screen.getByRole('contentinfo')
    expect(within(statusbar).getByRole('button', { name: /cw off/i })).toBeTruthy()
  })

  it('uses the shared menu trigger (variant: menu) with aria-haspopup', async () => {
    render(<Harness sessionId="session-a" />)

    const statusbar = screen.getByRole('contentinfo')
    const trigger = within(statusbar).getByRole('button', { name: /cw off/i })
    expect(trigger.getAttribute('aria-haspopup')).toBe('menu')

    fireEvent.pointerDown(trigger, { button: 0 })
    expect(await screen.findByRole('menuitemradio', { name: /keep working/i })).toBeTruthy()
    expect(screen.getByRole('menuitemradio', { name: /mode off/i })).toBeTruthy()
  })

  it('toggles only the target session and updates its own label', async () => {
    render(<Harness sessionId="session-a" />)

    const statusbar = screen.getByRole('contentinfo')
    fireEvent.pointerDown(within(statusbar).getByRole('button', { name: /cw off/i }), { button: 0 })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: /keep working/i }))

    expect(continuousWorkForSession('session-a')).toBe(true)
    expect(continuousWorkForSession('session-b')).toBe(false)
    expect(within(statusbar).getByRole('button', { name: /cw on/i })).toBeTruthy()
  })

  it('is isolated: setting session A never sets session B', () => {
    setContinuousWorkForSession('session-a', true)

    expect(continuousWorkForSession('session-a')).toBe(true)
    expect(continuousWorkForSession('session-b')).toBe(false)

    // The atom holds only session-a.
    expect(Object.keys($continuousWorkBySession.get())).toEqual(['session-a'])
  })

  it('clears a session flag when toggled off', () => {
    setContinuousWorkForSession('session-a', true)
    setContinuousWorkForSession('session-a', false)

    expect(continuousWorkForSession('session-a')).toBe(false)
    expect(Object.keys($continuousWorkBySession.get())).toEqual([])
  })
})

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it } from 'vitest'

import { StatusbarControls } from '@/app/shell/statusbar-controls'
import { $continuousWork } from '@/store/continuous-work'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { useContinuousWorkStatusbarItem } from './continuous-work-statusbar'

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
})

afterEach(() => {
  cleanup()
  $continuousWork.set(false)
})

function Harness() {
  const item = useContinuousWorkStatusbarItem()

  return (
    <MemoryRouter>
      <StatusbarControls items={[item]} />
    </MemoryRouter>
  )
}

describe('continuous work statusbar item', () => {
  it('renders a visible trigger button with the CW Off label', () => {
    render(<Harness />)

    const statusbar = screen.getByRole('contentinfo')
    const trigger = within(statusbar).getByRole('button', { name: /cw off/i })
    expect(trigger).toBeTruthy()
    expect(within(statusbar).getAllByRole('button')).toHaveLength(1)
  })

  it('uses the shared statusbar menu trigger (variant: menu) — not a bare button', async () => {
    render(<Harness />)

    const statusbar = screen.getByRole('contentinfo')
    const trigger = within(statusbar).getByRole('button', { name: /cw off/i })

    // The root cause of the invisible toggle was a missing `variant: 'menu'`.
    // The shared trigger carries aria-haspopup only when the menu variant is set.
    expect(trigger.getAttribute('aria-haspopup')).toBe('menu')

    fireEvent.pointerDown(trigger, { button: 0 })

    expect(await screen.findByRole('menuitemradio', { name: /keep working/i })).toBeTruthy()
    expect(screen.getByRole('menuitemradio', { name: /mode off/i })).toBeTruthy()
  })

  it('toggles on through the menu and updates the trigger label', async () => {
    render(<Harness />)

    const statusbar = screen.getByRole('contentinfo')
    fireEvent.pointerDown(within(statusbar).getByRole('button', { name: /cw off/i }), { button: 0 })

    fireEvent.click(await screen.findByRole('menuitemradio', { name: /keep working/i }))

    expect($continuousWork.get()).toBe(true)
    expect(within(statusbar).getByRole('button', { name: /cw on/i })).toBeTruthy()
  })
})

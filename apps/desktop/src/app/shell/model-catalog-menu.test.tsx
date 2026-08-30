import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import {
  $modelVisibilityOpen,
  $visibleModels,
  modelVisibilityKey,
  setModelVisibilityOpen,
  setVisibleModels
} from '@/store/model-visibility'

import { ModelCatalogMenu, type ModelMenuController } from './model-catalog-menu'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const getGlobalModelOptions = vi.fn()
const searchProviderModels = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args),
  searchProviderModels: (...args: unknown[]) => searchProviderModels(...args),
  setApiRequestProfile: vi.fn()
}))

beforeEach(() => {
  $visibleModels.set(null)
  setModelVisibilityOpen(false)
  getGlobalModelOptions.mockResolvedValue({
    providers: [{ models: ['gemini-3.1-pro', 'gemini-2.5-flash'], name: 'Google', slug: 'google' }]
  })
  // Default: no openrouter provider in most tests, so the live-search effect
  // never fires — set anyway so an unexpected call fails loudly instead of
  // hanging the test on an unresolved promise.
  searchProviderModels.mockResolvedValue({ models: [] })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

// A minimal controller — these tests are about the CATALOG's own behaviour
// (what it lists, what it offers), not about what any host does with a pick.
function renderMenu() {
  const select = vi.fn()

  const controller: ModelMenuController = {
    applyPreset: vi.fn(),
    current: { effort: '', fast: false, model: '', provider: '' },
    presetFor: () => ({}),
    select,
    setOptions: vi.fn()
  }

  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <DropdownMenu open>
        <DropdownMenuContent>
          <ModelCatalogMenu controller={controller} />
        </DropdownMenuContent>
      </DropdownMenu>
    </QueryClientProvider>
  )

  return select
}

// Curation is ONE global preference, so it belongs to the catalog rather than
// to whichever surface mounted it. If a host had to opt in, the composer and
// the kanban board would end up disagreeing about what "my models" means —
// which is exactly the drift extracting this component was meant to prevent.
describe('the catalog owns model curation', () => {
  it('honours the stored Edit Models shortlist', async () => {
    setVisibleModels(new Set([modelVisibilityKey('google', 'gemini-2.5-flash')]))

    renderMenu()

    await screen.findByText(/Gemini 2\.5 Flash/i)
    expect(screen.queryByText(/Gemini 3\.1 Pro/i)).toBeNull()
  })

  it('still finds a hidden model by search — curation narrows the default view, not the catalog', async () => {
    setVisibleModels(new Set([modelVisibilityKey('google', 'gemini-2.5-flash')]))

    renderMenu()
    await screen.findByText(/Gemini 2\.5 Flash/i)

    const input = screen.getByRole('textbox', { name: 'Search models' })

    fireEvent.change(input, { target: { value: 'gemini-3.1' } })

    await vi.waitFor(() => {
      expect(screen.queryByText(/Gemini 3\.1 Pro/i)).not.toBeNull()
    })
  })

  it('offers Edit Models without the host wiring it up', async () => {
    renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    fireEvent.click(screen.getByText('Edit models…'))

    expect($modelVisibilityOpen.get()).toBe(true)
  })
})

// The live-search debounce + focus-preservation logic shipped with two
// follow-up fix commits in its history (surfacing errors, then preserving
// focus against Radix's roving-focus-group) and had zero regression
// coverage before this — nothing here would have caught either bug
// recurring.
describe('live OpenRouter search', () => {
  beforeEach(() => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        { models: ['gemini-3.1-pro', 'gemini-2.5-flash'], name: 'Google', slug: 'google' },
        { models: ['anthropic/claude-sonnet-4'], name: 'OpenRouter', slug: 'openrouter' }
      ]
    })
  })

  it('fetches and shows live results for a query the curated catalog does not have', async () => {
    searchProviderModels.mockResolvedValue({ models: ['mistralai/mixtral-8x22b'] })

    renderMenu()
    await screen.findByText(/Gemini 2\.5 Flash/i)

    const input = screen.getByRole('textbox', { name: 'Search models' })
    fireEvent.change(input, { target: { value: 'mixtral' } })

    // HighlightMatches splits the match into separate text/mark nodes, so
    // match against the full rendered text rather than a single node.
    await vi.waitFor(() => {
      expect(document.body.textContent).toMatch(/mixtral-8x22b/i)
    }, { timeout: 2000 })
    expect(searchProviderModels).toHaveBeenCalledWith('openrouter', 'mixtral')
  })

  it('does not steal focus from the search input while live results load and arrive', async () => {
    let resolveSearch: ((v: { models: string[] }) => void) | undefined

    searchProviderModels.mockImplementation(
      () => new Promise(resolve => { resolveSearch = resolve })
    )

    renderMenu()
    await screen.findByText(/Gemini 2\.5 Flash/i)

    const input = screen.getByRole('textbox', { name: 'Search models' })
    input.focus()
    fireEvent.change(input, { target: { value: 'mixtral' } })

    // Loading skeleton mounts while the fetch is in flight — must not move
    // focus off the input the user is still typing in.
    await vi.waitFor(() => expect(searchProviderModels).toHaveBeenCalled(), { timeout: 2000 })
    expect(document.activeElement).toBe(input)

    resolveSearch?.({ models: ['mistralai/mixtral-8x22b'] })
    await vi.waitFor(() => {
      expect(document.body.textContent).toMatch(/mixtral-8x22b/i)
    }, { timeout: 2000 })
    // Results replacing the skeleton is exactly the DOM mutation that used
    // to yank focus via Radix's roving-focus-group (issue fixed upstream of
    // this test) — assert it stays put.
    expect(document.activeElement).toBe(input)
  })

  it('surfaces a live-search failure inline without crashing', async () => {
    searchProviderModels.mockRejectedValue(new Error('network down'))

    renderMenu()
    await screen.findByText(/Gemini 2\.5 Flash/i)

    const input = screen.getByRole('textbox', { name: 'Search models' })
    fireEvent.change(input, { target: { value: 'mixtral' } })

    await screen.findByText(/network down/i, undefined, { timeout: 2000 })
  })

  it('never queries live search for a provider outside openrouter', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [{ models: ['gemini-3.1-pro'], name: 'Google', slug: 'google' }]
    })

    renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    const input = screen.getByRole('textbox', { name: 'Search models' })
    fireEvent.change(input, { target: { value: 'mixtral' } })

    // Give the 300ms debounce window a chance to fire if it were going to.
    await new Promise(resolve => setTimeout(resolve, 400))
    expect(searchProviderModels).not.toHaveBeenCalled()
  })
})

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { ModelPickerDialog } from './model-picker'

// Radix Dialog + cmdk call scrollIntoView / pointer-capture / ResizeObserver
// APIs jsdom lacks.
class TestResizeObserver {
  disconnect() {}
  observe() {}
  unobserve() {}
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const getGlobalModelOptions = vi.fn()
const searchProviderModels = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args),
  searchProviderModels: (...args: unknown[]) => searchProviderModels(...args)
}))

beforeEach(() => {
  getGlobalModelOptions.mockResolvedValue({
    providers: [
      { models: ['gemini-3.1-pro', 'gemini-2.5-flash'], name: 'Google', slug: 'google' },
      { models: ['anthropic/claude-sonnet-4'], name: 'OpenRouter', slug: 'openrouter' }
    ]
  })
  // Default: fail loudly instead of hanging on an unresolved promise if a
  // test forgets to stub a call it triggers.
  searchProviderModels.mockResolvedValue({ models: [] })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderPicker() {
  const onSelect = vi.fn()
  const onOpenChange = vi.fn()
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <ModelPickerDialog
        currentModel="gemini-3.1-pro"
        currentProvider="google"
        onOpenChange={onOpenChange}
        onSelect={onSelect}
        open
      />
    </QueryClientProvider>
  )

  return { onOpenChange, onSelect }
}

// The debounced live-search + dedup-against-curated logic (model-picker.tsx)
// had zero regression coverage before this, despite being the same feature
// family as model-catalog-menu.tsx's live search (which shipped two
// focus/error follow-up fixes). Nothing here would have caught a regression
// in the fetch, dedup, or error path.
describe('live OpenRouter search', () => {
  it('fetches and shows live results not already in the curated list', async () => {
    searchProviderModels.mockResolvedValue({ models: ['mistralai/mixtral-8x22b'] })

    renderPicker()
    await screen.findByText(/gemini-2\.5-flash/i)

    const input = screen.getByPlaceholderText('Filter providers and models...')

    fireEvent.change(input, { target: { value: 'mixtral' } })

    await vi.waitFor(() => {
      expect(document.body.textContent).toMatch(/mixtral-8x22b/i)
    }, { timeout: 2000 })
    expect(searchProviderModels).toHaveBeenCalledWith('openrouter', 'mixtral')
  })

  it('filters out live results already present in the curated list, case-insensitively', async () => {
    // Curated openrouter list has 'anthropic/claude-sonnet-4' — the live
    // fetch echoing it back (in a different case) must be deduped, not
    // shown as a duplicate second entry.
    searchProviderModels.mockResolvedValue({
      models: ['Anthropic/Claude-Sonnet-4', 'mistralai/mixtral-8x22b']
    })

    renderPicker()
    await screen.findByText(/gemini-2\.5-flash/i)

    const input = screen.getByPlaceholderText('Filter providers and models...')

    fireEvent.change(input, { target: { value: 'claude' } })

    await vi.waitFor(() => {
      expect(document.body.textContent).toMatch(/mixtral-8x22b/i)
    }, { timeout: 2000 })

    // Only the curated occurrence should be present — no second "Live
    // Search" row for the same model under a different case. HighlightMatches
    // splits the matched substring into sibling text/mark nodes, so count
    // occurrences in the full rendered text rather than querying by node.
    expect(screen.queryByText(/No live results/i)).toBeNull()
    const occurrences = document.body.textContent?.match(/claude-sonnet-4/gi) ?? []

    expect(occurrences).toHaveLength(1)
  })

  it('surfaces a live-search failure inline without crashing', async () => {
    searchProviderModels.mockRejectedValue(new Error('network down'))

    renderPicker()
    await screen.findByText(/gemini-2\.5-flash/i)

    const input = screen.getByPlaceholderText('Filter providers and models...')

    fireEvent.change(input, { target: { value: 'mixtral' } })

    await screen.findByText(/network down/i, undefined, { timeout: 2000 })
  })

  it('never queries live search for a provider outside openrouter', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [{ models: ['gemini-3.1-pro'], name: 'Google', slug: 'google' }]
    })

    renderPicker()
    await screen.findByText(/gemini-3\.1-pro/i)

    const input = screen.getByPlaceholderText('Filter providers and models...')

    fireEvent.change(input, { target: { value: 'mixtral' } })

    // Give the 300ms debounce window a chance to fire if it were going to.
    await new Promise(resolve => setTimeout(resolve, 400))
    expect(searchProviderModels).not.toHaveBeenCalled()
  })

  it('does not query live search for a blank query', async () => {
    renderPicker()
    await screen.findByText(/gemini-2\.5-flash/i)

    const input = screen.getByPlaceholderText('Filter providers and models...')

    fireEvent.change(input, { target: { value: '   ' } })

    await new Promise(resolve => setTimeout(resolve, 400))
    expect(searchProviderModels).not.toHaveBeenCalled()
  })

  it('debounces rapid typing into a single request for the final query', async () => {
    renderPicker()
    await screen.findByText(/gemini-2\.5-flash/i)

    const input = screen.getByPlaceholderText('Filter providers and models...')

    fireEvent.change(input, { target: { value: 'm' } })
    fireEvent.change(input, { target: { value: 'mi' } })
    fireEvent.change(input, { target: { value: 'mix' } })

    await new Promise(resolve => setTimeout(resolve, 400))
    expect(searchProviderModels).toHaveBeenCalledTimes(1)
    expect(searchProviderModels).toHaveBeenCalledWith('openrouter', 'mix')
  })
})

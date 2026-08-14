import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ConfigFieldSchema } from '@/types/hermes'

import { ConfigField } from './config-field'
import { rankSearchOption, SearchableSelect } from './searchable-select'

// Radix Popover + cmdk call scrollIntoView / pointer-capture / ResizeObserver
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

const searchProviderModels = vi.fn()

vi.mock('@/hermes', () => ({
  searchProviderModels: (provider: string, query: string) => searchProviderModels(provider, query)
}))

beforeEach(() => {
  searchProviderModels.mockResolvedValue({ models: [] })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('rankSearchOption', () => {
  it('ranks a final-segment match above a mid-path match', () => {
    // "york" hits the city segment of America/New_York (score 2) but only a
    // mid-path segment of America/New_York/Special (score 1).
    expect(rankSearchOption('America/New_York', 'york')).toBe(2)
    expect(rankSearchOption('America/New_York/Special', 'york')).toBe(1)
    expect(rankSearchOption('America/New_York', 'york')).toBeGreaterThan(
      rankSearchOption('America/New_York/Special', 'york')
    )
  })

  it('is case-insensitive', () => {
    expect(rankSearchOption('Asia/Kolkata', 'KOLKATA')).toBe(2)
    expect(rankSearchOption('ASIA/KOLKATA', 'kolkata')).toBe(2)
  })

  it('scores a substring match anywhere as 1', () => {
    expect(rankSearchOption('America/New_York', 'amer')).toBe(1)
  })

  it('scores a slashless option by plain substring', () => {
    expect(rankSearchOption('UTC', 'ut')).toBe(1)
    expect(rankSearchOption('UTC', 'xyz')).toBe(0)
  })

  it('scores a non-match as 0', () => {
    expect(rankSearchOption('Europe/Berlin', 'tokyo')).toBe(0)
  })
})

describe('SearchableSelect', () => {
  const options = ['America/New_York', 'Asia/Kolkata', 'Europe/Berlin', 'UTC']

  it('opens, filters, and selects an option', () => {
    const onChange = vi.fn()

    render(<SearchableSelect onChange={onChange} options={options} placeholder="Search…" value="" />)

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.change(screen.getByPlaceholderText('Search…'), { target: { value: 'kolkata' } })
    fireEvent.click(screen.getByText('Asia/Kolkata'))

    expect(onChange).toHaveBeenCalledWith('Asia/Kolkata')
  })

  it('renders the clear item when clearLabel is set and selecting it resets to blank', () => {
    const onChange = vi.fn()

    render(<SearchableSelect clearLabel="System default" onChange={onChange} options={options} value="Asia/Kolkata" />)

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.click(screen.getByText('System default'))

    expect(onChange).toHaveBeenCalledWith('')
  })

  it('omits the clear item without clearLabel', () => {
    render(<SearchableSelect onChange={vi.fn()} options={options} value="" />)

    fireEvent.click(screen.getByRole('combobox'))

    expect(screen.queryByText('System default')).toBeNull()
  })

  it('shows the placeholder when the value is blank', () => {
    render(<SearchableSelect onChange={vi.fn()} options={options} placeholder="Search…" value="" />)

    expect(screen.getByRole('combobox').textContent).toContain('Search…')
  })
})

// Live OpenRouter search (liveSearchProvider) had zero regression coverage
// before this. Of particular concern: the live CommandItems carry a
// `__live__`-prefixed cmdk value (to keep them distinct from curated items),
// and this Command uses cmdk's own `filter={rankSearchOption}` — unlike the
// dropdown-menu-based live search elsewhere, which disables cmdk filtering
// entirely and filters manually. If the prefix ever broke rankSearchOption's
// scoring, cmdk would silently hide the very results it just fetched.
describe('SearchableSelect live search', () => {
  const options = ['anthropic/claude-sonnet-4']

  it('fetches and shows live results not already in the curated list', async () => {
    searchProviderModels.mockResolvedValue({ models: ['mistralai/mixtral-8x22b'] })

    render(
      <SearchableSelect liveSearchProvider="openrouter" onChange={vi.fn()} options={options} value="" />
    )

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.change(screen.getByPlaceholderText('Search…'), { target: { value: 'mixtral' } })

    await vi.waitFor(() => {
      expect(document.body.textContent).toMatch(/mixtral-8x22b/i)
    }, { timeout: 2000 })
    expect(searchProviderModels).toHaveBeenCalledWith('openrouter', 'mixtral')
  })

  it('selecting a live result reports the bare model id, not the internal __live__ value', async () => {
    searchProviderModels.mockResolvedValue({ models: ['mistralai/mixtral-8x22b'] })
    const onChange = vi.fn()

    render(
      <SearchableSelect liveSearchProvider="openrouter" onChange={onChange} options={options} value="" />
    )

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.change(screen.getByPlaceholderText('Search…'), { target: { value: 'mixtral' } })

    const result = await screen.findByText(/mixtral-8x22b/i, undefined, { timeout: 2000 })

    fireEvent.click(result)
    expect(onChange).toHaveBeenCalledWith('mistralai/mixtral-8x22b')
  })

  it('filters out live results already present in the curated list, case-insensitively', async () => {
    searchProviderModels.mockResolvedValue({
      models: ['Anthropic/Claude-Sonnet-4', 'mistralai/mixtral-8x22b']
    })

    render(
      <SearchableSelect liveSearchProvider="openrouter" onChange={vi.fn()} options={options} value="" />
    )

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.change(screen.getByPlaceholderText('Search…'), { target: { value: 'claude' } })

    await vi.waitFor(() => {
      expect(document.body.textContent).toMatch(/claude-sonnet-4/i)
    }, { timeout: 2000 })

    // Only the curated occurrence should render — no duplicate "Live Search"
    // entry for the same model under a different case.
    const occurrences = document.body.textContent?.match(/claude-sonnet-4/gi) ?? []

    expect(occurrences).toHaveLength(1)
  })

  it('surfaces a live-search failure inline without crashing', async () => {
    searchProviderModels.mockRejectedValue(new Error('network down'))

    render(
      <SearchableSelect liveSearchProvider="openrouter" onChange={vi.fn()} options={options} value="" />
    )

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.change(screen.getByPlaceholderText('Search…'), { target: { value: 'mixtral' } })

    await screen.findByText(/network down/i, undefined, { timeout: 2000 })
  })

  it('never queries live search without a liveSearchProvider', async () => {
    render(<SearchableSelect onChange={vi.fn()} options={options} value="" />)

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.change(screen.getByPlaceholderText('Search…'), { target: { value: 'mixtral' } })

    // Give the 300ms debounce window a chance to fire if it were going to.
    await new Promise(resolve => setTimeout(resolve, 400))
    expect(searchProviderModels).not.toHaveBeenCalled()
  })

  it('never queries live search for a provider other than openrouter', async () => {
    render(
      <SearchableSelect liveSearchProvider="some-other-provider" onChange={vi.fn()} options={options} value="" />
    )

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.change(screen.getByPlaceholderText('Search…'), { target: { value: 'mixtral' } })

    await new Promise(resolve => setTimeout(resolve, 400))
    expect(searchProviderModels).not.toHaveBeenCalled()
  })
})

describe('ConfigField searchable routing', () => {
  const searchableSchema: ConfigFieldSchema = {
    type: 'select',
    searchable: true,
    clearable: true,
    options: ['America/New_York', 'UTC']
  }

  it('routes searchable select schemas to SearchableSelect, not a free-text input', () => {
    const { container } = render(
      <ConfigField onChange={vi.fn()} schema={searchableSchema} schemaKey="timezone" value="UTC" />
    )

    // The searchable trigger renders; the generic free-text <Input> does not.
    expect(container.querySelector('[data-slot="searchable-select-trigger"]')).not.toBeNull()
    expect(container.querySelector('input[type="text"]')).toBeNull()
  })

  it('keeps plain string schemas on the free-text input', () => {
    const { container } = render(
      <ConfigField onChange={vi.fn()} schema={{ type: 'string' }} schemaKey="some.other.key" value="hello" />
    )

    expect(container.querySelector('[data-slot="searchable-select-trigger"]')).toBeNull()
    expect(screen.getByDisplayValue('hello')).not.toBeNull()
  })

  it('surfaces the clear item via schema.clearable and resets to blank', () => {
    const onChange = vi.fn()

    render(<ConfigField onChange={onChange} schema={searchableSchema} schemaKey="timezone" value="UTC" />)

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.click(screen.getByText('System default'))

    expect(onChange).toHaveBeenCalledWith('')
  })
})

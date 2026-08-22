import { useCallback, useEffect, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator
} from '@/components/ui/command'
import { controlVariants } from '@/components/ui/control'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Skeleton } from '@/components/ui/skeleton'
import { searchProviderModels } from '@/hermes'
import { cn } from '@/lib/utils'
import type { ModelPricing } from '@/types/hermes'

/**
 * cmdk filter score for one option. Case-insensitive substring match, with
 * the final path segment (after the last "/") ranked above matches anywhere
 * else so "york" ranks "America/New_York" over "America/New_York/Special".
 * Exported for tests.
 */
export function rankSearchOption(option: string, search: string): number {
  const lower = search.toLowerCase()
  const itemLower = option.toLowerCase()
  const slash = itemLower.lastIndexOf('/')

  if (slash !== -1 && itemLower.slice(slash + 1).includes(lower)) {
    return 2
  }

  if (itemLower.includes(lower)) {
    return 1
  }

  return 0
}

/**
 * Searchable select for large option lists (e.g. ~590 IANA timezones).
 * Built on Popover + cmdk Command — the same stack as Shadcn's Combobox.
 *
 * The trigger renders like the existing closed `<Select>` but opens into a
 * searchable Command palette. Closed-world only: the user must pick from the
 * list; arbitrary text entry is not supported.
 *
 * `ConfigField` routes here when `schema.searchable === true`.
 */
export function SearchableSelect({
  value,
  onChange,
  options,
  placeholder = 'Search…',
  emptyMessage = 'No results found.',
  clearLabel,
  liveSearchProvider,
  pricing
}: {
  value: string
  onChange: (value: string) => void
  options: readonly string[]
  placeholder?: string
  emptyMessage?: string
  /** When set, prepends a "clear" item that sets the value to ''.
   *  Matches the existing <Select> pattern of EMPTY_SELECT_VALUE + "(none)". */
  clearLabel?: string
  /** Optional provider slug to enable live model search (e.g. 'openrouter'). */
  liveSearchProvider?: string
  /** Optional pricing map keyed by model id — used to show $/Mtok prices in live results. */
  pricing?: Record<string, ModelPricing>
}) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const [liveResults, setLiveResults] = useState<string[]>([])
  const [liveLoading, setLiveLoading] = useState(false)
  const [liveError, setLiveError] = useState<string | null>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)

  // Debounced live search — fires 300ms after the user stops typing when the
  // dropdown is open and a liveSearchProvider is configured.
  useEffect(() => {
    if (!open || liveSearchProvider !== 'openrouter' || !search.trim()) {
      setLiveResults([])
      setLiveLoading(false)
      setLiveError(null)

      return
    }

    let cancelled = false
    setLiveLoading(true)
    setLiveError(null)

    const timer = window.setTimeout(() => {
      searchProviderModels('openrouter', search)
        .then(result => {
          if (!cancelled) {
            const staticModels = new Set(options.map(m => m.toLowerCase()))
            setLiveResults(result.models.filter(m => !staticModels.has(m.toLowerCase())))
          }
        })
        .catch((err: unknown) => {
          if (!cancelled) {
            setLiveResults([])
            setLiveError(err instanceof Error ? err.message : 'Live search failed')
          }
        })
        .finally(() => {
          if (!cancelled) {
            setLiveLoading(false)
          }
        })
    }, 300)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [search, open, liveSearchProvider, options])

  const handleSelect = useCallback(
    (selected: string) => {
      onChange(selected)
      setOpen(false)
    },
    [onChange]
  )

  const displayValue = value !== '' && value !== undefined ? value : placeholder

  return (
    <Popover onOpenChange={setOpen} open={open}>
      <PopoverTrigger asChild>
        <button
          aria-expanded={open}
          aria-haspopup="listbox"
          className={cn(
            controlVariants(),
            'flex items-center justify-between gap-2 whitespace-nowrap',
            !value && 'text-muted-foreground'
          )}
          data-slot="searchable-select-trigger"
          ref={triggerRef}
          role="combobox"
          type="button"
        >
          <span className="truncate">{displayValue}</span>
          <Codicon className="shrink-0 opacity-60" name={open ? 'chevron-up' : 'chevron-down'} size="1rem" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-[var(--radix-popover-trigger-width)] p-0">
        <Command filter={rankSearchOption}>
          <CommandInput autoFocus onValueChange={setSearch} placeholder={placeholder} value={search} />
          <CommandList>
            <CommandEmpty>{emptyMessage}</CommandEmpty>
            <CommandGroup>
              {clearLabel && (
                <CommandItem onSelect={() => handleSelect('')} value={clearLabel}>
                  <Codicon className={cn('mr-2 size-4', value === '' ? 'opacity-100' : 'opacity-0')} name="check" />
                  {clearLabel}
                </CommandItem>
              )}
              {options.map(option => {
                const p = pricing?.[option]

                return (
                  <CommandItem key={option} onSelect={() => handleSelect(option)} value={option}>
                    <Codicon
                      className={cn('mr-2 size-4', option === value ? 'opacity-100' : 'opacity-0')}
                      name="check"
                    />
                    <span className="flex-1 truncate">{option}</span>
                    {p && (
                      <span className="ml-2 shrink-0 text-xs text-muted-foreground">
                        {p.free ? 'free' : `${p.input} / ${p.output}`}
                      </span>
                    )}
                  </CommandItem>
                )
              })}
            </CommandGroup>
            {liveSearchProvider && (liveResults.length > 0 || liveLoading || liveError) && (
              <>
                <CommandSeparator />
                <CommandGroup heading="Live Search">
                  {liveError && !liveLoading && <div className="px-2 py-2 text-xs text-destructive">{liveError}</div>}
                  {liveLoading &&
                    Array.from({ length: 3 }).map((_, i) => (
                      <CommandItem disabled key={`skeleton-${i}`}>
                        <Skeleton className="h-4 w-full" />
                      </CommandItem>
                    ))}
                  {!liveLoading &&
                    liveResults.map(model => {
                      const p = pricing?.[model]

                      return (
                        <CommandItem key={model} onSelect={() => handleSelect(model)} value={`__live__${model}`}>
                          <Codicon
                            className={cn('mr-2 size-4', model === value ? 'opacity-100' : 'opacity-0')}
                            name="check"
                          />
                          <span className="flex-1 truncate">{model}</span>
                          {p && (
                            <span className="ml-2 shrink-0 text-xs text-muted-foreground">
                              {p.free ? 'free' : `${p.input} / ${p.output}`}
                            </span>
                          )}
                        </CommandItem>
                      )
                    })}
                </CommandGroup>
              </>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

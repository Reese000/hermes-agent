"""Point-of-use context discovery for ``read_file`` (W6b).

When the agent reads a source file, the next question is almost always the
same one: *what else touches this?*  Answering it today costs two or three
``search_files`` round trips.  This module answers it in the same tool
result, as a short footer listing the files that reference the symbols
defined here, and the files whose symbols this one references.

Design constraints this module is built around
----------------------------------------------

**Prompt caching is sacred.**  This footer rides in a per-turn *tool
result*, never in the system prompt.  Nothing here mutates the cached
prefix, swaps a toolset, or rebuilds a system block, so a long-lived
conversation keeps its cached prefix intact.  The guard tests in
``tests/agent/test_related_context.py`` enforce that: the module must not
be reachable from prompt assembly.

**The core is a narrow waist.**  This adds no model tool.  Discovery
happens at the point of use, inside a tool the agent already calls, which
is why it costs zero schema bytes on every API call.

**Never fails a read.**  Every entry point returns ``""`` on any error.  A
navigation hint is not worth failing a file read over.

**Bounded, always.**  The scan is shared with the session-start repo map
(:func:`agent.repo_map.build_index`, cached), and the emitted text is
capped by :data:`DEFAULT_CHAR_BUDGET`.

Known limits
------------

Matching is by symbol *name*, not by resolved import, because the
underlying scan is regex-based (see ``agent.repo_map`` for why parsing is
off the table here).  Two unrelated modules that both define
``format_report`` will therefore look related.  The mitigations are
ranking by the number of shared symbols, dropping short and generic names,
and saying plainly in the footer that this is a static index - not
pretending to a precision the scan cannot deliver.
"""

from __future__ import annotations

import os
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Hard cap on the emitted footer.  Roughly 150 tokens - large enough for
#: two useful lists, small enough that it cannot crowd out the file the
#: agent actually asked for.
DEFAULT_CHAR_BUDGET = 600

#: Most entries shown per direction.  Beyond a handful this stops being a
#: hint and starts being a search result the agent has to wade through.
MAX_ENTRIES = 6

#: Symbols listed alongside an outbound reference.
MAX_SYMBOLS_PER_ENTRY = 3

#: Shorter names are too weak to establish a relationship.  ``fmt``,
#: ``ref`` and friends collide across unrelated modules and produce
#: confident-looking nonsense.
MIN_SYMBOL_LEN = 4

def _rel(path: Path, root: Path) -> str:
    """Repo-relative, forward-slashed path text."""
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _same_file(a: Path, b: Path) -> bool:
    return os.path.normcase(str(a)) == os.path.normcase(str(b))


def _interesting(name: str, generic: frozenset) -> bool:
    """Whether a symbol is distinctive enough to be worth reporting.

    Private helpers and the repo's common vocabulary are dropped: a match on
    ``path`` or ``_helper`` says nothing about how two files relate.
    """
    from agent.repo_map import _NOISE_NAMES

    base = name.split(".")[-1]
    return not (
        base.startswith("_")
        or len(base) < MIN_SYMBOL_LEN
        or base in _NOISE_NAMES
        or base in generic
    )


_maps_lock = threading.Lock()
_maps_cache: Dict[str, Tuple[object, Dict[str, List[Path]], Dict[str, List[Path]]]] = {}


def _name_maps(index) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """Return ``(referenced_by, defined_in)`` inverted indexes for *index*.

    Without these, every ``read_file`` would rescan every file in the repo
    to answer one question about one file - measurably half a second on a
    two-thousand-file tree.  Built once per index and reused, keyed on the
    index's identity so a rebuilt index is never answered from stale maps.
    """
    key = os.path.normcase(str(index.root))
    with _maps_lock:
        hit = _maps_cache.get(key)
        if hit is not None and hit[0] is index:
            return hit[1], hit[2]

    referenced_by: Dict[str, List[Path]] = defaultdict(list)
    defined_in: Dict[str, List[Path]] = defaultdict(list)
    for path, (defined, referenced) in index.per_file.items():
        for name in referenced:
            referenced_by[name].append(path)
        for name in defined:
            defined_in[name.split(".")[-1]].append(path)

    with _maps_lock:
        _maps_cache[key] = (index, referenced_by, defined_in)
    return referenced_by, defined_in


def _is_test_path(path: Path) -> bool:
    """Whether *path* looks like a test module."""
    lowered = [part.lower() for part in path.parts]
    return any(part in ("tests", "test") for part in lowered) or lowered[
        -1
    ].startswith("test_")


def related_context(
    path: str | Path,
    root: Optional[str | Path],
    *,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    max_entries: int = MAX_ENTRIES,
) -> str:
    """Return a related-files footer for *path*, or ``""``.

    Never raises.  Returns ``""`` when there is no index, no repo root, the
    file is not a scanned source file, or nothing interesting relates to it.
    """
    if char_budget <= 0 or max_entries <= 0:
        return ""
    try:
        return _related_context(Path(path), root, char_budget, max_entries)
    except Exception:  # noqa: BLE001 - a hint must never fail a file read.
        return ""


def _related_context(
    target: Path,
    root: Optional[str | Path],
    char_budget: int,
    max_entries: int,
) -> str:
    from agent.repo_map import _SUPPORTED_SUFFIXES, build_index

    if target.suffix not in _SUPPORTED_SUFFIXES:
        return ""

    index = build_index(root)
    if index is None:
        return ""

    # The index is keyed by the paths the walk produced; match on a
    # normalised comparison so a differently-cased or non-normalised path
    # from the caller still finds its entry.
    entry = None
    indexed_path = None
    for candidate, value in index.per_file.items():
        if _same_file(candidate, target):
            entry = value
            indexed_path = candidate
            break
    if entry is None or indexed_path is None:
        return ""

    defined, referenced = entry
    generic = index.generic

    inbound = _inbound(index, indexed_path, defined, generic)
    outbound = _outbound(index, indexed_path, referenced, generic)
    if not inbound and not outbound:
        return ""

    return _render(index.root, inbound, outbound, char_budget, max_entries)


def _inbound(index, indexed_path: Path, defined, generic) -> List[Tuple[str, int]]:
    """Files that reference symbols this file defines, most-coupled first."""
    own = {n.split(".")[-1] for n in defined if _interesting(n, generic)}
    if not own:
        return []

    referenced_by, _defined_in = _name_maps(index)
    skip_tests = not _is_test_path(indexed_path)

    counts: Counter = Counter()
    for name in own:
        for other in referenced_by.get(name, ()):
            if _same_file(other, indexed_path):
                continue
            if skip_tests and _is_test_path(other):
                continue
            counts[other] += 1

    # Strongest coupling first, path as the deterministic tie-break.  Nothing
    # caps the list length here: repo-wide infrastructure is already removed
    # by the generic-vocabulary filter in _interesting, and max_entries bounds
    # what is actually rendered.
    hits = [(_rel(other, index.root), n) for other, n in counts.items()]
    hits.sort(key=lambda h: (-h[1], h[0]))
    return hits


def _outbound(
    index, indexed_path: Path, referenced, generic
) -> List[Tuple[str, List[str]]]:
    """Files defining symbols this file references, most-coupled first."""
    refs = {n for n in set(referenced) if _interesting(n, generic)}
    if not refs:
        return []

    _referenced_by, defined_in = _name_maps(index)
    skip_tests = not _is_test_path(indexed_path)

    per_target: Dict[Path, set] = defaultdict(set)
    for name in refs:
        for other in defined_in.get(name, ()):
            if _same_file(other, indexed_path):
                continue
            if skip_tests and _is_test_path(other):
                continue
            per_target[other].add(name)

    hits = [
        (_rel(other, index.root), sorted(syms))
        for other, syms in per_target.items()
    ]
    hits.sort(key=lambda h: (-len(h[1]), h[0]))
    return hits


def _render(
    root: Path,
    inbound: List[Tuple[str, int]],
    outbound: List[Tuple[str, List[str]]],
    char_budget: int,
    max_entries: int,
) -> str:
    header = (
        "Related files (static index, may be stale - re-read before relying "
        "on it):\n"
    )
    parts: List[str] = []
    used = len(header)

    def _add(label: str, items: List[str]) -> None:
        nonlocal used
        if not items:
            return
        line = f"  {label}: {', '.join(items)}\n"
        if used + len(line) > char_budget:
            return
        parts.append(line)
        used += len(line)

    _add("referenced by", [p for p, _n in inbound[:max_entries]])
    _add(
        "references",
        [
            f"{p} ({', '.join(syms[:MAX_SYMBOLS_PER_ENTRY])})"
            for p, syms in outbound[:max_entries]
        ],
    )

    if not parts:
        return ""
    out = header + "".join(parts)
    # The budget is a guarantee, not a target.
    return out[:char_budget]

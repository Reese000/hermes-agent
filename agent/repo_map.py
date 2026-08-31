"""Ranked repository symbol map for the system prompt (W6a).

Gives the model a bird's-eye view of the codebase — which files exist and
which symbols in them matter — so it can navigate without a scavenger hunt
of ``search`` calls.  This is Aider's "repo map" idea, adapted to Hermes'
constraints.

Design constraints this module is built around
----------------------------------------------

**Prompt caching is sacred.**  The map is built once, at session start, from
:func:`agent.coding_context.RuntimeMode.system_blocks`, and never re-probed
per turn.  Output is fully deterministic — identical repo state produces
byte-identical text — so the cached prefix stays valid for the whole
conversation.

**No new dependencies, and no parsing.**  Symbols come from line-anchored
regex scans rather than tree-sitter or ``ast``.  Parsing was both the
slowest step by far and an outright crash risk - deep AST traversal faults
CPython 3.11 on Windows with an access violation no ``except`` can catch.
Only Python files are scanned today; see ``_SUPPORTED_SUFFIXES``.

**Bounded cost, always.**  Three independent limits apply, and whichever
binds first wins: a wall-clock deadline (session start must never stall on
a huge repo), a file-count cap, and a hard character budget on the emitted
text.  A repo of any size produces a bounded block or an empty one.

**Never walks hostile directories.**  Vendor trees and live runtime state
(``data/`` holds a running browser profile whose cache files are locked)
are pruned before descent, not filtered after.
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ── Limits ──────────────────────────────────────────────────────────────────

#: Hard cap on emitted characters.  ~4 chars/token, so the default is roughly
#: an 800-token block.  Truncation is explicit and marked.
DEFAULT_CHAR_BUDGET = 3200

#: Stop walking after this many candidate files, however large the repo.
DEFAULT_MAX_FILES = 1500

#: Wall-clock deadline for the whole build.  Session start must not stall.
#: Sized with headroom: the largest repo tested here finishes in ~1.1s, and
#: the margin matters because a deadline that bites mid-build would make the
#: emitted map depend on machine load rather than repo contents.
DEFAULT_DEADLINE_S = 2.5

#: Files larger than this are skipped rather than parsed.
MAX_FILE_BYTES = 400_000

#: Upper bound on source characters scanned per file for references.
MAX_SCAN_CHARS = 200_000

#: Only these are parsed for symbols.  Everything else is out of scope for v1.
_SUPPORTED_SUFFIXES = frozenset({".py"})

#: Pruned before descent.  ``data`` is not merely noise: it holds a live
#: browser profile whose cache files the daemon keeps locked.
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".env", "build", "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", "site-packages", ".idea", ".vscode", "target", "vendor",
    "coverage", "htmlcov", ".next", ".nuxt", ".cache", "data", ".backups",
    "release", "win-unpacked", "out", ".gradle", "Pods", ".terraform",
})

#: Symbol names too generic to be worth ranking or showing.
_NOISE_NAMES = frozenset({
    "main", "run", "test", "setup", "__init__", "__main__", "wrapper",
    "inner", "handler", "callback", "helper", "get", "set", "value",
})


# ── Walking ─────────────────────────────────────────────────────────────────

def _iter_source_files(
    root: Path, *, max_files: int, deadline: float
) -> List[Path]:
    """Return parseable source files under *root*, bounded and deterministic.

    Directories in :data:`_SKIP_DIRS` and dotted directories are pruned
    before descent.  Results are sorted so the caller's output is stable.
    """
    found: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if time.monotonic() > deadline or len(found) >= max_files:
            break
        # Prune in place — os.walk honours mutation of dirnames.
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        )
        for fn in sorted(filenames):
            if len(found) >= max_files:
                break
            if Path(fn).suffix in _SUPPORTED_SUFFIXES:
                found.append(Path(dirpath) / fn)
    return sorted(found)


# ── Symbol extraction ───────────────────────────────────────────────────────

def _extract_symbols(path: Path) -> Tuple[List[str], List[str]]:
    """Return ``(defined_names, referenced_names)`` for a Python file.

    Only top-level and class-level defs/classes count as *defined* — locals
    are noise in a repo map.  Any failure (syntax error, encoding problem,
    unreadable file) yields empty lists: a repo map must never be able to
    break session start.
    """
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return [], []
        src = path.read_text(encoding="utf-8", errors="replace")
        return _scan_definitions(src), sorted(_scan_names(src))
    except (Exception, RecursionError, MemoryError):
        # Deliberately broad.  A repo map is a convenience: no single file is
        # worth failing over, and this runs during prompt assembly where a
        # raise would break session start.
        return [], []


#: Top-level ``def`` / ``async def`` / ``class`` at column 0.
_TOP_DEF_RE = re.compile(
    r"^(?:async\s+)?def\s+([A-Za-z_]\w*)|^class\s+([A-Za-z_]\w*)", re.M
)
#: A method: ``def`` indented exactly one level inside a class body.
_METHOD_RE = re.compile(r"^[ \t]{4}(?:async\s+)?def\s+([A-Za-z_]\w*)", re.M)


def _scan_definitions(src: str) -> List[str]:
    """Extract defined symbols by scanning source text, not by parsing it.

    Parsing every file with ``ast`` was both the slowest step by far (several
    seconds on a large repo, blowing the session-start deadline and making
    the emitted map depend on machine timing) and a crash risk: deep AST
    traversal faulted the interpreter on Windows.  A line-anchored scan is
    ~an order of magnitude faster, cannot crash, and is accurate enough for
    a navigation aid.

    Known limits, accepted deliberately: definitions inside ``if``/``try``
    blocks or nested one level deeper than a class body are missed, and a
    ``def`` inside a triple-quoted string would be counted.  Neither changes
    what this is for - pointing the model at the right file.
    """
    defined: List[str] = []
    current_class: Optional[str] = None
    for match in _TOP_DEF_RE.finditer(src):
        func, cls = match.group(1), match.group(2)
        if cls:
            current_class = cls
            defined.append(cls)
        elif func:
            current_class = None
            defined.append(func)

    # Methods are attributed to the class that most recently opened above
    # them, which is what a one-level-indent scan gives for ordinary layout.
    class_spans = [
        (m.start(), m.group(2))
        for m in _TOP_DEF_RE.finditer(src)
        if m.group(2)
    ]
    if class_spans:
        for m in _METHOD_RE.finditer(src):
            name = m.group(1)
            if name.startswith("_"):
                continue
            owner = None
            for start, cls in class_spans:
                if start < m.start():
                    owner = cls
                else:
                    break
            if owner:
                defined.append(f"{owner}.{name}")
    return defined


#: Identifiers, for the reference scan.  Deliberately simple.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _scan_names(src: str) -> set:
    """Collect referenced identifiers from *source text*, not from the AST.

    This does not walk the syntax tree, and that is the point.  Deep AST
    traversal of this repo's larger modules crashes CPython 3.11 on Windows
    with an access violation inside ``ast.iter_child_nodes`` - a hard
    interpreter fault that no ``except`` clause can catch, which would take
    session start down with it.  Definitions still come from ``ast.parse``
    (a shallow read of ``tree.body`` only, which is safe); references come
    from this scan.

    The trade-off is precision: identifiers inside comments and strings are
    counted too.  That is acceptable here because references feed only
    *document-frequency ranking* - a name's exact count never matters, just
    roughly how widely it is mentioned - and the generic-name filter removes
    the common vocabulary this over-counts anyway.
    """
    return set(_IDENT_RE.findall(src[:MAX_SCAN_CHARS]))


# ── Ranking ─────────────────────────────────────────────────────────────────

#: A name referenced in more than this fraction of files is treated as
#: generic vocabulary (``name``, ``start``, ``path``, ``read``) rather than a
#: distinctive symbol.  Without this the map fills up with accessors, because
#: raw reference counts reward the most common words in the language, not the
#: most important symbols in the repo.
_GENERIC_DF_RATIO = 0.02

#: Minimum file count before the generic-name filter engages at all.
_GENERIC_DF_FLOOR = 8


def _reference_stats(
    per_file: Dict[Path, Tuple[List[str], List[str]]]
) -> Tuple[Counter, frozenset]:
    """Return ``(document_frequency, generic_names)``.

    Document frequency counts *files* referencing a name, not occurrences, so
    one file mentioning a symbol 500 times cannot make it look universal.
    """
    df: Counter = Counter()
    for _defined, referenced in per_file.values():
        df.update(set(referenced))

    total = max(1, len(per_file))
    # The floor matters as much as the ratio.  Set too low, a small repo has
    # its *most important* symbol - referenced by everything - dismissed as
    # generic vocabulary, and the map comes out empty.  The floor keeps the
    # filter dormant until a repo is big enough for "referenced everywhere"
    # to genuinely mean "common word" rather than "central abstraction".
    cutoff = max(_GENERIC_DF_FLOOR, int(total * _GENERIC_DF_RATIO))
    generic = frozenset(n for n, c in df.items() if c > cutoff)
    return df, generic


def _rank(
    per_file: Dict[Path, Tuple[List[str], List[str]]],
    df: Counter,
    generic: frozenset,
) -> Dict[Path, float]:
    """Score files by how much the rest of the repo depends on their symbols.

    A cheap stand-in for Aider's PageRank: a definition is important when
    *other* files reference its name.  Self-references are excluded so a
    large file cannot inflate its own rank, and generic vocabulary is
    ignored entirely.
    """
    scores: Dict[Path, float] = {}
    for path, (defined, referenced) in per_file.items():
        own = set(referenced)
        score = 0.0
        for name in defined:
            base = name.split(".")[-1]
            if base in _NOISE_NAMES or base.startswith("_") or base in generic:
                continue
            external = df.get(base, 0) - (1 if base in own else 0)
            if external > 0:
                # Top-level definitions are the useful navigation anchors;
                # methods are supporting detail.
                weight = 1.0 if "." not in name else 0.5
                score += float(external) * weight
        scores[path] = score
    return scores


def _top_symbols(
    defined: Iterable[str], df: Counter, generic: frozenset, limit: int
) -> List[str]:
    """Most-referenced defined symbols, deterministically ordered.

    Generic vocabulary is excluded, and top-level definitions outrank
    methods at equal reference counts.
    """
    scored = []
    for n in defined:
        base = n.split(".")[-1]
        if base.startswith("_") or base in _NOISE_NAMES or base in generic:
            continue
        weight = 1.0 if "." not in n else 0.5
        scored.append((-df.get(base, 0) * weight, n))
    scored.sort()
    return [n for _s, n in scored[:limit]]


# ── Public API ──────────────────────────────────────────────────────────────

def build_repo_map(
    root: Optional[str | Path],
    *,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    max_files: int = DEFAULT_MAX_FILES,
    deadline_s: float = DEFAULT_DEADLINE_S,
    symbols_per_file: int = 4,
) -> str:
    """Return a ranked repo-map block, or ``""`` when there is nothing useful.

    The result is deterministic for a given repo state and never exceeds
    *char_budget* characters.  All failure modes return ``""`` rather than
    raising — this runs during prompt assembly and must not be able to
    break session start.
    """
    if not root:
        return ""
    try:
        root_path = Path(root)
        if not root_path.is_dir():
            return ""
    except (OSError, ValueError):
        return ""

    if char_budget <= 0 or max_files <= 0:
        return ""

    deadline = time.monotonic() + max(0.0, deadline_s)

    try:
        files = _iter_source_files(
            root_path, max_files=max_files, deadline=deadline
        )
    except OSError:
        return ""
    if not files:
        return ""

    per_file: Dict[Path, Tuple[List[str], List[str]]] = {}
    for path in files:
        if time.monotonic() > deadline:
            break
        defined, referenced = _extract_symbols(path)
        if defined:
            per_file[path] = (defined, referenced)
    if not per_file:
        return ""

    df, generic = _reference_stats(per_file)
    scores = _rank(per_file, df, generic)

    # Deterministic ordering: score desc, then path asc as the tie-break.
    ordered = sorted(
        per_file.keys(),
        key=lambda p: (-scores.get(p, 0.0), str(p).replace("\\", "/")),
    )

    header = (
        "# Repository map\n"
        "Most-referenced symbols per file, ranked. A navigation aid built "
        "once at session start - it is a snapshot, not live state, so "
        "re-read a file before relying on its contents.\n"
    )
    lines: List[str] = []
    used = len(header)
    truncated = False

    for path in ordered:
        if scores.get(path, 0.0) <= 0:
            continue
        try:
            rel = path.relative_to(root_path)
        except ValueError:
            rel = path
        rel_str = str(rel).replace("\\", "/")
        syms = _top_symbols(per_file[path][0], df, generic, symbols_per_file)
        if not syms:
            continue
        line = f"{rel_str}: {', '.join(syms)}\n"
        if used + len(line) > char_budget:
            truncated = True
            break
        lines.append(line)
        used += len(line)

    if not lines:
        return ""

    out = header + "".join(lines)
    if truncated:
        marker = "(truncated)\n"
        if len(out) + len(marker) <= char_budget:
            out += marker

    # Belt and braces: the budget is a hard guarantee, not a target.
    if len(out) > char_budget:
        out = out[:char_budget]
    return out

"""Edit keep-rate ledger (W7).

Cursor measures its models by how much of what they write the user keeps.
An agentic CLI has no accept/reject button, so the analogue has to be
derived from the edit stream itself: an edit is *kept* if nothing later
undoes it.

  * **superseded** - a later edit removed exactly what this edit added.
    The model rewrote its own work: churn, not progress.
  * **reverted** - a later edit added back exactly what this edit removed.
    The model undid itself.
  * **kept** - neither happened; the change survived the window.

Both relations are exact-content matches, which is deliberate. A fuzzy
"same region" heuristic would report churn that is really refinement, and
a number nobody trusts is worse than no number.

Coverage differs by edit format, and the report should be read with that
in mind:

  * ``replace`` - fully scored; the removed and added text are both known
    exactly from the tool arguments.
  * ``write_file`` - scored as an *overwriter*: a whole-file write means
    nothing earlier on that file survived. Its own content is recorded, so
    a later write to the same file supersedes it too.
  * ``patch`` (V4A) - counted in the applied/failed rates, but its
    per-file before/after text is not available at the tool boundary, so
    it is only marked superseded by a later whole-file write. It is never
    credited as churn it did not demonstrably cause.

What is stored
--------------
Hashes, never text.  This ledger is a long-lived file in the Hermes home
directory, and the text flowing through it is the user's source code.
:func:`_digest` is the only thing that ever sees content, and only its
output is written.  A test pins this.

The ledger is passive: recording is best-effort and never blocks or fails
an edit, and nothing here reads back into the agent loop.  It exists to be
reported (``/insights``), not to steer.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

_DB_LOCK = threading.Lock()

#: Schema version, bumped when the table shape changes.
_SCHEMA_VERSION = 1

#: Rows older than this are pruned on write.  Keep-rate is a trend signal;
#: nobody asks about an edit from last quarter.
MAX_EVENT_AGE_DAYS = 30

#: Hard ceiling on stored rows, enforced after the age prune.  A runaway
#: batch job must not grow this file without bound.
MAX_EVENTS = 50_000

#: Rows read per report.  Bounds memory on a very busy history.
MAX_REPORT_ROWS = 20_000

APPLIED = "applied"
FAILED = "failed"

#: Edit formats. ``write_file`` is special-cased in :func:`_classify`
#: because it replaces a whole file rather than a region.
REPLACE = "replace"
PATCH = "patch"
WRITE_FILE = "write_file"

KEPT = "kept"
SUPERSEDED = "superseded"
REVERTED = "reverted"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cutoff(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _digest(text: Optional[str]) -> str:
    """Stable short hash of *text*; ``""`` for nothing.

    The only function in this module that touches edit content.  Truncated
    to 16 hex chars: collisions would have to occur between two edits to
    the same file in the same session to matter at all, and at that rate
    the birthday bound is astronomically slack.
    """
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _norm(path: str) -> str:
    try:
        return os.path.normcase(os.path.abspath(path))
    except (OSError, ValueError, TypeError):
        return str(path)


def _db_path() -> Path:
    return get_hermes_home() / "edit_outcomes.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS edit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            session_id TEXT NOT NULL,
            path TEXT NOT NULL,
            edit_format TEXT NOT NULL,
            status TEXT NOT NULL,
            removed_hash TEXT NOT NULL DEFAULT '',
            added_hash TEXT NOT NULL DEFAULT '',
            added_chars INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_edit_events_scope
        ON edit_events(session_id, path, id)
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(_SCHEMA_VERSION),),
    )
    conn.commit()


def _prune(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM edit_events WHERE created_at < ?", (_cutoff(MAX_EVENT_AGE_DAYS),))
    conn.execute(
        """
        DELETE FROM edit_events WHERE id NOT IN (
            SELECT id FROM edit_events ORDER BY id DESC LIMIT ?
        )
        """,
        (MAX_EVENTS,),
    )


def record_edit(
    *,
    session_id: Optional[str],
    path: str,
    edit_format: str,
    status: str,
    removed_text: Optional[str] = None,
    added_text: Optional[str] = None,
) -> bool:
    """Append one edit outcome.  Returns whether it was stored.

    Best-effort by contract: any failure here is swallowed, because an
    unwritable ledger must never turn a working edit into a failed one.
    """
    if not path or status not in (APPLIED, FAILED):
        return False
    try:
        row = (
            _utc_now(),
            str(session_id or "default"),
            _norm(path),
            str(edit_format or "unknown"),
            status,
            _digest(removed_text),
            _digest(added_text),
            len(added_text or ""),
        )
        with _DB_LOCK:
            with _connect() as conn:
                conn.execute(
                    """
                    INSERT INTO edit_events(
                        created_at, session_id, path, edit_format, status,
                        removed_hash, added_hash, added_chars
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )
                _prune(conn)
                conn.commit()
        return True
    except Exception:  # noqa: BLE001 - a ledger write must never break an edit
        return False


def _classify(rows: list) -> dict[int, str]:
    """Label each applied edit ``kept`` / ``superseded`` / ``reverted``.

    Rows must be ordered by id.  Two applied edits interact only within one
    ``(session_id, path)`` group: an edit cannot be undone by a change to a
    different file, nor by a different agent's session.
    """
    groups: dict[tuple, list] = defaultdict(list)
    for row in rows:
        if row["status"] == APPLIED:
            groups[(row["session_id"], row["path"])].append(row)

    verdicts: dict[int, str] = {}
    for group in groups.values():
        for i, row in enumerate(group):
            verdict = KEPT
            for later in group[i + 1:]:
                took_back = bool(row["added_hash"]) and (
                    later["removed_hash"] == row["added_hash"]
                )
                put_back = bool(row["removed_hash"]) and (
                    later["added_hash"] == row["removed_hash"]
                )
                # Both at once is an exact undo, which is the more specific
                # verdict: the later edit did not merely rewrite this one's
                # output, it restored the state this one replaced. Check it
                # first, or every revert would be filed as churn.
                if took_back and put_back:
                    verdict = REVERTED
                    break
                # The later edit removed exactly what this one added.
                if took_back:
                    verdict = SUPERSEDED
                    break
                # The later edit put back exactly what this one removed.
                if put_back:
                    verdict = REVERTED
                    break
                # A later whole-file write replaces the entire file, so
                # nothing an earlier edit did to it survived. This is the
                # only rule that does not need the earlier edit's hashes,
                # which is what keeps V4A patches (recorded without
                # per-file text) from being scored as kept by default.
                if (
                    later["edit_format"] == WRITE_FILE
                    and later["added_hash"] != row["added_hash"]
                ):
                    verdict = SUPERSEDED
                    break
            verdicts[row["id"]] = verdict
    return verdicts


def keep_rate_report(days: int = 30, session_id: Optional[str] = None) -> dict[str, Any]:
    """Keep-rate summary over the last *days*.

    Returns zeros rather than raising when the ledger is missing or
    unreadable - this feeds a report, not a decision.
    """
    empty: dict[str, Any] = {
        "days": days,
        "total": 0,
        "applied": 0,
        "failed": 0,
        "kept": 0,
        "superseded": 0,
        "reverted": 0,
        "keep_rate": None,
        "apply_rate": None,
        "by_format": {},
    }
    try:
        with _DB_LOCK:
            with _connect() as conn:
                if session_id:
                    rows = conn.execute(
                        """
                        SELECT * FROM edit_events
                        WHERE created_at >= ? AND session_id = ?
                        ORDER BY id LIMIT ?
                        """,
                        (_cutoff(days), str(session_id), MAX_REPORT_ROWS),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM edit_events
                        WHERE created_at >= ?
                        ORDER BY id LIMIT ?
                        """,
                        (_cutoff(days), MAX_REPORT_ROWS),
                    ).fetchall()
    except Exception:  # noqa: BLE001
        return empty

    if not rows:
        return empty

    verdicts = _classify(rows)
    applied = sum(1 for r in rows if r["status"] == APPLIED)
    failed = sum(1 for r in rows if r["status"] == FAILED)
    counts = {KEPT: 0, SUPERSEDED: 0, REVERTED: 0}
    by_format: dict[str, dict[str, int]] = {}

    for row in rows:
        fmt = row["edit_format"]
        bucket = by_format.setdefault(
            fmt, {"applied": 0, "failed": 0, KEPT: 0, SUPERSEDED: 0, REVERTED: 0}
        )
        if row["status"] == FAILED:
            bucket["failed"] += 1
            continue
        bucket["applied"] += 1
        verdict = verdicts.get(row["id"], KEPT)
        counts[verdict] += 1
        bucket[verdict] += 1

    for bucket in by_format.values():
        bucket["keep_rate"] = (
            round(bucket[KEPT] / bucket["applied"], 4) if bucket["applied"] else None
        )

    return {
        "days": days,
        "total": len(rows),
        "applied": applied,
        "failed": failed,
        "kept": counts[KEPT],
        "superseded": counts[SUPERSEDED],
        "reverted": counts[REVERTED],
        "keep_rate": round(counts[KEPT] / applied, 4) if applied else None,
        "apply_rate": round(applied / len(rows), 4) if rows else None,
        "by_format": by_format,
    }


def format_report(report: dict[str, Any]) -> list[str]:
    """Render *report* as terminal lines.  Empty list when there is nothing."""
    if not report or not report.get("total"):
        return []
    lines = ["", "EDIT OUTCOMES", "-" * 40]
    keep = report.get("keep_rate")
    apply_rate = report.get("apply_rate")
    lines.append(
        f"  Edits: {report['total']}  "
        f"applied {report['applied']}  failed {report['failed']}"
        + (f"  ({apply_rate:.0%} applied)" if apply_rate is not None else "")
    )
    if keep is not None:
        lines.append(
            f"  Kept: {report['kept']}/{report['applied']} ({keep:.0%})  "
            f"superseded {report['superseded']}  reverted {report['reverted']}"
        )
    for fmt, bucket in sorted(report.get("by_format", {}).items()):
        rate = bucket.get("keep_rate")
        lines.append(
            f"    {fmt:<12} applied {bucket['applied']:>4}  "
            f"failed {bucket['failed']:>4}"
            + (f"  kept {rate:.0%}" if rate is not None else "")
        )
    return lines

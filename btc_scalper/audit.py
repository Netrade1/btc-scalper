"""SHA-256 hash-chained JSONL audit ledger.

Every record stores the hash of the previous record, and its own hash is
computed over its canonical JSON (sorted keys, including ``prev_hash``).
Editing, deleting or re-ordering any line breaks the chain, and
``verify_ledger`` reports the first bad line. Appending to an existing file
continues its chain.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from typing import Any

GENESIS = "0" * 64


def _clean(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "item"):  # numpy scalar
        return _clean(v.item())
    return v


def _canonical(rec: dict) -> str:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str)


def _digest(rec_without_hash: dict) -> str:
    return hashlib.sha256(_canonical(rec_without_hash).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | None = None, keep_in_memory: bool = True):
        self.path = path
        self.records: list[dict[str, Any]] | None = [] if keep_in_memory else None
        self.prev_hash = GENESIS
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            if os.path.exists(path):
                ok, n, err = verify_ledger(path)
                if not ok:
                    raise RuntimeError(f"Refusing to append to a broken audit ledger {path}: {err}")
                self.prev_hash = _last_hash(path)
            self._fh = open(path, "a", encoding="utf-8")

    def log(self, event: str, bar_ts: Any = None, **payload: Any) -> dict:
        rec = _clean({"event": event, "bar_ts": bar_ts, "wall_ts": datetime.now(timezone.utc), **payload})
        rec["prev_hash"] = self.prev_hash
        rec["hash"] = _digest(rec)
        self.prev_hash = rec["hash"]
        if self.records is not None:
            self.records.append(rec)
        if self._fh:
            self._fh.write(_canonical(rec) + "\n")
            self._fh.flush()
        return rec

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def _last_hash(path: str) -> str:
    last = GENESIS
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = json.loads(line)["hash"]
    return last


def verify_ledger(path: str) -> tuple[bool, int, str]:
    """Returns (ok, records_checked, error)."""
    prev, n = GENESIS, 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return False, n, f"line {lineno}: not valid JSON"
            h = rec.pop("hash", None)
            if rec.get("prev_hash") != prev:
                return False, n, f"line {lineno}: chain broken (prev_hash mismatch)"
            if _digest(rec) != h:
                return False, n, f"line {lineno}: content hash mismatch (record altered)"
            prev, n = h, n + 1
    return True, n, ""

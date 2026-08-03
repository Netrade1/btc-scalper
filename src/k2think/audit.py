from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any


class AuditValidationError(ValueError):
    """Raised when audit events are invalid."""


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    payload: dict[str, Any]
    created_at: str

    @classmethod
    def create(cls, event_type: str, payload: dict[str, Any]) -> "AuditEvent":
        if not event_type.strip():
            raise AuditValidationError("event_type must be non-empty")
        return cls(
            event_type=event_type,
            payload=payload,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


@dataclass(frozen=True)
class ChainedAuditRecord:
    index: int
    event: AuditEvent
    previous_hash: str
    record_hash: str


class AuditChain:
    """In-memory immutable hash chain for forecast and execution evidence."""

    def __init__(self) -> None:
        self._records: list[ChainedAuditRecord] = []

    @property
    def records(self) -> list[ChainedAuditRecord]:
        return list(self._records)

    def append(self, event: AuditEvent) -> ChainedAuditRecord:
        previous_hash = self._records[-1].record_hash if self._records else "GENESIS"
        idx = len(self._records)
        record_hash = self._hash_record(idx, event, previous_hash)
        record = ChainedAuditRecord(
            index=idx,
            event=event,
            previous_hash=previous_hash,
            record_hash=record_hash,
        )
        self._records.append(record)
        return record

    def verify(self) -> bool:
        previous_hash = "GENESIS"
        for idx, record in enumerate(self._records):
            if record.index != idx:
                return False
            if record.previous_hash != previous_hash:
                return False
            expected = self._hash_record(record.index, record.event, record.previous_hash)
            if expected != record.record_hash:
                return False
            previous_hash = record.record_hash
        return True

    @staticmethod
    def _hash_record(index: int, event: AuditEvent, previous_hash: str) -> str:
        payload = {
            "index": index,
            "event": asdict(event),
            "previous_hash": previous_hash,
        }
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(normalized).hexdigest()

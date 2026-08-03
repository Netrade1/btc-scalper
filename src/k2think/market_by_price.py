from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable


class OrderBookSequenceError(ValueError):
    """Raised when market data sequence constraints are violated."""


class OrderBookValidationError(ValueError):
    """Raised when normalized market data is invalid."""


@dataclass(frozen=True)
class PriceLevelUpdate:
    sequence: int
    side: str
    price: Decimal
    size: Decimal


class MarketByPriceBook:
    """Sequence-aware market-by-price book reconstruction."""

    def __init__(self) -> None:
        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}
        self._last_sequence: int | None = None

    @property
    def last_sequence(self) -> int | None:
        return self._last_sequence

    def apply(self, update: PriceLevelUpdate) -> None:
        self._validate(update)
        self._enforce_sequence(update.sequence)

        side_book = self._bids if update.side == "bid" else self._asks
        if update.size == 0:
            side_book.pop(update.price, None)
        else:
            side_book[update.price] = update.size

        self._last_sequence = update.sequence

    def replay(self, updates: Iterable[PriceLevelUpdate]) -> None:
        for update in updates:
            self.apply(update)

    def best_bid(self) -> tuple[Decimal, Decimal] | None:
        if not self._bids:
            return None
        price = max(self._bids)
        return price, self._bids[price]

    def best_ask(self) -> tuple[Decimal, Decimal] | None:
        if not self._asks:
            return None
        price = min(self._asks)
        return price, self._asks[price]

    def snapshot(self, depth: int = 10) -> dict[str, list[tuple[Decimal, Decimal]]]:
        bids = sorted(self._bids.items(), key=lambda row: row[0], reverse=True)[:depth]
        asks = sorted(self._asks.items(), key=lambda row: row[0])[:depth]
        return {"bids": bids, "asks": asks}

    def _enforce_sequence(self, sequence: int) -> None:
        if self._last_sequence is None:
            return
        if sequence < self._last_sequence:
            raise OrderBookSequenceError("out-of-order sequence")
        if sequence > self._last_sequence + 1:
            raise OrderBookSequenceError("sequence gap detected")

    @staticmethod
    def _validate(update: PriceLevelUpdate) -> None:
        if update.side not in {"bid", "ask"}:
            raise OrderBookValidationError("side must be 'bid' or 'ask'")
        if update.price <= 0:
            raise OrderBookValidationError("price must be positive")
        if update.size < 0:
            raise OrderBookValidationError("size must be non-negative")

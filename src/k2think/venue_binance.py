from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from .market_by_price import PriceLevelUpdate


class VenueAdapterError(ValueError):
    """Raised when venue payloads are malformed."""


class BinanceDiffDepthAdapter:
    """Normalize Binance diff-depth messages into price-level updates.

    Expected payload shape:
    {
      "U": <first_update_id>,
      "u": <final_update_id>,
      "b": [[price, qty], ...],
      "a": [[price, qty], ...]
    }
    """

    def normalize(self, payload: dict[str, Any]) -> list[PriceLevelUpdate]:
        first = self._read_int(payload, "U")
        final = self._read_int(payload, "u")
        if first > final:
            raise VenueAdapterError("U cannot be greater than u")

        sequence = final
        updates: list[PriceLevelUpdate] = []
        updates.extend(self._side_updates(sequence, "bid", payload.get("b", [])))
        updates.extend(self._side_updates(sequence, "ask", payload.get("a", [])))
        return updates

    @staticmethod
    def _read_int(payload: dict[str, Any], key: str) -> int:
        if key not in payload:
            raise VenueAdapterError(f"missing field: {key}")
        try:
            return int(payload[key])
        except (TypeError, ValueError) as exc:
            raise VenueAdapterError(f"invalid integer for {key}") from exc

    @staticmethod
    def _side_updates(sequence: int, side: str, levels: list[list[str]]) -> list[PriceLevelUpdate]:
        normalized: list[PriceLevelUpdate] = []
        for row in levels:
            if len(row) != 2:
                raise VenueAdapterError("each level row must have price and quantity")
            try:
                price = Decimal(str(row[0]))
                size = Decimal(str(row[1]))
            except (InvalidOperation, ValueError) as exc:
                raise VenueAdapterError("invalid decimal in price level") from exc
            normalized.append(
                PriceLevelUpdate(sequence=sequence, side=side, price=price, size=size)
            )
        return normalized

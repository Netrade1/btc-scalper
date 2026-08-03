from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Protocol


class AdapterContractError(ValueError):
    """Raised when adapter contracts are violated."""


@dataclass(frozen=True)
class Level2Order:
    side: str
    price: Decimal
    size: Decimal
    order_id: str


@dataclass(frozen=True)
class PaperOrderRequest:
    symbol: str
    side: str
    quantity: Decimal
    client_order_id: str | None = None


@dataclass(frozen=True)
class PaperExecution:
    order_id: str
    status: str
    filled_qty: Decimal


class Level2FeedAdapter(Protocol):
    def stream(self, symbol: str) -> Iterable[Level2Order]: ...


class PaperBrokerAdapter(Protocol):
    def submit_order(self, order: PaperOrderRequest) -> PaperExecution: ...


class PaperBrokerClient(Protocol):
    def submit_paper_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: str,
        client_order_id: str | None,
    ) -> dict[str, Any]: ...


class MockLevel2FeedAdapter:
    def __init__(self, orders: Iterable[Level2Order]) -> None:
        self._orders = list(orders)

    def stream(self, symbol: str) -> Iterable[Level2Order]:
        if not symbol.strip():
            raise AdapterContractError("symbol is required")
        return list(self._orders)


class MockPaperBrokerAdapter:
    def __init__(self) -> None:
        self._counter = 0

    def submit_order(self, order: PaperOrderRequest) -> PaperExecution:
        _validate_paper_order(order)

        self._counter += 1
        return PaperExecution(
            order_id=f"paper-{self._counter}",
            status="filled",
            filled_qty=order.quantity,
        )


class OfficialPaperBrokerAdapter:
    """Paper-broker adapter that wraps an official client implementation."""

    def __init__(self, client: PaperBrokerClient) -> None:
        self._client = client

    def submit_order(self, order: PaperOrderRequest) -> PaperExecution:
        _validate_paper_order(order)

        response = self._client.submit_paper_order(
            symbol=order.symbol,
            side=order.side,
            quantity=str(order.quantity),
            client_order_id=order.client_order_id,
        )

        order_id = response.get("order_id")
        status = response.get("status")
        filled_qty_raw = response.get("filled_qty")

        if not isinstance(order_id, str) or not order_id.strip():
            raise AdapterContractError("broker response missing valid order_id")
        if status not in {"filled", "partially_filled", "rejected", "canceled", "new"}:
            raise AdapterContractError("broker response has invalid status")

        try:
            filled_qty = Decimal(str(filled_qty_raw))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise AdapterContractError("broker response has invalid filled_qty") from exc

        if filled_qty < 0:
            raise AdapterContractError("filled_qty must be non-negative")

        return PaperExecution(order_id=order_id, status=status, filled_qty=filled_qty)


def _validate_paper_order(order: PaperOrderRequest) -> None:
    if not order.symbol.strip():
        raise AdapterContractError("symbol is required")
    if order.side not in {"buy", "sell"}:
        raise AdapterContractError("side must be 'buy' or 'sell'")
    if order.quantity <= 0:
        raise AdapterContractError("quantity must be positive")

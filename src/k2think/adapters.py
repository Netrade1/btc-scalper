from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Protocol


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


@dataclass(frozen=True)
class PaperExecution:
    order_id: str
    status: str
    filled_qty: Decimal


class Level2FeedAdapter(Protocol):
    def stream(self, symbol: str) -> Iterable[Level2Order]: ...


class PaperBrokerAdapter(Protocol):
    def submit_order(self, order: PaperOrderRequest) -> PaperExecution: ...


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
        if not order.symbol.strip():
            raise AdapterContractError("symbol is required")
        if order.side not in {"buy", "sell"}:
            raise AdapterContractError("side must be 'buy' or 'sell'")
        if order.quantity <= 0:
            raise AdapterContractError("quantity must be positive")

        self._counter += 1
        return PaperExecution(
            order_id=f"paper-{self._counter}",
            status="filled",
            filled_qty=order.quantity,
        )

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


class ReconciliationError(ValueError):
    """Raised when reconciliation inputs are invalid."""


@dataclass(frozen=True)
class ExpectedOrder:
    order_id: str
    quantity: Decimal


@dataclass(frozen=True)
class ExecutionRecord:
    order_id: str
    status: str
    filled_qty: Decimal


@dataclass(frozen=True)
class ReconciliationBreak:
    order_id: str
    reason: str
    expected_qty: Decimal | None
    filled_qty: Decimal | None


@dataclass(frozen=True)
class ReconciliationReport:
    total_orders: int
    matched_orders: int
    breaks: list[ReconciliationBreak]

    @property
    def is_clean(self) -> bool:
        return len(self.breaks) == 0


class ExecutionReconciliationEngine:
    """Reconcile expected paper orders against paper executions."""

    VALID_STATUSES = {"filled", "partially_filled", "rejected", "canceled", "new"}

    def reconcile(
        self,
        expected_orders: list[ExpectedOrder],
        executions: list[ExecutionRecord],
    ) -> ReconciliationReport:
        expected_by_id: dict[str, ExpectedOrder] = {}
        for row in expected_orders:
            if not row.order_id.strip():
                raise ReconciliationError("expected order_id must be non-empty")
            if row.quantity <= 0:
                raise ReconciliationError("expected quantity must be positive")
            if row.order_id in expected_by_id:
                raise ReconciliationError("duplicate expected order_id")
            expected_by_id[row.order_id] = row

        execution_by_id: dict[str, ExecutionRecord] = {}
        for row in executions:
            if not row.order_id.strip():
                raise ReconciliationError("execution order_id must be non-empty")
            if row.status not in self.VALID_STATUSES:
                raise ReconciliationError("invalid execution status")
            if row.filled_qty < 0:
                raise ReconciliationError("execution filled_qty must be non-negative")
            if row.order_id in execution_by_id:
                raise ReconciliationError("duplicate execution order_id")
            execution_by_id[row.order_id] = row

        breaks: list[ReconciliationBreak] = []
        matched = 0

        for order_id, expected in expected_by_id.items():
            execution = execution_by_id.get(order_id)
            if execution is None:
                breaks.append(
                    ReconciliationBreak(
                        order_id=order_id,
                        reason="missing_execution",
                        expected_qty=expected.quantity,
                        filled_qty=None,
                    )
                )
                continue

            reason = self._validate_pair(expected, execution)
            if reason is None:
                matched += 1
            else:
                breaks.append(
                    ReconciliationBreak(
                        order_id=order_id,
                        reason=reason,
                        expected_qty=expected.quantity,
                        filled_qty=execution.filled_qty,
                    )
                )

        for order_id, execution in execution_by_id.items():
            if order_id not in expected_by_id:
                breaks.append(
                    ReconciliationBreak(
                        order_id=order_id,
                        reason="unexpected_execution",
                        expected_qty=None,
                        filled_qty=execution.filled_qty,
                    )
                )

        return ReconciliationReport(
            total_orders=len(expected_orders),
            matched_orders=matched,
            breaks=breaks,
        )

    @staticmethod
    def _validate_pair(expected: ExpectedOrder, execution: ExecutionRecord) -> str | None:
        if execution.filled_qty > expected.quantity:
            return "overfill"

        if execution.status == "filled" and execution.filled_qty != expected.quantity:
            return "filled_qty_mismatch"

        if execution.status in {"rejected", "canceled", "new"} and execution.filled_qty != 0:
            return "invalid_nonzero_for_nonfill_status"

        if execution.status == "partially_filled" and execution.filled_qty in {
            Decimal("0"),
            expected.quantity,
        }:
            return "invalid_partial_fill_qty"

        return None

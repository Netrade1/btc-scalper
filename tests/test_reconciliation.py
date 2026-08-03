import unittest
from decimal import Decimal

from k2think.reconciliation import (
    ExecutionReconciliationEngine,
    ExecutionRecord,
    ExpectedOrder,
)


class ReconciliationTests(unittest.TestCase):
    def test_clean_reconciliation(self) -> None:
        engine = ExecutionReconciliationEngine()
        report = engine.reconcile(
            expected_orders=[ExpectedOrder("o1", Decimal("1.0"))],
            executions=[ExecutionRecord("o1", "filled", Decimal("1.0"))],
        )
        self.assertTrue(report.is_clean)
        self.assertEqual(1, report.matched_orders)

    def test_detects_missing_and_unexpected(self) -> None:
        engine = ExecutionReconciliationEngine()
        report = engine.reconcile(
            expected_orders=[ExpectedOrder("o1", Decimal("1.0"))],
            executions=[ExecutionRecord("o2", "filled", Decimal("1.0"))],
        )
        reasons = {item.reason for item in report.breaks}
        self.assertEqual({"missing_execution", "unexpected_execution"}, reasons)

    def test_detects_overfill(self) -> None:
        engine = ExecutionReconciliationEngine()
        report = engine.reconcile(
            expected_orders=[ExpectedOrder("o1", Decimal("1.0"))],
            executions=[ExecutionRecord("o1", "filled", Decimal("1.1"))],
        )
        self.assertFalse(report.is_clean)
        self.assertEqual("overfill", report.breaks[0].reason)


if __name__ == "__main__":
    unittest.main()

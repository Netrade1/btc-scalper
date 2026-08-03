import unittest
from decimal import Decimal

from k2think.adapters import (
    AdapterContractError,
    Level2Order,
    MockLevel2FeedAdapter,
    MockPaperBrokerAdapter,
    PaperOrderRequest,
)


class AdapterContractTests(unittest.TestCase):
    def test_mock_l2_stream_returns_orders(self) -> None:
        adapter = MockLevel2FeedAdapter(
            [
                Level2Order(side="bid", price=Decimal("100.0"), size=Decimal("1.2"), order_id="o1"),
                Level2Order(side="ask", price=Decimal("100.1"), size=Decimal("0.8"), order_id="o2"),
            ]
        )
        rows = list(adapter.stream("BTCUSDT"))
        self.assertEqual(2, len(rows))

    def test_mock_l2_stream_rejects_blank_symbol(self) -> None:
        adapter = MockLevel2FeedAdapter([])
        with self.assertRaises(AdapterContractError):
            list(adapter.stream(""))

    def test_mock_paper_broker_validates_and_fills(self) -> None:
        broker = MockPaperBrokerAdapter()
        execution = broker.submit_order(
            PaperOrderRequest(symbol="BTCUSDT", side="buy", quantity=Decimal("0.01"))
        )
        self.assertEqual("filled", execution.status)

    def test_mock_paper_broker_rejects_invalid_side(self) -> None:
        broker = MockPaperBrokerAdapter()
        with self.assertRaises(AdapterContractError):
            broker.submit_order(
                PaperOrderRequest(symbol="BTCUSDT", side="hold", quantity=Decimal("0.01"))
            )


if __name__ == "__main__":
    unittest.main()

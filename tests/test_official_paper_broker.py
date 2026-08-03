import unittest
from decimal import Decimal

from k2think.adapters import (
    AdapterContractError,
    OfficialPaperBrokerAdapter,
    PaperOrderRequest,
)


class FakeBrokerClient:
    def __init__(self, response: dict):
        self.response = response

    def submit_paper_order(self, **kwargs):
        return self.response


class OfficialPaperBrokerAdapterTests(unittest.TestCase):
    def test_submit_order_maps_valid_response(self) -> None:
        adapter = OfficialPaperBrokerAdapter(
            FakeBrokerClient(
                {"order_id": "ord-1", "status": "filled", "filled_qty": "0.50"}
            )
        )

        execution = adapter.submit_order(
            PaperOrderRequest(symbol="BTCUSDT", side="buy", quantity=Decimal("0.50"))
        )
        self.assertEqual("ord-1", execution.order_id)
        self.assertEqual(Decimal("0.50"), execution.filled_qty)

    def test_submit_order_rejects_invalid_response(self) -> None:
        adapter = OfficialPaperBrokerAdapter(FakeBrokerClient({"status": "filled"}))
        with self.assertRaises(AdapterContractError):
            adapter.submit_order(
                PaperOrderRequest(symbol="BTCUSDT", side="buy", quantity=Decimal("0.1"))
            )


if __name__ == "__main__":
    unittest.main()

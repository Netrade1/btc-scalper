import unittest
from decimal import Decimal

from k2think.market_by_price import MarketByPriceBook
from k2think.venue_binance import BinanceDiffDepthAdapter, VenueAdapterError


class BinanceDiffDepthAdapterTests(unittest.TestCase):
    def test_normalizes_payload(self) -> None:
        adapter = BinanceDiffDepthAdapter()
        payload = {
            "U": 100,
            "u": 100,
            "b": [["101.0", "0.7"]],
            "a": [["101.1", "0.6"]],
        }

        updates = adapter.normalize(payload)
        self.assertEqual(2, len(updates))
        self.assertEqual(Decimal("101.0"), updates[0].price)

        book = MarketByPriceBook()
        book.replay(updates)
        self.assertEqual((Decimal("101.0"), Decimal("0.7")), book.best_bid())
        self.assertEqual((Decimal("101.1"), Decimal("0.6")), book.best_ask())

    def test_rejects_invalid_update_window(self) -> None:
        adapter = BinanceDiffDepthAdapter()
        with self.assertRaises(VenueAdapterError):
            adapter.normalize({"U": 12, "u": 11, "b": [], "a": []})


if __name__ == "__main__":
    unittest.main()

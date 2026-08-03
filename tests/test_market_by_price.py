import unittest
from decimal import Decimal

from k2think.market_by_price import (
    MarketByPriceBook,
    OrderBookSequenceError,
    PriceLevelUpdate,
)


class MarketByPriceBookTests(unittest.TestCase):
    def test_reconstructs_and_tracks_best_levels(self) -> None:
        book = MarketByPriceBook()
        book.replay(
            [
                PriceLevelUpdate(10, "bid", Decimal("100.0"), Decimal("1.0")),
                PriceLevelUpdate(11, "ask", Decimal("100.5"), Decimal("1.2")),
                PriceLevelUpdate(12, "bid", Decimal("100.1"), Decimal("0.5")),
                PriceLevelUpdate(13, "ask", Decimal("100.5"), Decimal("0")),
            ]
        )

        self.assertEqual((Decimal("100.1"), Decimal("0.5")), book.best_bid())
        self.assertIsNone(book.best_ask())

    def test_rejects_sequence_gap(self) -> None:
        book = MarketByPriceBook()
        book.apply(PriceLevelUpdate(100, "bid", Decimal("99"), Decimal("1")))
        with self.assertRaises(OrderBookSequenceError):
            book.apply(PriceLevelUpdate(102, "ask", Decimal("101"), Decimal("1")))


if __name__ == "__main__":
    unittest.main()

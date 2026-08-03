"""K2Think institutional research platform primitives."""

from .adapters import (
    AdapterContractError,
    Level2FeedAdapter,
    MockLevel2FeedAdapter,
    MockPaperBrokerAdapter,
    PaperBrokerAdapter,
)
from .audit import AuditChain, AuditEvent
from .config import PlatformConfig
from .market_by_price import (
    MarketByPriceBook,
    OrderBookSequenceError,
    OrderBookValidationError,
    PriceLevelUpdate,
)
from .venue_binance import BinanceDiffDepthAdapter, VenueAdapterError

__all__ = [
    "PlatformConfig",
    "AuditChain",
    "AuditEvent",
    "AdapterContractError",
    "Level2FeedAdapter",
    "MockLevel2FeedAdapter",
    "PaperBrokerAdapter",
    "MockPaperBrokerAdapter",
    "PriceLevelUpdate",
    "MarketByPriceBook",
    "OrderBookSequenceError",
    "OrderBookValidationError",
    "BinanceDiffDepthAdapter",
    "VenueAdapterError",
]

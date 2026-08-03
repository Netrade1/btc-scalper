"""K2Think institutional research platform primitives."""

from .adapters import (
    AdapterContractError,
    Level2FeedAdapter,
    MockLevel2FeedAdapter,
    MockPaperBrokerAdapter,
    OfficialPaperBrokerAdapter,
    PaperBrokerAdapter,
    PaperBrokerClient,
)
from .audit import AuditChain, AuditEvent
from .config import PlatformConfig
from .market_by_price import (
    MarketByPriceBook,
    OrderBookSequenceError,
    OrderBookValidationError,
    PriceLevelUpdate,
)
from .reconciliation import (
    ExecutionReconciliationEngine,
    ExecutionRecord,
    ExpectedOrder,
    ReconciliationBreak,
    ReconciliationError,
    ReconciliationReport,
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
    "PaperBrokerClient",
    "MockPaperBrokerAdapter",
    "OfficialPaperBrokerAdapter",
    "PriceLevelUpdate",
    "MarketByPriceBook",
    "OrderBookSequenceError",
    "OrderBookValidationError",
    "ExecutionReconciliationEngine",
    "ExecutionRecord",
    "ExpectedOrder",
    "ReconciliationBreak",
    "ReconciliationError",
    "ReconciliationReport",
    "BinanceDiffDepthAdapter",
    "VenueAdapterError",
]

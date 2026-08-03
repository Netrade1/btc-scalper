"""K2Think institutional research platform primitives."""

from .config import PlatformConfig
from .audit import AuditChain, AuditEvent
from .adapters import (
    AdapterContractError,
    Level2FeedAdapter,
    MockLevel2FeedAdapter,
    MockPaperBrokerAdapter,
    PaperBrokerAdapter,
)

__all__ = [
    "PlatformConfig",
    "AuditChain",
    "AuditEvent",
    "AdapterContractError",
    "Level2FeedAdapter",
    "MockLevel2FeedAdapter",
    "PaperBrokerAdapter",
    "MockPaperBrokerAdapter",
]

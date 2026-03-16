from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Trade(Base):
    """Records each trade signal received from TradingView."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    action = Column(String, nullable=False)          # "buy" | "sell" | "close"
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    is_winner = Column(Boolean, nullable=True)        # filled after close
    pnl = Column(Float, nullable=True)               # profit/loss in base currency
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return (
            f"<Trade id={self.id} action={self.action} "
            f"price={self.price} pnl={self.pnl}>"
        )

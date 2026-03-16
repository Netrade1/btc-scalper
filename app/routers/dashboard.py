"""
Dashboard router – exposes trading statistics and ruin-probability alerts.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.analytics import ruin_probability, stats_from_trades
from app.config import get_settings, Settings
from app.database import get_db
from app.models import Trade

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats", summary="Aggregated trading statistics")
def get_stats(db: Session = Depends(get_db)) -> dict:
    """Return win rate, average P&L, and net P&L for all closed trades."""
    trades = db.query(Trade).filter(Trade.pnl.isnot(None)).all()
    pnl_values = [t.pnl for t in trades]
    return stats_from_trades(pnl_values)


@router.get("/ruin-probability", summary="Current probability of ruin")
def get_ruin_probability(
    capital: float = Query(..., gt=0, description="Current account balance"),
    unit_size: float = Query(..., gt=0, description="Size of one trade (stake)"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Calculate the probability of ruin given current capital and trade size.

    Raises an alert flag when the probability exceeds the configured threshold
    (``RUIN_PROBABILITY_ALERT_THRESHOLD``).
    """
    trades = db.query(Trade).filter(Trade.pnl.isnot(None)).all()
    wins = sum(1 for t in trades if t.pnl is not None and t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl is not None and t.pnl <= 0)

    prob = ruin_probability(wins, losses, capital, unit_size)
    threshold = settings.ruin_probability_alert_threshold
    alert = prob >= threshold

    return {
        "ruin_probability": round(prob, 6),
        "alert": alert,
        "alert_threshold": threshold,
        "wins": wins,
        "losses": losses,
        "capital": capital,
        "unit_size": unit_size,
        "message": (
            f"⚠️  Ruin probability {prob:.1%} exceeds threshold {threshold:.1%}!"
            if alert
            else f"Ruin probability {prob:.1%} is within safe limits."
        ),
    }


@router.get("/trades", summary="Recent trade history")
def get_trades(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list:
    """Return the most recent trades, newest first."""
    trades = (
        db.query(Trade)
        .order_by(Trade.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": t.id,
            "symbol": t.symbol,
            "action": t.action,
            "price": t.price,
            "quantity": t.quantity,
            "is_winner": t.is_winner,
            "pnl": t.pnl,
            "timestamp": t.timestamp.isoformat() if t.timestamp else None,
        }
        for t in trades
    ]

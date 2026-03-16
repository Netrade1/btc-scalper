"""
Celery application and task definitions.

Tasks process incoming trade signals asynchronously so that the HTTP
webhook endpoint can return immediately to TradingView.
"""

from celery import Celery
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "btc_scalper",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)


@celery_app.task(name="process_trade_signal")
def process_trade_signal(payload: dict) -> dict:
    """Persist a trade signal and log it.

    Args:
        payload: Validated webhook payload dict.

    Returns:
        dict with stored trade id and timestamp.
    """
    from datetime import datetime
    from app.database import SessionLocal
    from app.models import Trade

    db = SessionLocal()
    try:
        trade = Trade(
            symbol=payload["symbol"],
            action=payload["action"],
            price=payload["price"],
            quantity=payload["quantity"],
        )
        db.add(trade)
        db.commit()
        db.refresh(trade)
        return {"trade_id": trade.id, "timestamp": str(trade.timestamp)}
    finally:
        db.close()

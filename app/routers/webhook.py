"""
Webhook router – receives TradingView alerts.

Security: every request must include the shared secret token in the
``X-Webhook-Token`` header.  The token is configured via the
``WEBHOOK_SECRET`` environment variable (see .env.example) and must
also be embedded in the Pine Script ``webhook_message`` JSON body or
sent as an HTTP header from TradingView's "alert > notifications >
webhook URL" settings.
"""

import hmac
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, status
from pydantic import BaseModel, Field

from app.config import get_settings, Settings
from app.database import get_db
from app.tasks import process_trade_signal

router = APIRouter(prefix="/webhook", tags=["webhook"])


# ---------------------------------------------------------------------------
# Pydantic schema for the incoming alert body
# ---------------------------------------------------------------------------

class WebhookPayload(BaseModel):
    symbol: str = Field(..., examples=["BTCUSDT"])
    action: str = Field(..., pattern="^(buy|sell|close)$", examples=["buy"])
    price: float = Field(..., gt=0)
    quantity: float = Field(..., gt=0)
    # Optional: Pine Script can embed the token in the JSON body as well
    token: Optional[str] = Field(default=None, exclude=True)


# ---------------------------------------------------------------------------
# Token verification dependency
# ---------------------------------------------------------------------------

def verify_token(
    x_webhook_token: Optional[str] = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Raise 401 when the supplied token does not match the configured secret."""
    if not settings.webhook_secret or settings.webhook_secret == "changeme":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook secret is not configured on the server.",
        )
    if x_webhook_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Webhook-Token header.",
        )
    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(
        x_webhook_token.encode(), settings.webhook_secret.encode()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook token.",
        )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/tradingview",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Receive a TradingView alert and queue it for processing",
    dependencies=[Depends(verify_token)],
)
async def tradingview_webhook(
    payload: WebhookPayload,
) -> dict:
    """Queue a trade signal for async processing via Celery."""
    task = process_trade_signal.delay(payload.model_dump(exclude={"token"}))
    return {"status": "queued", "task_id": task.id}

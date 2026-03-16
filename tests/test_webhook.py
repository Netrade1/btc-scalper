"""
Tests for the TradingView webhook endpoint.

Covers:
- Missing token → 401
- Wrong token → 401
- Invalid payload fields → 422
- Valid request → 202 (queued)
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Override settings before importing the app
import os
os.environ["WEBHOOK_SECRET"] = "test-secret-token"
os.environ["DATABASE_URL"] = "sqlite:///./test_btc_scalper.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379/0"
os.environ["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/1"

from app.main import app  # noqa: E402


VALID_PAYLOAD = {
    "symbol": "BTCUSDT",
    "action": "buy",
    "price": 65000.0,
    "quantity": 0.001,
}

HEADERS_OK = {"X-Webhook-Token": "test-secret-token"}


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test and drop them after."""
    from app.database import engine
    from app.models import Base
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── Token security ───────────────────────────────────────────────────────────

def test_missing_token_returns_401(client):
    resp = client.post("/webhook/tradingview", json=VALID_PAYLOAD)
    assert resp.status_code == 401
    assert "Missing" in resp.json()["detail"]


def test_wrong_token_returns_401(client):
    resp = client.post(
        "/webhook/tradingview",
        json=VALID_PAYLOAD,
        headers={"X-Webhook-Token": "wrong-token"},
    )
    assert resp.status_code == 401
    assert "Invalid" in resp.json()["detail"]


# ── Payload validation ───────────────────────────────────────────────────────

def test_invalid_action_returns_422(client):
    payload = {**VALID_PAYLOAD, "action": "hodl"}
    resp = client.post("/webhook/tradingview", json=payload, headers=HEADERS_OK)
    assert resp.status_code == 422


def test_negative_price_returns_422(client):
    payload = {**VALID_PAYLOAD, "price": -1.0}
    resp = client.post("/webhook/tradingview", json=payload, headers=HEADERS_OK)
    assert resp.status_code == 422


def test_zero_quantity_returns_422(client):
    payload = {**VALID_PAYLOAD, "quantity": 0.0}
    resp = client.post("/webhook/tradingview", json=payload, headers=HEADERS_OK)
    assert resp.status_code == 422


# ── Valid request ────────────────────────────────────────────────────────────

def test_valid_webhook_returns_202(client):
    mock_result = MagicMock()
    mock_result.id = "task-abc-123"
    with patch("app.routers.webhook.process_trade_signal.delay", return_value=mock_result):
        resp = client.post(
            "/webhook/tradingview",
            json=VALID_PAYLOAD,
            headers=HEADERS_OK,
        )
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "queued"
    assert data["task_id"] == "task-abc-123"


def test_valid_sell_signal(client):
    payload = {**VALID_PAYLOAD, "action": "sell"}
    mock_result = MagicMock()
    mock_result.id = "task-sell-456"
    with patch("app.routers.webhook.process_trade_signal.delay", return_value=mock_result):
        resp = client.post(
            "/webhook/tradingview",
            json=payload,
            headers=HEADERS_OK,
        )
    assert resp.status_code == 202


def test_valid_close_signal(client):
    payload = {**VALID_PAYLOAD, "action": "close"}
    mock_result = MagicMock()
    mock_result.id = "task-close-789"
    with patch("app.routers.webhook.process_trade_signal.delay", return_value=mock_result):
        resp = client.post(
            "/webhook/tradingview",
            json=payload,
            headers=HEADERS_OK,
        )
    assert resp.status_code == 202

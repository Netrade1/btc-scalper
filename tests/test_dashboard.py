"""
Tests for the dashboard endpoints and ruin-probability analytics.
"""

import pytest
import os

os.environ.setdefault("WEBHOOK_SECRET", "test-secret-token")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_btc_scalper.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
os.environ.setdefault("RUIN_PROBABILITY_ALERT_THRESHOLD", "0.10")

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
from app.analytics import ruin_probability, stats_from_trades  # noqa: E402


@pytest.fixture(autouse=True)
def setup_db():
    from app.database import engine
    from app.models import Base
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_with_trades():
    """Seed the database with a mix of winning and losing trades."""
    from app.database import SessionLocal
    from app.models import Trade
    db = SessionLocal()
    try:
        trades = [
            Trade(symbol="BTCUSDT", action="close", price=65000, quantity=0.001,
                  is_winner=True, pnl=120.0),
            Trade(symbol="BTCUSDT", action="close", price=64500, quantity=0.001,
                  is_winner=True, pnl=80.0),
            Trade(symbol="BTCUSDT", action="close", price=63000, quantity=0.001,
                  is_winner=False, pnl=-60.0),
            Trade(symbol="BTCUSDT", action="close", price=62000, quantity=0.001,
                  is_winner=False, pnl=-50.0),
            Trade(symbol="BTCUSDT", action="close", price=66000, quantity=0.001,
                  is_winner=True, pnl=200.0),
        ]
        db.add_all(trades)
        db.commit()
    finally:
        db.close()


# ── /dashboard/stats ─────────────────────────────────────────────────────────

def test_stats_empty_db(client):
    resp = client.get("/dashboard/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_trades"] == 0
    assert data["win_rate"] == 0.0


def test_stats_with_trades(client, db_with_trades):
    resp = client.get("/dashboard/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_trades"] == 5
    assert data["wins"] == 3
    assert data["losses"] == 2
    assert abs(data["win_rate"] - 0.6) < 1e-6
    assert data["net_pnl"] == pytest.approx(290.0)


# ── /dashboard/ruin-probability ──────────────────────────────────────────────

def test_ruin_probability_requires_params(client):
    resp = client.get("/dashboard/ruin-probability")
    assert resp.status_code == 422


def test_ruin_probability_empty_db(client):
    resp = client.get("/dashboard/ruin-probability?capital=10000&unit_size=100")
    assert resp.status_code == 200
    data = resp.json()
    # No trade history → probability is 0
    assert data["ruin_probability"] == 0.0
    assert data["alert"] is False


def test_ruin_probability_with_trades(client, db_with_trades):
    # 3 wins / 5 trades = 60% win rate, strong positive edge → very low ruin probability
    resp = client.get("/dashboard/ruin-probability?capital=10000&unit_size=100")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ruin_probability"] < 0.10
    assert data["alert"] is False
    assert "wins" in data
    assert "losses" in data


def test_ruin_probability_alert_fires(client):
    """Seed trades with a near-50% win rate to push ruin probability above threshold."""
    from app.database import SessionLocal
    from app.models import Trade
    db = SessionLocal()
    try:
        # 51 wins vs 49 losses → win rate = 0.51, tiny positive edge
        # With small capital / large unit, ruin probability will be high
        for _ in range(51):
            db.add(Trade(symbol="BTCUSDT", action="close", price=65000,
                         quantity=0.001, is_winner=True, pnl=1.0))
        for _ in range(49):
            db.add(Trade(symbol="BTCUSDT", action="close", price=65000,
                         quantity=0.001, is_winner=False, pnl=-1.0))
        db.commit()
    finally:
        db.close()

    # capital=1000, unit_size=1000 → N=1 → ruin probability = ((1-p)/p)^1 ≈ 0.96
    resp = client.get("/dashboard/ruin-probability?capital=1000&unit_size=1000")
    assert resp.status_code == 200
    data = resp.json()
    assert data["alert"] is True
    assert data["ruin_probability"] >= 0.10
    assert "⚠️" in data["message"]


# ── /dashboard/trades ────────────────────────────────────────────────────────

def test_trades_empty(client):
    resp = client.get("/dashboard/trades")
    assert resp.status_code == 200
    assert resp.json() == []


def test_trades_returns_records(client, db_with_trades):
    resp = client.get("/dashboard/trades?limit=3")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    assert all("symbol" in t for t in data)


# ── analytics unit tests ──────────────────────────────────────────────────────

class TestRuinProbability:
    def test_no_trades(self):
        assert ruin_probability(0, 0, 10000, 100) == 0.0

    def test_zero_capital(self):
        assert ruin_probability(60, 40, 0, 100) == 1.0

    def test_zero_unit_size(self):
        assert ruin_probability(60, 40, 10000, 0) == 0.0

    def test_all_losses(self):
        assert ruin_probability(0, 100, 10000, 100) == 1.0

    def test_all_wins(self):
        assert ruin_probability(100, 0, 10000, 100) == 0.0

    def test_positive_edge_low_ruin(self):
        pr = ruin_probability(60, 40, 10000, 100)
        assert 0.0 <= pr <= 0.10

    def test_negative_edge_certain_ruin(self):
        pr = ruin_probability(40, 60, 10000, 100)
        assert pr == 1.0

    def test_result_clamped_to_unit_interval(self):
        pr = ruin_probability(51, 49, 100, 100)
        assert 0.0 <= pr <= 1.0


class TestStatsFromTrades:
    def test_empty(self):
        s = stats_from_trades([])
        assert s["total_trades"] == 0
        assert s["win_rate"] == 0.0

    def test_mixed(self):
        s = stats_from_trades([100, -50, 200, -30, 80])
        assert s["total_trades"] == 5
        assert s["wins"] == 3
        assert s["losses"] == 2
        assert abs(s["win_rate"] - 0.6) < 1e-9
        assert s["net_pnl"] == pytest.approx(300.0)

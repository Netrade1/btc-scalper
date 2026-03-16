"""
BTC Scalper — FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.database import init_db
from app.routers import webhook, dashboard


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="BTC Scalper",
    description=(
        "Receives TradingView webhook alerts, queues them via Celery, "
        "and exposes a dashboard with ruin-probability alerts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(webhook.router)
app.include_router(dashboard.router)

# Serve the dashboard UI from the frontend directory
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_dashboard():
        return FileResponse(os.path.join(_frontend_dir, "dashboard.html"))


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok"}

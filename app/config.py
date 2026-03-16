from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Webhook secret token — must match the value sent by TradingView/Pine Script
    webhook_secret: str = "changeme"

    # Redis / Celery
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    # Database
    database_url: str = "sqlite:///./btc_scalper.db"

    # Dashboard alert threshold (0–1)
    ruin_probability_alert_threshold: float = 0.10

    # API host / port (used by uvicorn in docker-compose)
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()

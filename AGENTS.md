# btc-scalper

A Bitcoin scalping backend: a FastAPI service receives TradingView webhook alerts,
queues them via Celery (Redis broker), a Celery worker persists trades to a database
(SQLite by default), and a dashboard API + static UI expose win/loss stats and
ruin-probability alerts.

## Cursor Cloud specific instructions

These notes are for agents running in an environment where the update script has
already installed dependencies (Python deps into `.venv`, plus `redis-server`).
Standard commands live in `README.md`; only the non-obvious caveats are below.

### Services

| Service | Run command (from repo root) | Notes |
|---------|------------------------------|-------|
| `redis` | `sudo service redis-server start` | Celery broker + result backend. Must be running before the worker. Verify with `redis-cli ping` → `PONG`. |
| `api` (FastAPI) | `.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000` | Serves webhook endpoints + dashboard UI at `http://localhost:8000`. |
| `worker` (Celery) | `.venv/bin/celery -A app.tasks.celery_app worker --loglevel=info --concurrency=2` | Consumes queued signals and writes trades. Required for end-to-end flow. |

Run each service in its own tmux session; `api` and `worker` are long-running.

### Environment / `.env` caveats (non-obvious)

- A `.env` file is required and is gitignored. Copy `.env.example` to `.env`.
- The default `REDIS_URL` / Celery URLs point at host `redis` (the docker-compose
  service name). For non-Docker local runs you MUST override them to `localhost`,
  e.g. `REDIS_URL=redis://localhost:6379/0`, `CELERY_BROKER_URL=redis://localhost:6379/0`,
  `CELERY_RESULT_BACKEND=redis://localhost:6379/1`.
- `WEBHOOK_SECRET` must be set to something other than `changeme`, otherwise
  `POST /webhook/tradingview` returns HTTP 500 (the server refuses an unconfigured secret).
- The default `DATABASE_URL` (`sqlite:////app/data/btc_scalper.db`) is a Docker path.
  For local runs use a writable absolute path the api and worker share, e.g.
  `DATABASE_URL=sqlite:////workspace/data/btc_scalper.db` (create `data/` first). The DB
  schema is auto-created on api startup and on worker task execution via `init_db()`/SQLAlchemy.

### Testing

- `.venv/bin/python -m pytest tests/ -v` runs the full suite (26 tests).
- Tests mock Celery's `.delay()` and use their own SQLite file, so a running Redis/worker
  is not strictly required for pytest, but the tests set `REDIS_URL` to `localhost`.

### Behavior gotchas

- `GET /dashboard/stats` and `/dashboard/ruin-probability` only count trades whose `pnl`
  is non-null. Trades created by the webhook flow have `pnl = null` (P&L is not computed
  on ingest), so they appear in `GET /dashboard/trades` but NOT in `/stats`. This is
  expected — empty stats with populated `/trades` is correct behavior.

### Tooling

- No linter/formatter is configured (no ruff/flake8/black config; none in `requirements.txt`).

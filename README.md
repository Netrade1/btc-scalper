# btc-scalper

A production-ready BTC scalping backend that:

- Receives **TradingView webhook alerts** and queues them via **Celery**
- Exposes a **dashboard API** with win-rate stats and **ruin-probability alerts**
- Is packaged as **Docker Compose** services for easy cloud deployment
- Comes with a **Pine Script** strategy that fires the webhooks from TradingView

---

## Architecture

```
TradingView (Pine Script)
        │  POST /webhook/tradingview
        │  X-Webhook-Token: <secret>
        ▼
┌──────────────────┐      ┌───────────────┐
│  FastAPI  (api)  │─────▶│  Redis broker │
└──────────────────┘      └───────┬───────┘
        │                         │
        │  /dashboard/*           ▼
        │                 ┌───────────────┐
        ▼                 │ Celery worker │
  Browser dashboard       └───────┬───────┘
                                  │ writes
                                  ▼
                           SQLite / PostgreSQL
```

---

## Quick start

### 1. Clone and configure

```bash
git clone https://github.com/Netrade1/btc-scalper.git
cd btc-scalper
cp .env.example .env
```

Edit `.env` and set a strong `WEBHOOK_SECRET`:

```dotenv
WEBHOOK_SECRET=your_super_secret_token_here
```

### 2. Build and run with Docker Compose

```bash
docker compose up --build -d
```

Services started:

| Service | Port | Description |
|---------|------|-------------|
| `api`   | 8000 | FastAPI + dashboard UI |
| `worker`| –    | Celery trade processor |
| `redis` | 6379 | Message broker (internal) |

Open the dashboard at **http://localhost:8000** (or your server's IP/domain).

### 3. Scale Celery workers

```bash
docker compose up --scale worker=4 -d
```

---

## Deployment on a cloud VM (AWS EC2 / DigitalOcean)

1. **Provision** an Ubuntu 22.04 VM (≥1 vCPU, 1 GB RAM).
2. **Install Docker** and Docker Compose:
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   ```
3. **Clone** the repo and copy `.env.example` → `.env` with your secret.
4. **Run** `docker compose up --build -d`.
5. **Configure a reverse proxy** (nginx / Caddy) to forward HTTPS traffic on
   port 443 to the API on port 8000.  Example nginx snippet:
   ```nginx
   server {
       listen 443 ssl;
       server_name your.domain.com;
       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header X-Forwarded-For $remote_addr;
       }
   }
   ```
6. **Obtain a TLS certificate** with Let's Encrypt / Certbot.

---

## TradingView webhook configuration

1. Open the Pine Script in `pine_script/btc_scalper.pine` and add it to your
   TradingView chart.
2. Create an alert on the strategy:
   - **Webhook URL**: `https://your.domain.com/webhook/tradingview`
   - **Message**: use the built-in `{{strategy.order.alert_message}}` placeholder
     (the Pine Script fills it in).
3. Add a **custom HTTP header** in the TradingView alert form:
   ```
   X-Webhook-Token: your_super_secret_token_here
   ```
   This must match `WEBHOOK_SECRET` in your `.env`.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/webhook/tradingview` | Receive a TradingView alert |
| `GET`  | `/dashboard/stats` | Aggregated win/loss statistics |
| `GET`  | `/dashboard/ruin-probability?capital=&unit_size=` | Ruin probability + alert flag |
| `GET`  | `/dashboard/trades?limit=` | Recent trade history |
| `GET`  | `/health` | Health check |
| `GET`  | `/docs` | Interactive Swagger UI |

### Webhook payload

```json
{
  "symbol": "BTCUSDT",
  "action": "buy",
  "price": 65000.0,
  "quantity": 0.001
}
```

`action` must be one of `buy`, `sell`, or `close`.

### Ruin probability alert

```
GET /dashboard/ruin-probability?capital=10000&unit_size=100
```

```json
{
  "ruin_probability": 0.000123,
  "alert": false,
  "alert_threshold": 0.1,
  "wins": 60,
  "losses": 40,
  "capital": 10000,
  "unit_size": 100,
  "message": "Ruin probability 0.0% is within safe limits."
}
```

When `alert` is `true` (probability ≥ `RUIN_PROBABILITY_ALERT_THRESHOLD`), the
dashboard highlights the metric in red and shows an on-screen warning.  Tune
the threshold via the `RUIN_PROBABILITY_ALERT_THRESHOLD` environment variable.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WEBHOOK_SECRET` | `changeme` | Shared secret for webhook authentication |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `CELERY_BROKER_URL` | same as above | Celery broker |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/1` | Celery result store |
| `DATABASE_URL` | `sqlite:////app/data/btc_scalper.db` | SQLAlchemy DSN |
| `RUIN_PROBABILITY_ALERT_THRESHOLD` | `0.10` | Alert threshold (0–1) |

---

## Development

```bash
# Install deps
pip install -r requirements.txt

# Run locally (requires Redis on localhost)
uvicorn app.main:app --reload

# Run tests
pytest tests/ -v
```

---

## Security notes

- The `X-Webhook-Token` header is verified with a **constant-time comparison**
  (`hmac.compare_digest`) to prevent timing attacks.
- The server rejects requests if `WEBHOOK_SECRET` is still set to `changeme`.
- All secret values are read from environment variables — never hardcoded.

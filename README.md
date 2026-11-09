# btc-scalper

A walk-forward machine-learning trading system for BTC: research → backtest → **paper trading**.
It is a rework of the "Agentic Autonomous ML Trading System" script. That script looked complete
but had several bugs that made its backtest results meaningless and its live loop unsafe (listed below).

> **Paper trading only.** `btc_scalper/safety.py` sets `PAPER_TRADING_ONLY = True` and `FAIL_CLOSED = True`.
> The real-order router (`LiveRouter`) is implemented and tested against a fake exchange, but it raises
> `PermissionError` until a replay on real history and a forward paper run both pass and the Risk Chair approves.

## What was missing / wrong in the original script

Ordered by how much each one distorted results:

| # | Problem in the original | Effect | Fix here |
|---|---|---|---|
| 1 | **The live loop traded the still-forming candle.** `fetch_ohlcv` returns the unclosed bar last. | Signals repaint; the "no repainting" claim was false. | Bars whose close time is in the future are dropped (`exchange.py`). |
| 2 | **Live order flow was fake.** `buy_volume = sell_volume = 0.5 * volume` → CVD ≡ 0 → `cvd_roc` = 0/0 = NaN → `dropna()` deleted **every row**. | The live daemon could never produce a signal (it crashed on `iloc[-1]`). | Real taker-buy volume from Binance raw klines (index 9). If it is missing, the system **fails closed**. Proxy only by explicit opt-in. |
| 3 | **The label did not match the trade.** It trained on "next close higher", then traded a 1.5 ATR stop / 3 ATR target bracket and fed that probability into a 2:1 expectancy formula. | The expectancy number was meaningless. | Triple-barrier labels: "does this exact bracket hit TP before SL within H bars?" A bar touching both barriers = loss. |
| 4 | **Costs ignored in the decision.** Expectancy was computed before fees/slippage. On 5m BTC a 1.5 ATR stop makes round-trip costs **≈0.5R (perp 0.05%) to ≈1R (spot 0.10%)**. | Most "edge" trades were losers after costs. | `E = p·RR − (1−p) − cost_R` with `cost_R` from the real fee schedule. |
| 5 | **No out-of-sample check on the model.** Fit and predict on the same window, no purge, uncalibrated probabilities. | Overfit noise traded as signal. | Purged walk-forward, calibration, and a health gate (AUC confidence bound + Brier skill on the forward track record). |
| 6 | Same-close fills: the signal on bar t filled at bar t's close. | Optimistic fills. | Fill at bar t+1 open, with slippage + square-root impact. Gaps through a stop fill at the open. |
| 7 | Live loop polled every 60 s on 5m bars, re-running the model ~5× per bar. | Duplicate decisions. | Sleep aligned to bar close; each bar processed exactly once (de-duplicated by timestamp). |
| 8 | `submit_live_order` had the real call commented out but still "filled" in memory; SL/TP existed only in memory. | Unprotected positions if the process died. | Exchange-side reduce-only stop + take-profit; flatten if they can't be placed; reconcile every bar; `clientOrderId`. |
| 9 | Cumulative CVD/VWAP from the start of the data window. | Live features drift away from the backtested ones as the window rolls. | Only finite rolling windows. A test checks values don't depend on the window start. |
| 10 | Also: kill switch only on realized PnL, no leverage cap (tiny ATR → huge size), fees charged at the bar close instead of the exit price, comment said 0.5% but code used 0.1%, `MarketBar` timestamp taken from the wrong bar, xgboost refit every bar. | — | Mark-to-market kill switch, leverage cap, correct fee accounting, scheduled retrains. |

## Architecture

```
btc_scalper/
  config.py      typed, validated config (TOML); unknown keys rejected
  data.py        validation, integrity gate, CSV I/O, synthetic market generator
  features.py    causal, scale-free features (returns, ATR%, RSI, rolling VWAP, order-flow imbalance, …)
  labeling.py    triple-barrier labels matching the traded bracket
  predictor.py   purged walk-forward XGBoost, long & short models, 2-stage calibration, health gate
  strategy.py    cost-aware expectancy → signal
  risk.py        Risk Governor (final veto): sizing + every hard limit
  broker.py      SimBroker: next-open fills, slippage/impact, gaps, ambiguous bars, fees, funding
  exchange.py    CCXT data feed (closed bars, real taker volume) + LiveRouter (blocked)
  engine.py      one step per closed bar: the same code for backtest and paper
  live.py        paper daemon: bar-aligned, de-duplicated, state persisted, fail-closed
  audit.py       SHA-256 hash-chained JSONL ledger + verify
  metrics.py     Sharpe, Sortino, Calmar, PSR, drawdown, trade stats, Monte-Carlo drawdowns
  safety.py      PAPER_TRADING_ONLY, FAIL_CLOSED
```

**Model.** Two classifiers estimate P(long bracket wins) and P(short bracket wins). Training only
uses rows whose labels were resolved `H + embargo` bars before the decision (purging). Inside each fit, a
purged hold-out slice calibrates the probabilities. Every forward prediction is recorded, and once its
outcome is known, that walk-forward track record is used to **recalibrate** (this fixed a 62% forecast
that realized 42%). A side may trade only when:
hold-out AUC ≥ `min_auc` **and** its 95% lower bound > 0.5 (effective sample size corrected for overlapping labels),
**and** the forward record has ≥ `min_oos_samples` resolved predictions with AUC lower bound > 0.5 **and** cross-fitted Brier skill > 0.

**Risk Governor (final veto).** Fixed-fractional risk (0.75%) capped by ¼-Kelly; linear drawdown throttle;
max 3× notional leverage; 15% mark-to-market drawdown kill switch (latches across restarts and flattens);
3% daily loss limit; cooldown after 5 consecutive losses; ATR% volatility band; data-gap gate.

**Paper warm start.** On first launch the daemon fetches `warmup_bars` (8,000) of history and replays the
model walk-forward over it (no trades). That builds the forward track record the health gate needs, so the
system doesn't have to sit idle for days.

**Fail-closed loop.** No new decision if the feature window has gaps, duplicates, malformed bars,
out-of-range flow or a stale feed. Auth/permission/bad-symbol errors halt immediately. Network errors spend
an error budget, then halt. The loop stops when the circuit breaker trips. Everything goes to the hash-chained ledger.

## Usage

```bash
pip install -r requirements.txt        # if pip hangs on xgboost GPU deps:
                                       #   pip download xgboost --no-deps && pip install xgboost*.whl --no-deps

python -m pytest -q                    # 38 tests

# Backtest on synthetic data (alpha=0 is a pure random walk; alpha>0 plants an order-flow edge)
python -m btc_scalper backtest --synthetic-bars 20000 --alpha 1.0 --config config.perp.example.toml

# Real data (Kraken, default): build 5m bars with real taker volume from public trades.
# About 1 minute per day of history; resumable (re-run the same command to continue).
python -m btc_scalper fetch-trades --days 45
python -m btc_scalper backtest --csv data/kraken_BTCUSD_5m_trades.csv

# Binance (where available) has taker volume in its candles directly:
#   [data] exchange = "binance", symbol = "BTC/USDT"   then:
python -m btc_scalper fetch --days 90 --out data/btc_5m.csv

# Paper trading against live closed bars (state in state/, ledger in logs/)
python -m btc_scalper paper
python -m btc_scalper verify-ledger logs/audit.jsonl

python -m btc_scalper live --i-accept-the-risk   # → PermissionError (PAPER_TRADING_ONLY)
```

Defaults are **Kraken BTC/USD spot, long-only, 0.40% taker** (Kraken Pro base tier; set `[execution]
taker_fee` to your real tier). Kraken candles don't include taker volume, so bars are built from the public
trade feed, where each trade carries the taker's side (`btc_scalper/trades.py`). The paper loop keeps that
store up to date incrementally. `config.perp.example.toml` switches to a Binance perpetual (0.05% taker,
long/short, funding). Venues with no taker data fail closed unless you set `allow_flow_proxy = true`.

**Fees vs. timeframe, the main constraint.** Round-trip cost in R with a 1.5×ATR stop on 5m BTC (ATR ≈ 0.15%):

| Taker fee per side | Round-trip cost |
|---|---|
| 0.05% (perp / VIP) | ≈ 0.5R |
| 0.10% | ≈ 1R |
| 0.26% | ≈ 2.4R |
| 0.40% (Kraken base) | ≈ 3.6R |

At 0.40%, a 5m bracket would need a win rate above ~90% just to break even, so expect the system to make
**no trades**. That's the correct answer at those fees. Options are a higher fee tier, maker (limit) entries,
or longer bars (a wider ATR shrinks the cost in R).

## Synthetic replay results

From `python scripts/replay_report.py` (20,000 × 5m bars ≈ 69 days per run, default model settings):

| market | cost level | seed | trades | win % | avg R | return | max DD | Sharpe |
|---|---|---|---|---|---|---|---|---|
| random walk | perp 0.05% L/S | 7 | 0 | 0% | +0.00 | +0.0% | -0.0% | +0.00 |
| random walk | perp 0.05% L/S | 11 | 0 | 0% | +0.00 | +0.0% | -0.0% | +0.00 |
| random walk | perp 0.05% L/S | 23 | 0 | 0% | +0.00 | +0.0% | -0.0% | +0.00 |
| random walk | spot 0.10% long | 7 | 0 | 0% | +0.00 | +0.0% | -0.0% | +0.00 |
| random walk | spot 0.10% long | 11 | 0 | 0% | +0.00 | +0.0% | -0.0% | +0.00 |
| random walk | spot 0.10% long | 23 | 0 | 0% | +0.00 | +0.0% | -0.0% | +0.00 |
| planted edge a=1.0 | perp 0.05% L/S | 7 | 58 | 57% | +0.47 | +20.5% | 4.5% | +6.57 |
| planted edge a=1.0 | perp 0.05% L/S | 11 | 55 | 40% | -0.12 | -3.0% | 8.1% | -1.17 |
| planted edge a=1.0 | perp 0.05% L/S | 23 | 114 | 53% | +0.31 | +24.1% | 6.4% | +5.70 |
| planted edge a=1.0 | spot 0.10% long | 7 | 12 | 75% | +0.80 | +7.1% | 1.5% | +5.71 |
| planted edge a=1.0 | spot 0.10% long | 11 | 31 | 39% | -0.35 | -6.0% | 7.9% | -3.10 |
| planted edge a=1.0 | spot 0.10% long | 23 | 14 | 64% | +0.43 | +4.3% | 4.2% | +3.04 |
| planted edge a=1.5 | perp 0.05% L/S | 7 | 518 | 57% | +0.43 | +328.3% | 6.3% | +17.41 |
| planted edge a=1.5 | perp 0.05% L/S | 11 | 376 | 58% | +0.43 | +194.6% | 9.3% | +16.13 |
| planted edge a=1.5 | spot 0.10% long | 7 | 117 | 53% | +0.27 | +20.3% | 5.7% | +4.85 |
| planted edge a=1.5 | spot 0.10% long | 11 | 88 | 53% | +0.09 | +2.9% | 7.1% | +0.98 |

Reading it: on a **random walk the system never trades** (0 trades in all 6 runs), because the health gate
correctly rejects noise. With a planted edge it trades and is mostly profitable. A weak edge (a=1.0) still
loses on one seed out of three, which is realistic. **Costs matter a lot.** Going from 0.05% to 0.10% taker
cuts the trade count 3–5× and the return by up to 10× on the same edge.

**What this proves:** the mechanics work. The system stays flat on a random walk, finds a planted edge
without look-ahead, and responds correctly to costs. **What it does not prove:** that any real BTC edge
exists. The synthetic edge was put there on purpose.

## Next gates before any real money
1. Replay on real exchange history (`fetch` + `backtest --csv`). Check that the health gate opens at all,
   forward AUC/Brier skill, realized vs. predicted R, and the cost drag.
2. A forward paper run (`paper`) for several weeks; verify the ledger; compare paper fills with the backtest.
3. Risk Chair review. Only then edit `safety.py`.

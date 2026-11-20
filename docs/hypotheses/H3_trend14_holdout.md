# H3: 14-day trend rule on unseen history (pre-registered)

Written and committed **before** the 2018–2024 data was downloaded. Do not edit after the run except to
append results.

## Why this test
In H2 the 28-day rule failed. The 14-day lookback looked better on the same 2024–2026 data, but it was
found by looking at that data. H3 tests it once on history the rule has never seen.

## Hypothesis
Holding BTC only while its daily close is above the close 14 days earlier (otherwise cash) earns a higher
risk-adjusted return than buy-and-hold, after Kraken's 0.40% base taker fee.

## Fixed design (identical to H2 except the lookback and the data)
- Data: Kraken BTC/USD public trades, 2018-01-01 to 2024-11-12 (UTC), combined into daily bars and
  stored apart from the 2024–2026 data: `data/holdout/`. Before the run: no daily path, return or
  statistic of this window has been computed in this project.
- Signal at the close of day t: `close[t] / close[t-14] - 1 > 0` → long, else flat.
- Execution at the open of day t+1; daily return = open-to-open; long-only, 100% notional, no leverage.
- Costs: 0.40% taker fee + 1 bp slippage per side on every change in position.
- Benchmark: buy-and-hold over the same days, entry cost charged once. First 14 days are warm-up.
- Data must pass the merge checks (no gaps, no overlaps, no missing prices) before the test is run.

## Pass rule (all three, same as H2)
1. Strategy annualized Sharpe (365 days) > buy-and-hold Sharpe.
2. Paired circular block bootstrap of the daily Sharpe difference, block 20 days, 10,000 resamples,
   seed 0: P(difference ≤ 0) < 5%.
3. Strategy max drawdown no worse than buy-and-hold's.

## Reported but not used to decide
Fees 0.26% and 0.16%; lookbacks 28 and 56; results by calendar year.

## What follows
- PASS: paper run of the rule under the Risk Governor (daily bars), plus a fee check against your
  actual Kraken tier. Still no live trading without Risk Chair approval.
- FAIL: the 14-day result in H2 was noise; stop testing trend variants on this data.

## Result (appended after the run): FAIL

Data: 722,016 five-minute bars, 2018-01-01 to 2024-11-11, merged with no gaps or overlaps. The long
empty stretches (2018-01-11/13, about 2 days; 2024-04-14, 7h) are Kraken trading halts, confirmed against
Kraken's API.

`python scripts/trend_test.py data/holdout/kraken_BTCUSD_5m_trades.csv --lookback 14`

Daily bars 2018-01-01 -> 2024-11-11 (2507); evaluated 2491 days; time in market 53%; switches 291

| | Sharpe | total return | max drawdown |
|---|---|---|---|
| trend 14d, 0.40% | +0.76 | +440.4% | -61.0% |
| buy and hold | +0.73 | +488.2% | -76.7% |

- [PASS] Sharpe above buy-and-hold
- [FAIL] bootstrap P(diff <= 0) = 0.468 < 0.05
- [PASS] drawdown no worse than buy-and-hold

VERDICT (14d, 0.40%): FAIL

By calendar year (reported, not used to decide):

| year | trend return | buy-and-hold return | time in market |
|---|---|---|---|
| 2018 | -39.2% | -73.0% | 37% |
| 2019 | +55.2% | +94.2% | 56% |
| 2020 | +241.1% | +304.0% | 69% |
| 2021 | +56.9% | +59.4% | 52% |
| 2022 | -51.9% | -64.2% | 41% |
| 2023 | +73.8% | +155.6% | 63% |
| 2024 | +28.1% | +90.4% | 56% |

Sensitivity (reported, not used to decide):

| lookback | fee | Sharpe | total return | max DD | switches | P(diff<=0) |
|---|---|---|---|---|---|---|
| 14d | 0.40% | +0.76 | +440.4% | -61.0% | 291 | 0.468 |
| 14d | 0.26% | +0.89 | +713.4% | -57.8% | 291 | 0.288 |
| 14d | 0.16% | +0.98 | +989.0% | -55.3% | 291 | 0.184 |
| 28d | 0.40% | +0.89 | +722.2% | -66.9% | 187 | 0.363 |
| 28d | 0.26% | +0.97 | +968.4% | -65.1% | 187 | 0.263 |
| 28d | 0.16% | +1.03 | +1187.9% | -63.8% | 187 | 0.202 |
| 56d | 0.40% | +1.09 | +1345.9% | -49.3% | 127 | 0.170 |
| 56d | 0.26% | +1.15 | +1627.8% | -47.4% | 127 | 0.127 |
| 56d | 0.16% | +1.19 | +1861.9% | -46.0% | 127 | 0.100 |

The rule matched buy-and-hold on Sharpe (+0.76 vs +0.73) and cut the worst drawdown from −77% to −61%,
but the Sharpe difference is indistinguishable from luck (P = 0.47; the rule needed < 0.05). No lookback
or fee in the sensitivity table reaches 0.05 either. As pre-registered, trend variants are no longer
tested on this data.

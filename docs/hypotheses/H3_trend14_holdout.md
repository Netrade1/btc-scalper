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

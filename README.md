# btc-scalper

Institutional research/paper-trading platform bootstrap for K2Think.ai.

## Included in this slice

- Safety-first platform configuration with explicit separation:
  - `PRODUCTION_APPROVED` (governed research/paper environments)
  - `LIVE_CAPITAL_APPROVED` (separate live-capital authorization state)
- Immutable hash-chained audit evidence records
- Mock Level2 feed adapter and mock paper broker adapter
- One real-venue Level2 adapter normalization layer (`BinanceDiffDepthAdapter`)
- Sequence-aware market-by-price order book reconstruction
- Official paper-broker client wrapper (`OfficialPaperBrokerAdapter`)
- Execution reconciliation engine for order-vs-execution integrity checks
- Shared contract tests for configuration, audit, adapters, replay integrity, and reconciliation

## Controls (defaulted safe)

- `RESEARCH_ONLY = true`
- `PAPER_TRADING_ONLY = true`
- `LIVE_EXECUTION = false`
- `HUMAN_APPROVAL_REQUIRED = true`
- `AUTOMATIC_MODEL_PROMOTION = false`
- `FAIL_CLOSED = true`

## Run tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

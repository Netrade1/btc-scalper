# btc-scalper

Institutional research/paper-trading platform bootstrap for K2Think.ai.

## Included in this slice

- Safety-first platform configuration with explicit separation:
  - `PRODUCTION_APPROVED` (governed research/paper environments)
  - `LIVE_CAPITAL_APPROVED` (separate live-capital authorization state)
- Immutable hash-chained audit evidence records
- Mock Level2 feed adapter and mock paper broker adapter
- Shared contract tests for configuration, audit, and adapters

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

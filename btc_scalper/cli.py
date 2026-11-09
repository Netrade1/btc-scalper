"""Command line: backtest | fetch | paper | live."""
from __future__ import annotations

import argparse
import logging
import sys

from .config import SystemConfig


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="btc_scalper", description="Walk-forward ML trading system")
    ap.add_argument("--config", help="TOML config file (defaults are used if omitted)")
    ap.add_argument("--log-level", default="INFO")
    sub = ap.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="Walk-forward backtest on CSV or synthetic data")
    bt.add_argument("--csv", help="OHLCV CSV (timestamp,open,high,low,close,volume[,buy_volume])")
    bt.add_argument("--synthetic-bars", type=int, default=20_000)
    bt.add_argument("--alpha", type=float, default=0.0, help="planted edge strength for synthetic data")
    bt.add_argument("--seed", type=int, default=7)
    bt.add_argument("--out", default="reports/latest")

    fe = sub.add_parser("fetch", help="Download historical closed bars to CSV")
    fe.add_argument("--days", type=float, default=60)
    fe.add_argument("--out", required=True)

    vl = sub.add_parser("verify-ledger", help="Verify a hash-chained audit ledger")
    vl.add_argument("path")

    for name in ("paper", "live"):
        p = sub.add_parser(name, help=f"{name} trading daemon")
        p.add_argument("--reset-kill-switch", action="store_true", help="operator override after review")
        p.add_argument("--max-cycles", type=int, default=None)
        if name == "live":
            p.add_argument("--i-accept-the-risk", action="store_true", required=True,
                           help="required: real orders will be sent")

    args = ap.parse_args(argv)
    _setup_logging(args.log_level)
    cfg = SystemConfig.load(args.config)
    log = logging.getLogger("btc_scalper")

    if args.cmd == "backtest":
        from .backtest import run_backtest
        from .data import generate_synthetic_ohlcv, load_csv
        from .metrics import format_summary

        if args.csv:
            raw = load_csv(args.csv, cfg.bar_seconds, cfg.data.allow_flow_proxy)
        else:
            log.info("No --csv given: using %d synthetic bars (alpha=%.2f)", args.synthetic_bars, args.alpha)
            raw = generate_synthetic_ohlcv(args.synthetic_bars, cfg.bar_seconds, args.seed, args.alpha)
        res = run_backtest(raw, cfg, args.out)
        print("\n=== Walk-forward backtest ===\n" + format_summary(res.summary))
        print(f"\nReports written to {args.out}/")
        return 0

    if args.cmd == "verify-ledger":
        from .audit import verify_ledger

        ok, n, err = verify_ledger(args.path)
        print(f"{'OK' if ok else 'TAMPERED'}: {n} records verified" + (f" - {err}" if err else ""))
        return 0 if ok else 2

    from .exchange import CCXTClient

    if args.cmd == "fetch":
        import pandas as pd

        from .data import save_csv

        client = CCXTClient(cfg.data)
        start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.days)
        df = client.fetch_range(start)
        save_csv(df, args.out)
        log.info("Saved %d bars to %s", len(df), args.out)
        return 0

    from . import safety
    from .live import LiveRunner

    live = args.cmd == "live"
    if live and safety.PAPER_TRADING_ONLY:
        raise PermissionError("PAPER_TRADING_ONLY: live trading is disabled in btc_scalper/safety.py. "
                              "Run `paper` first; enabling live needs Risk Chair approval.")
    client = CCXTClient(cfg.data, cfg.live, authenticated=live)
    LiveRunner(cfg, client, live=live, reset_kill_switch=args.reset_kill_switch).run(args.max_cycles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Backfill long trade history in parallel chunks, then merge into the main trade store.

    python scripts/backfill_history.py --start 2024-11-12 --workers 4
    python scripts/backfill_history.py --merge-only

Each chunk is its own resumable store (data/chunks/). Re-run the same command to
resume after an error. The merge fails closed if chunks leave gaps or overlap.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

from btc_scalper.config import SystemConfig
from btc_scalper.exchange import CCXTClient
from btc_scalper.trades import TradeBarStore, default_store_path


def chunk_paths(cfg, n):
    base = os.path.join(cfg.data.trade_store_dir, "chunks")
    os.makedirs(base, exist_ok=True)
    name = os.path.basename(default_store_path(cfg.data.trade_store_dir, cfg.data.exchange,
                                               cfg.data.symbol, cfg.data.timeframe))
    return [os.path.join(base, name.replace(".csv", f"_{i:02d}.csv")) for i in range(n)]


def run_chunk(job):
    i, path, start, end, cfg_path = job
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [chunk {i}] %(message)s")
    cfg = SystemConfig.load(cfg_path) if cfg_path else SystemConfig()
    client = CCXTClient(cfg.data, cfg.live)
    for attempt in range(20):
        store = TradeBarStore(client.exchange, cfg.data.symbol, cfg.bar_seconds, path,
                              side_is_maker=cfg.data.trade_side_is_maker)
        done = store.last_bar_end_ms()
        if done is not None and done >= int(end.timestamp() * 1000):
            return i, "done"
        try:
            store.update(start=start, end=end, progress_every=200)
            return i, "done"
        except Exception as e:  # resumable: back off and continue from the last bar
            logging.warning("chunk %d error (%s); resuming in %ds", i, e, 30 * (attempt + 1))
            time.sleep(30 * (attempt + 1))
    return i, "failed"


def merge(cfg, paths, main_path):
    frames = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    if os.path.exists(main_path):
        frames.append(pd.read_csv(main_path))
    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df = df.assign(_ts=ts).sort_values("_ts")
    if df["_ts"].duplicated().any():
        raise SystemExit(f"Merge refused: {int(df['_ts'].duplicated().sum())} overlapping bars")
    step = df["_ts"].diff().dt.total_seconds().iloc[1:]
    if (step != cfg.bar_seconds).any():
        raise SystemExit(f"Merge refused: {int((step != cfg.bar_seconds).sum())} gaps between bars")
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise SystemExit("Merge refused: bars with missing prices")
    tmp = main_path + ".tmp"
    df.drop(columns="_ts").to_csv(tmp, index=False)
    os.replace(tmp, main_path)
    print(f"Merged {len(df)} bars {df['_ts'].iloc[0]} -> {df['_ts'].iloc[-1]}, "
          f"empty bars {int((df['trades'] == 0).sum())} -> {main_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-11-12")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=None)
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    cfg = SystemConfig.load(a.config) if a.config else SystemConfig()
    main_path = default_store_path(cfg.data.trade_store_dir, cfg.data.exchange, cfg.data.symbol,
                                   cfg.data.timeframe)
    first = pd.read_csv(main_path, nrows=1)["timestamp"].iloc[0] if os.path.exists(main_path) else None
    if first is None:
        raise SystemExit(f"{main_path} missing: run `fetch-trades` first so the chunks have an end point")
    start, end = pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(first)
    paths = chunk_paths(cfg, a.workers)
    if not a.merge_only:
        edges = pd.date_range(start, end, periods=a.workers + 1)
        edges = [e.floor(f"{cfg.bar_seconds}s") for e in edges[:-1]] + [end]
        jobs = [(i, paths[i], edges[i], edges[i + 1], a.config) for i in range(a.workers)]
        with Pool(a.workers) as pool:
            results = pool.map(run_chunk, jobs)
        print(results)
        if any(r != "done" for _, r in results):
            sys.exit("Some chunks failed; re-run to resume")
    merge(cfg, paths, main_path)


if __name__ == "__main__":
    main()

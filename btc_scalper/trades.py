"""Build OHLCV bars with REAL taker-buy volume from an exchange's public trade feed.

Kraken, Coinbase and most venues don't put taker-buy volume in their candles,
but every public trade carries the taker's side. This module paginates the trade
feed through ccxt, aggregates it into bars and appends *completed* bars to a
local CSV store, so:

  * a backfill can be stopped and resumed (it restarts at the last complete bar);
  * the paper loop only fetches the few trades since its last bar each cycle;
  * bars with no trades are written explicitly (flat, zero volume) so the index
    has no artificial gaps.

ccxt's ``side`` on a public trade is the taker side for Kraken (``b``/``s``).
Check before using another venue: on some, the reported side is the maker's
(set ``data.trade_side_is_maker = true`` for those).
"""
from __future__ import annotations

import csv
import logging
import os
import time
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "buy_volume", "trades"]


def default_store_path(directory: str, exchange: str, symbol: str, timeframe: str) -> str:
    safe = symbol.replace("/", "").replace(":", "_")
    return os.path.join(directory, f"{exchange}_{safe}_{timeframe}_trades.csv")


class TradeBarStore:
    def __init__(self, exchange: Any, symbol: str, bar_seconds: int, path: str,
                 side_is_maker: bool = False, page_limit: int = 1000,
                 publish_delay_seconds: float = 3.0, now_ms: Callable[[], int] | None = None):
        self.ex, self.symbol, self.bar_ms = exchange, symbol, bar_seconds * 1000
        self.path, self.side_is_maker, self.page_limit = path, side_is_maker, page_limit
        self.publish_delay_ms = int(publish_delay_seconds * 1000)
        self.now_ms = now_ms or (lambda: int(exchange.milliseconds()))

    # ------------------------------------------------------------ store I/O
    def last_bar_end_ms(self) -> int | None:
        if not os.path.exists(self.path):
            return None
        last = None
        with open(self.path, newline="") as fh:
            for row in csv.DictReader(fh):
                last = row["timestamp"]
        return None if last is None else int(pd.Timestamp(last).timestamp() * 1000) + self.bar_ms

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("timestamp"), utc=True), name="timestamp")
        return df

    def _append(self, bars: list[list]) -> None:
        if not bars:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        new = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(COLUMNS)
            for b in bars:
                w.writerow([pd.Timestamp(b[0], unit="ms", tz="UTC").isoformat()] + b[1:])

    # ---------------------------------------------------------- aggregation
    def _bars_between(self, trades: list[dict], start_ms: int, end_ms: int, prev_close: float | None) -> list[list]:
        """Aggregate trades into every bar in [start_ms, end_ms). Empty bars are flat at the prior close."""
        out, i, n = [], 0, len(trades)
        for t0 in range(start_ms, end_ms, self.bar_ms):
            t1 = t0 + self.bar_ms
            o = h = l = c = None
            vol = buy = 0.0
            cnt = 0
            while i < n and trades[i]["timestamp"] < t1:
                tr = trades[i]
                px, amt = float(tr["price"]), float(tr["amount"])
                if o is None:
                    o = h = l = px
                h, l, c = max(h, px), min(l, px), px
                vol += amt
                taker_buy = (tr["side"] == "buy") != self.side_is_maker
                buy += amt if taker_buy else 0.0
                cnt += 1
                i += 1
            if o is None:
                if prev_close is None:
                    continue  # nothing to anchor a flat bar to yet
                o = h = l = c = prev_close
            out.append([t0, o, h, l, c, vol, buy, cnt])
            prev_close = c
        return out

    def _last_close(self) -> float | None:
        if not os.path.exists(self.path):
            return None
        last = None
        with open(self.path, newline="") as fh:
            for row in csv.DictReader(fh):
                last = row
        return float(last["close"]) if last else None

    # -------------------------------------------------------------- update
    def update(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
               max_pages: int | None = None, progress_every: int = 50) -> int:
        """Fetch trades from the last stored bar (or ``start``) and append completed bars.

        Returns the number of bars written. Safe to interrupt and call again.
        """
        resume = self.last_bar_end_ms()
        if resume is None:
            if start is None:
                raise ValueError(f"Trade store {self.path} is empty: run `fetch-trades` to backfill first")
            resume = int(start.timestamp() * 1000) // self.bar_ms * self.bar_ms
        end_ms = int(end.timestamp() * 1000) if end is not None else None
        bar_start, cursor = resume, resume
        buf: list[dict] = []
        seen_at_cursor: set = set()
        prev_close = self._last_close()
        written = pages = 0
        t_start = time.time()
        while True:
            from .exchange import _retry  # local import to avoid a cycle

            page = _retry(self.ex.fetch_trades, self.symbol, cursor, self.page_limit)
            pages += 1
            fresh = [t for t in page if t["timestamp"] >= cursor
                     and not (t["timestamp"] == cursor and t.get("id") in seen_at_cursor)]
            fresh.sort(key=lambda t: t["timestamp"])
            if end_ms is not None:
                fresh = [t for t in fresh if t["timestamp"] < end_ms]
            buf.extend(fresh)
            if fresh:
                new_cursor = fresh[-1]["timestamp"]
                ids = {t.get("id") for t in fresh if t["timestamp"] == new_cursor}
                seen_at_cursor = ids if new_cursor != cursor else seen_at_cursor | ids
                cursor = new_cursor
            caught_up = len(fresh) == 0 or (end_ms is not None and cursor >= end_ms - 1)
            # Bars ending at or before the newest trade are complete; at the live edge,
            # bars that ended (plus a publish delay) before now are complete too.
            limit_ms = cursor
            if caught_up:
                limit_ms = end_ms if end_ms is not None else self.now_ms() - self.publish_delay_ms
            complete_end = limit_ms // self.bar_ms * self.bar_ms
            if complete_end > bar_start:
                k = 0
                while k < len(buf) and buf[k]["timestamp"] < complete_end:
                    k += 1
                bars = self._bars_between(buf[:k], bar_start, complete_end, prev_close)
                self._append(bars)
                if bars:
                    prev_close = bars[-1][4]
                written += len(bars)
                buf = buf[k:]
                bar_start = complete_end
            if pages % progress_every == 0:
                rate = pages / max(time.time() - t_start, 1e-9)
                logger.info("Trades backfill: %d pages (%.1f/s), bars up to %s", pages, rate,
                            pd.Timestamp(bar_start, unit="ms", tz="UTC"))
            if caught_up or (max_pages is not None and pages >= max_pages):
                break
        return written

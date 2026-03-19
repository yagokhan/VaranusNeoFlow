#!/usr/bin/env python3
"""
data_fetcher.py — Historical OHLCV Fetcher for Varanus Neo-Flow

Fetches 5m, 30m, 1h, 4h data from Binance public REST API for 15 Tier-2
assets across three test periods:

  Training/Backtest : 2024-01-01 → 2024-12-31
  Validation        : 2025-01-01 → 2025-06-30
  Blind Test        : 2025-07-01 → 2026-03-15

Saves one Parquet file per asset/timeframe in varanus_neo_flow/data/.
Resumable: skips date ranges already present in existing files.

Usage
-----
    # Fetch everything (resumes automatically):
    python data_fetcher.py

    # Fetch a single asset:
    python data_fetcher.py --asset ADAUSDT

    # Fetch a single timeframe:
    python data_fetcher.py --tf 5m

    # Dry run (show what would be fetched):
    python data_fetcher.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent / "data"

ASSETS = [
    "ADAUSDT",   "AVAXUSDT",  "LINKUSDT",  "DOTUSDT",   "TRXUSDT",
    "SOLUSDT",   "ATOMUSDT",  "NEARUSDT",  "ALGOUSDT",  "UNIUSDT",
    "ICPUSDT",   "HBARUSDT",  "SANDUSDT",  "MANAUSDT",  "THETAUSDT",
]

TIMEFRAMES = ["5m", "30m", "1h", "4h"]

# Binance kline intervals → milliseconds per bar
TF_MS = {
    "5m":   5 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "4h":  4 * 60 * 60 * 1000,
}

# Global date range — aligned with v5.7.1 methodology:
#   Training (8-fold WFV): 2023-01-01 → 2025-10-31
#   Blind Test:            2025-11-01 → 2026-03-15
GLOBAL_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
GLOBAL_END   = datetime(2026, 3, 19, 15, 0, 0, tzinfo=timezone.utc)

# Binance public API
KLINES_URL   = "https://api.binance.com/api/v3/klines"
MAX_CANDLES   = 1000     # Binance limit per request
RATE_LIMIT_S  = 0.15     # 150ms between requests (~400 req/min, well under 1200)

# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("data_fetcher")

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ts_ms(dt: datetime) -> int:
    """Datetime → Binance-style millisecond timestamp."""
    return int(dt.timestamp() * 1000)


def _parquet_path(symbol: str, tf: str) -> Path:
    return DATA_DIR / f"{symbol}_{tf}.parquet"


def _load_existing(path: Path) -> pd.DataFrame | None:
    """Load existing Parquet file if it exists."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            return None
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    return df


def _fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    """
    Fetch klines from Binance with pagination.
    Handles the 1000-bar limit by issuing multiple requests.
    Returns list of raw kline arrays.
    """
    all_klines: list[list] = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": cursor,
            "endTime":   end_ms,
            "limit":     MAX_CANDLES,
        }

        for attempt in range(5):
            try:
                resp = requests.get(KLINES_URL, params=params, timeout=15)

                # Rate limit hit — back off
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    logger.warning(
                        "Rate limited — sleeping %ds (attempt %d/5)",
                        retry_after, attempt + 1,
                    )
                    time.sleep(retry_after)
                    continue

                # IP ban — longer back off
                if resp.status_code == 418:
                    logger.error("IP banned (418) — sleeping 120s")
                    time.sleep(120)
                    continue

                resp.raise_for_status()
                batch = resp.json()
                break

            except requests.RequestException as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Request error (attempt %d/5): %s — retrying in %ds",
                    attempt + 1, exc, wait,
                )
                time.sleep(wait)
                if attempt == 4:
                    raise
        else:
            raise RuntimeError(f"Failed after 5 retries: {symbol} {interval}")

        if not batch:
            break

        all_klines.extend(batch)

        # Advance cursor past the last received bar
        last_open_ms = int(batch[-1][0])
        next_cursor = last_open_ms + TF_MS[interval]

        if next_cursor <= cursor:
            # Safety: avoid infinite loop
            break
        cursor = next_cursor

        # Rate limiting
        time.sleep(RATE_LIMIT_S)

    return all_klines


def _klines_to_df(raw: list[list]) -> pd.DataFrame:
    """Convert raw Binance kline arrays to a clean DataFrame."""
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    df = df[["open", "high", "low", "close", "volume"]]
    df = df.sort_index()

    # Drop duplicates (can happen at pagination boundaries)
    df = df[~df.index.duplicated(keep="first")]

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Core Fetch Logic
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_symbol_tf(
    symbol: str,
    tf: str,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Fetch (or resume) OHLCV data for one symbol/timeframe.

    If a Parquet file already exists, only fetches the missing tail.
    Merges new data with existing and overwrites the file.
    """
    path = _parquet_path(symbol, tf)
    existing = _load_existing(path)

    global_start_ms = _ts_ms(GLOBAL_START)
    global_end_ms   = _ts_ms(GLOBAL_END)

    if existing is not None and not existing.empty:
        first_ts = existing.index[0]
        last_ts  = existing.index[-1]

        first_ms = int(first_ts.timestamp() * 1000)
        tail_resume_ms = int(last_ts.timestamp() * 1000) + TF_MS[tf]

        need_head = first_ms > global_start_ms  # Missing older data at start
        need_tail = tail_resume_ms < global_end_ms  # Missing newer data at end

        if not need_head and not need_tail:
            logger.info(
                "  %-12s %-3s  SKIP — already complete (%d bars, %s → %s)",
                symbol, tf, len(existing), first_ts, last_ts,
            )
            return existing

        bars_have = len(existing)
        if need_head:
            logger.info(
                "  %-12s %-3s  BACKFILL %s → %s + tail (%d bars cached)",
                symbol, tf, GLOBAL_START.date(), first_ts.date(), bars_have,
            )
        else:
            logger.info(
                "  %-12s %-3s  RESUME from %s (%d bars cached)",
                symbol, tf, pd.Timestamp(tail_resume_ms, unit="ms", tz="UTC"), bars_have,
            )
    else:
        need_head = True
        need_tail = True
        first_ms = global_end_ms
        tail_resume_ms = global_start_ms
        logger.info("  %-12s %-3s  FULL FETCH %s → %s", symbol, tf, GLOBAL_START.date(), GLOBAL_END.date())

    if dry_run:
        return existing if existing is not None else pd.DataFrame()

    # Fetch missing head (older data before existing start)
    head_df = pd.DataFrame()
    if need_head and existing is not None and not existing.empty:
        head_end_ms = first_ms - TF_MS[tf]
        raw_head = _fetch_klines(symbol, tf, global_start_ms, head_end_ms)
        head_df = _klines_to_df(raw_head)
        if not head_df.empty:
            logger.info("  %-12s %-3s  HEAD fetched %d bars", symbol, tf, len(head_df))

    # Fetch missing tail (newer data after existing end)
    tail_df = pd.DataFrame()
    if need_tail:
        fetch_start = tail_resume_ms if (existing is not None and not existing.empty) else global_start_ms
        raw_tail = _fetch_klines(symbol, tf, fetch_start, global_end_ms)
        tail_df = _klines_to_df(raw_tail)

    # Combine: head + existing + tail
    parts = [df for df in [head_df, existing, tail_df] if df is not None and not df.empty]
    if not parts:
        logger.warning("  %-12s %-3s  No data returned", symbol, tf)
        return existing if existing is not None else pd.DataFrame()
    new_df = pd.concat(parts)

    # Deduplicate and sort
    combined = new_df[~new_df.index.duplicated(keep="last")]
    combined = combined.sort_index()

    # Trim to global range
    combined = combined[
        (combined.index >= pd.Timestamp(GLOBAL_START))
        & (combined.index <= pd.Timestamp(GLOBAL_END))
    ]

    # Save
    combined.to_parquet(path, engine="pyarrow")
    logger.info(
        "  %-12s %-3s  SAVED %d bars (%s → %s)",
        symbol, tf, len(combined),
        combined.index[0].strftime("%Y-%m-%d"),
        combined.index[-1].strftime("%Y-%m-%d %H:%M"),
    )

    return combined


def fetch_all(
    assets: list[str] | None = None,
    timeframes: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Fetch all assets × all timeframes.

    Returns nested dict: {symbol: {tf: DataFrame}}.
    """
    assets = assets or ASSETS
    timeframes = timeframes or TIMEFRAMES

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total = len(assets) * len(timeframes)
    done = 0
    results: dict[str, dict[str, pd.DataFrame]] = {}

    logger.info("=" * 70)
    logger.info("Varanus Neo-Flow Data Fetcher")
    logger.info("  Assets:     %d", len(assets))
    logger.info("  Timeframes: %s", ", ".join(timeframes))
    logger.info("  Range:      %s → %s", GLOBAL_START.date(), GLOBAL_END.date())
    logger.info("  Output:     %s", DATA_DIR)
    logger.info("  Dry run:    %s", dry_run)
    logger.info("=" * 70)

    for symbol in assets:
        results[symbol] = {}
        logger.info("[%s]", symbol)

        for tf in timeframes:
            df = fetch_symbol_tf(symbol, tf, dry_run=dry_run)
            results[symbol][tf] = df
            done += 1

        logger.info("")  # blank line between assets

    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    for symbol in assets:
        parts = []
        for tf in timeframes:
            df = results[symbol].get(tf)
            n = len(df) if df is not None and not df.empty else 0
            parts.append(f"{tf}={n:>7,}")
        logger.info("  %-12s  %s", symbol, "  ".join(parts))

    # Expected bar counts (approximate)
    logger.info("")
    logger.info("Expected bar counts (2024-01-01 → 2026-03-15):")
    days = (GLOBAL_END - GLOBAL_START).days
    for tf in timeframes:
        expected = days * 24 * 60 // (TF_MS[tf] // 60000)
        logger.info("  %-3s  ~%d bars", tf, expected)

    logger.info("=" * 70)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Varanus Neo-Flow — Historical OHLCV Data Fetcher",
    )
    parser.add_argument(
        "--asset", type=str, default=None,
        help="Fetch a single asset (e.g. ADAUSDT). Default: all 15.",
    )
    parser.add_argument(
        "--tf", type=str, default=None,
        help="Fetch a single timeframe (5m, 30m, 1h, 4h). Default: all.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be fetched without downloading.",
    )
    args = parser.parse_args()

    assets = [args.asset.upper()] if args.asset else None
    timeframes = [args.tf] if args.tf else None

    fetch_all(assets=assets, timeframes=timeframes, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

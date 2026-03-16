#!/usr/bin/env python3
"""Quick test: run Neo-Flow scanner against v5.7.1 cached data."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/gokhan/varanus_neo_flow")

import pandas as pd
import numpy as np
from pathlib import Path
from neo_flow.adaptive_engine import (
    calc_log_regression, find_best_regression, scan_asset,
    scan_universe, print_scan_report, TIER2_UNIVERSE, SCAN_TIMEFRAMES,
)

CACHE = Path("/home/gokhan/varanus_v571/varanus/data/cache")

def load_asset(asset: str) -> dict[str, pd.DataFrame]:
    """Load from v5.7.1 cache — 4h parquet + resample to 1h from 1h parquet."""
    result = {}

    # 4h data
    p4h = CACHE / f"{asset}_USDT.parquet"
    if p4h.exists():
        df = pd.read_parquet(p4h)
        df.columns = [c.lower() for c in df.columns]
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        result["4h"] = df

        # Resample 4h → "1h" placeholder (we don't have real 1h for all)
        # Actually use the 1h parquet if available

    # 1h data
    p1h = CACHE / f"{asset}_USDT_1h.parquet"
    if p1h.exists():
        df1h = pd.read_parquet(p1h)
        df1h.columns = [c.lower() for c in df1h.columns]
        if not isinstance(df1h.index, pd.DatetimeIndex):
            df1h = df1h.set_index("timestamp")
        df1h.index = pd.to_datetime(df1h.index, utc=True)
        df1h = df1h.sort_index()
        result["1h"] = df1h

        # Resample 1h → 30m not possible (need finer data)
        # Resample 1h → simulate 30m by using 1h as-is for testing
        # In production, will fetch real 30m and 5m from Binance

    return result

# ── Unit test: regression math ───────────────────────────────────────────
print("=== Unit Test: calc_log_regression ===")
np.random.seed(42)
n = 100
trend = np.exp(np.linspace(0, 0.5, n))  # rising trend
prices = trend * (1 + 0.01 * np.random.randn(n))
std_dev, pearson_r, slope, intercept = calc_log_regression(prices, 50)
print(f"Rising trend:  slope={slope:.6f}  R={pearson_r:.4f}  (expect slope<0, |R|~1)")

trend_down = np.exp(np.linspace(0.5, 0, n))
prices_down = trend_down * (1 + 0.01 * np.random.randn(n))
std_dev, pearson_r, slope, intercept = calc_log_regression(prices_down, 50)
print(f"Falling trend: slope={slope:.6f}  R={pearson_r:.4f}  (expect slope>0, |R|~1)")

flat = 100 + 2 * np.random.randn(n)
std_dev, pearson_r, slope, intercept = calc_log_regression(flat, 50)
print(f"Flat/noisy:    slope={slope:.6f}  R={pearson_r:.4f}  (expect |R|~0)")
print()

# ── Load data and run scan ───────────────────────────────────────────────
print("=== Loading cached data ===")
market_data = {}
for asset in TIER2_UNIVERSE:
    data = load_asset(asset)
    if data:
        market_data[asset] = data
        tfs = list(data.keys())
        print(f"  {asset}: {tfs}")

print()

# ── Print scan report ────────────────────────────────────────────────────
print("=== Scan Report (top 3 per asset) ===")
print_scan_report(market_data, top_n=3)

# ── Run full scanner ─────────────────────────────────────────────────────
print("=== Full Scanner ===")
signals = scan_universe(market_data, open_positions={})
if signals:
    for s in signals:
        d = "LONG" if s.direction == 1 else "SHORT"
        print(f"  {s.asset:<8} {d:>5}  |R|={s.confidence:.4f}  TF={s.best_tf}  P={s.best_period}  "
              f"entry={s.entry_price:.6f}  SL={s.sl_price:.6f}")
else:
    print("  No signals generated (all assets filtered out)")

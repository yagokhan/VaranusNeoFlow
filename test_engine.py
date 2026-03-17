#!/usr/bin/env python3
"""Test Neo-Flow engine against fetched historical data (5m, 30m, 1h, 4h)."""
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/yagokhan/VaranusNeoFlow")

import pandas as pd
import numpy as np
from pathlib import Path
from neo_flow.adaptive_engine import (
    calc_log_regression, calc_linear_regression,
    compute_pvt, compute_pvt_regression, check_pvt_alignment,
    find_best_regression, scan_asset, scan_universe,
    print_scan_report, TIER2_UNIVERSE, SCAN_TIMEFRAMES,
    trim_to_7d, BARS_7D,
)

DATA_DIR = Path("/home/yagokhan/VaranusNeoFlow/data")

def load_asset(asset: str) -> dict[str, pd.DataFrame]:
    """Load all timeframes from fetched parquet files."""
    result = {}
    symbol = f"{asset}USDT"
    for tf in ["5m", "30m", "1h", "4h"]:
        path = DATA_DIR / f"{symbol}_{tf}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            result[tf] = df
    return result

# ── Unit tests ───────────────────────────────────────────────────────────
print("=" * 70)
print("UNIT TEST: calc_log_regression")
print("=" * 70)
np.random.seed(42)
n = 200
trend = np.exp(np.linspace(0, 0.5, n))
prices = trend * (1 + 0.005 * np.random.randn(n))
std_dev, pearson_r, slope, intercept = calc_log_regression(prices, 50)
print(f"  Rising trend:  slope={slope:.6f}  R={pearson_r:.4f}  (expect slope<0, |R|~1)")
assert slope < 0, f"Rising trend should have negative slope, got {slope}"
assert abs(pearson_r) > 0.9, f"Rising trend should have |R|~1, got {pearson_r}"
print("  PASS")

trend_down = np.exp(np.linspace(0.5, 0, n))
prices_down = trend_down * (1 + 0.005 * np.random.randn(n))
std_dev, pearson_r, slope, intercept = calc_log_regression(prices_down, 50)
print(f"  Falling trend: slope={slope:.6f}  R={pearson_r:.4f}  (expect slope>0, |R|~1)")
assert slope > 0, f"Falling trend should have positive slope, got {slope}"
assert abs(pearson_r) > 0.9, f"Falling trend should have |R|~1, got {pearson_r}"
print("  PASS")

flat = 100 + np.random.randn(n)
std_dev, pearson_r, slope, intercept = calc_log_regression(flat, 50)
print(f"  Flat/noisy:    slope={slope:.6f}  R={pearson_r:.4f}  (expect |R|~0)")
assert abs(pearson_r) < 0.5, f"Flat should have low |R|, got {pearson_r}"
print("  PASS")
print()

print("=" * 70)
print("UNIT TEST: PVT computation")
print("=" * 70)
# Synthetic: steadily rising close with steady volume
df_test = pd.DataFrame({
    "close": np.exp(np.linspace(0, 0.3, 100)),
    "volume": np.ones(100) * 1000,
    "open": np.exp(np.linspace(0, 0.3, 100)),
    "high": np.exp(np.linspace(0, 0.3, 100)) * 1.01,
    "low": np.exp(np.linspace(0, 0.3, 100)) * 0.99,
})
pvt = compute_pvt(df_test)
print(f"  PVT[0]={pvt[0]:.2f}  PVT[-1]={pvt[-1]:.2f}  (expect rising)")
assert pvt[-1] > pvt[0], "PVT should rise with rising prices"
pvt_reg = compute_pvt_regression(df_test, 50)
print(f"  PVT regression: R={pvt_reg.pearson_r:.4f}  slope={pvt_reg.slope:.6f}  dir={pvt_reg.direction}")
assert pvt_reg.direction == 1, f"Expected bullish PVT direction, got {pvt_reg.direction}"
print("  PASS")
print()

print("=" * 70)
print("UNIT TEST: PVT alignment gate")
print("=" * 70)
# Case 1: Aligned - LONG with rising PVT, strong R
from neo_flow.adaptive_engine import PVTResult
pvt_ok = PVTResult(pearson_r=-0.85, slope=-0.01, direction=1)
passes, reason = check_pvt_alignment(1, 0.90, pvt_ok)
print(f"  Aligned LONG+rising PVT: passes={passes}  ({reason})")
assert passes, f"Should pass, got: {reason}"

# Case 2: Divergence - LONG with falling PVT
pvt_bad = PVTResult(pearson_r=0.80, slope=0.01, direction=-1)
passes, reason = check_pvt_alignment(1, 0.90, pvt_bad)
print(f"  Divergence LONG+falling PVT: passes={passes}  ({reason})")
assert not passes, "Should be blocked"

# Case 3: Weak PVT
pvt_weak = PVTResult(pearson_r=-0.40, slope=-0.001, direction=1)
passes, reason = check_pvt_alignment(1, 0.90, pvt_weak)
print(f"  Weak PVT: passes={passes}  ({reason})")
assert not passes, "Should be blocked (weak volume)"

# Case 4: Direction match but PVT R below minimum
pvt_low = PVTResult(pearson_r=-0.60, slope=-0.005, direction=1)
passes, reason = check_pvt_alignment(1, 0.82, pvt_low)
print(f"  PVT R below min: passes={passes}  ({reason})")
assert not passes, "Should be blocked (PVT too weak)"

print("  ALL PVT GATE TESTS PASS")
print()

# ── Load real data ───────────────────────────────────────────────────────
print("=" * 70)
print("LOADING HISTORICAL DATA")
print("=" * 70)
market_data = {}
for asset in TIER2_UNIVERSE:
    data = load_asset(asset)
    if data:
        market_data[asset] = data
        tfs = {tf: len(df) for tf, df in data.items()}
        print(f"  {asset:<8} {tfs}")

print(f"\n  Loaded {len(market_data)}/{len(TIER2_UNIVERSE)} assets")
print()

# ── Test 7-day trim ─────────────────────────────────────────────────────
print("=" * 70)
print("UNIT TEST: 7-day window trim")
print("=" * 70)
sample_asset = list(market_data.keys())[0]
for tf in SCAN_TIMEFRAMES:
    df = market_data[sample_asset].get(tf)
    if df is not None:
        trimmed = trim_to_7d(df, tf)
        expected = BARS_7D[tf]
        print(f"  {sample_asset} {tf}: {len(df)} → {len(trimmed)} bars (expected {expected})")
        assert len(trimmed) == expected, f"Expected {expected}, got {len(trimmed)}"
print("  PASS")
print()

# ── Scan performance ────────────────────────────────────────────────────
print("=" * 70)
print("PERFORMANCE: Full universe scan")
print("=" * 70)
t0 = time.perf_counter()
signals = scan_universe(market_data, open_positions={})
elapsed = (time.perf_counter() - t0) * 1000
print(f"  Scan time: {elapsed:.1f}ms  ({len(market_data)} assets)")
print(f"  Signals: {len(signals)}")
if signals:
    for s in signals:
        d = "LONG" if s.direction == 1 else "SHORT"
        print(f"    {s.asset:<8} {d:>5}  |R|={s.confidence:.4f}  pvt|R|={s.pvt_r:.4f}  "
              f"TF={s.best_tf}  P={s.best_period}  entry={s.entry_price:.4f}  SL={s.sl_price:.4f}")
else:
    print("  (No signals — check report below for details)")
print()

# ── Print diagnostic report ─────────────────────────────────────────────
print("=" * 70)
print("SCAN REPORT (top 3 per asset)")
print("=" * 70)
print_scan_report(market_data, top_n=3)

print("\nAll tests passed.")

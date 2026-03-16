# Varanus Neo-Flow — Master Design Document

## System Overview

Varanus Neo-Flow is an autonomous multi-timeframe trend-following engine.
It replaces the fixed-TF approach of v5.7.1 with **adaptive timeframe/period selection**
driven by Logarithmic Linear Regression and Pearson's R correlation, confirmed by
**Price Volume Trend (PVT) alignment**.

The system scans 5m, 30m, and 1h data (standardized 7-day rolling window) for 15 assets,
finds the strongest linear trend across all timeframe/period combinations (periods 20–200),
validates with PVT directional alignment, and executes — gated by the 4H higher-timeframe filter.

---

## Architecture

```
varanus_neo_flow/
  claude.md                         # This file
  neo_flow/
    __init__.py
    adaptive_engine.py              # Core: LogReg + PVT + multi-TF scanner + signal gen
    trend_filter.py                 # 4H MSS + EMA alignment filter (future)
    trailing_stop.py                # Adaptive trailing stop manager (future)
    binance_client.py               # Exchange wrapper (shared from v5.7.1)
    risk.py                         # Position sizing, SL management (future)
    config/
      params.json                   # Tunable parameters
      binance.env                   # API credentials
      telegram.env                  # Alert credentials
  run_scanner.py                    # CLI runner (future)
  run_live.py                       # Live trading runner (future)
```

---

## Core Algorithm: Logarithmic Linear Regression

### Origin

Translated from TradingView Pine Script `calcDev()` function.
Performs OLS linear regression on log(price), returning:

| Output        | Meaning                                              |
|---------------|------------------------------------------------------|
| `std_dev`     | Standard deviation of residuals (channel width)      |
| `pearson_r`   | Correlation between actual data and fitted line       |
| `slope`       | Rate of change in log space (trend speed + direction) |
| `intercept`   | Fitted value at the most recent bar = regression tip  |

### Indexing Convention

Pine Script uses reverse indexing (`[0]` = newest bar).
Python/Pandas uses forward indexing (last element = newest bar).
The engine flips internally: `log_src = log(prices[-length:][::-1])`.

### Slope Interpretation

Since x=1 maps to the newest bar and x=length maps to the oldest:
- **Negative slope** → price rising over the window → **uptrend**
- **Positive slope** → price falling over the window → **downtrend**

### Pearson's R Interpretation

|R| measures how well the data fits a straight line in log space:
- |R| > 0.90 → very strong trend (high-conviction entry zone)
- |R| 0.80–0.90 → strong trend (standard entry zone)
- |R| < 0.80 → weak/noisy → no trade

R is always positive in this formulation (data and fitted line always move together).
Direction is determined from slope sign only, not R sign.

---

## Price Volume Trend (PVT) Integration

### Calculation

```
PVT[i] = PVT[i-1] + ((Close[i] - Close[i-1]) / Close[i-1]) * Volume[i]
PVT[0] = 0
```

PVT is a cumulative indicator that combines price momentum with volume.
Rising PVT = volume flowing into the asset (bullish).
Falling PVT = volume flowing out (bearish).

### PVT Regression

For the same (TF, period) that wins the price regression scan, we also compute
**linear regression** on the PVT curve (not log — PVT can be negative).
This yields `pvt_pearson_r` and `pvt_slope`.

### PVT Entry Filter (Alignment Gate)

A trade can only be executed if:
1. **PVT direction matches price direction**: PVT slope sign == price slope sign
2. **PVT strength**: `|pvt_pearson_r| >= 0.70` (MIN_PVT_PEARSON_R)

This ensures the price trend is backed by consistent volume flow.

### Volume-Price Divergence Check

If price |R| is strong (>= 0.85) but PVT is either:
- Moving in the **opposite direction** (divergence), or
- Has **weak fit** (`|pvt_R| < 0.50`)

→ The signal is **suppressed** as a "Volume-Price Divergence".
This filters out low-volume traps and reduces Hard SL hits.

### Gate Summary

| Price |R| | PVT Direction | PVT |R| | Result |
|-----------|---------------|---------|--------|
| >= 0.80   | Same          | >= 0.70 | PASS   |
| >= 0.85   | Opposite      | any     | BLOCK (divergence) |
| >= 0.85   | Same          | < 0.50  | BLOCK (weak volume) |
| >= 0.80   | Same          | < 0.70  | BLOCK (insufficient PVT) |
| < 0.80    | any           | any     | BLOCK (weak trend) |

---

## Standardized 7-Day Rolling Window

All timeframes maintain exactly 7 calendar days of data for scanning.
This ensures consistent lookback depth regardless of TF granularity.

| Timeframe | 7-Day Bars | Purpose         |
|-----------|------------|-----------------|
| 5m        | 2,016      | Scan execution  |
| 30m       | 336        | Scan execution  |
| 1h        | 168        | Scan execution  |
| 4h        | 42         | HTF filter      |

Period range 20–200 fits within all scan TFs (minimum 336 bars on 30m).

---

## Scanning Logic

### Per-Cycle Pipeline

```
For each asset in TIER2_UNIVERSE (15 assets):
  1. Fetch 7-day rolling windows: 5m (2016), 30m (336), 1h (168), 4h (42)
  2. For each scan TF in [5m, 30m, 1h]:
       For each period in [20, 21, ..., 200]:
         calc_log_regression(close, period) → (std_dev, R, slope, intercept)
  3. Select the (TF, period) with the HIGHEST |R|
  4. Compute PVT on the winning (TF, period)
  5. Run PVT alignment gate + divergence check
  6. Apply 4H trend filter (MSS + EMA21/55 alignment)
  7. Determine direction from slope sign
  8. Apply Combined Gate (suppress if opposing |R| > 0.65)
  9. If all gates pass → generate ScanSignal
```

### Autonomous Timeframe Selection

The system does NOT pre-select a timeframe. It tests all 3 scan TFs and
181 periods per TF (= 543 regressions per asset, 8,145 total per cycle).
NumPy vectorization keeps total scan time under 200ms.

The winning (TF, period) determines:
- The regression midline used for the adaptive trailing stop
- The channel width (std_dev) used for SL placement
- The PVT curve used for volume confirmation
- The refresh rate for recalculating the regression

---

## Entry Rules

### Direction

| Slope Sign | Direction |
|------------|-----------|
| Negative   | LONG      |
| Positive   | SHORT     |

Direction is determined solely from slope sign. R is always positive (goodness-of-fit).

### Gate Conditions (ALL must pass)

1. **Trend Strength**: `|pearson_r| >= 0.80` (MIN_PEARSON_R)
2. **PVT Alignment**: PVT slope same direction AND `|pvt_R| >= 0.70`
3. **PVT Divergence**: No volume-price divergence (price |R| >= 0.85 with opposing/weak PVT)
4. **4H Filter**: MSS direction matches + EMA21 > EMA55 (long) or EMA21 < EMA55 (short)
5. **Combined Gate**: No opposing TF/period with `|R| > 0.65` in the opposite direction
6. **Max Positions**: <= 4 concurrent positions
7. **Leverage Cap**: Portfolio leverage <= 2.5x

### Confidence Mapping

`confidence = |pearson_r|` of the best (TF, period) combination.

| Confidence  | Leverage |
|-------------|----------|
| [0.80, 0.85) | 1x     |
| [0.85, 0.90) | 2x     |
| [0.90, 0.95) | 3x     |
| [0.95, 1.00] | 5x     |

---

## Exit Rules

### 1. Adaptive Trailing Stop (target: 71.2% of exits)

The trailing stop tracks the **regression midline** of the selected TF/period:
- Midline = `exp(intercept)` = fitted price at the current bar
- Each bar: recalculate regression → update midline
- LONG trail: `midline - trail_buffer * std_dev`
- SHORT trail: `midline + trail_buffer * std_dev`
- The trail only moves in the favorable direction (ratchets)

This is adaptive because as the trend steepens (midline moves faster),
the trailing stop follows more aggressively. In consolidation (midline
flattens), the trail tightens.

### 2. Hard Stop Loss (server-side STOP_MARKET)

Placed at entry time at:
- LONG: `entry - hard_sl_atr_mult * ATR(14)`
- SHORT: `entry + hard_sl_atr_mult * ATR(14)`

Touch-based execution — v5.7.1 logs confirm `exit_price == stop_loss` is optimal.
This is the safety net; the adaptive trail should exit before the hard SL.
PVT filter aims to reduce Hard SL hits by filtering low-volume traps.

### 3. Trend Exhaustion Exit

If |R| drops below 0.50 on the active (TF, period) → close immediately.
The trend that justified the entry no longer exists.

### 4. Time Barrier

Maximum holding period = 200 bars of the selected TF.
- 5m: 200 * 5m = ~16.7 hours
- 30m: 200 * 30m = ~4.2 days
- 1h: 200 * 1h = ~8.3 days

---

## Risk Management

### Position Sizing

```
pos_usd = capital * 0.10 * leverage * vol_scalar
```

Where `vol_scalar = 0.75` for high-volatility assets (ICP), `1.0` otherwise.

### Circuit Breaker

- Daily loss > 5% of equity → halt all new entries for 24h
- Drawdown from peak > 15% → halt until manual reset

### Combined Gate Detail

For each candidate signal, check all OTHER (TF, period) results:
- If LONG signal, and any (TF, period) shows SHORT with |R| > 0.65 → suppress
- If SHORT signal, and any (TF, period) shows LONG with |R| > 0.65 → suppress

This prevents entering when timeframes disagree.

---

## Performance Targets (Heritage from v5.7.1)

| Metric             | Target  | PVT Impact                        |
|--------------------|---------|-----------------------------------|
| Win Rate           | ~65%    | PVT filter improves by removing traps |
| Trailing Stop Exits| 71.2%   | Maintained                        |
| Hard SL Exits      | <20%    | PVT reduces low-volume SL hits    |
| Time Barrier Exits | <10%    | Maintained                        |
| Max Drawdown       | <15%    | PVT divergence check reduces DD   |
| Sharpe Ratio       | >1.5    | Improved signal quality           |

---

## Date Ranges (aligned with v5.7.1)

| Period | Start | End | Purpose |
|--------|-------|-----|---------|
| Training (8-fold WFV) | 2023-01-01 | 2025-10-31 | Model training + walk-forward validation |
| Blind Test | 2025-11-01 | 2026-03-15 | Forward test on unseen data |

Walk-forward: 8 folds, 40% train / 30% validation / 30% test, 24-candle embargo gap.

## Data Requirements

| Timeframe | 7-Day Bars (Live) | Historical Bars | Purpose |
|-----------|-------------------|-----------------|---------|
| 5m        | 2,016             | ~337,000        | Scan + PVT computation |
| 30m       | 336               | ~56,000         | Scan + PVT computation |
| 1h        | 168               | ~28,000         | Scan + PVT computation |
| 4h        | 42                | ~7,000          | HTF filter (MSS+EMA) |

---

## Asset Universe (15 Tier-2)

ADA, AVAX, LINK, DOT, TRX, SOL, ATOM, NEAR, ALGO, UNI, ICP, HBAR, SAND, MANA, THETA

---

## Implementation Status

- [x] claude.md — Master design document
- [x] neo_flow/adaptive_engine.py — Core LogReg + PVT + scanner
- [x] data_fetcher.py — Historical OHLCV fetcher (all 15 assets, 4 TFs)
- [ ] neo_flow/trend_filter.py — 4H MSS + EMA filter module
- [ ] neo_flow/trailing_stop.py — Adaptive trailing stop manager
- [ ] neo_flow/risk.py — Position sizing + circuit breaker
- [ ] neo_flow/binance_client.py — Exchange wrapper
- [ ] run_scanner.py — CLI test scanner
- [ ] run_live.py — Live trading runner
- [ ] Backtesting framework
- [ ] Parameter optimization

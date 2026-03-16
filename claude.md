# Varanus Neo-Flow — Master Design Document

## System Overview

Varanus Neo-Flow is an autonomous multi-timeframe trend-following engine.
It replaces the fixed-TF approach of v5.7.1 with **adaptive timeframe/period selection**
driven by Logarithmic Linear Regression and Pearson's R correlation.

The system scans 5m, 30m, and 1h data for 15 assets, finds the strongest
linear trend across all timeframe/period combinations (periods 20–200),
and executes on the single best configuration — gated by the 4H higher-timeframe filter.

---

## Architecture

```
varanus_neo_flow/
  claude.md                         # This file
  neo_flow/
    __init__.py
    adaptive_engine.py              # Core: LogReg math, multi-TF scanner, signal gen
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

The sign of R echoes the slope: negative R = uptrend, positive R = downtrend.

---

## Scanning Logic

### Per-Cycle Pipeline

```
For each asset in TIER2_UNIVERSE (15 assets):
  1. Fetch 5m, 30m, 1h, 4h OHLCV data
  2. For each scan TF in [5m, 30m, 1h]:
       For each period in [20, 21, ..., 200]:
         calc_log_regression(close, period) → (std_dev, R, slope, intercept)
  3. Select the (TF, period) with the HIGHEST |R|
  4. Apply 4H trend filter (MSS + EMA21/55 alignment)
  5. Determine direction from slope sign
  6. Apply Combined Gate (suppress if opposing |R| > 0.65)
  7. If all gates pass → generate ScanSignal
```

### Autonomous Timeframe Selection

The system does NOT pre-select a timeframe. It tests all 3 scan TFs and
181 periods per TF (= 543 regressions per asset, 8,145 total per cycle).
NumPy vectorization keeps total scan time under 200ms.

The winning (TF, period) determines:
- The regression midline used for the adaptive trailing stop
- The channel width (std_dev) used for SL placement
- The refresh rate for recalculating the regression

---

## Entry Rules

### Direction

| Slope Sign | Pearson R Sign | Direction |
|------------|----------------|-----------|
| Negative   | Negative       | LONG      |
| Positive   | Positive       | SHORT     |

### Gate Conditions (ALL must pass)

1. **Trend Strength**: `|pearson_r| >= 0.80` (MIN_PEARSON_R)
2. **4H Filter**: MSS direction matches + EMA21 > EMA55 (long) or EMA21 < EMA55 (short)
3. **Combined Gate**: No opposing TF/period with `|R| > 0.65` in the opposite direction
4. **Max Positions**: <= 4 concurrent positions
5. **Leverage Cap**: Portfolio leverage <= 2.5x

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

| Metric             | Target  |
|--------------------|---------|
| Win Rate           | ~65%    |
| Trailing Stop Exits| 71.2%   |
| Hard SL Exits      | <20%    |
| Time Barrier Exits | <10%    |
| Max Drawdown       | <15%    |
| Sharpe Ratio       | >1.5    |

---

## Data Requirements

| Timeframe | Bars Needed | Duration     | Purpose            |
|-----------|-------------|--------------|---------------------|
| 5m        | 250         | ~20.8 hours  | Scan (period 200+50 buffer) |
| 30m       | 250         | ~5.2 days    | Scan               |
| 1h        | 250         | ~10.4 days   | Scan               |
| 4h        | 200         | ~33 days     | HTF filter (MSS+EMA)|

---

## Implementation Status

- [x] claude.md — Master design document
- [x] neo_flow/adaptive_engine.py — Core LogReg + scanner
- [ ] neo_flow/trend_filter.py — 4H MSS + EMA filter module
- [ ] neo_flow/trailing_stop.py — Adaptive trailing stop manager
- [ ] neo_flow/risk.py — Position sizing + circuit breaker
- [ ] neo_flow/binance_client.py — Exchange wrapper
- [ ] run_scanner.py — CLI test scanner
- [ ] run_live.py — Live trading runner
- [ ] Backtesting framework
- [ ] Parameter optimization

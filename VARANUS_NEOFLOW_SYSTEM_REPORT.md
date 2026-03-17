# Varanus Neo-Flow: Comprehensive System Report

**Version:** 1.0
**Date:** March 2026
**Author:** Varanus Quantitative Research
**Repository:** github.com/yagokhan/VaranusNeoFlow

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Core Methodology: Logarithmic Linear Regression](#3-core-methodology-logarithmic-linear-regression)
4. [Price Volume Trend (PVT) Confirmation Layer](#4-price-volume-trend-pvt-confirmation-layer)
5. [4-Hour Higher-Timeframe Filter](#5-4-hour-higher-timeframe-filter)
6. [Combined Gate: Cross-Timeframe Conflict Suppression](#6-combined-gate-cross-timeframe-conflict-suppression)
7. [Multi-Timeframe Scanning Pipeline](#7-multi-timeframe-scanning-pipeline)
8. [Position Entry Logic](#8-position-entry-logic)
9. [Position Management & Exit Mechanisms](#9-position-management--exit-mechanisms)
10. [Risk Management Framework](#10-risk-management-framework)
11. [Walk-Forward Validation (WFV) Framework](#11-walk-forward-validation-wfv-framework)
12. [Optuna Parameter Optimization](#12-optuna-parameter-optimization)
13. [Date Selection & Fold Construction](#13-date-selection--fold-construction)
14. [Optimization Results: 8-Fold WFV](#14-optimization-results-8-fold-wfv)
15. [Consensus Parameters](#15-consensus-parameters)
16. [Blind Test Results](#16-blind-test-results)
17. [Pre-Computed Feature Store (Fast Engine)](#17-pre-computed-feature-store-fast-engine)
18. [Live Trading Bot Architecture](#18-live-trading-bot-architecture)
19. [Data Infrastructure](#19-data-infrastructure)
20. [Complete Parameter Reference](#20-complete-parameter-reference)
21. [Performance Metrics Definitions](#21-performance-metrics-definitions)
22. [File Structure & Codebase Map](#22-file-structure--codebase-map)

---

## 1. Executive Summary

Varanus Neo-Flow is an autonomous multi-timeframe trend-following system designed for Binance USDT-M perpetual futures. It scans 15 Tier-2 cryptocurrency assets every hour, combining **logarithmic linear regression** with **Price Volume Trend (PVT) confirmation**, filtering through a **4-hour higher-timeframe bias**, and managing positions with **adaptive trailing stops** that track the regression midline.

### Key Performance Highlights

| Metric | WFV Aggregate (8 Folds) | Blind Test (Out-of-Sample) |
|--------|--------------------------|----------------------------|
| Win Rate | 67.93% | 70.74% |
| Sharpe Ratio | 14.06 | 12.60 |
| Max Drawdown | -5.79% | -5.36% |
| Profit Factor | 4.61 | 4.51 |
| Trail Exit % | 99.16% | 99.18% |
| Hard SL Exit % | 0.80% | 0.82% |
| Total Trades | 12,149 | 1,097 |
| PnL (per fold avg) | 737.1% | 381.1% |

### Asset Universe

15 Tier-2 cryptocurrencies: **ADA, AVAX, LINK, DOT, TRX, SOL, ATOM, NEAR, ALGO, UNI, ICP, HBAR, SAND, MANA, THETA**

### Scan Timeframes

3 scan timeframes (5m, 30m, 1h) with 4h as higher-timeframe filter. 181 regression periods (20-200) evaluated per timeframe per asset, yielding **543 regressions per asset** and **8,145 regressions per scan cycle** across the full universe.

---

## 2. System Architecture Overview

```
                    SCAN CYCLE (every 1 hour)
                            |
            +---------------+---------------+
            |                               |
    CHECK OPEN POSITIONS            SCAN FOR NEW ENTRIES
    (sub-bar exit logic)            (15 assets x 3 TFs x 181 periods)
            |                               |
    [Trail/SL/Exhaust/Time]         [Best |R| per asset]
            |                               |
        EXIT if hit                 GATE 1: Min Pearson R >= 0.83
                                            |
                                    GATE 2: PVT Alignment
                                    (direction + strength)
                                            |
                                    GATE 3: 4H HTF Filter
                                    (MSS + EMA alignment)
                                            |
                                    GATE 4: Combined Gate
                                    (no opposing TF with |R| > 0.80)
                                            |
                                    POSITION SIZING
                                    (confidence -> leverage tier)
                                            |
                                    ENTRY + STOP PLACEMENT
                                    (hard SL + adaptive trail)
```

---

## 3. Core Methodology: Logarithmic Linear Regression

### 3.1 Mathematical Foundation

The system implements an exact translation of Pine Script's `calcDev()` function for Ordinary Least Squares (OLS) regression on log-transformed prices. This is the cornerstone of the entire signal generation process.

**Input:**
- `prices`: 1-D array of OHLCV close values, oldest-first (Pandas convention)
- `length`: Window size (20-200 bars)

**Step 1 — Log Transformation & Reversal:**
```
log_src = log(prices[-length:])   # Take last 'length' prices, apply natural log
log_src = reverse(log_src)        # Flip to newest-first (Pine Script convention)
```

**Step 2 — OLS Regression (Pine Script Formulation):**
```
x = [1, 2, 3, ..., length]       # x=1 is newest bar, x=length is oldest

sum_x  = length * (length + 1) / 2
sum_xx = length * (length + 1) * (2*length + 1) / 6
sum_yx = SUM(x_i * log_src_i)    for i in 1..length
sum_y  = SUM(log_src_i)           for i in 1..length

denom     = length * sum_xx - sum_x^2
slope     = (length * sum_yx - sum_x * sum_y) / denom
average   = sum_y / length
intercept = average - slope * (sum_x / length) + slope
```

**Step 3 — Outputs:**

1. **Standard Deviation (channel width):**
```
regres   = intercept + slope * (length - 1) * 0.5
reg_line = [intercept + i * slope  for i in 0..length-1]
residuals = log_src - reg_line
std_dev  = sqrt(SUM(residuals^2) / (length - 1))
```

2. **Pearson Correlation Coefficient (goodness-of-fit):**
```
dxt = log_src - average          # deviations from mean
dyt = reg_line - regres          # deviations from regression mean
pearson_r = SUM(dxt * dyt) / sqrt(SUM(dxt^2) * SUM(dyt^2))
```

3. **Slope:** Trend direction and steepness
4. **Intercept:** Log-fitted value at current bar; `midline = exp(intercept)` is the regression tip price in real price space

### 3.2 Slope Sign Convention

Since x=1 is the newest bar (most recent) and x=length is the oldest bar:

| Slope Sign | Price Action | Signal |
|------------|-------------|--------|
| slope < 0 | Price increasing over window | **LONG** (direction = +1) |
| slope > 0 | Price decreasing over window | **SHORT** (direction = -1) |
| slope ~ 0 | Flat/no trend | No signal (direction = 0) |

### 3.3 Pearson R Interpretation

The absolute value |R| measures how well the price data fits the regression line. It is **always positive** in this formulation (goodness-of-fit, not directional correlation). The trend direction comes from the slope sign.

| |R| Range | Classification | Action |
|-----------|----------------|--------|
| 0.95 - 1.00 | Extremely strong trend | 5x leverage |
| 0.90 - 0.95 | Very strong trend | 3x leverage |
| 0.85 - 0.90 | Strong trend | 2x leverage |
| 0.80 - 0.85 | Good trend | 1x leverage |
| < 0.80 | Weak/insufficient trend | **NO TRADE** |

### 3.4 Why Logarithmic Regression?

1. **Percentage-based fitting:** Log transformation makes the regression measure percentage price changes rather than absolute dollar changes. A $0.01 move on a $0.10 coin (10%) is weighted the same as a $10 move on a $100 coin (10%).
2. **Compounding assumption:** Financial returns compound multiplicatively, not additively. Log space is the natural domain for compounding.
3. **Channel symmetry:** Standard deviation bands in log space produce symmetric percentage channels above and below the midline.
4. **Cross-asset comparability:** Pearson R values are directly comparable across assets with vastly different price scales (ADA at $0.40 vs SOL at $150).

---

## 4. Price Volume Trend (PVT) Confirmation Layer

### 4.1 PVT Calculation

PVT is a cumulative volume-weighted momentum indicator that tracks whether volume is flowing INTO or OUT OF an asset:

```
close_pct_change[i] = (close[i] - close[i-1]) / close[i-1]
pvt_increment[i]    = close_pct_change[i] * volume[i]
pvt[i]              = pvt[i-1] + pvt_increment[i]     (pvt[0] = 0)
```

**Interpretation:**
- **Rising PVT:** Cumulative volume flowing in, prices supported by buying pressure
- **Falling PVT:** Cumulative volume flowing out, prices weakened by selling pressure
- **Flat PVT:** No consistent directional volume flow, unreliable trend

### 4.2 PVT Linear Regression

For the same (timeframe, period) combination that won the price regression scan, a separate **unlogged** linear regression is applied to the cumulative PVT array:

```
pvt_std_dev, pvt_pearson_r, pvt_slope, pvt_intercept = calc_linear_regression(pvt_array, best_period)

pvt_direction:
  -1 if pvt_slope < 0  (PVT rising → volume accumulation)
  +1 if pvt_slope > 0  (PVT declining → volume distribution)
   0 if |pvt_slope| ~ 0
```

Note: PVT regression is **not** log-transformed because PVT values can be negative (cumulative signed volume).

### 4.3 Three-Tier PVT Validation Gate

**Gate 1 — Direction Alignment:**
```
IF pvt_direction != price_direction:
  IF |price_R| >= 0.85:
    REJECT → "VOLUME-PRICE DIVERGENCE"
    (Strong price trend but volume flowing opposite → trap/reversal risk)
  ELSE:
    REJECT → "PVT direction mismatch"
```

**Gate 2 — Volume Strength for High-Confidence Prices:**
```
IF |price_R| >= 0.85 AND |pvt_R| < 0.50:
  REJECT → "WEAK VOLUME"
  (Strong price trend but insufficient volume backing → hollow rally/selloff)
```

**Gate 3 — Minimum PVT Strength:**
```
IF |pvt_R| < MIN_PVT_PEARSON_R (0.80):
  REJECT → "PVT too weak"
ELSE:
  PASS → proceed to 4H filter
```

### 4.4 Why PVT Matters

Without PVT confirmation, the system would enter on any strong regression fit. But:
- A 0.92 |R| uptrend with declining volume is likely a **distribution top**
- A 0.88 |R| downtrend with rising volume suggests **short squeeze risk**
- Only when price AND volume trends align is the directional conviction high enough for entry

---

## 5. 4-Hour Higher-Timeframe Filter

### 5.1 Market Structure Shift (MSS) Detection

Detects if price has broken a swing high/low on the 4-hour chart, indicating a structural trend change:

```
lookback = 30 bars (MSS_LOOKBACK on 4h = ~5 days)
swing_high = MAX(high[current_idx - 30 : current_idx])
swing_low  = MIN(low[current_idx - 30 : current_idx])
current_close = close[current_idx]

IF current_close > swing_high:  MSS = +1  (BULLISH STRUCTURAL BREAK)
ELIF current_close < swing_low: MSS = -1  (BEARISH STRUCTURAL BREAK)
ELSE:                           MSS =  0  (NO STRUCTURAL SHIFT)
```

### 5.2 EMA (Exponential Moving Average) Alignment

Evaluates the trend bias on 4H using fast/slow EMA crossover:

```
ema_fast = EMA(close_4h, 21)    (EMA_FAST = 21 periods)
ema_slow = EMA(close_4h, 55)    (EMA_SLOW = 55 periods)

ema_bullish = (ema_fast > ema_slow at current bar)
ema_bearish = (ema_fast < ema_slow at current bar)
```

### 5.3 Combined HTF Bias

**Both MSS and EMA must agree for a directional bias:**

```
IF MSS == +1 AND ema_bullish:   htf_bias = +1  (BULLISH)
ELIF MSS == -1 AND ema_bearish: htf_bias = -1  (BEARISH)
ELSE:                           htf_bias =  0  (NEUTRAL/CONFLICTING)
```

**Filter Rule:**
```
IF htf_bias == 0:                    REJECT → "4H bias neutral"
ELIF htf_bias != signal_direction:   REJECT → "4H conflicts with signal"
ELSE:                                PASS
```

### 5.4 Why This Filter Exists

The lower timeframes (5m, 30m, 1h) can show strong regression fits during counter-trend pullbacks. The 4H filter ensures the system only trades **with** the higher-timeframe structural trend, avoiding mean-reversion traps within a larger opposing move.

---

## 6. Combined Gate: Cross-Timeframe Conflict Suppression

### 6.1 Purpose

Even after passing the PVT and 4H gates, a signal can still be unreliable if another timeframe or period shows a strong opposing trend. The combined gate prevents entries when significant cross-timeframe disagreement exists.

### 6.2 Logic

After finding the best regression across all TF/period combinations:

```
FOR each regression result across ALL timeframes and periods:
  result_direction = sign(result.slope)

  IF result_direction != 0 AND result_direction != signal_direction:
    IF abs(result.pearson_r) > COMBINED_GATE_THRESHOLD (0.80):
      REJECT → "Combined gate blocked by opposing TF/period with R=X.XX"
      BREAK

IF no opposing signal exceeds threshold:
  PASS
```

### 6.3 Example

- Best signal: LONG (5m, period=45) with |R| = 0.92
- Another result: SHORT (30m, period=80) with |R| = 0.82
- 0.82 > 0.80 → **BLOCK** the LONG entry

This prevents the system from taking a short-term LONG when a medium-term trend is clearly SHORT with strong confidence.

---

## 7. Multi-Timeframe Scanning Pipeline

### 7.1 Rolling Window: 7 Calendar Days

All scan calculations use exactly 7 calendar days of data, standardized across timeframes:

| Timeframe | Bars per 7 Days | Calculation |
|-----------|-----------------|-------------|
| 5-minute | 2,016 | 7 x 24 x 12 |
| 30-minute | 336 | 7 x 24 x 2 |
| 1-hour | 168 | 7 x 24 x 1 |
| 4-hour | 42 | 7 x 24 / 4 |

The 7-day window ensures:
- Even the longest regression period (200) fits within the smallest scan window (336 bars on 30m)
- Enough data for ATR and EMA computations on 4H
- Recent-enough data that the regression reflects current market dynamics

### 7.2 Full Pipeline Per Asset

```
STEP 1: Find Best Regression Across All TFs and Periods
        FOR tf in ["5m", "30m", "1h"]:
          FOR period in range(20, 201):    # 181 periods
            std_dev, pearson_r, slope, intercept = calc_log_regression(close, period)
            IF |pearson_r| > best_so_far:
              Record as best (tf, period, R, slope, intercept, std_dev, midline)
        Total: 3 TFs x 181 periods = 543 regressions per asset

STEP 2: Check Minimum Strength
        IF best |R| < MIN_PEARSON_R (0.83):   REJECT

STEP 3: Determine Direction
        direction = +1 if slope < 0 (LONG), -1 if slope > 0 (SHORT)

STEP 4: PVT Alignment Gate
        Compute PVT regression on best (tf, period)
        Check 3-tier PVT validation

STEP 5: 4H Higher-Timeframe Filter
        Compute MSS + EMA on 4H data
        Check directional alignment

STEP 6: Combined Gate
        Check if any opposing regression exceeds threshold

STEP 7: Compute ATR & Stop Levels
        atr = ATR(14) on best TF
        hard_sl = entry +/- HARD_SL_ATR_MULT * atr
        trail_sl = midline +/- TRAIL_BUFFER_STD * (midline * std_dev)

STEP 8: Generate ScanSignal
        Package all results into a ScanSignal object
```

### 7.3 Scan Throughput

- **15 assets x 543 regressions = 8,145 regressions per scan cycle**
- Computation is NumPy-vectorized (batch regression across all periods simultaneously)
- Total scan time: ~200ms on modern hardware (excluding data fetch)
- With Binance REST data fetch: ~30-35 seconds per cycle

---

## 8. Position Entry Logic

### 8.1 Leverage Tier Mapping

Leverage is discretized based on regression confidence:

```python
def get_leverage(confidence: float) -> int:
    if confidence >= 0.95: return 5    # Extremely strong trend
    if confidence >= 0.90: return 3    # Very strong trend
    if confidence >= 0.85: return 2    # Strong trend
    if confidence >= 0.80: return 1    # Good trend
    return 0                            # No trade
```

### 8.2 Position Sizing (Compounding)

```
equity = initial_capital + realized_pnl     # Current equity (compounding)
vol_scalar = 0.75 if asset in HIGH_VOL_ASSETS else 1.0
position_usd = equity * pos_frac * leverage * vol_scalar

HIGH_VOL_ASSETS = {"ICP"}   # 25% size reduction for high-volatility assets
```

**Live Configuration:**
- `pos_frac = 0.10` (10% of equity per position)
- `initial_capital = $1,000`
- Maximum 4 concurrent positions

**Example:**
- Equity: $1,200 (after some profits)
- ADA signal with |R| = 0.88 → leverage = 2x
- Position = $1,200 x 0.10 x 2 x 1.0 = **$240 notional**

### 8.3 Entry Price

- **Backtest:** Entry at the **open of the next bar** on the signal's winning timeframe
- **Live:** Entry at current market price via market order

### 8.4 Maximum Concurrent Positions

The system caps at **4 simultaneous positions** (`MAX_CONCURRENT = 4`). If 4 positions are open, no scanning occurs until one closes. This bounds total portfolio exposure.

---

## 9. Position Management & Exit Mechanisms

Once a position is opened, it is managed on its **winning timeframe's sub-bars** (not on 1-hour bars). This provides granular exit precision, especially for 5m entries where price can move significantly within an hour.

### 9.1 Adaptive Trailing Stop (Primary Exit — ~99.2% of exits)

The adaptive trailing stop tracks the regression **midline** (fitted price) of the active (TF, period) combination. It recalculates at every bar on the position's native timeframe.

**Recalculation (every sub-bar):**
```
# Recompute regression on the latest data including new bars since entry
close_arr = data[best_tf].close[: current_bar_idx + 1]
IF len(close_arr) >= best_period:
  std_dev, pearson_r, slope, intercept = calc_log_regression(close_arr, best_period)
  midline = exp(intercept)           # Updated fitted price in real space
  std_dev = std_dev                  # Updated channel width
  peak_r = max(peak_r, |pearson_r|) # Track max R seen during hold
```

**Trail Update (ratcheting — only tightens, never loosens):**
```
std_dev_price = midline * std_dev    # Convert log-space std_dev to price space

IF direction == +1 (LONG):
  new_trail = midline - TRAIL_BUFFER_STD * std_dev_price
  trail_sl = MAX(trail_sl, new_trail)    # Ratchet: only moves UP
ELIF direction == -1 (SHORT):
  new_trail = midline + TRAIL_BUFFER_STD * std_dev_price
  trail_sl = MIN(trail_sl, new_trail)    # Ratchet: only moves DOWN
```

**Exit Trigger:**
```
IF direction == +1 AND bar_low <= trail_sl:
  EXIT at trail_sl → reason: "ADAPTIVE_TRAIL_HIT"
ELIF direction == -1 AND bar_high >= trail_sl:
  EXIT at trail_sl → reason: "ADAPTIVE_TRAIL_HIT"
```

**Key Properties:**
- The trail **adapts to the regression curve**, not a fixed percentage
- As the regression midline rises (for LONG), the trail follows it up
- The std_dev naturally widens in volatile conditions and tightens in quiet conditions
- The ratchet mechanism ensures the trail never moves against the trade direction
- `TRAIL_BUFFER_STD = 0.5` means the trail sits 0.5 standard deviations below (LONG) or above (SHORT) the midline

### 9.2 Hard Stop Loss (Emergency Exit — ~0.8% of exits)

A fixed stop placed at entry time using ATR (Average True Range):

```
atr = ATR(14)  on the position's timeframe

IF direction == +1 (LONG):
  hard_sl = entry_price - HARD_SL_ATR_MULT * atr
ELIF direction == -1 (SHORT):
  hard_sl = entry_price + HARD_SL_ATR_MULT * atr

HARD_SL_ATR_MULT = 2.5 (consensus)
```

**Exit Trigger:**
```
IF direction == +1 AND bar_low <= hard_sl:
  EXIT at hard_sl → reason: "HARD_SL_HIT"
ELIF direction == -1 AND bar_high >= hard_sl:
  EXIT at hard_sl → reason: "HARD_SL_HIT"
```

The hard SL is a **safety net** — with proper trailing stop behavior, it should rarely be hit. The 0.8% hit rate in testing confirms this.

### 9.3 Trend Exhaustion (Proactive Exit)

If the Pearson R on the active (TF, period) drops below a threshold, the trend that justified entry no longer exists:

```
IF |current_pearson_r| < EXHAUST_R (0.425):
  EXIT at bar_close → reason: "TREND_EXHAUSTION"
```

This exit is proactive — it closes the position before the trailing stop is hit, when the regression fit deteriorates (indicating the trend is breaking down).

### 9.4 Time Barrier (Maximum Holding Period)

A maximum hold duration of 200 bars on the active timeframe:

```
IF bars_held >= 200:
  EXIT at bar_close → reason: "TIME_BARRIER"
```

**Effective Maximum Hold Durations:**

| Timeframe | Calculation | Max Duration |
|-----------|-------------|-------------|
| 5-minute | 200 x 5m | ~16.7 hours |
| 30-minute | 200 x 30m | ~4.2 days |
| 1-hour | 200 x 1h | ~8.3 days |

### 9.5 End-of-Period Exit (Backtest Only)

At the end of the backtest period, all open positions are force-closed at the last available close price:

```
reason: "END_OF_PERIOD"
```

### 9.6 Exit Priority Order

Within each sub-bar, exits are checked in this order:
1. **Hard SL** (highest priority — checked against bar low/high)
2. **Adaptive Trail** (checked against bar low/high)
3. **Trend Exhaustion** (checked against |R|)
4. **Time Barrier** (checked against bars_held)

If multiple conditions trigger on the same bar, the first one in priority order determines the exit.

---

## 10. Risk Management Framework

### 10.1 Position-Level Risk

| Control | Value | Purpose |
|---------|-------|---------|
| Hard SL (ATR-based) | 2.5 x ATR(14) | Maximum loss per trade |
| Adaptive Trail | 0.5 std_dev from midline | Dynamic profit protection |
| Trend Exhaustion | |R| < 0.425 | Close failing trends early |
| Time Barrier | 200 bars | Prevent infinite holds |
| Leverage Cap | 5x maximum | Confidence-tiered exposure |
| High-Vol Scalar | 0.75x for ICP | Reduce size on volatile assets |

### 10.2 Portfolio-Level Risk

| Control | Value | Purpose |
|---------|-------|---------|
| Max Concurrent | 4 positions | Limit total exposure |
| Position Fraction | 10% of equity | Diversification per trade |
| Circuit Breaker (Daily) | 5% daily loss | Halt entries for 24 hours |
| Circuit Breaker (DD) | 15% drawdown from peak | Halt until manual reset |

### 10.3 Signal-Level Risk

| Gate | Threshold | Purpose |
|------|-----------|---------|
| Min Pearson R | >= 0.83 | Only trade strong trends |
| PVT Alignment | Direction + |R| >= 0.80 | Volume must confirm |
| PVT Divergence | Reject if |price_R| >= 0.85 but volume opposes | Avoid traps |
| 4H HTF Filter | MSS + EMA alignment | Trade with structure |
| Combined Gate | Block if opposing |R| > 0.80 | No conflicting TFs |

---

## 11. Walk-Forward Validation (WFV) Framework

### 11.1 Why Walk-Forward?

Traditional train/test splits suffer from:
- **Look-ahead bias:** Single split point can land on favorable/unfavorable market
- **Overfitting:** Parameters tuned to one market regime may fail in others
- **Non-stationarity:** Crypto markets evolve; parameters that worked in 2023 may not work in 2025

Walk-forward validation addresses all three by:
1. Creating **multiple overlapping folds** across the entire date range
2. **Optimizing separately** on each fold's validation period
3. **Testing on unseen data** for each fold
4. Taking the **median (consensus)** of all optimized parameters
5. Running a final **blind test** on completely untouched data

### 11.2 Configuration

```
WFV_START       = 2023-01-01 00:00:00 UTC
WFV_END         = 2025-10-31 23:00:00 UTC
BLIND_START      = 2025-11-01 00:00:00 UTC
BLIND_END        = 2026-03-15 23:00:00 UTC

Number of Folds  = 8
Train Fraction   = 40%
Validation Frac  = 30%
Test Fraction    = 30%
Embargo          = 24 bars (24 hours between train/val and val/test)
```

### 11.3 Fold Construction

Each fold covers ~60% of the total date range, with the 8 folds sliding across time:

```
Total WFV period: 2023-01-01 → 2025-10-31 (~34 months)
Fold window: ~60% of total = ~20.4 months per fold
Stride: ~1.94 months between fold starts

For each fold k (0-7):
  offset = k * stride
  fold.train_start = WFV_START + offset
  fold.train_end   = train_start + (40% of fold_window)
  fold.val_start   = train_end + 24h embargo
  fold.val_end     = val_start + (30% of fold_window)
  fold.test_start  = val_end + 24h embargo
  fold.test_end    = test_start + (30% of fold_window)
```

The **embargo** of 24 bars between periods prevents any information leakage from one period to the next.

---

## 12. Optuna Parameter Optimization

### 12.1 Search Space

For each fold, Optuna explores the following 7-dimensional parameter space:

| Parameter | Range | Step | Description |
|-----------|-------|------|-------------|
| min_pearson_r | [0.75, 0.92] | 0.01 | Minimum trend strength to enter |
| min_pvt_r | [0.50, 0.85] | 0.05 | Minimum PVT alignment strength |
| combined_gate | [0.50, 0.80] | 0.05 | Block threshold for opposing TFs |
| hard_sl_mult | [1.0, 3.0] | 0.25 | Hard SL distance in ATR multiples |
| trail_buffer | [0.5, 2.0] | 0.25 | Trail offset in std_dev multiples |
| exhaust_r | [0.30, 0.65] | 0.05 | Exhaustion |R| threshold |
| pos_frac | [0.05, 0.15] | 0.01 | Position size as fraction of equity |

### 12.2 Sampler: Tree-Structured Parzen Estimator (TPE)

Optuna uses TPE, a Bayesian optimization algorithm that:
1. Builds probability models of "good" and "bad" parameter regions
2. Samples more densely from regions that historically produced good scores
3. Balances exploration (trying new regions) with exploitation (refining good regions)
4. Is more sample-efficient than random search or grid search

**Seed:** `42 + fold_id` (deterministic but different per fold)

### 12.3 Pruner: MedianPruner

Trials that perform below the median of completed trials are stopped early:
- **Startup trials:** 10 (run at least 10 full trials before pruning)
- This saves computation by abandoning clearly poor parameter combinations

### 12.4 Objective Function (Multi-Criteria)

The optimizer runs a full backtest on the **validation period** for each trial:

```python
# Primary: Sharpe Ratio
score = sharpe_ratio

# Minimum sample size
IF total_trades < 10:
  RETURN -10.0    # Not enough trades for statistical significance

# Bonus for meeting heritage targets
IF win_rate >= 60%:          score += 0.5
IF trail_exit_pct >= 60%:    score += 0.3
IF max_drawdown > -15%:      score += 0.2

# Penalty for excessive drawdown
IF max_drawdown < -20%:      score -= 2.0

RETURN score
```

**Why this objective?**
- **Sharpe ratio** as primary metric rewards risk-adjusted returns, not just raw PnL
- **Win rate bonus** encourages parameters that maintain the system's edge
- **Trail exit bonus** rewards parameters where the adaptive trail works as designed
- **Drawdown penalty** prevents parameters that achieve high returns through excessive risk
- **Minimum trades** ensures statistical validity

### 12.5 Trials Per Fold

100 trials per fold (default). Each trial runs a complete backtest on the validation period with sampled parameters.

---

## 13. Date Selection & Fold Construction

### 13.1 Why These Dates?

**WFV Period (2023-01-01 to 2025-10-31):**
- Covers ~34 months of market history
- Includes multiple market regimes: recovery (early 2023), bull (late 2023-2024), correction, rally
- Provides sufficient data for 8 overlapping folds

**Blind Test Period (2025-11-01 to 2026-03-15):**
- ~4.5 months of completely untouched data
- Never seen during optimization or fold testing
- Represents the "true" out-of-sample forward test
- Tests whether consensus parameters generalize to unseen market conditions

### 13.2 Fold Date Ranges (Approximate)

| Fold | Train Period | Validation Period | Test Period |
|------|-------------|-------------------|-------------|
| 0 | Jan 2023 - Sep 2023 | Sep 2023 - Mar 2024 | Mar 2024 - Sep 2024 |
| 1 | Mar 2023 - Nov 2023 | Nov 2023 - May 2024 | May 2024 - Nov 2024 |
| 2 | May 2023 - Jan 2024 | Jan 2024 - Jul 2024 | Jul 2024 - Jan 2025 |
| 3 | Jul 2023 - Mar 2024 | Mar 2024 - Sep 2024 | Sep 2024 - Mar 2025 |
| 4 | Sep 2023 - May 2024 | May 2024 - Nov 2024 | Nov 2024 - May 2025 |
| 5 | Nov 2023 - Jul 2024 | Jul 2024 - Jan 2025 | Jan 2025 - Jul 2025 |
| 6 | Jan 2024 - Sep 2024 | Sep 2024 - Mar 2025 | Mar 2025 - Sep 2025 |
| 7 | Mar 2024 - Nov 2024 | Nov 2024 - May 2025 | May 2025 - Nov 2025 |

Each fold has 24-hour embargo gaps between train/val and val/test to prevent information leakage.

### 13.3 Train/Val/Test Purpose

- **Train (40%):** Available for learning market behavior (not directly used in TPE, but part of the data context)
- **Validation (30%):** Optuna runs backtests here to score each parameter combination
- **Test (30%):** After optimization, the best parameters are evaluated here to check generalization

---

## 14. Optimization Results: 8-Fold WFV

### 14.1 Per-Fold Best Parameters

| Fold | min_R | pvt_R | gate | sl_mult | trail_buf | exhaust | pos_frac |
|------|-------|-------|------|---------|-----------|---------|----------|
| 0 | 0.84 | 0.85 | 0.80 | 2.25 | 0.50 | 0.50 | 0.05 |
| 1 | 0.81 | 0.70 | 0.70 | 2.50 | 0.50 | 0.35 | 0.06 |
| 2 | 0.76 | 0.80 | 0.80 | 2.50 | 0.50 | 0.55 | 0.05 |
| 3 | 0.89 | 0.80 | 0.80 | 2.75 | 0.50 | 0.30 | 0.05 |
| 4 | 0.83 | 0.85 | 0.80 | 2.00 | 0.50 | 0.45 | 0.05 |
| 5 | 0.83 | 0.80 | 0.80 | 2.25 | 0.50 | 0.40 | 0.05 |
| 6 | 0.76 | 0.60 | 0.80 | 2.75 | 0.50 | 0.40 | 0.06 |
| 7 | 0.87 | 0.55 | 0.50 | 2.75 | 0.50 | 0.60 | 0.05 |

**Notable observations:**
- `trail_buffer` converged to **0.50 across all 8 folds** — extremely stable parameter
- `combined_gate` converged to **0.80 in 6 of 8 folds** — strong consensus
- `pos_frac` converged to **0.05 in 6 of 8 folds** — consistent sizing
- `min_pearson_r` varied from 0.76 to 0.89 — the median (0.83) balances selectivity vs opportunity
- `hard_sl_mult` ranged 2.0-2.75 — wider stops outperformed tighter ones

### 14.2 Per-Fold Test Results

| Fold | Trades | Win Rate | Sharpe | Max DD | PnL % |
|------|--------|----------|--------|--------|-------|
| 0 | 1,490 | 69.73% | 14.17 | -5.38% | 551.1% |
| 1 | 1,707 | 68.54% | 16.64 | -4.27% | 737.6% |
| 2 | 1,742 | 68.08% | 20.01 | -3.75% | 935.3% |
| 3 | 1,633 | 68.28% | 18.22 | -3.46% | 989.0% |
| 4 | 1,471 | 66.89% | 13.62 | -4.54% | 917.6% |
| 5 | 1,380 | 67.39% | 12.39 | -8.40% | 629.4% |
| 6 | 1,538 | 68.14% | 9.79 | -8.09% | 668.2% |
| 7 | 1,188 | 66.41% | 7.62 | -8.44% | 468.6% |

### 14.3 Train/Val/Test Performance Consistency

Checking for overfitting by comparing train, validation, and test performance:

| Fold | Train Sharpe | Val Sharpe | Test Sharpe | Overfit? |
|------|-------------|------------|-------------|----------|
| 0 | 13.86 | 15.61 | 14.17 | No (val > train, test stable) |
| 1 | 10.18 | 13.70 | 16.64 | No (test > val > train) |
| 2 | 13.14 | 13.90 | 20.01 | No (test > val > train) |
| 3 | 13.79 | 11.39 | 18.22 | No (test highest) |
| 4 | 14.89 | 16.49 | 13.62 | Mild (test slightly below) |
| 5 | 11.65 | 19.96 | 12.39 | Mild (val inflated) |
| 6 | 12.92 | 17.28 | 9.79 | Some (test noticeably below val) |
| 7 | 10.22 | 13.80 | 7.62 | Some (test below val) |

**Key takeaway:** Most folds show **no overfitting** — test performance often exceeds training performance, which is unusual and indicates the strategy's edge is genuine rather than curve-fitted. Later folds (6, 7) show some degradation, which is expected as they test on later time periods.

### 14.4 Aggregate Test Metrics

| Metric | Value |
|--------|-------|
| Average Win Rate | 67.93% |
| Average Sharpe Ratio | 14.06 |
| Average Max Drawdown | -5.79% |
| Average Profit Factor | 4.61 |
| Total Trades (all folds) | 12,149 |
| Average Trail Exit % | 99.16% |
| Average Hard SL Exit % | 0.80% |

---

## 15. Consensus Parameters

Consensus parameters are computed by taking the **median** across all 8 fold-optimal parameters. Median is chosen over mean because:
- More robust to outlier folds
- Less sensitive to one fold finding an extreme value
- Better represents the "central tendency" of the search space

### Final Consensus Values

| Parameter | Consensus Value | Description |
|-----------|----------------|-------------|
| min_pearson_r | **0.83** | Minimum |R| for entry signal |
| min_pvt_r | **0.80** | Minimum PVT |R| for volume confirmation |
| combined_gate | **0.80** | Block threshold for opposing signals |
| hard_sl_mult | **2.50** | Hard SL at 2.5x ATR(14) from entry |
| trail_buffer | **0.50** | Trail at 0.5 std_dev from midline |
| exhaust_r | **0.425** | Close if |R| drops below 0.425 |
| pos_frac | **0.05** | 5% of equity per position (optimized) |

**Live Override:** `pos_frac` is overridden to **0.10** (10%) for live trading to increase position sizing.

---

## 16. Blind Test Results

### 16.1 Configuration

- **Period:** 2025-11-01 to 2026-03-15 (~4.5 months)
- **Parameters:** Consensus from 8-fold WFV
- **Data:** Never seen during optimization
- **Engine:** FastBacktestEngine with pre-computed features
- **Capital:** $10,000 (backtest default)
- **Scan Interval:** 1 hour

### 16.2 Results

| Metric | Value |
|--------|-------|
| **Total Trades** | 1,097 |
| **Win Rate** | 70.74% |
| **Total PnL** | +381.13% |
| **Sharpe Ratio** | 12.60 |
| **Max Drawdown** | -5.36% |
| **Profit Factor** | 4.51 |
| **Trail Exit %** | 99.18% |
| **Hard SL Exit %** | 0.82% |

### 16.3 Blind Test vs WFV Aggregate Comparison

| Metric | WFV Aggregate | Blind Test | Delta |
|--------|---------------|------------|-------|
| Win Rate | 67.93% | 70.74% | +2.81% |
| Sharpe | 14.06 | 12.60 | -1.46 |
| Max DD | -5.79% | -5.36% | +0.43% better |
| Profit Factor | 4.61 | 4.51 | -0.10 |
| Trail Exit % | 99.16% | 99.18% | +0.02% |

**Interpretation:** The blind test performance is **remarkably consistent** with the WFV aggregate. Win rate actually improved (+2.8%), while Sharpe ratio decreased slightly (-1.46) — both within normal statistical variation. Max drawdown was slightly better. This consistency strongly suggests the consensus parameters generalize well to unseen data.

### 16.4 Blind Test Trade Sample

From the 1,097 blind test trades (first 4 shown):

| ID | Asset | Dir | Entry Price | Exit Price | TF | Period | |R| | Leverage | Reason | PnL% |
|----|-------|-----|------------|------------|-----|--------|-----|----------|--------|------|
| 1 | ICP | LONG | $3.680 | $3.683 | 5m | 27 | 0.967 | 5x | Trail | +0.44% |
| 2 | ICP | LONG | $3.868 | $3.855 | 5m | 39 | 0.975 | 5x | Trail | -1.72% |
| 3 | ICP | LONG | $4.026 | $4.023 | 5m | 51 | 0.985 | 5x | Trail | -0.43% |
| 4 | ICP | LONG | $3.957 | $4.012 | 5m | 68 | 0.950 | 3x | Trail | +4.19% |

---

## 17. Pre-Computed Feature Store (Fast Engine)

### 17.1 Purpose

The Optuna optimizer runs 100 trials per fold, each requiring a full backtest. With 8 folds, that's 800 backtests. Each backtest scans 15 assets x 543 regressions per scan x hundreds of 1-hour steps. To make this tractable, all regressions are **pre-computed once** and stored as structured numpy arrays.

### 17.2 Feature Structure

For each 1-hour timestamp, for each asset, the following features are pre-computed:

```
FEATURE_DTYPE = [
  ("timestamp_ns", int64),     # 1h bar timestamp in nanoseconds
  ("best_r", float32),         # Best |pearson_r| across all TF/periods
  ("best_slope", float32),     # Slope of best regression
  ("best_std", float32),       # Std dev of best regression
  ("best_midline", float32),   # exp(intercept) of best regression
  ("best_period", int16),      # Period of best regression
  ("best_tf_idx", int8),       # 0=5m, 1=30m, 2=1h
  ("best_direction", int8),    # +1 LONG, -1 SHORT
  ("pvt_r", float32),          # |pvt_pearson_r| on best TF/period
  ("pvt_direction", int8),     # +1 rising, -1 falling, 0 flat
  ("htf_bias", int8),          # +1 bull, -1 bear, 0 neutral
  ("max_opposing_r", float32), # Max |R| among opposing regressions
  ("atr_best", float32),       # ATR(14) on best TF
  ("close_best", float32),     # Close price on best TF
]
```

### 17.3 Performance Gain

| Operation | Regular Engine | Fast Engine |
|-----------|---------------|-------------|
| Scan per asset | 543 regressions | 1 array lookup (O(1)) |
| Scan per cycle | ~200ms | ~0.01ms |
| Full backtest | ~5 minutes | ~2 seconds |
| 100 trials | ~8 hours | ~3 minutes |
| 8-fold WFV | ~64 hours | ~25 minutes |

### 17.4 Position Exit in Fast Engine

While scan signals use pre-computed features, **position exit still computes regressions in real-time** because:
- Exit tracking needs bar-by-bar midline updates on the native TF
- The pre-computed features only capture 1h-resolution snapshots
- Real-time regression on a single (TF, period) is fast (~0.1ms per bar)

---

## 18. Live Trading Bot Architecture

### 18.1 Operational Modes

| Mode | Flag | Description |
|------|------|-------------|
| DRY-RUN | (default) | Simulates trades, no real orders |
| LIVE | --live | Executes real orders on Binance Futures |
| SINGLE SCAN | --once | Run one scan cycle and exit |
| STATUS | --status | Send status to Telegram and exit |

### 18.2 Main Loop

```
INITIALIZE:
  Load consensus params from wfv_results.json
  Load Binance API credentials (if --live)
  Connect Telegram bot
  Load persisted state from live_state.json

EVERY scan_interval MINUTES:
  1. Fetch 15 assets x 4 TFs from Binance REST API
     - 5m: 2016 bars, 30m/1h/4h: 500 bars
     - ~30-35 seconds per fetch cycle

  2. Check open positions for exits
     - Walk sub-bars since last check
     - Apply trail/SL/exhaustion/time checks
     - Close positions via market order (if live)
     - Send Telegram notification on close

  3. Scan for new entries
     - Run full pipeline (regression → PVT → 4H → combined gate)
     - Open positions up to MAX_CONCURRENT
     - Place market order + stop order (if live)
     - Send Telegram notification on entry

  4. Send scan summary to Telegram
     - Cycle number, equity, open positions
     - Next scan time

  5. Persist state to live_state.json

  SLEEP until next scan interval
```

### 18.3 Telegram Integration

**Bot:** @Varanusneoflowbot

**Automated Notifications:**
- New position opened (with entry price, leverage, SL levels)
- Position closed (with PnL, exit reason, duration)
- Scan cycle summary (equity, open positions, next scan time)
- Circuit breaker tripped
- Errors

**User Commands:**

| Command | Description |
|---------|-------------|
| /status | Full system status summary |
| /positions | Detailed open position info |
| /pnl | P&L breakdown (realized + unrealized) |
| /today | Today's trade activity |
| /params | Current consensus parameters |
| /help | List available commands |

### 18.4 State Persistence

State is saved to `live_state.json` after every scan cycle:

```json
{
  "trade_counter": 0,
  "realized_pnl": 0.0,
  "initial_capital": 1000.0,
  "peak_equity": 1000.0,
  "circuit_breaker": false,
  "last_scan_ts": "2026-03-17T19:14:11+00:00",
  "total_scans": 1,
  "positions": {}
}
```

All trades are logged to `live_trades.csv` for post-analysis.

---

## 19. Data Infrastructure

### 19.1 Historical Data (Backtest)

- **Source:** Binance Futures USDT-M via REST API
- **Assets:** 15 Tier-2 cryptocurrencies
- **Timeframes:** 5m, 30m, 1h, 4h
- **Date Range:** 2022-12-25 to 2026-03-15 (includes 7-day warmup before WFV start)
- **Format:** Parquet files per asset per timeframe
- **Storage:** ~2GB total, numpy arrays in memory (~60MB)

### 19.2 Live Data

- **Source:** Binance Futures REST API (`/fapi/v1/klines`)
- **Fetch per cycle:** 15 assets x 4 TFs = 60 API requests
- **5m bars:** 2016 (7 days, needed for regression periods up to 200)
- **30m/1h/4h bars:** 500 (more than sufficient)
- **Fetch time:** ~30-35 seconds per cycle
- **Rate limiting:** Sequential requests with exponential backoff on errors

### 19.3 Data Processing

1. Raw klines (OHLCV) converted to numpy arrays (float64 prices, int64 timestamps)
2. PVT computed cumulatively from close + volume arrays
3. All data kept in memory for the duration of the scan
4. No database; state persisted as JSON, trades as CSV

---

## 20. Complete Parameter Reference

### 20.1 Signal Generation Parameters

| Parameter | Default | Consensus | Range (Optuna) | Description |
|-----------|---------|-----------|----------------|-------------|
| MIN_PEARSON_R | 0.80 | 0.83 | [0.75, 0.92] | Min |R| for entry |
| MIN_PVT_PEARSON_R | 0.70 | 0.80 | [0.50, 0.85] | Min PVT |R| |
| COMBINED_GATE_THRESHOLD | 0.65 | 0.80 | [0.50, 0.80] | Opposing TF block |
| PVT_DIVERGENCE_PRICE_R | 0.85 | 0.85 | Fixed | Divergence check threshold |
| PVT_DIVERGENCE_WEAK_R | 0.50 | 0.50 | Fixed | Weak volume threshold |

### 20.2 Position Management Parameters

| Parameter | Default | Consensus | Range (Optuna) | Description |
|-----------|---------|-----------|----------------|-------------|
| HARD_SL_ATR_MULT | 1.50 | 2.50 | [1.0, 3.0] | Hard SL in ATR units |
| TRAIL_BUFFER_STD | 1.00 | 0.50 | [0.5, 2.0] | Trail offset in std_devs |
| TREND_EXHAUST_R | 0.50 | 0.425 | [0.30, 0.65] | Exhaustion |R| threshold |
| MAX_BARS | 200 | 200 | Fixed | Max holding period |

### 20.3 Portfolio Parameters

| Parameter | Default | Consensus | Live Override | Description |
|-----------|---------|-----------|---------------|-------------|
| pos_frac | 0.10 | 0.05 | 0.10 | Equity fraction per trade |
| MAX_CONCURRENT | 4 | 4 | 4 | Max simultaneous positions |
| initial_capital | 10,000 | 10,000 | 1,000 | Starting capital |

### 20.4 Scanning Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| SCAN_TIMEFRAMES | [5m, 30m, 1h] | Timeframes for regression scan |
| HTF_TIMEFRAME | 4h | Higher-timeframe filter |
| PERIOD_RANGE | [20, 200] | Regression period range |
| ROLLING_WINDOW | 7 days | Data window for each scan |
| SCAN_INTERVAL | 1 hour | Time between scan cycles |

### 20.5 HTF Filter Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| EMA_FAST | 21 | Fast EMA on 4H |
| EMA_SLOW | 55 | Slow EMA on 4H |
| MSS_LOOKBACK | 30 | Swing point lookback on 4H |

### 20.6 Leverage Tiers

| Confidence (|R|) | Leverage |
|-------------------|----------|
| >= 0.95 | 5x |
| >= 0.90 | 3x |
| >= 0.85 | 2x |
| >= 0.80 | 1x |
| < 0.80 | 0 (no trade) |

---

## 21. Performance Metrics Definitions

### 21.1 Trade Metrics

| Metric | Formula |
|--------|---------|
| Win Rate | winning_trades / total_trades x 100 |
| Profit Factor | gross_profit / gross_loss |
| Average Winner | mean(pnl_pct where pnl > 0) |
| Average Loser | mean(pnl_pct where pnl < 0) |
| Largest Win | max(pnl_pct) |
| Largest Loss | min(pnl_pct) |

### 21.2 Risk Metrics

| Metric | Formula |
|--------|---------|
| Sharpe Ratio | (mean(hourly_returns) / std(hourly_returns)) x sqrt(8760) |
| Max Drawdown | min((equity - peak_equity) / peak_equity) x 100 |
| Calmar Ratio | annualized_return / abs(max_drawdown) |

### 21.3 Exit Distribution

| Metric | Formula |
|--------|---------|
| Trail Exit % | trail_exits / total_trades x 100 |
| Hard SL Exit % | hard_sl_exits / total_trades x 100 |
| Exhaustion Exit % | exhaustion_exits / total_trades x 100 |
| Time Barrier Exit % | time_barrier_exits / total_trades x 100 |

### 21.4 Duration Metrics

| Metric | Formula |
|--------|---------|
| Average Duration | mean(exit_ts - entry_ts) in hours |
| Average Bars Held | mean(bars_held) |

---

## 22. File Structure & Codebase Map

```
varanus_neo_flow_git/
│
├── neo_flow/                           # Core algorithm modules
│   ├── __init__.py
│   ├── adaptive_engine.py              # ~900 lines
│   │   ├── calc_log_regression()       # Pine Script log regression
│   │   ├── calc_pvt()                  # PVT computation
│   │   ├── calc_pvt_regression()       # PVT linear regression
│   │   ├── check_pvt_alignment()       # 3-tier PVT gate
│   │   ├── get_htf_bias()             # 4H MSS + EMA filter
│   │   ├── scan_asset()               # Full pipeline per asset
│   │   ├── scan_universe()            # All 15 assets
│   │   ├── compute_atr()              # ATR(14)
│   │   ├── compute_hard_sl()          # Hard SL placement
│   │   └── get_leverage()             # Confidence → leverage tier
│   │
│   └── precompute_features.py          # ~630 lines
│       ├── batch_regression()          # Vectorized multi-period regression
│       ├── precompute_asset_features() # Per-asset feature computation
│       └── precompute_all_features()   # Full universe pre-computation
│
├── backtest/                           # Backtesting framework
│   ├── __init__.py
│   ├── data_loader.py                  # ~315 lines
│   │   ├── load_all_assets()          # Load parquet → numpy
│   │   ├── generate_wfv_folds()       # 8-fold WFV construction
│   │   ├── build_scan_dataframes()    # 7-day window extraction
│   │   └── build_htf_dataframe()      # 4H data extraction
│   │
│   ├── engine.py                       # ~504 lines
│   │   └── BacktestEngine             # Regular engine (on-the-fly regression)
│   │       ├── run()                  # Main backtest loop
│   │       ├── _update_positions()    # Sub-bar exit checks
│   │       └── _scan_and_enter()      # Signal generation + entry
│   │
│   ├── engine_fast.py                  # ~357 lines
│   │   └── FastBacktestEngine         # Pre-computed feature engine
│   │       ├── run()                  # Fast backtest loop
│   │       └── _lookup_features()     # O(1) feature lookup
│   │
│   ├── metrics.py                      # ~262 lines
│   │   ├── compute_metrics()          # Full metric computation
│   │   ├── print_metrics()            # Formatted output
│   │   └── trades_to_csv()            # CSV export
│   │
│   ├── optimize.py                     # ~414 lines
│   │   ├── _create_objective()        # Optuna objective function
│   │   ├── optimize_fold()            # Single fold optimization
│   │   ├── compute_consensus_params() # Median across folds
│   │   └── run_wfv()                  # Full 8-fold WFV
│   │
│   └── optimize_fast.py                # ~294 lines
│       └── run_wfv_fast()             # Parallel WFV with pre-computed features
│
├── live_bot.py                         # ~1300 lines — Live trading bot
│   ├── LiveBot                        # Main bot class
│   │   ├── run_cycle()               # One scan cycle
│   │   ├── _check_exits()            # Exit management
│   │   ├── _scan_and_enter()         # Signal + entry
│   │   ├── send_status()             # Telegram /status
│   │   ├── send_positions()          # Telegram /positions
│   │   └── send_pnl()               # Telegram /pnl
│   ├── fetch_live_data()             # Binance REST data fetch
│   ├── tg_send()                     # Telegram message sender
│   └── start_telegram_listener()     # Command polling thread
│
├── run_backtest.py                     # CLI entry point
├── data_fetcher.py                     # Historical data download
├── dashboard.py                        # Streamlit visualization
├── test_scan.py                        # Unit tests
├── test_engine.py                      # Integration tests
├── stress_test.py                      # Parameter stress tests
│
├── wfv_results.json                    # 8-fold WFV results + consensus
├── blind_test_trades.csv               # 1,097 blind test trades
├── live_state.json                     # Bot persistence
├── live_trades.csv                     # Live trade log
├── .env                                # Telegram + Binance credentials
│
└── data/ -> /home/gokhan/varanus_neo_flow/data/  # Symlink to historical data
    ├── ADA_5m.parquet
    ├── ADA_30m.parquet
    ├── ADA_1h.parquet
    ├── ADA_4h.parquet
    ├── ... (15 assets x 4 TFs = 60 files)
    └── THETA_4h.parquet
```

---

## Appendix A: Key Formulas Quick Reference

### Logarithmic Regression
```
midline = exp(intercept)
std_dev_price = midline * std_dev
upper_band = midline * exp(+std_dev)
lower_band = midline * exp(-std_dev)
```

### Position Sizing
```
position_usd = equity * pos_frac * leverage * vol_scalar
```

### Hard Stop Loss
```
LONG:  hard_sl = entry - hard_sl_mult * ATR(14)
SHORT: hard_sl = entry + hard_sl_mult * ATR(14)
```

### Adaptive Trail
```
LONG:  trail = MAX(trail, midline - trail_buffer * midline * std_dev)
SHORT: trail = MIN(trail, midline + trail_buffer * midline * std_dev)
```

### P&L
```
pnl_pct = (exit - entry) / entry * direction * 100
pnl_pct_leveraged = pnl_pct * leverage
pnl_usd = pnl_pct_leveraged / 100 * position_usd
```

### Sharpe Ratio
```
sharpe = (mean(hourly_returns) / std(hourly_returns)) * sqrt(8760)
```

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| ATR | Average True Range — volatility measure over 14 periods |
| Combined Gate | Filter that blocks entry when opposing timeframes show strong signals |
| Consensus Parameters | Median of optimal parameters across all WFV folds |
| EMA | Exponential Moving Average |
| Embargo | 24-hour gap between WFV periods to prevent information leakage |
| HTF | Higher Timeframe (4-hour) |
| MSS | Market Structure Shift — breakout above swing high or below swing low |
| OLS | Ordinary Least Squares — linear regression method |
| Pearson R | Correlation coefficient measuring regression fit quality (0-1) |
| PVT | Price Volume Trend — cumulative volume-weighted momentum indicator |
| Ratchet | Mechanism where trailing stop only moves in the profitable direction |
| TPE | Tree-structured Parzen Estimator — Bayesian optimization algorithm |
| WFV | Walk-Forward Validation — rolling train/validate/test methodology |

---

*End of Report*

"""
neo_flow/adaptive_engine.py — Adaptive Multi-TF Trend Engine

Core of Varanus Neo-Flow.  Implements:
  1. Logarithmic Linear Regression (exact Pine Script calcDev translation)
  2. Multi-TF period scanner (5m, 30m, 1h × periods 20–200)
  3. 4H trend filter (MSS + EMA21/55 alignment)
  4. Adaptive trailing stop (regression midline based)
  5. Combined gate (suppress opposing signals)

All math is vectorised with NumPy for sub-200ms full scans.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SCAN_TIMEFRAMES = ["5m", "30m", "1h"]
HTF_TIMEFRAME   = "4h"
PERIOD_MIN      = 20
PERIOD_MAX      = 200
PERIOD_RANGE    = range(PERIOD_MIN, PERIOD_MAX + 1)

TIER2_UNIVERSE = [
    "ADA", "AVAX", "LINK", "DOT", "TRX", "NEAR", "UNI",
    "SUI", "ARB", "OP", "POL", "APT", "ATOM", "FIL", "ICP",
]

HIGH_VOL_ASSETS = {"ICP"}

# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_PEARSON_R           = 0.80   # Minimum |R| to generate a signal
COMBINED_GATE_THRESHOLD = 0.65   # Suppress if opposing direction |R| > this
TREND_EXHAUST_R         = 0.50   # Close if active |R| drops below this
HARD_SL_ATR_MULT        = 1.5    # Hard SL distance in ATR(14) multiples
TRAIL_BUFFER_STD        = 1.0    # Trailing stop = midline ± buffer * std_dev
MAX_CONCURRENT          = 4
MAX_LEVERAGE            = 2.5

# ── EMA periods for HTF filter ────────────────────────────────────────────────

EMA_FAST = 21
EMA_SLOW = 55
MSS_LOOKBACK = 30   # Bars to search for swing points on 4H


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegressionResult:
    """Output of a single log-regression computation."""
    std_dev:   float
    pearson_r: float
    slope:     float
    intercept: float
    midline:   float       # exp(intercept) — current fitted price
    period:    int
    timeframe: str


@dataclass
class ScanSignal:
    """Candidate trade signal from the scanner."""
    asset:       str
    direction:   int        # +1 LONG, -1 SHORT
    confidence:  float      # |pearson_r| of best (TF, period)
    best_tf:     str
    best_period: int
    entry_price: float
    sl_price:    float      # Hard SL (server-side STOP_MARKET)
    midline:     float      # Regression midline for adaptive trailing
    std_dev:     float      # Channel width for trail buffer
    atr:         float      # ATR(14) at entry
    regression:  RegressionResult


@dataclass
class ActiveTrade:
    """State of an open position managed by the adaptive trailing stop."""
    asset:           str
    direction:       int
    entry_price:     float
    hard_sl:         float
    trail_sl:        float       # Current trailing stop level
    best_trail:      float       # Best midline seen (for ratchet)
    midline:         float
    std_dev:         float
    best_tf:         str
    best_period:     int
    entry_ts:        pd.Timestamp
    bars_held:       int = 0
    peak_r:          float = 0.0  # Highest |R| seen during trade


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Logarithmic Linear Regression — Pine Script calcDev() Translation
# ═══════════════════════════════════════════════════════════════════════════════

def calc_log_regression(
    prices: np.ndarray,
    length: int,
) -> tuple[float, float, float, float]:
    """
    Exact translation of Pine Script ``calcDev(source, length)``.

    Parameters
    ----------
    prices : 1-D float array, **oldest-first** (standard Pandas convention).
             Must have at least ``length`` elements.
    length : Regression window (number of bars).

    Returns
    -------
    (std_dev, pearson_r, slope, intercept)

    Notes
    -----
    Pine Script indexes ``[0]`` = newest bar.  We flip internally so callers
    can pass data in the natural Pandas order.

    Slope sign convention (because x=1 → newest, x=length → oldest):
      - slope < 0 → prices are RISING  → uptrend   → LONG
      - slope > 0 → prices are FALLING → downtrend  → SHORT
    """
    # ── Flip to Pine Script order: index 0 = newest ──────────────────────
    log_src = np.log(prices[-length:].astype(np.float64)[::-1])

    # ── First pass: OLS on log(price) ────────────────────────────────────
    x = np.arange(1, length + 1, dtype=np.float64)

    sum_x  = x.sum()
    sum_xx = (x * x).sum()
    sum_yx = (x * log_src).sum()
    sum_y  = log_src.sum()

    denom = length * sum_xx - sum_x * sum_x
    slope = (length * sum_yx - sum_x * sum_y) / denom

    average   = sum_y / length
    intercept = average - slope * sum_x / length + slope
    # NOTE: Pine Script intercept = regression value at x=1 (newest bar).
    #       Standard OLS intercept would omit the final ``+ slope``.

    # ── Second pass: residuals + Pearson R ───────────────────────────────
    period_1 = length - 1
    regres   = intercept + slope * period_1 * 0.5   # midpoint of regression line

    # Regression line values: intercept, intercept+slope, ..., intercept+(length-1)*slope
    i_vals   = np.arange(length, dtype=np.float64)
    reg_line = intercept + i_vals * slope

    # Deviations
    dxt = log_src - average       # actual deviation from mean
    dyt = reg_line - regres       # fitted deviation from midpoint

    # Residuals (actual − fitted)
    residuals = log_src - reg_line

    sum_dxx = (dxt * dxt).sum()
    sum_dyy = (dyt * dyt).sum()
    sum_dyx = (dxt * dyt).sum()
    sum_dev = (residuals * residuals).sum()

    std_dev = np.sqrt(sum_dev / period_1) if period_1 > 0 else 0.0

    d = np.sqrt(sum_dxx * sum_dyy)
    pearson_r = (sum_dyx / d) if d > 1e-15 else 0.0

    return std_dev, pearson_r, slope, intercept


def calc_log_regression_full(
    prices: np.ndarray,
    length: int,
    timeframe: str = "",
) -> RegressionResult:
    """Convenience wrapper returning a RegressionResult dataclass."""
    std_dev, pearson_r, slope, intercept = calc_log_regression(prices, length)
    return RegressionResult(
        std_dev=std_dev,
        pearson_r=pearson_r,
        slope=slope,
        intercept=intercept,
        midline=np.exp(intercept),
        period=length,
        timeframe=timeframe,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Multi-TF Period Scanner
# ═══════════════════════════════════════════════════════════════════════════════

def scan_all_periods(
    close: np.ndarray,
    timeframe: str,
    periods: range = PERIOD_RANGE,
) -> list[RegressionResult]:
    """
    Compute log-regression for every period in *periods* on one close array.

    Parameters
    ----------
    close     : 1-D float array (oldest-first).
    timeframe : Label string ("5m", "30m", "1h").
    periods   : Range of periods to test.

    Returns
    -------
    List of RegressionResult, one per valid period.
    """
    results: list[RegressionResult] = []
    n = len(close)

    for p in periods:
        if n < p:
            break  # periods are ascending — no point continuing
        std_dev, pearson_r, slope, intercept = calc_log_regression(close, p)
        results.append(RegressionResult(
            std_dev=std_dev,
            pearson_r=pearson_r,
            slope=slope,
            intercept=intercept,
            midline=np.exp(intercept),
            period=p,
            timeframe=timeframe,
        ))

    return results


def find_best_regression(
    data: dict[str, pd.DataFrame],
) -> tuple[Optional[RegressionResult], list[RegressionResult]]:
    """
    Scan all timeframes × all periods and return the single best result.

    Parameters
    ----------
    data : ``{timeframe_str: ohlcv_dataframe}`` for scan TFs (5m, 30m, 1h).

    Returns
    -------
    (best_result, all_results)
    best_result is None if no valid regression was found.
    """
    all_results: list[RegressionResult] = []

    for tf in SCAN_TIMEFRAMES:
        df = data.get(tf)
        if df is None or df.empty:
            continue
        close = df["close"].values.astype(np.float64)
        if len(close) < PERIOD_MIN:
            continue
        results = scan_all_periods(close, tf)
        all_results.extend(results)

    if not all_results:
        return None, []

    best = max(all_results, key=lambda r: abs(r.pearson_r))
    return best, all_results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 4H Trend Filter — MSS + EMA Alignment
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_mss(df: pd.DataFrame, lookback: int = MSS_LOOKBACK) -> int:
    """
    Detect Market Structure Shift on the most recent bar.

    Returns +1 (bullish MSS), -1 (bearish MSS), or 0 (no shift).

    Bullish MSS: current close breaks above the highest high of the
                 lookback window (excluding current bar).
    Bearish MSS: current close breaks below the lowest low of the
                 lookback window (excluding current bar).
    """
    if len(df) < lookback + 1:
        return 0

    window = df.iloc[-(lookback + 1):-1]
    current = df.iloc[-1]

    swing_high = window["high"].max()
    swing_low  = window["low"].min()

    if current["close"] > swing_high:
        return 1    # Bullish MSS
    if current["close"] < swing_low:
        return -1   # Bearish MSS
    return 0


def get_htf_bias(df_4h: pd.DataFrame) -> int:
    """
    Combined 4H higher-timeframe bias: MSS direction + EMA alignment.

    Returns +1 (bullish), -1 (bearish), or 0 (neutral/conflicting).
    Both MSS and EMA must agree for a non-zero bias.
    """
    if df_4h is None or len(df_4h) < max(EMA_SLOW, MSS_LOOKBACK) + 1:
        return 0

    mss = detect_mss(df_4h, MSS_LOOKBACK)

    ema_fast = compute_ema(df_4h["close"], EMA_FAST)
    ema_slow = compute_ema(df_4h["close"], EMA_SLOW)

    ema_bullish = float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1])
    ema_bearish = float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1])

    if mss == 1 and ema_bullish:
        return 1
    if mss == -1 and ema_bearish:
        return -1
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Combined Gate — Suppress Conflicting Signals
# ═══════════════════════════════════════════════════════════════════════════════

def check_combined_gate(
    direction: int,
    all_results: list[RegressionResult],
    threshold: float = COMBINED_GATE_THRESHOLD,
) -> bool:
    """
    Check whether any TF/period shows a strong opposing signal.

    Parameters
    ----------
    direction   : Proposed trade direction (+1 or -1).
    all_results : Every RegressionResult from the scan.
    threshold   : Suppress if opposing |R| exceeds this.

    Returns
    -------
    True if the gate PASSES (no strong opposition).
    False if the signal should be suppressed.
    """
    for r in all_results:
        # Determine what direction this regression implies
        #   slope < 0 → uptrend → direction +1
        #   slope > 0 → downtrend → direction -1
        r_direction = -1 if r.slope > 0 else (1 if r.slope < 0 else 0)

        # Is this result opposing our proposed direction?
        if r_direction != 0 and r_direction != direction:
            if abs(r.pearson_r) > threshold:
                logger.debug(
                    "Combined gate BLOCKED: opposing %s %s period=%d |R|=%.4f > %.2f",
                    r.timeframe, "SHORT" if r_direction == -1 else "LONG",
                    r.period, abs(r.pearson_r), threshold,
                )
                return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Leverage & Position Sizing
# ═══════════════════════════════════════════════════════════════════════════════

def get_leverage(confidence: float) -> int:
    """
    Map |pearson_r| confidence to leverage tier.

    | Confidence    | Leverage |
    |---------------|----------|
    | [0.80, 0.85)  | 1x       |
    | [0.85, 0.90)  | 2x       |
    | [0.90, 0.95)  | 3x       |
    | [0.95, 1.00]  | 5x       |
    """
    if confidence >= 0.95:
        return 5
    if confidence >= 0.90:
        return 3
    if confidence >= 0.85:
        return 2
    if confidence >= 0.80:
        return 1
    return 0   # Below minimum — should not reach here


def compute_position_size(
    capital: float,
    confidence: float,
    asset: str,
) -> tuple[float, int]:
    """
    Calculate USD position size and leverage.

    Formula: pos_usd = capital * 0.10 * leverage * vol_scalar

    Returns (pos_usd, leverage).
    """
    lev = get_leverage(confidence)
    vol_scalar = 0.75 if asset in HIGH_VOL_ASSETS else 1.0
    pos_usd = capital * 0.10 * lev * vol_scalar
    return pos_usd, lev


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Adaptive Trailing Stop
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trail_sl(
    direction: int,
    midline: float,
    std_dev: float,
    current_trail: float,
    buffer_mult: float = TRAIL_BUFFER_STD,
) -> float:
    """
    Compute new adaptive trailing stop based on regression midline.

    The trail ratchets — it only moves in the favourable direction.

    LONG:  trail = max(current_trail, midline - buffer_mult * std_dev_price)
    SHORT: trail = min(current_trail, midline + buffer_mult * std_dev_price)

    std_dev is in log space, so we convert: std_dev_price = midline * std_dev
    """
    # Convert log-space std_dev to price-space
    std_dev_price = midline * std_dev

    if direction == 1:  # LONG
        new_trail = midline - buffer_mult * std_dev_price
        return max(current_trail, new_trail)  # Ratchet up
    else:               # SHORT
        new_trail = midline + buffer_mult * std_dev_price
        return min(current_trail, new_trail)  # Ratchet down


def check_trail_hit(
    direction: int,
    trail_sl: float,
    bar: pd.Series,
) -> bool:
    """Check if the trailing stop was hit during this bar."""
    if direction == 1:   # LONG — trail is below, hit if low touches
        return float(bar["low"]) <= trail_sl
    else:                # SHORT — trail is above, hit if high touches
        return float(bar["high"]) >= trail_sl


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Hard Stop Loss Calculation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hard_sl(
    entry_price: float,
    atr: float,
    direction: int,
    mult: float = HARD_SL_ATR_MULT,
) -> float:
    """
    Server-side STOP_MARKET placement (touch-based).

    LONG:  SL = entry - mult * ATR
    SHORT: SL = entry + mult * ATR
    """
    if direction == 1:
        return entry_price - mult * atr
    return entry_price + mult * atr


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Full Asset Scanner
# ═══════════════════════════════════════════════════════════════════════════════

def scan_asset(
    asset: str,
    scan_data: dict[str, pd.DataFrame],
    df_4h: pd.DataFrame,
) -> Optional[ScanSignal]:
    """
    Run the full Neo-Flow pipeline for a single asset.

    Parameters
    ----------
    asset     : Ticker symbol (e.g. "DOT").
    scan_data : ``{"5m": df, "30m": df, "1h": df}`` — OHLCV for scan TFs.
    df_4h     : 4H OHLCV DataFrame for the HTF filter.

    Returns
    -------
    ScanSignal if all gates pass, else None.
    """
    # ── Step 1: Find best regression across all TFs and periods ──────────
    best, all_results = find_best_regression(scan_data)

    if best is None:
        logger.debug("[%s] No valid regression found", asset)
        return None

    abs_r = abs(best.pearson_r)

    # ── Step 2: Minimum trend strength ───────────────────────────────────
    if abs_r < MIN_PEARSON_R:
        logger.debug(
            "[%s] Best |R|=%.4f < %.2f (TF=%s, P=%d) — skip",
            asset, abs_r, MIN_PEARSON_R, best.timeframe, best.period,
        )
        return None

    # ── Step 3: Direction from slope ─────────────────────────────────────
    #   slope < 0 → uptrend → LONG (+1)
    #   slope > 0 → downtrend → SHORT (-1)
    direction = 1 if best.slope < 0 else -1

    # ── Step 4: 4H trend filter ──────────────────────────────────────────
    htf_bias = get_htf_bias(df_4h)
    if htf_bias == 0:
        logger.debug("[%s] 4H bias neutral — skip", asset)
        return None
    if htf_bias != direction:
        logger.debug(
            "[%s] Direction %+d conflicts with 4H bias %+d — skip",
            asset, direction, htf_bias,
        )
        return None

    # ── Step 5: Combined gate ────────────────────────────────────────────
    if not check_combined_gate(direction, all_results):
        logger.debug("[%s] Combined gate blocked", asset)
        return None

    # ── Step 6: Build signal ─────────────────────────────────────────────
    # Use the best TF's latest close as entry price
    best_df = scan_data[best.timeframe]
    entry_price = float(best_df.iloc[-1]["close"])
    atr = float(compute_atr(best_df, 14).iloc[-1])

    if np.isnan(atr) or atr <= 0:
        logger.debug("[%s] ATR invalid", asset)
        return None

    sl_price = compute_hard_sl(entry_price, atr, direction)

    # Initial trailing stop = midline ± buffer
    std_dev_price = best.midline * best.std_dev
    if direction == 1:
        initial_trail = best.midline - TRAIL_BUFFER_STD * std_dev_price
    else:
        initial_trail = best.midline + TRAIL_BUFFER_STD * std_dev_price

    signal = ScanSignal(
        asset=asset,
        direction=direction,
        confidence=abs_r,
        best_tf=best.timeframe,
        best_period=best.period,
        entry_price=entry_price,
        sl_price=sl_price,
        midline=best.midline,
        std_dev=best.std_dev,
        atr=atr,
        regression=best,
    )

    dir_str = "LONG" if direction == 1 else "SHORT"
    logger.info(
        "[%s] SIGNAL  %s  |R|=%.4f  TF=%s  P=%d  entry=%.6f  SL=%.6f  midline=%.6f",
        asset, dir_str, abs_r, best.timeframe, best.period,
        entry_price, sl_price, best.midline,
    )

    return signal


def scan_universe(
    market_data: dict[str, dict[str, pd.DataFrame]],
    open_positions: dict[str, ActiveTrade],
) -> list[ScanSignal]:
    """
    Scan all 15 Tier-2 assets and return qualifying signals.

    Parameters
    ----------
    market_data    : ``{asset: {"5m": df, "30m": df, "1h": df, "4h": df}}``
    open_positions : Currently open trades (to skip already-held assets).

    Returns
    -------
    List of ScanSignal sorted by confidence (descending).
    """
    signals: list[ScanSignal] = []

    for asset in TIER2_UNIVERSE:
        if asset in open_positions:
            continue

        asset_data = market_data.get(asset)
        if asset_data is None:
            continue

        scan_tfs = {tf: asset_data[tf] for tf in SCAN_TIMEFRAMES if tf in asset_data}
        df_4h    = asset_data.get("4h")

        if not scan_tfs or df_4h is None:
            continue

        signal = scan_asset(asset, scan_tfs, df_4h)
        if signal is not None:
            signals.append(signal)

    # Sort by confidence — strongest trends first
    signals.sort(key=lambda s: s.confidence, reverse=True)

    if signals:
        logger.info(
            "Scan complete: %d signal(s) — best: %s %s |R|=%.4f TF=%s P=%d",
            len(signals), signals[0].asset,
            "LONG" if signals[0].direction == 1 else "SHORT",
            signals[0].confidence, signals[0].best_tf, signals[0].best_period,
        )
    else:
        logger.info("Scan complete: 0 signals across %d assets", len(TIER2_UNIVERSE))

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Trade Management — Update Trailing Stop
# ═══════════════════════════════════════════════════════════════════════════════

def update_active_trade(
    trade: ActiveTrade,
    scan_data: dict[str, pd.DataFrame],
) -> ActiveTrade:
    """
    Recalculate regression on the active (TF, period) and update the
    adaptive trailing stop.

    Call this each bar of the active TF for open positions.
    """
    df = scan_data.get(trade.best_tf)
    if df is None or len(df) < trade.best_period:
        return trade

    close = df["close"].values.astype(np.float64)
    std_dev, pearson_r, slope, intercept = calc_log_regression(close, trade.best_period)

    midline = np.exp(intercept)

    trade.midline  = midline
    trade.std_dev  = std_dev
    trade.peak_r   = max(trade.peak_r, abs(pearson_r))
    trade.bars_held += 1

    # Update trailing stop (ratchet)
    trade.trail_sl = compute_trail_sl(
        trade.direction, midline, std_dev, trade.trail_sl,
    )

    return trade


def check_exit_conditions(
    trade: ActiveTrade,
    current_bar: pd.Series,
    current_r: float,
) -> Optional[str]:
    """
    Check all exit conditions for an active trade.

    Returns exit reason string or None if trade should stay open.
    """
    # 1. Hard SL (touch-based — checked against bar high/low)
    if trade.direction == 1 and float(current_bar["low"]) <= trade.hard_sl:
        return "HARD_SL_HIT"
    if trade.direction == -1 and float(current_bar["high"]) >= trade.hard_sl:
        return "HARD_SL_HIT"

    # 2. Adaptive trailing stop
    if check_trail_hit(trade.direction, trade.trail_sl, current_bar):
        return "ADAPTIVE_TRAIL_HIT"

    # 3. Trend exhaustion
    if abs(current_r) < TREND_EXHAUST_R:
        return "TREND_EXHAUSTION"

    # 4. Time barrier (200 bars of the active TF)
    if trade.bars_held >= 200:
        return "TIME_BARRIER"

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Diagnostic / Debug Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def print_scan_report(
    market_data: dict[str, dict[str, pd.DataFrame]],
    top_n: int = 5,
) -> None:
    """
    Print a diagnostic table of the top-N regressions per asset.
    Useful for offline analysis and debugging.
    """
    print()
    print("=" * 110)
    print(f"{'ASSET':<8} {'TF':<5} {'PERIOD':>6} {'|R|':>8} {'SLOPE':>12} "
          f"{'MIDLINE':>12} {'STD_DEV':>10} {'DIR':>6} {'4H_BIAS':>8}")
    print("=" * 110)

    for asset in TIER2_UNIVERSE:
        asset_data = market_data.get(asset)
        if asset_data is None:
            print(f"{asset:<8} NO DATA")
            continue

        scan_tfs = {tf: asset_data[tf] for tf in SCAN_TIMEFRAMES if tf in asset_data}
        _, all_results = find_best_regression(scan_tfs)

        if not all_results:
            print(f"{asset:<8} NO RESULTS")
            continue

        # Sort by |R| descending
        sorted_results = sorted(all_results, key=lambda r: abs(r.pearson_r), reverse=True)

        # 4H bias
        df_4h = asset_data.get("4h")
        htf_bias = get_htf_bias(df_4h) if df_4h is not None else 0
        bias_str = {1: "BULL", -1: "BEAR", 0: "NEUTRAL"}[htf_bias]

        for i, r in enumerate(sorted_results[:top_n]):
            abs_r = abs(r.pearson_r)
            direction = "LONG" if r.slope < 0 else "SHORT"
            marker = " ***" if abs_r >= MIN_PEARSON_R else ""

            if i == 0:
                print(f"{asset:<8} {r.timeframe:<5} {r.period:>6} {abs_r:>8.4f} "
                      f"{r.slope:>12.8f} {r.midline:>12.6f} {r.std_dev:>10.6f} "
                      f"{direction:>6} {bias_str:>8}{marker}")
            else:
                print(f"{'':8} {r.timeframe:<5} {r.period:>6} {abs_r:>8.4f} "
                      f"{r.slope:>12.8f} {r.midline:>12.6f} {r.std_dev:>10.6f} "
                      f"{direction:>6} {'':>8}{marker}")

    print("=" * 110)
    print(f"  *** = |R| >= {MIN_PEARSON_R} (signal threshold)")
    print()

"""
neo_flow/adaptive_engine.py — Adaptive Multi-TF Trend Engine

Core of Varanus Neo-Flow.  Implements:
  1. Logarithmic Linear Regression (exact Pine Script calcDev translation)
  2. Linear Regression on PVT (Price Volume Trend) for volume confirmation
  3. Multi-TF period scanner with standardized 7-day rolling window
  4. PVT alignment gate + volume-price divergence detection
  5. 4H trend filter (MSS + EMA21/55 alignment)
  6. Adaptive trailing stop (regression midline based)
  7. Combined gate (suppress opposing signals)

All math is vectorised with NumPy for sub-200ms full scans.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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

# ── Standardized 7-Day Rolling Window ────────────────────────────────────────
ROLLING_WINDOW_DAYS = 7
BARS_7D = {
    "5m":  2016,    # 7 * 24 * 12
    "30m": 336,     # 7 * 24 * 2
    "1h":  168,     # 7 * 24
    "4h":  42,      # 7 * 6
}

TIER2_UNIVERSE = [
    "ADA", "AVAX", "LINK", "DOT", "TRX",
    "SOL", "ATOM", "NEAR", "ALGO", "UNI",
    "ICP", "HBAR", "SAND", "MANA", "THETA",
]

HIGH_VOL_ASSETS = {"ICP"}

# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_PEARSON_R           = 0.80   # Minimum price |R| to generate a signal
MIN_PVT_PEARSON_R       = 0.70   # Minimum PVT |R| for alignment gate
PVT_DIVERGENCE_PRICE_R  = 0.85   # Price |R| above this triggers divergence check
PVT_DIVERGENCE_WEAK_R   = 0.50   # PVT |R| below this = weak volume confirmation
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
class PVTResult:
    """Output of PVT linear regression."""
    pearson_r: float
    slope:     float
    direction: int          # +1 rising PVT, -1 falling PVT


@dataclass
class ScanSignal:
    """Candidate trade signal from the scanner."""
    asset:        str
    direction:    int        # +1 LONG, -1 SHORT
    confidence:   float      # |pearson_r| of best (TF, period)
    pvt_r:        float      # |pvt_pearson_r| — volume confirmation strength
    best_tf:      str
    best_period:  int
    entry_price:  float
    sl_price:     float      # Hard SL (server-side STOP_MARKET)
    midline:      float      # Regression midline for adaptive trailing
    std_dev:      float      # Channel width for trail buffer
    atr:          float      # ATR(14) at entry
    regression:   RegressionResult
    pvt:          PVTResult


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

    Slope sign convention (x=1 → newest, x=length → oldest):
      - slope < 0 → prices RISING  → uptrend  → LONG
      - slope > 0 → prices FALLING → downtrend → SHORT
    """
    log_src = np.log(prices[-length:].astype(np.float64)[::-1])

    x = np.arange(1, length + 1, dtype=np.float64)

    sum_x  = x.sum()
    sum_xx = (x * x).sum()
    sum_yx = (x * log_src).sum()
    sum_y  = log_src.sum()

    denom = length * sum_xx - sum_x * sum_x
    slope = (length * sum_yx - sum_x * sum_y) / denom

    average   = sum_y / length
    intercept = average - slope * sum_x / length + slope

    period_1 = length - 1
    regres   = intercept + slope * period_1 * 0.5

    i_vals   = np.arange(length, dtype=np.float64)
    reg_line = intercept + i_vals * slope

    dxt = log_src - average
    dyt = reg_line - regres
    residuals = log_src - reg_line

    sum_dxx = (dxt * dxt).sum()
    sum_dyy = (dyt * dyt).sum()
    sum_dyx = (dxt * dyt).sum()
    sum_dev = (residuals * residuals).sum()

    std_dev = np.sqrt(sum_dev / period_1) if period_1 > 0 else 0.0

    d = np.sqrt(sum_dxx * sum_dyy)
    pearson_r = (sum_dyx / d) if d > 1e-15 else 0.0

    return std_dev, pearson_r, slope, intercept


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Linear Regression (for PVT — no log transform)
# ═══════════════════════════════════════════════════════════════════════════════

def calc_linear_regression(
    values: np.ndarray,
    length: int,
) -> tuple[float, float, float, float]:
    """
    Linear regression on raw values (not log-transformed).
    Used for PVT which can be negative.

    Same structure as calc_log_regression but operates on raw values.

    Returns
    -------
    (std_dev, pearson_r, slope, intercept)
    """
    src = values[-length:].astype(np.float64)[::-1]  # newest-first

    x = np.arange(1, length + 1, dtype=np.float64)

    sum_x  = x.sum()
    sum_xx = (x * x).sum()
    sum_yx = (x * src).sum()
    sum_y  = src.sum()

    denom = length * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-15:
        return 0.0, 0.0, 0.0, 0.0

    slope = (length * sum_yx - sum_x * sum_y) / denom

    average   = sum_y / length
    intercept = average - slope * sum_x / length + slope

    period_1 = length - 1
    regres   = intercept + slope * period_1 * 0.5

    i_vals   = np.arange(length, dtype=np.float64)
    reg_line = intercept + i_vals * slope

    dxt = src - average
    dyt = reg_line - regres
    residuals = src - reg_line

    sum_dxx = (dxt * dxt).sum()
    sum_dyy = (dyt * dyt).sum()
    sum_dyx = (dxt * dyt).sum()
    sum_dev = (residuals * residuals).sum()

    std_dev = np.sqrt(sum_dev / period_1) if period_1 > 0 else 0.0

    d = np.sqrt(sum_dxx * sum_dyy)
    pearson_r = (sum_dyx / d) if d > 1e-15 else 0.0

    return std_dev, pearson_r, slope, intercept


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Price Volume Trend (PVT)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pvt(df: pd.DataFrame) -> np.ndarray:
    """
    Compute cumulative Price Volume Trend.

    PVT[i] = PVT[i-1] + ((Close[i] - Close[i-1]) / Close[i-1]) * Volume[i]
    PVT[0] = 0

    Parameters
    ----------
    df : OHLCV DataFrame (oldest-first).

    Returns
    -------
    1-D float array, same length as df.
    """
    close  = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    pct_change = np.zeros(len(close))
    pct_change[1:] = (close[1:] - close[:-1]) / close[:-1]

    pvt_increments = pct_change * volume
    pvt = np.cumsum(pvt_increments)

    return pvt


def compute_pvt_regression(
    df: pd.DataFrame,
    period: int,
) -> PVTResult:
    """
    Compute PVT then run linear regression on the last ``period`` bars.

    Returns PVTResult with pearson_r, slope, and direction.
    """
    pvt = compute_pvt(df)

    if len(pvt) < period:
        return PVTResult(pearson_r=0.0, slope=0.0, direction=0)

    _, pearson_r, slope, _ = calc_linear_regression(pvt, period)

    # PVT slope sign convention (same as price):
    #   slope < 0 → PVT rising (newest > oldest) → bullish
    #   slope > 0 → PVT falling → bearish
    if abs(slope) < 1e-15:
        direction = 0
    elif slope < 0:
        direction = 1    # Rising PVT → bullish
    else:
        direction = -1   # Falling PVT → bearish

    return PVTResult(pearson_r=pearson_r, slope=slope, direction=direction)


def check_pvt_alignment(
    price_direction: int,
    price_abs_r: float,
    pvt: PVTResult,
) -> tuple[bool, str]:
    """
    PVT alignment gate + volume-price divergence check.

    Returns
    -------
    (passes, reason)
    passes : True if PVT confirms the price signal.
    reason : Human-readable rejection reason (empty if passes).
    """
    pvt_abs_r = abs(pvt.pearson_r)

    # Gate 1: PVT direction must match price direction
    if pvt.direction != price_direction:
        # Divergence check: strong price but opposing PVT
        if price_abs_r >= PVT_DIVERGENCE_PRICE_R:
            return False, (
                f"VOLUME-PRICE DIVERGENCE: price |R|={price_abs_r:.4f} "
                f"but PVT direction opposing (pvt_R={pvt.pearson_r:.4f})"
            )
        return False, (
            f"PVT direction mismatch: price={'LONG' if price_direction == 1 else 'SHORT'} "
            f"but PVT={'RISING' if pvt.direction == 1 else 'FALLING' if pvt.direction == -1 else 'FLAT'}"
        )

    # Gate 2: Strong price with weak PVT = suppressed
    if price_abs_r >= PVT_DIVERGENCE_PRICE_R and pvt_abs_r < PVT_DIVERGENCE_WEAK_R:
        return False, (
            f"WEAK VOLUME: price |R|={price_abs_r:.4f} but PVT |R|={pvt_abs_r:.4f} < {PVT_DIVERGENCE_WEAK_R}"
        )

    # Gate 3: PVT minimum strength
    if pvt_abs_r < MIN_PVT_PEARSON_R:
        return False, (
            f"PVT too weak: |pvt_R|={pvt_abs_r:.4f} < {MIN_PVT_PEARSON_R}"
        )

    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Multi-TF Period Scanner
# ═══════════════════════════════════════════════════════════════════════════════

def trim_to_7d(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Trim a DataFrame to the standardized 7-day rolling window.
    Returns the last N bars where N = BARS_7D[tf].
    """
    n = BARS_7D.get(tf)
    if n is None or df is None or df.empty:
        return df
    if len(df) <= n:
        return df
    return df.iloc[-n:]


def scan_all_periods(
    close: np.ndarray,
    timeframe: str,
    periods: range = PERIOD_RANGE,
) -> list[RegressionResult]:
    """
    Compute log-regression for every period in *periods* on one close array.
    """
    results: list[RegressionResult] = []
    n = len(close)

    for p in periods:
        if n < p:
            break
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
    Data is trimmed to 7-day windows before scanning.
    """
    all_results: list[RegressionResult] = []

    for tf in SCAN_TIMEFRAMES:
        df = data.get(tf)
        if df is None or df.empty:
            continue
        df_7d = trim_to_7d(df, tf)
        close = df_7d["close"].values.astype(np.float64)
        if len(close) < PERIOD_MIN:
            continue
        results = scan_all_periods(close, tf)
        all_results.extend(results)

    if not all_results:
        return None, []

    best = max(all_results, key=lambda r: abs(r.pearson_r))
    return best, all_results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 4H Trend Filter — MSS + EMA Alignment
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
    """
    if len(df) < lookback + 1:
        return 0

    window = df.iloc[-(lookback + 1):-1]
    current = df.iloc[-1]

    swing_high = window["high"].max()
    swing_low  = window["low"].min()

    if current["close"] > swing_high:
        return 1
    if current["close"] < swing_low:
        return -1
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
# 6. Combined Gate — Suppress Conflicting Signals
# ═══════════════════════════════════════════════════════════════════════════════

def check_combined_gate(
    direction: int,
    all_results: list[RegressionResult],
    threshold: float = COMBINED_GATE_THRESHOLD,
) -> bool:
    """
    Check whether any TF/period shows a strong opposing signal.
    Returns True if the gate PASSES.
    """
    for r in all_results:
        r_direction = -1 if r.slope > 0 else (1 if r.slope < 0 else 0)

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
# 7. Leverage & Position Sizing
# ═══════════════════════════════════════════════════════════════════════════════

def get_leverage(confidence: float) -> int:
    """Map |pearson_r| confidence to leverage tier."""
    if confidence >= 0.95:
        return 5
    if confidence >= 0.90:
        return 3
    if confidence >= 0.85:
        return 2
    if confidence >= 0.80:
        return 1
    return 0


def compute_position_size(
    capital: float,
    confidence: float,
    asset: str,
) -> tuple[float, int]:
    """pos_usd = capital * 0.10 * leverage * vol_scalar"""
    lev = get_leverage(confidence)
    vol_scalar = 0.75 if asset in HIGH_VOL_ASSETS else 1.0
    pos_usd = capital * 0.10 * lev * vol_scalar
    return pos_usd, lev


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Adaptive Trailing Stop
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
    std_dev is in log space, converted: std_dev_price = midline * std_dev
    """
    std_dev_price = midline * std_dev

    if direction == 1:
        new_trail = midline - buffer_mult * std_dev_price
        return max(current_trail, new_trail)
    else:
        new_trail = midline + buffer_mult * std_dev_price
        return min(current_trail, new_trail)


def check_trail_hit(
    direction: int,
    trail_sl: float,
    bar: pd.Series,
) -> bool:
    """Check if the trailing stop was hit during this bar."""
    if direction == 1:
        return float(bar["low"]) <= trail_sl
    else:
        return float(bar["high"]) >= trail_sl


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Hard Stop Loss Calculation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hard_sl(
    entry_price: float,
    atr: float,
    direction: int,
    mult: float = HARD_SL_ATR_MULT,
) -> float:
    """Server-side STOP_MARKET placement (touch-based)."""
    if direction == 1:
        return entry_price - mult * atr
    return entry_price + mult * atr


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Full Asset Scanner
# ═══════════════════════════════════════════════════════════════════════════════

def scan_asset(
    asset: str,
    scan_data: dict[str, pd.DataFrame],
    df_4h: pd.DataFrame,
) -> Optional[ScanSignal]:
    """
    Run the full Neo-Flow pipeline for a single asset.

    Pipeline: Price Regression → PVT Alignment → 4H Filter → Combined Gate.
    """
    # ── Step 1: Find best price regression across all TFs and periods ────
    best, all_results = find_best_regression(scan_data)

    if best is None:
        logger.debug("[%s] No valid regression found", asset)
        return None

    abs_r = abs(best.pearson_r)

    # ── Step 2: Minimum price trend strength ─────────────────────────────
    if abs_r < MIN_PEARSON_R:
        logger.debug(
            "[%s] Best |R|=%.4f < %.2f (TF=%s, P=%d) — skip",
            asset, abs_r, MIN_PEARSON_R, best.timeframe, best.period,
        )
        return None

    # ── Step 3: Direction from price slope ───────────────────────────────
    direction = 1 if best.slope < 0 else -1

    # ── Step 4: PVT alignment gate ───────────────────────────────────────
    best_df = scan_data.get(best.timeframe)
    if best_df is None:
        return None

    best_df_7d = trim_to_7d(best_df, best.timeframe)
    pvt = compute_pvt_regression(best_df_7d, best.period)

    pvt_passes, pvt_reason = check_pvt_alignment(direction, abs_r, pvt)
    if not pvt_passes:
        logger.debug("[%s] PVT BLOCKED: %s", asset, pvt_reason)
        return None

    # ── Step 5: 4H trend filter ──────────────────────────────────────────
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

    # ── Step 6: Combined gate ────────────────────────────────────────────
    if not check_combined_gate(direction, all_results):
        logger.debug("[%s] Combined gate blocked", asset)
        return None

    # ── Step 7: Build signal ─────────────────────────────────────────────
    entry_price = float(best_df_7d.iloc[-1]["close"])
    atr = float(compute_atr(best_df_7d, 14).iloc[-1])

    if np.isnan(atr) or atr <= 0:
        logger.debug("[%s] ATR invalid", asset)
        return None

    sl_price = compute_hard_sl(entry_price, atr, direction)

    signal = ScanSignal(
        asset=asset,
        direction=direction,
        confidence=abs_r,
        pvt_r=abs(pvt.pearson_r),
        best_tf=best.timeframe,
        best_period=best.period,
        entry_price=entry_price,
        sl_price=sl_price,
        midline=best.midline,
        std_dev=best.std_dev,
        atr=atr,
        regression=best,
        pvt=pvt,
    )

    dir_str = "LONG" if direction == 1 else "SHORT"
    logger.info(
        "[%s] SIGNAL  %s  |R|=%.4f  pvt|R|=%.4f  TF=%s  P=%d  "
        "entry=%.6f  SL=%.6f  midline=%.6f",
        asset, dir_str, abs_r, abs(pvt.pearson_r), best.timeframe,
        best.period, entry_price, sl_price, best.midline,
    )

    return signal


def scan_universe(
    market_data: dict[str, dict[str, pd.DataFrame]],
    open_positions: dict[str, ActiveTrade],
) -> list[ScanSignal]:
    """
    Scan all 15 Tier-2 assets and return qualifying signals.
    Data is trimmed to 7-day windows internally.
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

    signals.sort(key=lambda s: s.confidence, reverse=True)

    if signals:
        logger.info(
            "Scan complete: %d signal(s) — best: %s %s |R|=%.4f pvt|R|=%.4f TF=%s P=%d",
            len(signals), signals[0].asset,
            "LONG" if signals[0].direction == 1 else "SHORT",
            signals[0].confidence, signals[0].pvt_r,
            signals[0].best_tf, signals[0].best_period,
        )
    else:
        logger.info("Scan complete: 0 signals across %d assets", len(TIER2_UNIVERSE))

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Trade Management — Update Trailing Stop
# ═══════════════════════════════════════════════════════════════════════════════

def update_active_trade(
    trade: ActiveTrade,
    scan_data: dict[str, pd.DataFrame],
) -> ActiveTrade:
    """
    Recalculate regression on the active (TF, period) and update the
    adaptive trailing stop.  Data is trimmed to 7-day window.
    """
    df = scan_data.get(trade.best_tf)
    if df is None:
        return trade

    df_7d = trim_to_7d(df, trade.best_tf)
    if len(df_7d) < trade.best_period:
        return trade

    close = df_7d["close"].values.astype(np.float64)
    std_dev, pearson_r, slope, intercept = calc_log_regression(close, trade.best_period)

    midline = np.exp(intercept)

    trade.midline  = midline
    trade.std_dev  = std_dev
    trade.peak_r   = max(trade.peak_r, abs(pearson_r))
    trade.bars_held += 1

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
    # 1. Hard SL (touch-based)
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
# 12. Diagnostic / Debug Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def print_scan_report(
    market_data: dict[str, dict[str, pd.DataFrame]],
    top_n: int = 3,
) -> None:
    """
    Print a diagnostic table with price regression + PVT for each asset.
    """
    print()
    print("=" * 130)
    print(f"{'ASSET':<8} {'TF':<5} {'PERIOD':>6} {'|R|':>8} {'SLOPE':>12} "
          f"{'MIDLINE':>12} {'PVT_R':>8} {'PVT_DIR':>8} {'DIR':>6} {'4H':>8} {'STATUS'}")
    print("=" * 130)

    for asset in TIER2_UNIVERSE:
        asset_data = market_data.get(asset)
        if asset_data is None:
            print(f"{asset:<8} NO DATA")
            continue

        scan_tfs = {tf: asset_data[tf] for tf in SCAN_TIMEFRAMES if tf in asset_data}
        best, all_results = find_best_regression(scan_tfs)

        if best is None:
            print(f"{asset:<8} NO RESULTS")
            continue

        # 4H bias
        df_4h = asset_data.get("4h")
        htf_bias = get_htf_bias(df_4h) if df_4h is not None else 0
        bias_str = {1: "BULL", -1: "BEAR", 0: "NEUTRAL"}[htf_bias]

        # Sort by |R| descending
        sorted_results = sorted(all_results, key=lambda r: abs(r.pearson_r), reverse=True)

        for i, r in enumerate(sorted_results[:top_n]):
            abs_r = abs(r.pearson_r)
            direction = 1 if r.slope < 0 else -1
            dir_str = "LONG" if direction == 1 else "SHORT"

            # PVT for this TF/period
            tf_df = scan_tfs.get(r.timeframe)
            if tf_df is not None:
                tf_7d = trim_to_7d(tf_df, r.timeframe)
                pvt = compute_pvt_regression(tf_7d, r.period)
                pvt_r_str = f"{abs(pvt.pearson_r):>8.4f}"
                pvt_dir = {1: "RISING", -1: "FALLING", 0: "FLAT"}[pvt.direction]

                # Check gates
                pvt_ok, pvt_reason = check_pvt_alignment(direction, abs_r, pvt)
                if abs_r < MIN_PEARSON_R:
                    status = "WEAK_R"
                elif not pvt_ok:
                    status = pvt_reason.split(":")[0] if ":" in pvt_reason else "PVT_FAIL"
                elif htf_bias == 0:
                    status = "4H_NEUTRAL"
                elif htf_bias != direction:
                    status = "4H_CONFLICT"
                else:
                    status = "SIGNAL ***"
            else:
                pvt_r_str = "N/A"
                pvt_dir = "N/A"
                status = "NO_DATA"

            row_asset = asset if i == 0 else ""
            row_bias = bias_str if i == 0 else ""

            print(f"{row_asset:<8} {r.timeframe:<5} {r.period:>6} {abs_r:>8.4f} "
                  f"{r.slope:>12.8f} {r.midline:>12.6f} {pvt_r_str} {pvt_dir:>8} "
                  f"{dir_str:>6} {row_bias:>8} {status}")

    print("=" * 130)
    print()

"""
neo_flow/precompute_features.py — Pre-compute all scan features at 1h resolution.

Eliminates redundant regression math during Optuna trials.
The backtest engine performs fast lookups instead of computing 543 regressions per scan step.

Usage:
    from neo_flow.precompute_features import precompute_all_features
    features = precompute_all_features(all_data)
    # features: dict[str, np.ndarray]  — {asset: structured array}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Feature dtype (one row per 1h timestamp per asset) ─────────────────────
FEATURE_DTYPE = np.dtype([
    ("timestamp_ns", "i8"),
    ("best_r", "f4"),
    ("best_slope", "f4"),
    ("best_std", "f4"),
    ("best_midline", "f4"),
    ("best_period", "i2"),
    ("best_tf_idx", "i1"),       # 0=5m, 1=30m, 2=1h
    ("best_direction", "i1"),    # +1 LONG, -1 SHORT
    ("pvt_r", "f4"),             # abs(pvt pearson_r) for the best TF/period
    ("pvt_direction", "i1"),     # +1 rising, -1 falling, 0 flat
    ("htf_bias", "i1"),          # +1 bull, -1 bear, 0 neutral
    ("max_opposing_r", "f4"),    # max |R| among results with opposing direction
    ("atr_best", "f4"),          # ATR(14) on the best TF
    ("close_best", "f4"),        # latest close on the best TF
])

TF_LIST = ["5m", "30m", "1h"]
TF_TO_IDX = {"5m": 0, "30m": 1, "1h": 2}
BARS_7D = {"5m": 2016, "30m": 336, "1h": 168, "4h": 42}
PERIOD_MIN = 20
PERIOD_MAX = 200


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorized batch regression — computes ALL windows of a given period at once
# ═══════════════════════════════════════════════════════════════════════════════

def _batch_log_regression(
    log_close_rev: np.ndarray,
    period: int,
    eval_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute log-regression for specific window positions (Pine Script convention).

    Parameters
    ----------
    log_close_rev : log(close) array, REVERSED (newest-first, matching Pine).
    period : regression window length.
    eval_indices : indices into log_close_rev where each window STARTS (newest bar).
                   Window is log_close_rev[idx : idx + period].

    Returns
    -------
    (std_dev, pearson_r, slope, intercept) — each shape (len(eval_indices),)
    """
    n = period
    x = np.arange(1, n + 1, dtype=np.float64)
    sum_x = x.sum()
    sum_xx = (x * x).sum()

    # Extract windows: shape (n_eval, period)
    idx_2d = eval_indices[:, None] + np.arange(n)[None, :]
    windows = log_close_rev[idx_2d]  # (n_eval, period)

    sum_y = windows.sum(axis=1)
    sum_yx = (windows * x[None, :]).sum(axis=1)

    denom = n * sum_xx - sum_x * sum_x
    slope = (n * sum_yx - sum_x * sum_y) / denom

    average = sum_y / n
    intercept = average - slope * sum_x / n + slope

    period_1 = n - 1
    regres = intercept + slope * period_1 * 0.5

    i_vals = np.arange(n, dtype=np.float64)
    reg_line = intercept[:, None] + slope[:, None] * i_vals[None, :]

    dxt = windows - average[:, None]
    dyt = reg_line - regres[:, None]
    residuals = windows - reg_line

    sum_dxx = (dxt * dxt).sum(axis=1)
    sum_dyy = (dyt * dyt).sum(axis=1)
    sum_dyx = (dxt * dyt).sum(axis=1)
    sum_dev = (residuals * residuals).sum(axis=1)

    std_dev = np.where(period_1 > 0, np.sqrt(sum_dev / period_1), 0.0)

    d = np.sqrt(sum_dxx * sum_dyy)
    pearson_r = np.where(d > 1e-15, sum_dyx / d, 0.0)

    return std_dev, pearson_r, slope, intercept


def _batch_linear_regression(
    values_rev: np.ndarray,
    period: int,
    eval_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear regression (no log) for PVT. Same Pine convention.

    Returns
    -------
    (pearson_r, slope) — each shape (len(eval_indices),)
    """
    n = period
    x = np.arange(1, n + 1, dtype=np.float64)
    sum_x = x.sum()
    sum_xx = (x * x).sum()

    idx_2d = eval_indices[:, None] + np.arange(n)[None, :]
    windows = values_rev[idx_2d]

    sum_y = windows.sum(axis=1)
    sum_yx = (windows * x[None, :]).sum(axis=1)

    denom = n * sum_xx - sum_x * sum_x
    safe_denom = np.where(np.abs(denom) < 1e-15, 1.0, denom)
    slope = np.where(np.abs(denom) < 1e-15, 0.0,
                     (n * sum_yx - sum_x * sum_y) / safe_denom)

    average = sum_y / n
    intercept = average - slope * sum_x / n + slope
    period_1 = n - 1
    regres = intercept + slope * period_1 * 0.5

    i_vals = np.arange(n, dtype=np.float64)
    reg_line = intercept[:, None] + slope[:, None] * i_vals[None, :]

    dxt = windows - average[:, None]
    dyt = reg_line - regres[:, None]

    sum_dxx = (dxt * dxt).sum(axis=1)
    sum_dyy = (dyt * dyt).sum(axis=1)
    sum_dyx = (dxt * dyt).sum(axis=1)

    d = np.sqrt(sum_dxx * sum_dyy)
    pearson_r = np.where(d > 1e-15, sum_dyx / d, 0.0)

    return pearson_r, slope


# ═══════════════════════════════════════════════════════════════════════════════
# 4H HTF bias — vectorized over all 1h timestamps
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_htf_bias_series(
    ad_4h,
    hourly_ns: np.ndarray,
    ema_fast: int = 21,
    ema_slow: int = 55,
    mss_lookback: int = 30,
) -> np.ndarray:
    """Compute HTF bias for every 1h timestamp. Returns int8 array."""
    close_4h = ad_4h.close
    high_4h = ad_4h.high
    low_4h = ad_4h.low
    ts_4h = ad_4h.timestamps

    # Pre-compute full EMA series on 4h
    df_4h = pd.Series(close_4h)
    ema_f = df_4h.ewm(span=ema_fast, adjust=False).mean().values
    ema_s = df_4h.ewm(span=ema_slow, adjust=False).mean().values

    # For each 1h timestamp, find the corresponding 4h bar index
    idx_4h = np.searchsorted(ts_4h, hourly_ns, side="right") - 1
    idx_4h = np.clip(idx_4h, 0, len(ts_4h) - 1)

    bias = np.zeros(len(hourly_ns), dtype=np.int8)
    min_bars = max(ema_slow, mss_lookback) + 1

    for i, idx in enumerate(idx_4h):
        if idx < min_bars:
            continue

        # MSS detection
        window_high = high_4h[idx - mss_lookback:idx]
        window_low = low_4h[idx - mss_lookback:idx]
        current_close = close_4h[idx]

        swing_high = window_high.max()
        swing_low = window_low.min()

        if current_close > swing_high:
            mss = 1
        elif current_close < swing_low:
            mss = -1
        else:
            mss = 0

        # EMA alignment
        ema_bull = ema_f[idx] > ema_s[idx]
        ema_bear = ema_f[idx] < ema_s[idx]

        if mss == 1 and ema_bull:
            bias[i] = 1
        elif mss == -1 and ema_bear:
            bias[i] = -1

    return bias


# ═══════════════════════════════════════════════════════════════════════════════
# ATR computation — vectorized
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_atr_array(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute rolling ATR. Returns float64 array same length as input (NaN-padded)."""
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    atr = np.full_like(tr, np.nan)
    # Simple moving average for ATR
    cumsum = np.cumsum(tr)
    atr[period - 1:] = (cumsum[period - 1:] - np.concatenate([[0.0], cumsum[:-period]])) / period
    return atr


# ═══════════════════════════════════════════════════════════════════════════════
# Main precomputation for one asset
# ═══════════════════════════════════════════════════════════════════════════════

def _precompute_asset(
    asset: str,
    asset_data: dict,
    hourly_ns: np.ndarray,
) -> np.ndarray:
    """
    Pre-compute all scan features for one asset at every 1h timestamp.

    For each timestamp:
    1. Find bar indices on each TF corresponding to that 1h time
    2. Compute regressions for all 181 periods on each TF
    3. Find the best |R| across all TFs/periods
    4. Compute PVT regression for the best TF/period
    5. Compute max opposing |R|
    6. Store HTF bias and ATR

    Returns structured numpy array of shape (n_hours,) with FEATURE_DTYPE.
    """
    n_hours = len(hourly_ns)
    features = np.zeros(n_hours, dtype=FEATURE_DTYPE)
    features["timestamp_ns"] = hourly_ns

    # Pre-compute reversed log(close) and reversed PVT for each scan TF
    tf_data = {}
    for tf in TF_LIST:
        ad = asset_data.get(tf)
        if ad is None:
            continue
        # Reversed arrays (newest-first, Pine convention)
        # For eval at bar index `idx`, the reversed window starts at len-1-idx
        # But we need to build windows ending at specific bars.
        # Actually: for bar idx in original (oldest-first) data,
        # the window of `period` bars ending at idx is close[idx-period+1 : idx+1].
        # Reversed: close[idx-period+1 : idx+1][::-1] = newest first.
        # We'll store a full reversed array and compute indices.
        tf_data[tf] = ad

    # Map each 1h timestamp to bar indices on each TF
    tf_bar_indices = {}
    for tf in TF_LIST:
        ad = tf_data.get(tf)
        if ad is None:
            tf_bar_indices[tf] = None
            continue
        # For each hourly timestamp, find the last bar at or before it
        idx = np.searchsorted(ad.timestamps, hourly_ns, side="right") - 1
        idx = np.clip(idx, 0, len(ad.timestamps) - 1)
        tf_bar_indices[tf] = idx

    # Pre-compute ATR for each TF (full series)
    tf_atr = {}
    for tf in TF_LIST:
        ad = tf_data.get(tf)
        if ad is None:
            continue
        tf_atr[tf] = _compute_atr_array(ad.high, ad.low, ad.close, 14)

    # Pre-compute HTF bias
    ad_4h = asset_data.get("4h")
    if ad_4h is not None:
        htf_bias_arr = _compute_htf_bias_series(ad_4h, hourly_ns)
    else:
        htf_bias_arr = np.zeros(n_hours, dtype=np.int8)
    features["htf_bias"] = htf_bias_arr

    # ── For each TF, batch-compute regressions for all periods ──────────
    # Store: best_abs_r[n_hours], corresponding slope/std/midline/period/tf
    # Also track max opposing |R| per direction

    # Initialize tracking arrays
    best_abs_r = np.zeros(n_hours, dtype=np.float64)
    best_slope = np.zeros(n_hours, dtype=np.float64)
    best_std = np.zeros(n_hours, dtype=np.float64)
    best_midline = np.zeros(n_hours, dtype=np.float64)
    best_period = np.zeros(n_hours, dtype=np.int16)
    best_tf_idx = np.zeros(n_hours, dtype=np.int8)
    best_direction = np.zeros(n_hours, dtype=np.int8)
    # For combined gate: track max |R| for LONG results and SHORT results separately
    max_r_long = np.zeros(n_hours, dtype=np.float64)   # max |R| among slope<0 (LONG)
    max_r_short = np.zeros(n_hours, dtype=np.float64)  # max |R| among slope>0 (SHORT)

    for tf_idx, tf in enumerate(TF_LIST):
        ad = tf_data.get(tf)
        if ad is None:
            continue
        bar_indices = tf_bar_indices[tf]
        if bar_indices is None:
            continue

        close = ad.close
        n_bars_tf = len(close)
        n7d = BARS_7D[tf]

        # Reversed log(close) for the entire series
        log_close = np.log(close.astype(np.float64))

        for period in range(PERIOD_MIN, PERIOD_MAX + 1):
            # Which hourly timestamps have enough data for this period?
            # Bar index must be >= period - 1 (to have `period` bars)
            # Also must be within 7-day window: bar_idx >= n_bars_tf - n7d (approx)
            valid_mask = bar_indices >= (period - 1)
            if not valid_mask.any():
                continue

            valid_hours = np.where(valid_mask)[0]
            valid_bar_idx = bar_indices[valid_hours]

            # Build reversed windows ending at each valid_bar_idx
            # For bar idx, reversed window = close[idx-period+1 : idx+1][::-1]
            # In the reversed-full array (close[::-1]), the start index is (n-1-idx).
            # But it's simpler to just build index arrays directly.

            # Window indices: for bar `b`, we want bars [b-period+1, ..., b] reversed
            # = [b, b-1, ..., b-period+1]
            offsets = np.arange(0, -period, -1, dtype=np.int64)  # [0, -1, -2, ..., -(period-1)]
            win_indices = valid_bar_idx[:, None] + offsets[None, :]  # (n_valid, period)

            # Extract windows from log_close
            log_windows = log_close[win_indices]  # (n_valid, period)

            # ── Vectorized Pine-style regression on all windows ──
            n = period
            x = np.arange(1, n + 1, dtype=np.float64)
            sum_x = x.sum()
            sum_xx = (x * x).sum()

            sum_y = log_windows.sum(axis=1)
            sum_yx = (log_windows * x[None, :]).sum(axis=1)

            denom_val = n * sum_xx - sum_x * sum_x
            sl = (n * sum_yx - sum_x * sum_y) / denom_val

            average = sum_y / n
            intercept_val = average - sl * sum_x / n + sl

            period_1 = n - 1
            regres = intercept_val + sl * period_1 * 0.5

            i_vals = np.arange(n, dtype=np.float64)
            reg_line = intercept_val[:, None] + sl[:, None] * i_vals[None, :]

            dxt = log_windows - average[:, None]
            dyt = reg_line - regres[:, None]
            residuals = log_windows - reg_line

            sum_dxx = (dxt * dxt).sum(axis=1)
            sum_dyy = (dyt * dyt).sum(axis=1)
            sum_dyx = (dxt * dyt).sum(axis=1)
            sum_dev = (residuals * residuals).sum(axis=1)

            sd = np.where(period_1 > 0, np.sqrt(sum_dev / period_1), 0.0)
            d = np.sqrt(sum_dxx * sum_dyy)
            pr = np.where(d > 1e-15, sum_dyx / d, 0.0)
            ml = np.exp(intercept_val)

            abs_pr = np.abs(pr)

            # Update best tracking
            better = abs_pr > best_abs_r[valid_hours]
            update_idx = valid_hours[better]
            if len(update_idx) > 0:
                better_local = np.where(better)[0]
                best_abs_r[update_idx] = abs_pr[better_local]
                best_slope[update_idx] = sl[better_local]
                best_std[update_idx] = sd[better_local]
                best_midline[update_idx] = ml[better_local]
                best_period[update_idx] = period
                best_tf_idx[update_idx] = tf_idx

            # Update max opposing R tracking
            # LONG results: slope < 0
            long_mask = sl < 0
            short_mask = sl > 0

            # Update max_r_long for valid_hours where this result is LONG
            if long_mask.any():
                long_hours = valid_hours[long_mask]
                long_r = abs_pr[long_mask]
                np.maximum.at(max_r_long, long_hours, long_r)

            if short_mask.any():
                short_hours = valid_hours[short_mask]
                short_r = abs_pr[short_mask]
                np.maximum.at(max_r_short, short_hours, short_r)

    # Direction from slope
    best_direction = np.where(best_slope < 0, 1, -1).astype(np.int8)

    # Max opposing R: if best is LONG (+1), opposing is SHORT results
    max_opposing = np.where(best_direction == 1, max_r_short, max_r_long).astype(np.float32)

    # ── PVT regression for the best TF/period at each timestamp ─────────
    pvt_r_arr = np.zeros(n_hours, dtype=np.float32)
    pvt_dir_arr = np.zeros(n_hours, dtype=np.int8)

    # Group by (tf, period) for batch processing
    for tf_idx, tf in enumerate(TF_LIST):
        ad = tf_data.get(tf)
        if ad is None:
            continue
        bar_indices = tf_bar_indices[tf]
        if bar_indices is None:
            continue

        pvt_full = ad.pvt  # pre-computed cumulative PVT

        for period in range(PERIOD_MIN, PERIOD_MAX + 1):
            # Find hours where this (tf, period) is the best
            mask = (best_tf_idx == tf_idx) & (best_period == period)
            if not mask.any():
                continue

            hours_idx = np.where(mask)[0]
            valid_bar_idx = bar_indices[hours_idx]

            # Filter: need enough bars
            enough = valid_bar_idx >= (period - 1)
            if not enough.any():
                continue
            hours_idx = hours_idx[enough]
            valid_bar_idx = valid_bar_idx[enough]

            # Build reversed PVT windows
            offsets = np.arange(0, -period, -1, dtype=np.int64)
            win_indices = valid_bar_idx[:, None] + offsets[None, :]
            pvt_windows = pvt_full[win_indices].astype(np.float64)

            # Linear regression on PVT (no log)
            n = period
            x = np.arange(1, n + 1, dtype=np.float64)
            sum_x = x.sum()
            sum_xx = (x * x).sum()

            sum_y = pvt_windows.sum(axis=1)
            sum_yx = (pvt_windows * x[None, :]).sum(axis=1)

            denom_val = n * sum_xx - sum_x * sum_x
            safe_denom = np.where(np.abs(denom_val) < 1e-15, 1.0, denom_val)
            sl = np.where(np.abs(denom_val) < 1e-15, 0.0,
                          (n * sum_yx - sum_x * sum_y) / safe_denom)

            average = sum_y / n
            intercept_val = average - sl * sum_x / n + sl
            period_1 = n - 1
            regres = intercept_val + sl * period_1 * 0.5

            i_vals = np.arange(n, dtype=np.float64)
            reg_line = intercept_val[:, None] + sl[:, None] * i_vals[None, :]

            dxt = pvt_windows - average[:, None]
            dyt = reg_line - regres[:, None]

            sum_dxx = (dxt * dxt).sum(axis=1)
            sum_dyy = (dyt * dyt).sum(axis=1)
            sum_dyx = (dxt * dyt).sum(axis=1)

            d = np.sqrt(sum_dxx * sum_dyy)
            pr = np.where(d > 1e-15, sum_dyx / d, 0.0)

            pvt_r_arr[hours_idx] = np.abs(pr).astype(np.float32)
            pvt_dir = np.where(np.abs(sl) < 1e-15, 0,
                               np.where(sl < 0, 1, -1)).astype(np.int8)
            pvt_dir_arr[hours_idx] = pvt_dir

    # ── ATR and close for the best TF ───────────────────────────────────
    atr_best = np.zeros(n_hours, dtype=np.float32)
    close_best = np.zeros(n_hours, dtype=np.float32)

    for tf_idx, tf in enumerate(TF_LIST):
        ad = tf_data.get(tf)
        if ad is None:
            continue
        bar_indices = tf_bar_indices[tf]
        if bar_indices is None:
            continue

        mask = best_tf_idx == tf_idx
        if not mask.any():
            continue

        hours_idx = np.where(mask)[0]
        bidx = bar_indices[hours_idx]

        atr_full = tf_atr.get(tf)
        if atr_full is not None:
            atr_vals = atr_full[bidx]
            atr_vals = np.where(np.isnan(atr_vals), 0.0, atr_vals)
            atr_best[hours_idx] = atr_vals.astype(np.float32)

        close_best[hours_idx] = ad.close[bidx].astype(np.float32)

    # ── Pack into structured array ──────────────────────────────────────
    features["best_r"] = best_abs_r.astype(np.float32)
    features["best_slope"] = best_slope.astype(np.float32)
    features["best_std"] = best_std.astype(np.float32)
    features["best_midline"] = best_midline.astype(np.float32)
    features["best_period"] = best_period
    features["best_tf_idx"] = best_tf_idx
    features["best_direction"] = best_direction
    features["pvt_r"] = pvt_r_arr
    features["pvt_direction"] = pvt_dir_arr
    features["max_opposing_r"] = max_opposing
    features["atr_best"] = atr_best
    features["close_best"] = close_best

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_all_features(
    all_data: dict,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Pre-compute scan features for all assets.

    Parameters
    ----------
    all_data : {asset: {tf: AssetData}}
    start_ns, end_ns : optional nanosecond bounds to limit computation

    Returns
    -------
    {asset: structured numpy array with FEATURE_DTYPE}
    """
    t0 = time.perf_counter()

    # Get 1h timestamps from any asset
    hourly_ns = None
    for asset_data in all_data.values():
        ad_1h = asset_data.get("1h")
        if ad_1h is not None:
            hourly_ns = ad_1h.timestamps.copy()
            break

    if hourly_ns is None:
        raise ValueError("No 1h data found")

    # Optionally filter to range
    if start_ns is not None:
        hourly_ns = hourly_ns[hourly_ns >= start_ns]
    if end_ns is not None:
        hourly_ns = hourly_ns[hourly_ns <= end_ns]

    logger.info("Pre-computing features for %d 1h timestamps...", len(hourly_ns))

    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1)(
            delayed(_precompute_asset)(asset, asset_data, hourly_ns)
            for asset, asset_data in all_data.items()
        )
        features = dict(zip(all_data.keys(), results))
    except ImportError:
        logger.warning("joblib not found, falling back to sequential pre-computation")
        features = {}
        for asset, asset_data in all_data.items():
            t1 = time.perf_counter()
            features[asset] = _precompute_asset(asset, asset_data, hourly_ns)
            elapsed = time.perf_counter() - t1
            logger.info("  %s: %.1fs", asset, elapsed)

    total = time.perf_counter() - t0
    mem_mb = sum(f.nbytes for f in features.values()) / 1e6
    logger.info(
        "Pre-computation complete: %.1fs, %d assets, %.1f MB",
        total, len(features), mem_mb,
    )
    return features


def save_features(features: dict[str, np.ndarray], output_dir: str) -> None:
    """Save pre-computed features to .npz files."""
    from pathlib import Path
    out = Path(output_dir) / "features"
    out.mkdir(parents=True, exist_ok=True)
    for asset, arr in features.items():
        np.save(out / f"{asset}_scan_features.npy", arr)
    logger.info("Saved features to %s", out)


def load_features(feature_dir: str) -> dict[str, np.ndarray]:
    """Load pre-computed features from .npy files."""
    from pathlib import Path
    d = Path(feature_dir) / "features"
    features = {}
    for f in sorted(d.glob("*_scan_features.npy")):
        asset = f.stem.replace("_scan_features", "")
        features[asset] = np.load(f)
    logger.info("Loaded features for %d assets from %s", len(features), d)
    return features

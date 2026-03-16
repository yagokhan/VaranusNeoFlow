"""
backtest/engine.py — Core backtest loop with trade management.

Walks through 1h bars, scans for signals, manages active positions
with adaptive trailing stops, and records all trades.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from neo_flow.adaptive_engine import (
    scan_asset,
    calc_log_regression,
    compute_trail_sl,
    check_trail_hit,
    check_exit_conditions,
    compute_hard_sl,
    compute_atr,
    get_leverage,
    TRAIL_BUFFER_STD,
    HARD_SL_ATR_MULT,
    TREND_EXHAUST_R,
    MAX_CONCURRENT,
    HIGH_VOL_ASSETS,
    SCAN_TIMEFRAMES,
    MIN_PEARSON_R,
    MIN_PVT_PEARSON_R,
    PVT_DIVERGENCE_PRICE_R,
    PVT_DIVERGENCE_WEAK_R,
    COMBINED_GATE_THRESHOLD,
)

from backtest.data_loader import (
    AssetData,
    load_all_assets,
    build_scan_dataframes,
    build_htf_dataframe,
    get_1h_timestamps,
    get_sub_bars,
    find_bar_index,
    ts_to_ns,
    BARS_7D,
    ASSETS,
    TF_HOURS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Backtest Parameters (overridable for optimization)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestParams:
    """All tunable parameters for one backtest run."""
    min_pearson_r: float = 0.80
    min_pvt_r: float = 0.70
    combined_gate: float = 0.65
    hard_sl_mult: float = 1.5
    trail_buffer: float = 1.0
    exhaust_r: float = 0.50
    pos_frac: float = 0.10
    initial_capital: float = 10_000.0
    max_concurrent: int = 4
    scan_interval_hours: int = 1  # scan every N hours


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Record
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """Complete record of one backtest trade."""
    trade_id: int
    asset: str
    direction: int
    entry_ts: pd.Timestamp
    entry_price: float
    best_tf: str
    best_period: int
    confidence: float
    pvt_r: float
    leverage: int
    position_usd: float
    hard_sl: float
    initial_trail_sl: float
    # Filled on exit
    exit_ts: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    bars_held: int = 0
    peak_r: float = 0.0
    final_trail_sl: float = 0.0
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Active Position (mutable during backtest)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ActivePosition:
    """Mutable state of an open position."""
    trade_id: int
    asset: str
    direction: int
    entry_price: float
    entry_ts: pd.Timestamp
    hard_sl: float
    trail_sl: float
    midline: float
    std_dev: float
    best_tf: str
    best_period: int
    confidence: float
    pvt_r: float
    leverage: int
    position_usd: float
    bars_held: int = 0
    peak_r: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Backtest Engine
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Event-driven backtest engine.

    Walks 1h bars, scans all assets, manages positions on their native TFs.
    """

    def __init__(
        self,
        all_data: dict[str, dict[str, AssetData]],
        params: BacktestParams | None = None,
    ):
        self.all_data = all_data
        self.params = params or BacktestParams()
        self.trades: list[TradeRecord] = []
        self.positions: dict[str, ActivePosition] = {}  # asset -> position
        self.equity_curve: list[float] = []
        self.capital = self.params.initial_capital
        self.realized_pnl = 0.0
        self._trade_counter = 0
        self._scan_count = 0
        self._prev_1h_ns: int | None = None

    def _apply_params_to_engine(self):
        """Temporarily patch adaptive_engine globals with our params."""
        import neo_flow.adaptive_engine as ae
        ae.MIN_PEARSON_R = self.params.min_pearson_r
        ae.MIN_PVT_PEARSON_R = self.params.min_pvt_r
        ae.COMBINED_GATE_THRESHOLD = self.params.combined_gate
        ae.HARD_SL_ATR_MULT = self.params.hard_sl_mult
        ae.TRAIL_BUFFER_STD = self.params.trail_buffer
        ae.TREND_EXHAUST_R = self.params.exhaust_r

    def _restore_engine_defaults(self):
        """Restore original adaptive_engine globals."""
        import neo_flow.adaptive_engine as ae
        ae.MIN_PEARSON_R = 0.80
        ae.MIN_PVT_PEARSON_R = 0.70
        ae.COMBINED_GATE_THRESHOLD = 0.65
        ae.HARD_SL_ATR_MULT = 1.5
        ae.TRAIL_BUFFER_STD = 1.0
        ae.TREND_EXHAUST_R = 0.50

    def _get_entry_price(self, asset: str, tf: str, signal_ns: int) -> float | None:
        """Get the open of the next bar on the signal's TF after signal_ns."""
        ad = self.all_data.get(asset, {}).get(tf)
        if ad is None:
            return None
        # Find first bar after signal_ns
        idx = np.searchsorted(ad.timestamps, signal_ns, side="right")
        if idx >= len(ad.timestamps):
            return None
        return float(ad.open_[idx])

    def _close_position(
        self,
        pos: ActivePosition,
        exit_price: float,
        exit_ts: pd.Timestamp,
        exit_reason: str,
    ):
        """Close a position, record the trade, update capital."""
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * pos.direction * 100.0
        pnl_pct_lev = pnl_pct * pos.leverage
        pnl_usd = pnl_pct_lev / 100.0 * pos.position_usd

        self.realized_pnl += pnl_usd

        # Update the trade record
        for tr in reversed(self.trades):
            if tr.trade_id == pos.trade_id:
                tr.exit_ts = exit_ts
                tr.exit_price = exit_price
                tr.exit_reason = exit_reason
                tr.bars_held = pos.bars_held
                tr.peak_r = pos.peak_r
                tr.final_trail_sl = pos.trail_sl
                tr.pnl_pct = pnl_pct_lev
                tr.pnl_usd = pnl_usd
                break

        del self.positions[pos.asset]

    def _update_positions(self, current_1h_ns: int):
        """
        Update all active positions with sub-bar data.
        For each position, iterate through all bars of its native TF
        between the previous 1h bar and current 1h bar.
        """
        prev_ns = self._prev_1h_ns
        if prev_ns is None:
            return

        to_close: list[tuple[ActivePosition, float, pd.Timestamp, str]] = []

        for asset, pos in list(self.positions.items()):
            ad = self.all_data.get(asset, {}).get(pos.best_tf)
            if ad is None:
                continue

            # Get sub-bar indices in (prev_ns, current_ns]
            sub_indices = get_sub_bars(ad, prev_ns, current_1h_ns)

            for idx in sub_indices:
                pos.bars_held += 1

                # Update regression + trailing stop
                close_arr = ad.close[:idx + 1]
                if len(close_arr) >= pos.best_period:
                    std_dev, pearson_r, slope, intercept = calc_log_regression(
                        close_arr, pos.best_period
                    )
                    midline = np.exp(intercept)
                    pos.midline = midline
                    pos.std_dev = std_dev
                    pos.peak_r = max(pos.peak_r, abs(pearson_r))

                    pos.trail_sl = compute_trail_sl(
                        pos.direction, midline, std_dev,
                        pos.trail_sl, self.params.trail_buffer,
                    )

                    current_r = pearson_r
                else:
                    current_r = 0.0

                # Check exit conditions
                bar_high = float(ad.high[idx])
                bar_low = float(ad.low[idx])
                bar_close = float(ad.close[idx])
                bar_ts = pd.Timestamp(ad.timestamps[idx], tz="UTC")

                # Hard SL (touch-based)
                if pos.direction == 1 and bar_low <= pos.hard_sl:
                    to_close.append((pos, pos.hard_sl, bar_ts, "HARD_SL_HIT"))
                    break
                if pos.direction == -1 and bar_high >= pos.hard_sl:
                    to_close.append((pos, pos.hard_sl, bar_ts, "HARD_SL_HIT"))
                    break

                # Adaptive trail (touch-based)
                if pos.direction == 1 and bar_low <= pos.trail_sl:
                    to_close.append((pos, pos.trail_sl, bar_ts, "ADAPTIVE_TRAIL_HIT"))
                    break
                if pos.direction == -1 and bar_high >= pos.trail_sl:
                    to_close.append((pos, pos.trail_sl, bar_ts, "ADAPTIVE_TRAIL_HIT"))
                    break

                # Trend exhaustion
                if abs(current_r) < self.params.exhaust_r:
                    to_close.append((pos, bar_close, bar_ts, "TREND_EXHAUSTION"))
                    break

                # Time barrier
                if pos.bars_held >= 200:
                    to_close.append((pos, bar_close, bar_ts, "TIME_BARRIER"))
                    break

        for pos, exit_price, exit_ts, reason in to_close:
            if pos.asset in self.positions:
                self._close_position(pos, exit_price, exit_ts, reason)

    def _scan_and_enter(self, current_1h_ns: int):
        """Scan all assets and open positions for qualifying signals."""
        if len(self.positions) >= self.params.max_concurrent:
            return

        current_ts = pd.Timestamp(current_1h_ns, tz="UTC")

        for asset in ASSETS:
            if asset in self.positions:
                continue
            if len(self.positions) >= self.params.max_concurrent:
                break

            asset_data = self.all_data.get(asset)
            if asset_data is None:
                continue

            # Build scan DataFrames (7-day windows)
            scan_dfs = build_scan_dataframes(asset_data, current_1h_ns)
            df_4h = build_htf_dataframe(asset_data, current_1h_ns)

            if not scan_dfs or df_4h is None:
                continue

            # Run the full pipeline
            signal = scan_asset(asset, scan_dfs, df_4h)
            if signal is None:
                continue

            # Entry at next bar open on signal's TF
            entry_price = self._get_entry_price(asset, signal.best_tf, current_1h_ns)
            if entry_price is None:
                continue

            # Position sizing
            lev = get_leverage(signal.confidence)
            vol_scalar = 0.75 if asset in HIGH_VOL_ASSETS else 1.0
            pos_usd = self.capital * self.params.pos_frac * lev * vol_scalar

            if pos_usd <= 0 or lev == 0:
                continue

            # Hard SL based on ATR at entry
            ad = asset_data.get(signal.best_tf)
            idx = find_bar_index(ad.timestamps, current_1h_ns)
            # Build small DF for ATR computation
            start_idx = max(0, idx - 20)
            atr_df = pd.DataFrame({
                "high": ad.high[start_idx:idx + 1],
                "low": ad.low[start_idx:idx + 1],
                "close": ad.close[start_idx:idx + 1],
            })
            atr_val = float(compute_atr(atr_df, 14).iloc[-1])
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            hard_sl = compute_hard_sl(entry_price, atr_val, signal.direction, self.params.hard_sl_mult)

            # Initial trailing stop
            std_dev_price = signal.midline * signal.std_dev
            if signal.direction == 1:
                initial_trail = signal.midline - self.params.trail_buffer * std_dev_price
            else:
                initial_trail = signal.midline + self.params.trail_buffer * std_dev_price

            # Record trade
            self._trade_counter += 1
            entry_ts = current_ts

            tr = TradeRecord(
                trade_id=self._trade_counter,
                asset=asset,
                direction=signal.direction,
                entry_ts=entry_ts,
                entry_price=entry_price,
                best_tf=signal.best_tf,
                best_period=signal.best_period,
                confidence=signal.confidence,
                pvt_r=signal.pvt_r,
                leverage=lev,
                position_usd=pos_usd,
                hard_sl=hard_sl,
                initial_trail_sl=initial_trail,
                peak_r=signal.confidence,
            )
            self.trades.append(tr)

            pos = ActivePosition(
                trade_id=self._trade_counter,
                asset=asset,
                direction=signal.direction,
                entry_price=entry_price,
                entry_ts=entry_ts,
                hard_sl=hard_sl,
                trail_sl=initial_trail,
                midline=signal.midline,
                std_dev=signal.std_dev,
                best_tf=signal.best_tf,
                best_period=signal.best_period,
                confidence=signal.confidence,
                pvt_r=signal.pvt_r,
                leverage=lev,
                position_usd=pos_usd,
            )
            self.positions[asset] = pos

        self._scan_count += 1

    def _compute_equity(self) -> float:
        """Current equity = starting capital + realized PnL + unrealized PnL."""
        unrealized = 0.0
        for pos in self.positions.values():
            ad = self.all_data.get(pos.asset, {}).get(pos.best_tf)
            if ad is None:
                continue
            # Use the latest close as mark-to-market
            current_close = float(ad.close[find_bar_index(ad.timestamps, ts_to_ns(pos.entry_ts))])
            # Actually use the most recent bar we've processed
            # This will be updated each 1h bar
            pnl_pct = (pos.midline - pos.entry_price) / pos.entry_price * pos.direction
            unrealized += pnl_pct * pos.leverage * pos.position_usd

        return self.capital + self.realized_pnl + unrealized

    def run(
        self,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> list[TradeRecord]:
        """
        Run the backtest over [start_ts, end_ts].

        Returns list of completed TradeRecords.
        """
        self._apply_params_to_engine()

        try:
            # Reset state
            self.trades = []
            self.positions = {}
            self.equity_curve = []
            self.realized_pnl = 0.0
            self._trade_counter = 0
            self._scan_count = 0
            self._prev_1h_ns = None

            # Get 1h timestamps in range
            bar_timestamps = get_1h_timestamps(self.all_data, start_ts, end_ts)
            if len(bar_timestamps) == 0:
                logger.warning("No 1h bars in range %s → %s", start_ts, end_ts)
                return []

            total_bars = len(bar_timestamps)
            t0 = time.perf_counter()
            scan_interval = self.params.scan_interval_hours

            logger.info(
                "Backtest: %s → %s (%d 1h bars, scan every %dh)",
                start_ts.date(), end_ts.date(), total_bars, scan_interval,
            )

            for i, current_ns in enumerate(bar_timestamps):
                # 1. Update active positions (sub-bar)
                self._update_positions(current_ns)

                # 2. Scan for new signals (at scan interval)
                if i % scan_interval == 0:
                    self._scan_and_enter(current_ns)

                # 3. Track equity
                self.equity_curve.append(self._compute_equity())

                self._prev_1h_ns = current_ns

                # Progress logging every 5000 bars
                if (i + 1) % 5000 == 0:
                    elapsed = time.perf_counter() - t0
                    pct = (i + 1) / total_bars * 100
                    trades_so_far = len([t for t in self.trades if t.exit_ts is not None])
                    logger.info(
                        "  %5.1f%% (%d/%d bars, %.1fs) — %d trades, %d open",
                        pct, i + 1, total_bars, elapsed,
                        trades_so_far, len(self.positions),
                    )

            # Close any remaining positions at end
            for asset, pos in list(self.positions.items()):
                ad = self.all_data.get(asset, {}).get(pos.best_tf)
                if ad is not None:
                    last_idx = find_bar_index(ad.timestamps, bar_timestamps[-1])
                    exit_price = float(ad.close[last_idx])
                    exit_ts = pd.Timestamp(bar_timestamps[-1], tz="UTC")
                    self._close_position(pos, exit_price, exit_ts, "END_OF_PERIOD")

            elapsed = time.perf_counter() - t0
            completed = [t for t in self.trades if t.exit_ts is not None]
            logger.info(
                "Backtest complete: %.1fs, %d scans, %d trades",
                elapsed, self._scan_count, len(completed),
            )

            return self.trades

        finally:
            self._restore_engine_defaults()

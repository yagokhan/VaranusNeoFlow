#!/usr/bin/env python3
"""Run blind tests with different scan start offsets (10min steps)."""

import sys
import time
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import pandas as pd
from backtest.data_loader import load_all_assets, AssetData, find_bar_index, ts_to_ns, BARS_7D, ASSETS, TF_HOURS, build_scan_dataframes, build_htf_dataframe, get_sub_bars
from backtest.engine import BacktestEngine, BacktestParams
from backtest.metrics import compute_metrics, print_metrics

# Consensus params
PARAMS = BacktestParams(
    min_pearson_r=0.83,
    min_pvt_r=0.80,
    combined_gate=0.80,
    hard_sl_mult=2.5,
    trail_buffer=0.5,
    exhaust_r=0.425,
    pos_frac=0.05,
    initial_capital=10_000.0,
    scan_interval_hours=1,
)

START = pd.Timestamp("2025-11-01", tz="UTC")
END = pd.Timestamp("2026-03-17 23:00:00", tz="UTC")


class OffsetBacktestEngine(BacktestEngine):
    """BacktestEngine that uses 5m bar grid with configurable hourly offset."""

    def run_with_offset(self, start_ts, end_ts, offset_minutes=0):
        """Run backtest scanning every 60 min but offset by offset_minutes."""
        self._apply_params_to_engine()
        try:
            self.trades = []
            self.positions = {}
            self.equity_curve = []
            self.realized_pnl = 0.0
            self._trade_counter = 0
            self._scan_count = 0
            self._prev_1h_ns = None

            # Get 5m timestamps for position updates
            ref_asset = None
            for asset_data in self.all_data.values():
                if "5m" in asset_data:
                    ref_asset = asset_data
                    break
            if ref_asset is None:
                return []

            ad_5m = ref_asset["5m"]
            start_ns = ts_to_ns(start_ts)
            end_ns = ts_to_ns(end_ts)
            mask = (ad_5m.timestamps >= start_ns) & (ad_5m.timestamps <= end_ns)
            all_5m_bars = ad_5m.timestamps[mask]

            # Build scan timestamps: every 12th 5m bar (= 60 min) with offset
            offset_bars = offset_minutes // 5  # 10min = 2 bars, 20min = 4 bars, etc.
            scan_set = set()
            for i in range(offset_bars, len(all_5m_bars), 12):
                scan_set.add(all_5m_bars[i])

            # Also get 1h bar grid for position updates (use 1h bars as update ticks)
            ad_1h = ref_asset.get("1h")
            if ad_1h is None:
                return []
            mask_1h = (ad_1h.timestamps >= start_ns) & (ad_1h.timestamps <= end_ns)
            bar_timestamps = ad_1h.timestamps[mask_1h]

            total_bars = len(bar_timestamps)
            t0 = time.perf_counter()

            for i, current_ns in enumerate(bar_timestamps):
                # Update positions on 1h grid (sub-bar updates handled inside)
                self._update_positions(current_ns)

                # Scan only if this timestamp (or nearby 5m bar) is in our offset scan set
                # Find closest 5m bar to current_ns
                should_scan = False
                for offset_ns in scan_set:
                    # Check if there's a scan point within this hour
                    if self._prev_1h_ns is not None:
                        if self._prev_1h_ns < offset_ns <= current_ns:
                            should_scan = True
                            break
                    elif offset_ns <= current_ns:
                        should_scan = True
                        break

                if should_scan:
                    # Use the offset scan time for building windows
                    scan_ns = offset_ns
                    self._scan_and_enter_at(current_ns, scan_ns)

                self.equity_curve.append(self._compute_equity())
                self._prev_1h_ns = current_ns

            # Close remaining positions
            for asset, pos in list(self.positions.items()):
                ad = self.all_data.get(asset, {}).get(pos.best_tf)
                if ad is not None:
                    last_idx = find_bar_index(ad.timestamps, bar_timestamps[-1])
                    exit_price = float(ad.close[last_idx])
                    exit_ts = pd.Timestamp(bar_timestamps[-1], tz="UTC")
                    self._close_position(pos, exit_price, exit_ts, "END_OF_PERIOD")

            elapsed = time.perf_counter() - t0
            completed = [t for t in self.trades if t.exit_ts is not None]
            return self.trades
        finally:
            self._restore_engine_defaults()

    def _scan_and_enter_at(self, current_1h_ns, scan_ns):
        """Scan using scan_ns for window building but current_1h_ns for entry."""
        if len(self.positions) >= self.params.max_concurrent:
            return

        from neo_flow.adaptive_engine import scan_asset, get_leverage, compute_hard_sl, compute_atr, HIGH_VOL_ASSETS
        from backtest.engine import TradeRecord, ActivePosition

        current_ts = pd.Timestamp(current_1h_ns, tz="UTC")

        for asset in ASSETS:
            if asset in self.positions:
                continue
            if len(self.positions) >= self.params.max_concurrent:
                break

            asset_data = self.all_data.get(asset)
            if asset_data is None:
                continue

            scan_dfs = build_scan_dataframes(asset_data, scan_ns)
            df_4h = build_htf_dataframe(asset_data, scan_ns)

            if not scan_dfs or df_4h is None:
                continue

            signal = scan_asset(asset, scan_dfs, df_4h)
            if signal is None:
                continue

            entry_price = self._get_entry_price(asset, signal.best_tf, current_1h_ns)
            if entry_price is None:
                continue

            lev = get_leverage(signal.confidence)
            vol_scalar = 0.75 if asset in HIGH_VOL_ASSETS else 1.0
            pos_usd = self.capital * self.params.pos_frac * lev * vol_scalar

            if pos_usd <= 0 or lev == 0:
                continue

            ad = asset_data.get(signal.best_tf)
            idx = find_bar_index(ad.timestamps, current_1h_ns)
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

            std_dev_price = signal.midline * signal.std_dev
            if signal.direction == 1:
                initial_trail = signal.midline - self.params.trail_buffer * std_dev_price
            else:
                initial_trail = signal.midline + self.params.trail_buffer * std_dev_price

            self._trade_counter += 1

            tr = TradeRecord(
                trade_id=self._trade_counter, asset=asset, direction=signal.direction,
                entry_ts=current_ts, entry_price=entry_price, best_tf=signal.best_tf,
                best_period=signal.best_period, confidence=signal.confidence,
                pvt_r=signal.pvt_r, leverage=lev, position_usd=pos_usd,
                hard_sl=hard_sl, initial_trail_sl=initial_trail, peak_r=signal.confidence,
            )
            self.trades.append(tr)

            pos = ActivePosition(
                trade_id=self._trade_counter, asset=asset, direction=signal.direction,
                entry_price=entry_price, entry_ts=current_ts, hard_sl=hard_sl,
                trail_sl=initial_trail, midline=signal.midline, std_dev=signal.std_dev,
                best_tf=signal.best_tf, best_period=signal.best_period,
                confidence=signal.confidence, pvt_r=signal.pvt_r,
                leverage=lev, position_usd=pos_usd,
            )
            self.positions[asset] = pos

        self._scan_count += 1


def main():
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)-5s | %(message)s")

    print("Loading data...")
    all_data = load_all_assets()
    print("Data loaded.\n")

    offsets = [0, 10, 20, 30, 40, 50]
    results = []

    for offset in offsets:
        print(f"Running offset={offset:02d}min...", end=" ", flush=True)
        t0 = time.perf_counter()

        engine = OffsetBacktestEngine(all_data, PARAMS)

        if offset == 0:
            trades = engine.run(START, END)
        else:
            trades = engine.run_with_offset(START, END, offset_minutes=offset)

        completed = [t for t in trades if t.exit_ts is not None]
        m = compute_metrics(completed, engine.equity_curve, PARAMS.initial_capital)
        elapsed = time.perf_counter() - t0

        # Export trades CSV for this offset
        from backtest.metrics import trades_to_csv
        csv_path = f"blind_test_offset_{offset:02d}min.csv"
        trades_to_csv(completed, csv_path)

        results.append({
            "offset": offset,
            "trades": m.total_trades,
            "wr": m.win_rate,
            "pnl_pct": m.total_pnl_pct,
            "sharpe": m.sharpe_ratio,
            "max_dd": m.max_drawdown_pct,
            "pf": m.profit_factor,
            "trail_pct": m.trail_exit_pct,
            "hard_sl_pct": m.hard_sl_exit_pct,
        })

        print(f"done in {elapsed:.0f}s — {m.total_trades} trades, WR={m.win_rate:.1f}%, PnL={m.total_pnl_pct:+.1f}%, Sharpe={m.sharpe_ratio:.2f}")

    print("\n" + "=" * 90)
    print(f"{'Offset':>8} {'Trades':>7} {'WR%':>7} {'PnL%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'PF':>6} {'Trail%':>7} {'HardSL%':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r['offset']:>5}min {r['trades']:>7} {r['wr']:>6.1f}% {r['pnl_pct']:>+8.1f}% {r['sharpe']:>8.2f} {r['max_dd']:>7.2f}% {r['pf']:>6.2f} {r['trail_pct']:>6.1f}% {r['hard_sl_pct']:>7.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()

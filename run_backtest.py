#!/usr/bin/env python3
"""
run_backtest.py — CLI entry point for Varanus Neo-Flow backtester.

Usage:
    # Full backtest (2023-01-01 → 2025-10-31):
    python run_backtest.py

    # Custom date range:
    python run_backtest.py --start 2024-01-01 --end 2024-12-31

    # Blind test period:
    python run_backtest.py --blind

    # With custom parameters:
    python run_backtest.py --min-r 0.85 --trail-buffer 1.5

    # Scan every 4h instead of 1h (faster):
    python run_backtest.py --scan-interval 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/gokhan/varanus_neo_flow")

import pandas as pd

from backtest.data_loader import (
    load_all_assets,
    generate_wfv_folds,
    WFV_START, WFV_END,
    BLIND_START, BLIND_END,
)
from backtest.engine import BacktestEngine, BacktestParams
from backtest.metrics import compute_metrics, print_metrics, trades_to_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Varanus Neo-Flow — Backtester",
    )
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--blind", action="store_true", help="Run blind test period (2025-11-01 → 2026-03-15)")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Starting capital (default: 10000)")
    parser.add_argument("--scan-interval", type=int, default=1, help="Scan every N hours (default: 1)")
    parser.add_argument("--min-r", type=float, default=0.80, help="MIN_PEARSON_R (default: 0.80)")
    parser.add_argument("--min-pvt-r", type=float, default=0.70, help="MIN_PVT_PEARSON_R (default: 0.70)")
    parser.add_argument("--gate", type=float, default=0.65, help="COMBINED_GATE_THRESHOLD (default: 0.65)")
    parser.add_argument("--hard-sl", type=float, default=1.5, help="HARD_SL_ATR_MULT (default: 1.5)")
    parser.add_argument("--trail-buffer", type=float, default=1.0, help="TRAIL_BUFFER_STD (default: 1.0)")
    parser.add_argument("--exhaust-r", type=float, default=0.50, help="TREND_EXHAUST_R (default: 0.50)")
    parser.add_argument("--pos-frac", type=float, default=0.10, help="Position fraction (default: 0.10)")
    parser.add_argument("--csv", type=str, default=None, help="Export trades to CSV file")
    parser.add_argument("--wfv", action="store_true", help="Run walk-forward validation (8 folds)")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna optimization + WFV")
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna trials per fold (default: 100)")
    args = parser.parse_args()

    # Determine date range
    if args.blind:
        start_ts = BLIND_START
        end_ts = BLIND_END
    elif args.start and args.end:
        start_ts = pd.Timestamp(args.start, tz="UTC")
        end_ts = pd.Timestamp(args.end + " 23:00:00", tz="UTC")
    else:
        start_ts = WFV_START
        end_ts = WFV_END

    params = BacktestParams(
        min_pearson_r=args.min_r,
        min_pvt_r=args.min_pvt_r,
        combined_gate=args.gate,
        hard_sl_mult=args.hard_sl,
        trail_buffer=args.trail_buffer,
        exhaust_r=args.exhaust_r,
        pos_frac=args.pos_frac,
        initial_capital=args.capital,
        scan_interval_hours=args.scan_interval,
    )

    # Load data
    logger.info("Loading historical data...")
    all_data = load_all_assets()

    if args.optimize:
        # Full Optuna optimization + WFV
        from backtest.optimize import run_wfv
        run_wfv(
            all_data=all_data,
            n_trials=args.n_trials,
            scan_interval=args.scan_interval,
            initial_capital=args.capital,
        )
    elif args.wfv:
        # Walk-forward validation with fixed params (no optimization)
        folds = generate_wfv_folds()
        logger.info("Running %d-fold walk-forward validation (fixed params)", len(folds))
        logger.info("")

        all_fold_metrics = []
        for fold in folds:
            logger.info(
                "=== FOLD %d: train %s→%s | val %s→%s | test %s→%s ===",
                fold.fold_id,
                fold.train_start.date(), fold.train_end.date(),
                fold.val_start.date(), fold.val_end.date(),
                fold.test_start.date(), fold.test_end.date(),
            )

            engine = BacktestEngine(all_data, params)
            trades = engine.run(fold.test_start, fold.test_end)
            m = compute_metrics(trades, engine.equity_curve, params.initial_capital)
            all_fold_metrics.append(m)
            print_metrics(m, params.initial_capital)

        # Aggregate
        if all_fold_metrics:
            print("\n" + "=" * 70)
            print("WALK-FORWARD AGGREGATE (across test periods)")
            print("=" * 70)
            avg_wr = sum(m.win_rate for m in all_fold_metrics) / len(all_fold_metrics)
            avg_sharpe = sum(m.sharpe_ratio for m in all_fold_metrics) / len(all_fold_metrics)
            avg_dd = sum(m.max_drawdown_pct for m in all_fold_metrics) / len(all_fold_metrics)
            total_trades = sum(m.total_trades for m in all_fold_metrics)
            print(f"  Avg Win Rate:    {avg_wr:.1f}%")
            print(f"  Avg Sharpe:      {avg_sharpe:.2f}")
            print(f"  Avg Max DD:      {avg_dd:.2f}%")
            print(f"  Total Trades:    {total_trades}")
            print("=" * 70)
    else:
        # Single run
        engine = BacktestEngine(all_data, params)
        trades = engine.run(start_ts, end_ts)
        m = compute_metrics(trades, engine.equity_curve, params.initial_capital)
        print_metrics(m, params.initial_capital)

        if args.csv:
            trades_to_csv(trades, args.csv)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_optimize.py — High-Performance Bias-Free WFV Optimization

Runs 8-fold Walk-Forward Validation with Optuna using the bias-free
engine (closed-bar logic). Designed for multi-core machines.

Features:
  - Parallel fold optimization via joblib (n_jobs=-1 = all cores)
  - SQLite storage for resumable studies (one DB per fold)
  - Bias-free engine: scan_ns-1 for scan TFs, scan_ns-4h for HTF
  - Consensus params via median across folds
  - Blind test with consensus params
  - Full results saved to wfv_results_v2.json

Usage:
    # Full optimization (all cores, 200 trials per fold):
    python run_optimize.py --n-trials 200

    # Quick test (50 trials, 2 cores):
    python run_optimize.py --n-trials 50 --n-jobs 2

    # Resume interrupted run (reads existing .db files):
    python run_optimize.py --n-trials 200 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: optuna is required. Install: pip install optuna")
    sys.exit(1)

from backtest.data_loader import (
    load_all_assets,
    generate_wfv_folds,
    WFVFold,
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

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "optuna_studies"
DB_DIR.mkdir(exist_ok=True)

# Parameter search space (same as original, proven ranges)
PARAM_SPACE = {
    "min_pearson_r": (0.75, 0.92, 0.01),
    "min_pvt_r":     (0.50, 0.85, 0.05),
    "combined_gate": (0.50, 0.80, 0.05),
    "hard_sl_mult":  (1.0, 3.0, 0.25),
    "trail_buffer":  (0.5, 2.0, 0.25),
    "exhaust_r":     (0.30, 0.65, 0.05),
    "pos_frac":      (0.05, 0.15, 0.01),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Objective function
# ═══════════════════════════════════════════════════════════════════════════════

def create_objective(all_data, fold, scan_interval=1, initial_capital=10_000.0):
    """Create an Optuna objective for one WFV fold (validates on val period)."""

    def objective(trial):
        params = BacktestParams(
            min_pearson_r=trial.suggest_float("min_pearson_r", *PARAM_SPACE["min_pearson_r"]),
            min_pvt_r=trial.suggest_float("min_pvt_r", *PARAM_SPACE["min_pvt_r"]),
            combined_gate=trial.suggest_float("combined_gate", *PARAM_SPACE["combined_gate"]),
            hard_sl_mult=trial.suggest_float("hard_sl_mult", *PARAM_SPACE["hard_sl_mult"]),
            trail_buffer=trial.suggest_float("trail_buffer", *PARAM_SPACE["trail_buffer"]),
            exhaust_r=trial.suggest_float("exhaust_r", *PARAM_SPACE["exhaust_r"]),
            pos_frac=trial.suggest_float("pos_frac", *PARAM_SPACE["pos_frac"]),
            initial_capital=initial_capital,
            scan_interval_hours=scan_interval,
        )

        engine = BacktestEngine(all_data, params)
        trades = engine.run(fold.val_start, fold.val_end)
        m = compute_metrics(trades, engine.equity_curve, initial_capital)

        if m.total_trades < 10:
            return -10.0

        score = m.sharpe_ratio
        if m.win_rate >= 60:
            score += 0.5
        if m.trail_exit_pct >= 60:
            score += 0.3
        if m.max_drawdown_pct > -15:
            score += 0.2
        if m.max_drawdown_pct < -20:
            score -= 2.0

        return score

    return objective


# ═══════════════════════════════════════════════════════════════════════════════
# Single fold optimization
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_fold(all_data, fold, n_trials=200, scan_interval=1,
                  initial_capital=10_000.0, resume=False):
    """Optimize one fold with SQLite-backed study (resumable)."""

    db_path = DB_DIR / f"fold_{fold.fold_id}.db"
    storage = f"sqlite:///{db_path}"

    study_name = f"wfv_fold_{fold.fold_id}"

    if resume and db_path.exists():
        logger.info("Fold %d: resuming from %s", fold.fold_id, db_path)
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
        completed = len(study.trials)
        remaining = max(0, n_trials - completed)
        logger.info("Fold %d: %d trials completed, %d remaining", fold.fold_id, completed, remaining)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=TPESampler(seed=42 + fold.fold_id),
            load_if_exists=True,
        )
        completed = len(study.trials)
        remaining = max(0, n_trials - completed)

    if remaining > 0:
        logger.info(
            "Fold %d: optimizing on val %s→%s (%d trials)...",
            fold.fold_id, fold.val_start.date(), fold.val_end.date(), remaining,
        )
        objective = create_objective(all_data, fold, scan_interval, initial_capital)
        t0 = time.perf_counter()
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
        elapsed = time.perf_counter() - t0
        logger.info("Fold %d: optimization done in %.1f min (best=%.4f)",
                     fold.fold_id, elapsed / 60, study.best_value)
    else:
        logger.info("Fold %d: already has %d trials, skipping", fold.fold_id, completed)

    best = study.best_params
    logger.info("Fold %d best params: %s", fold.fold_id, best)

    # Evaluate best params on train/val/test
    best_params = BacktestParams(
        min_pearson_r=best["min_pearson_r"],
        min_pvt_r=best["min_pvt_r"],
        combined_gate=best["combined_gate"],
        hard_sl_mult=best["hard_sl_mult"],
        trail_buffer=best["trail_buffer"],
        exhaust_r=best["exhaust_r"],
        pos_frac=best["pos_frac"],
        initial_capital=initial_capital,
        scan_interval_hours=scan_interval,
    )

    results = {}
    for period_name, start, end in [
        ("train", fold.train_start, fold.train_end),
        ("val", fold.val_start, fold.val_end),
        ("test", fold.test_start, fold.test_end),
    ]:
        engine = BacktestEngine(all_data, best_params)
        trades = engine.run(start, end)
        m = compute_metrics(trades, engine.equity_curve, initial_capital)
        results[period_name] = m

    logger.info(
        "Fold %d — Train: WR=%.1f%% Sh=%.2f | Val: WR=%.1f%% Sh=%.2f | Test: WR=%.1f%% Sh=%.2f",
        fold.fold_id,
        results["train"].win_rate, results["train"].sharpe_ratio,
        results["val"].win_rate, results["val"].sharpe_ratio,
        results["test"].win_rate, results["test"].sharpe_ratio,
    )

    return {
        "fold_id": fold.fold_id,
        "best_params": best,
        "n_trials": len(study.trials),
        "train": results["train"],
        "val": results["val"],
        "test": results["test"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel fold runner
# ═══════════════════════════════════════════════════════════════════════════════

def _fold_worker(args):
    """Worker function for parallel fold optimization."""
    fold, n_trials, scan_interval, initial_capital, resume, data_dir = args

    # Each worker loads its own copy of the data (fork-safe)
    import warnings
    warnings.filterwarnings("ignore")

    # Suppress optuna noise in workers
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)

    all_data = load_all_assets(data_dir=data_dir)
    return optimize_fold(all_data, fold, n_trials, scan_interval, initial_capital, resume)


# ═══════════════════════════════════════════════════════════════════════════════
# Consensus computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_consensus(fold_results):
    """Median of best params across all folds."""
    param_keys = fold_results[0]["best_params"].keys()
    consensus = {}
    for key in param_keys:
        values = [fr["best_params"][key] for fr in fold_results]
        consensus[key] = round(float(np.median(values)), 4)
    return consensus


# ═══════════════════════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(fold_results, consensus, blind_metrics, output_path):
    """Save full WFV results to JSON."""

    def to_py(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    test_metrics = [fr["test"] for fr in fold_results]
    aggregate = {
        "avg_win_rate": float(np.mean([m.win_rate for m in test_metrics])),
        "avg_sharpe": float(np.mean([m.sharpe_ratio for m in test_metrics])),
        "avg_max_dd": float(np.mean([m.max_drawdown_pct for m in test_metrics])),
        "avg_profit_factor": float(np.mean([m.profit_factor for m in test_metrics])),
        "total_trades": int(sum(m.total_trades for m in test_metrics)),
        "avg_trail_exit_pct": float(np.mean([m.trail_exit_pct for m in test_metrics])),
    }

    data = {
        "engine_version": "bias-free (closed-bar, 4h-offset)",
        "consensus_params": consensus,
        "aggregate_test_metrics": aggregate,
        "blind_test": {
            "total_trades": blind_metrics.total_trades,
            "win_rate": blind_metrics.win_rate,
            "total_pnl_pct": blind_metrics.total_pnl_pct,
            "sharpe_ratio": blind_metrics.sharpe_ratio,
            "max_drawdown_pct": blind_metrics.max_drawdown_pct,
            "profit_factor": blind_metrics.profit_factor,
            "trail_exit_pct": blind_metrics.trail_exit_pct,
            "hard_sl_exit_pct": blind_metrics.hard_sl_exit_pct,
        },
        "folds": [],
    }

    for fr in fold_results:
        fold_data = {
            "fold_id": fr["fold_id"],
            "best_params": fr["best_params"],
            "n_trials": fr["n_trials"],
            "train": {"trades": fr["train"].total_trades, "win_rate": fr["train"].win_rate,
                       "sharpe": fr["train"].sharpe_ratio, "max_dd": fr["train"].max_drawdown_pct,
                       "pnl_pct": fr["train"].total_pnl_pct},
            "val":   {"trades": fr["val"].total_trades, "win_rate": fr["val"].win_rate,
                       "sharpe": fr["val"].sharpe_ratio, "max_dd": fr["val"].max_drawdown_pct,
                       "pnl_pct": fr["val"].total_pnl_pct},
            "test":  {"trades": fr["test"].total_trades, "win_rate": fr["test"].win_rate,
                       "sharpe": fr["test"].sharpe_ratio, "max_dd": fr["test"].max_drawdown_pct,
                       "pnl_pct": fr["test"].total_pnl_pct},
        }
        data["folds"].append(fold_data)

    Path(output_path).write_text(json.dumps(data, indent=2, default=to_py))
    logger.info("Results saved to %s", output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary printer
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(fold_results, consensus, blind_metrics):
    """Print formatted WFV summary table."""
    print()
    print("=" * 90)
    print("WALK-FORWARD VALIDATION SUMMARY (Bias-Free Engine)")
    print("=" * 90)
    print()

    print(f"{'Fold':>4} {'Trials':>7}  {'Train WR':>8} {'Val WR':>8} {'Test WR':>8}  "
          f"{'Train Sh':>8} {'Val Sh':>8} {'Test Sh':>8}  {'Test DD':>8}")
    print("-" * 90)

    for fr in fold_results:
        print(f"{fr['fold_id']:>4} {fr['n_trials']:>7}  "
              f"{fr['train'].win_rate:>7.1f}% {fr['val'].win_rate:>7.1f}% "
              f"{fr['test'].win_rate:>7.1f}%  "
              f"{fr['train'].sharpe_ratio:>8.2f} {fr['val'].sharpe_ratio:>8.2f} "
              f"{fr['test'].sharpe_ratio:>8.2f}  "
              f"{fr['test'].max_drawdown_pct:>7.2f}%")

    print("-" * 90)

    print(f"\nConsensus Parameters (median across {len(fold_results)} folds):")
    for k, v in consensus.items():
        print(f"  {k}: {v}")

    print(f"\nBlind Test ({BLIND_START.date()} → {BLIND_END.date()}):")
    print(f"  Trades:        {blind_metrics.total_trades}")
    print(f"  Win Rate:      {blind_metrics.win_rate:.1f}%")
    print(f"  PnL:           {blind_metrics.total_pnl_pct:+.1f}%")
    print(f"  Sharpe:        {blind_metrics.sharpe_ratio:.2f}")
    print(f"  Max DD:        {blind_metrics.max_drawdown_pct:.2f}%")
    print(f"  Profit Factor: {blind_metrics.profit_factor:.2f}")
    print(f"  Trail Exits:   {blind_metrics.trail_exit_pct:.1f}%")
    print("=" * 90)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Varanus Neo-Flow — Bias-Free WFV Optimization")
    parser.add_argument("--n-trials", type=int, default=200, help="Optuna trials per fold (default: 200)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers (-1=all cores, 1=sequential)")
    parser.add_argument("--scan-interval", type=int, default=1, help="Scan every N hours (default: 1)")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Starting capital (default: 10000)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing .db files")
    parser.add_argument("--blind-only", action="store_true", help="Skip optimization, run blind test with existing consensus")
    args = parser.parse_args()

    data_dir = BASE_DIR / "data"

    if args.blind_only:
        # Load existing consensus from wfv_results_v2.json
        result_path = BASE_DIR / "wfv_results_v2.json"
        if not result_path.exists():
            logger.error("No wfv_results_v2.json found. Run optimization first.")
            return
        with open(result_path) as f:
            data = json.load(f)
        consensus = data["consensus_params"]
        logger.info("Loaded consensus: %s", consensus)
    else:
        # Load data once for sequential mode (parallel mode loads per-worker)
        logger.info("Loading historical data...")
        all_data = load_all_assets(data_dir=str(data_dir))

        folds = generate_wfv_folds()
        logger.info("Generated %d WFV folds:", len(folds))
        for f in folds:
            logger.info(
                "  Fold %d: train %s→%s | val %s→%s | test %s→%s",
                f.fold_id,
                f.train_start.date(), f.train_end.date(),
                f.val_start.date(), f.val_end.date(),
                f.test_start.date(), f.test_end.date(),
            )

        n_jobs = args.n_jobs
        if n_jobs == -1:
            import os
            n_jobs = os.cpu_count() or 1

        fold_results = []
        t0_total = time.perf_counter()

        if n_jobs > 1 and len(folds) > 1:
            # Parallel execution via multiprocessing
            import multiprocessing
            ctx = multiprocessing.get_context("spawn")

            logger.info("Running %d folds in parallel (%d workers)...", len(folds), n_jobs)

            worker_args = [
                (fold, args.n_trials, args.scan_interval, args.capital, args.resume, str(data_dir))
                for fold in folds
            ]

            with ctx.Pool(min(n_jobs, len(folds))) as pool:
                fold_results = pool.map(_fold_worker, worker_args)
        else:
            # Sequential execution
            logger.info("Running %d folds sequentially...", len(folds))
            for fold in folds:
                result = optimize_fold(
                    all_data, fold,
                    n_trials=args.n_trials,
                    scan_interval=args.scan_interval,
                    initial_capital=args.capital,
                    resume=args.resume,
                )
                fold_results.append(result)

        elapsed_total = time.perf_counter() - t0_total
        logger.info("All folds complete in %.1f min", elapsed_total / 60)

        # Consensus params
        consensus = compute_consensus(fold_results)
        logger.info("Consensus params: %s", consensus)

    # Blind test with consensus params
    logger.info("Running blind test with consensus params...")
    if 'all_data' not in locals():
        all_data = load_all_assets(data_dir=str(data_dir))

    blind_params = BacktestParams(
        min_pearson_r=consensus["min_pearson_r"],
        min_pvt_r=consensus["min_pvt_r"],
        combined_gate=consensus["combined_gate"],
        hard_sl_mult=consensus["hard_sl_mult"],
        trail_buffer=consensus["trail_buffer"],
        exhaust_r=consensus["exhaust_r"],
        pos_frac=consensus["pos_frac"],
        initial_capital=args.capital,
        scan_interval_hours=args.scan_interval,
    )

    engine_blind = BacktestEngine(all_data, blind_params)
    trades_blind = engine_blind.run(BLIND_START, BLIND_END)
    blind_metrics = compute_metrics(trades_blind, engine_blind.equity_curve, args.capital)

    print_metrics(blind_metrics, args.capital)
    trades_to_csv(trades_blind, str(BASE_DIR / "blind_test_v2.csv"))

    if not args.blind_only:
        # Save everything
        output_path = BASE_DIR / "wfv_results_v2.json"
        save_results(fold_results, consensus, blind_metrics, output_path)
        print_summary(fold_results, consensus, blind_metrics)

    logger.info("Done.")


if __name__ == "__main__":
    main()

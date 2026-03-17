"""
backtest/optimize_fast.py — Optuna optimization with pre-computed features + parallel folds.

Uses FastBacktestEngine for O(1) scan lookups.
Parallelizes 8 WFV folds across CPU cores.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from backtest.data_loader import (
    AssetData,
    WFVFold,
    generate_wfv_folds,
    load_all_assets,
    BLIND_START, BLIND_END,
)
from backtest.engine import BacktestParams, TradeRecord
from backtest.engine_fast import FastBacktestEngine
from backtest.metrics import compute_metrics, print_metrics, BacktestMetrics
from backtest.optimize import FoldResult, WFVResult, compute_consensus_params, _save_results, _print_wfv_summary

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Worker init for multiprocessing (fork shares memory via COW)
# ═══════════════════════════════════════════════════════════════════════════════

_worker_all_data = None
_worker_features = None


def _init_worker(all_data, features):
    global _worker_all_data, _worker_features
    _worker_all_data = all_data
    _worker_features = features


def _optimize_fold_worker(args):
    """Worker function for parallel fold optimization."""
    fold, n_trials, scan_interval, initial_capital = args
    return _optimize_fold_fast(
        _worker_all_data, _worker_features, fold,
        n_trials, scan_interval, initial_capital,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Fast fold optimization
# ═══════════════════════════════════════════════════════════════════════════════

def _optimize_fold_fast(
    all_data: dict,
    features: dict,
    fold: WFVFold,
    n_trials: int = 100,
    scan_interval: int = 1,
    initial_capital: float = 10_000.0,
) -> FoldResult:
    """Optimize one fold using FastBacktestEngine."""
    if not HAS_OPTUNA:
        raise ImportError("optuna required")

    logger.info(
        "=== FOLD %d: Optimizing on val %s → %s (%d trials) ===",
        fold.fold_id, fold.val_start.date(), fold.val_end.date(), n_trials,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42 + fold.fold_id),
        pruner=MedianPruner(n_startup_trials=10),
    )

    def objective(trial):
        params = BacktestParams(
            min_pearson_r=trial.suggest_float("min_pearson_r", 0.75, 0.92, step=0.01),
            min_pvt_r=trial.suggest_float("min_pvt_r", 0.50, 0.85, step=0.05),
            combined_gate=trial.suggest_float("combined_gate", 0.50, 0.80, step=0.05),
            hard_sl_mult=trial.suggest_float("hard_sl_mult", 1.0, 3.0, step=0.25),
            trail_buffer=trial.suggest_float("trail_buffer", 0.5, 2.0, step=0.25),
            exhaust_r=trial.suggest_float("exhaust_r", 0.30, 0.65, step=0.05),
            pos_frac=trial.suggest_float("pos_frac", 0.05, 0.15, step=0.01),
            initial_capital=initial_capital,
            scan_interval_hours=scan_interval,
        )

        engine = FastBacktestEngine(all_data, params, features)
        trades = engine.run(fold.val_start, fold.val_end)
        metrics = compute_metrics(trades, engine.equity_curve, initial_capital)

        if metrics.total_trades < 10:
            return -10.0

        score = metrics.sharpe_ratio
        if metrics.win_rate >= 60:
            score += 0.5
        if metrics.trail_exit_pct >= 60:
            score += 0.3
        if metrics.max_drawdown_pct > -15:
            score += 0.2
        if metrics.max_drawdown_pct < -20:
            score -= 2.0

        return score

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.perf_counter() - t0

    best_params_dict = study.best_params
    logger.info(
        "  Fold %d complete: %.1f min, best score=%.4f",
        fold.fold_id, elapsed / 60, study.best_value,
    )

    # Evaluate best params on train/val/test
    best_params = BacktestParams(
        min_pearson_r=best_params_dict["min_pearson_r"],
        min_pvt_r=best_params_dict["min_pvt_r"],
        combined_gate=best_params_dict["combined_gate"],
        hard_sl_mult=best_params_dict["hard_sl_mult"],
        trail_buffer=best_params_dict["trail_buffer"],
        exhaust_r=best_params_dict["exhaust_r"],
        pos_frac=best_params_dict["pos_frac"],
        initial_capital=initial_capital,
        scan_interval_hours=scan_interval,
    )

    engine_train = FastBacktestEngine(all_data, best_params, features)
    trades_train = engine_train.run(fold.train_start, fold.train_end)
    train_metrics = compute_metrics(trades_train, engine_train.equity_curve, initial_capital)

    engine_val = FastBacktestEngine(all_data, best_params, features)
    trades_val = engine_val.run(fold.val_start, fold.val_end)
    val_metrics = compute_metrics(trades_val, engine_val.equity_curve, initial_capital)

    engine_test = FastBacktestEngine(all_data, best_params, features)
    trades_test = engine_test.run(fold.test_start, fold.test_end)
    test_metrics = compute_metrics(trades_test, engine_test.equity_curve, initial_capital)

    logger.info(
        "  Fold %d results — Train: WR=%.1f%% Sharpe=%.2f | "
        "Val: WR=%.1f%% Sharpe=%.2f | Test: WR=%.1f%% Sharpe=%.2f",
        fold.fold_id,
        train_metrics.win_rate, train_metrics.sharpe_ratio,
        val_metrics.win_rate, val_metrics.sharpe_ratio,
        test_metrics.win_rate, test_metrics.sharpe_ratio,
    )

    return FoldResult(
        fold_id=fold.fold_id,
        best_params=best_params_dict,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        n_trials=n_trials,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry: parallel WFV
# ═══════════════════════════════════════════════════════════════════════════════

def run_wfv_fast(
    all_data: dict | None = None,
    features: dict | None = None,
    n_trials: int = 100,
    scan_interval: int = 1,
    initial_capital: float = 10_000.0,
    output_dir: str = "/home/yagokhan/VaranusNeoFlow",
    n_jobs: int = 0,
) -> WFVResult:
    """
    Run walk-forward validation with pre-computed features and optional parallelism.

    Parameters
    ----------
    n_jobs : Number of parallel workers. 0 = auto (cpu_count), 1 = sequential.
    """
    if all_data is None:
        logger.info("Loading data...")
        all_data = load_all_assets()

    if features is None:
        logger.info("Pre-computing features...")
        from neo_flow.precompute_features import precompute_all_features
        features = precompute_all_features(all_data)

    folds = generate_wfv_folds()
    logger.info("Generated %d WFV folds", len(folds))
    for f in folds:
        logger.info(
            "  Fold %d: train %s→%s | val %s→%s | test %s→%s",
            f.fold_id,
            f.train_start.date(), f.train_end.date(),
            f.val_start.date(), f.val_end.date(),
            f.test_start.date(), f.test_end.date(),
        )

    if n_jobs == 0:
        n_jobs = min(mp.cpu_count(), len(folds))

    fold_args = [
        (fold, n_trials, scan_interval, initial_capital)
        for fold in folds
    ]

    if n_jobs > 1:
        logger.info("Optimizing %d folds in parallel (%d workers)...", len(folds), n_jobs)
        # Use fork start method for COW memory sharing
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=n_jobs,
            initializer=_init_worker,
            initargs=(all_data, features),
        ) as pool:
            fold_results = pool.map(_optimize_fold_worker, fold_args)
    else:
        logger.info("Optimizing %d folds sequentially...", len(folds))
        fold_results = []
        for args in fold_args:
            fold, nt, si, ic = args
            result = _optimize_fold_fast(all_data, features, fold, nt, si, ic)
            fold_results.append(result)

    # Consensus params
    consensus = compute_consensus_params(fold_results)
    logger.info("Consensus params: %s", consensus)

    # Aggregate test metrics
    test_metrics_list = [fr.test_metrics for fr in fold_results]
    aggregate = {
        "avg_win_rate": float(np.mean([m.win_rate for m in test_metrics_list])),
        "avg_sharpe": float(np.mean([m.sharpe_ratio for m in test_metrics_list])),
        "avg_max_dd": float(np.mean([m.max_drawdown_pct for m in test_metrics_list])),
        "avg_profit_factor": float(np.mean([m.profit_factor for m in test_metrics_list])),
        "total_trades": int(sum(m.total_trades for m in test_metrics_list)),
        "avg_trail_exit_pct": float(np.mean([m.trail_exit_pct for m in test_metrics_list])),
        "avg_hard_sl_pct": float(np.mean([m.hard_sl_exit_pct for m in test_metrics_list])),
    }

    # Blind test with consensus params
    logger.info("Running blind test with consensus params (%s → %s)...",
                BLIND_START.date(), BLIND_END.date())
    blind_params = BacktestParams(
        min_pearson_r=consensus["min_pearson_r"],
        min_pvt_r=consensus["min_pvt_r"],
        combined_gate=consensus["combined_gate"],
        hard_sl_mult=consensus["hard_sl_mult"],
        trail_buffer=consensus["trail_buffer"],
        exhaust_r=consensus["exhaust_r"],
        pos_frac=consensus["pos_frac"],
        initial_capital=initial_capital,
        scan_interval_hours=scan_interval,
    )
    engine_blind = FastBacktestEngine(all_data, blind_params, features)
    trades_blind = engine_blind.run(BLIND_START, BLIND_END)
    blind_metrics = compute_metrics(trades_blind, engine_blind.equity_curve, initial_capital)

    logger.info("Blind test results:")
    print_metrics(blind_metrics, initial_capital)

    # Save
    result = WFVResult(
        fold_results=fold_results,
        aggregate_test_metrics=aggregate,
        consensus_params=consensus,
    )

    output_path = Path(output_dir) / "wfv_results.json"
    _save_results(result, blind_metrics, output_path)
    _print_wfv_summary(result, blind_metrics)

    return result

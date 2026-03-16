"""
backtest/optimize.py — Parameter optimization with Optuna + Walk-Forward Validation.

Searches the parameter space using TPE (Tree-structured Parzen Estimator),
evaluates on validation periods, scores on test periods.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
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
from backtest.engine import BacktestEngine, BacktestParams
from backtest.metrics import compute_metrics, print_metrics, BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Result of one WFV fold."""
    fold_id: int
    best_params: dict
    train_metrics: BacktestMetrics
    val_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    n_trials: int


@dataclass
class WFVResult:
    """Aggregate result of all folds."""
    fold_results: list[FoldResult]
    aggregate_test_metrics: dict
    consensus_params: dict


def _create_objective(
    all_data: dict[str, dict[str, AssetData]],
    fold: WFVFold,
    scan_interval: int = 4,
    initial_capital: float = 10_000.0,
):
    """Create an Optuna objective function for one fold."""

    def objective(trial: optuna.Trial) -> float:
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

        # Run on validation period
        engine = BacktestEngine(all_data, params)
        trades = engine.run(fold.val_start, fold.val_end)
        metrics = compute_metrics(trades, engine.equity_curve, initial_capital)

        # Primary objective: Sharpe ratio
        # Penalize if too few trades (need statistical significance)
        if metrics.total_trades < 10:
            return -10.0

        # Multi-objective score: Sharpe with penalties
        score = metrics.sharpe_ratio

        # Bonus for meeting heritage targets
        if metrics.win_rate >= 60:
            score += 0.5
        if metrics.trail_exit_pct >= 60:
            score += 0.3
        if metrics.max_drawdown_pct > -15:
            score += 0.2

        # Penalty for excessive drawdown
        if metrics.max_drawdown_pct < -20:
            score -= 2.0

        return score

    return objective


def optimize_fold(
    all_data: dict[str, dict[str, AssetData]],
    fold: WFVFold,
    n_trials: int = 100,
    scan_interval: int = 4,
    initial_capital: float = 10_000.0,
) -> FoldResult:
    """
    Run Optuna optimization for one fold.

    1. Optimize params on val period
    2. Evaluate best params on train, val, and test periods
    """
    if not HAS_OPTUNA:
        raise ImportError("optuna is required for optimization. Install: pip install optuna")

    logger.info(
        "=== FOLD %d: Optimizing on val %s → %s (%d trials) ===",
        fold.fold_id, fold.val_start.date(), fold.val_end.date(), n_trials,
    )

    # Suppress optuna logging noise
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42 + fold.fold_id),
        pruner=MedianPruner(n_startup_trials=10),
    )

    objective = _create_objective(all_data, fold, scan_interval, initial_capital)

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.perf_counter() - t0

    best_params_dict = study.best_params
    logger.info(
        "  Fold %d optimization complete: %.1f min, best score=%.4f",
        fold.fold_id, elapsed / 60, study.best_value,
    )
    logger.info("  Best params: %s", best_params_dict)

    # Build BacktestParams from best
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

    # Evaluate on all three periods
    logger.info("  Evaluating best params on train/val/test...")

    engine_train = BacktestEngine(all_data, best_params)
    trades_train = engine_train.run(fold.train_start, fold.train_end)
    train_metrics = compute_metrics(trades_train, engine_train.equity_curve, initial_capital)

    engine_val = BacktestEngine(all_data, best_params)
    trades_val = engine_val.run(fold.val_start, fold.val_end)
    val_metrics = compute_metrics(trades_val, engine_val.equity_curve, initial_capital)

    engine_test = BacktestEngine(all_data, best_params)
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


def compute_consensus_params(fold_results: list[FoldResult]) -> dict:
    """
    Compute consensus parameters by taking the median across folds.
    Median is more robust than mean to outlier folds.
    """
    param_keys = fold_results[0].best_params.keys()
    consensus = {}
    for key in param_keys:
        values = [fr.best_params[key] for fr in fold_results]
        consensus[key] = round(float(np.median(values)), 4)
    return consensus


def run_wfv(
    all_data: dict[str, dict[str, AssetData]] | None = None,
    n_trials: int = 100,
    scan_interval: int = 4,
    initial_capital: float = 10_000.0,
    output_dir: str = "/home/gokhan/varanus_neo_flow",
) -> WFVResult:
    """
    Run full walk-forward validation.

    1. Generate 8 folds
    2. Optimize each fold
    3. Compute consensus params
    4. Run blind test with consensus params
    5. Save results
    """
    if all_data is None:
        logger.info("Loading data...")
        all_data = load_all_assets()

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

    # Optimize each fold
    fold_results = []
    for fold in folds:
        result = optimize_fold(
            all_data, fold,
            n_trials=n_trials,
            scan_interval=scan_interval,
            initial_capital=initial_capital,
        )
        fold_results.append(result)

    # Consensus params
    consensus = compute_consensus_params(fold_results)
    logger.info("Consensus params: %s", consensus)

    # Aggregate test metrics
    test_metrics_list = [fr.test_metrics for fr in fold_results]
    aggregate = {
        "avg_win_rate": np.mean([m.win_rate for m in test_metrics_list]),
        "avg_sharpe": np.mean([m.sharpe_ratio for m in test_metrics_list]),
        "avg_max_dd": np.mean([m.max_drawdown_pct for m in test_metrics_list]),
        "avg_profit_factor": np.mean([m.profit_factor for m in test_metrics_list]),
        "total_trades": sum(m.total_trades for m in test_metrics_list),
        "avg_trail_exit_pct": np.mean([m.trail_exit_pct for m in test_metrics_list]),
        "avg_hard_sl_pct": np.mean([m.hard_sl_exit_pct for m in test_metrics_list]),
    }

    # Blind test with consensus params
    logger.info("Running blind test with consensus params (%s → %s)...", BLIND_START.date(), BLIND_END.date())
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
    engine_blind = BacktestEngine(all_data, blind_params)
    trades_blind = engine_blind.run(BLIND_START, BLIND_END)
    blind_metrics = compute_metrics(trades_blind, engine_blind.equity_curve, initial_capital)

    logger.info("Blind test results:")
    print_metrics(blind_metrics, initial_capital)

    # Save results
    result = WFVResult(
        fold_results=fold_results,
        aggregate_test_metrics=aggregate,
        consensus_params=consensus,
    )

    output_path = Path(output_dir) / "wfv_results.json"
    _save_results(result, blind_metrics, output_path)

    # Print summary
    _print_wfv_summary(result, blind_metrics)

    return result


def _save_results(result: WFVResult, blind_metrics: BacktestMetrics, path: Path):
    """Save WFV results to JSON."""
    data = {
        "folds": [],
        "aggregate_test_metrics": result.aggregate_test_metrics,
        "consensus_params": result.consensus_params,
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
    }

    for fr in result.fold_results:
        fold_data = {
            "fold_id": fr.fold_id,
            "best_params": fr.best_params,
            "n_trials": fr.n_trials,
            "train": {
                "trades": fr.train_metrics.total_trades,
                "win_rate": fr.train_metrics.win_rate,
                "sharpe": fr.train_metrics.sharpe_ratio,
                "max_dd": fr.train_metrics.max_drawdown_pct,
                "pnl_pct": fr.train_metrics.total_pnl_pct,
            },
            "val": {
                "trades": fr.val_metrics.total_trades,
                "win_rate": fr.val_metrics.win_rate,
                "sharpe": fr.val_metrics.sharpe_ratio,
                "max_dd": fr.val_metrics.max_drawdown_pct,
                "pnl_pct": fr.val_metrics.total_pnl_pct,
            },
            "test": {
                "trades": fr.test_metrics.total_trades,
                "win_rate": fr.test_metrics.win_rate,
                "sharpe": fr.test_metrics.sharpe_ratio,
                "max_dd": fr.test_metrics.max_drawdown_pct,
                "pnl_pct": fr.test_metrics.total_pnl_pct,
            },
        }
        data["folds"].append(fold_data)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    path.write_text(json.dumps(data, indent=2, default=convert))
    logger.info("WFV results saved: %s", path)


def _print_wfv_summary(result: WFVResult, blind_metrics: BacktestMetrics):
    """Print formatted WFV summary."""
    print()
    print("=" * 80)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 80)
    print()

    # Per-fold table
    print(f"{'Fold':>4} {'Trials':>6}  {'Train WR':>8} {'Val WR':>8} {'Test WR':>8}  "
          f"{'Train Sh':>8} {'Val Sh':>8} {'Test Sh':>8}  {'Test DD':>8}")
    print("-" * 80)

    for fr in result.fold_results:
        print(f"{fr.fold_id:>4} {fr.n_trials:>6}  "
              f"{fr.train_metrics.win_rate:>7.1f}% {fr.val_metrics.win_rate:>7.1f}% "
              f"{fr.test_metrics.win_rate:>7.1f}%  "
              f"{fr.train_metrics.sharpe_ratio:>8.2f} {fr.val_metrics.sharpe_ratio:>8.2f} "
              f"{fr.test_metrics.sharpe_ratio:>8.2f}  "
              f"{fr.test_metrics.max_drawdown_pct:>7.2f}%")

    print("-" * 80)

    agg = result.aggregate_test_metrics
    print(f"\nAggregate Test Metrics:")
    print(f"  Avg Win Rate:      {agg['avg_win_rate']:.1f}%")
    print(f"  Avg Sharpe:        {agg['avg_sharpe']:.2f}")
    print(f"  Avg Max DD:        {agg['avg_max_dd']:.2f}%")
    print(f"  Avg Profit Factor: {agg['avg_profit_factor']:.2f}")
    print(f"  Avg Trail Exit:    {agg['avg_trail_exit_pct']:.1f}%")
    print(f"  Total Trades:      {agg['total_trades']}")

    print(f"\nConsensus Parameters:")
    for k, v in result.consensus_params.items():
        print(f"  {k}: {v}")

    print(f"\nBlind Test ({BLIND_START.date()} → {BLIND_END.date()}):")
    print(f"  Trades:       {blind_metrics.total_trades}")
    print(f"  Win Rate:     {blind_metrics.win_rate:.1f}%")
    print(f"  PnL:          {blind_metrics.total_pnl_pct:+.1f}%")
    print(f"  Sharpe:       {blind_metrics.sharpe_ratio:.2f}")
    print(f"  Max DD:       {blind_metrics.max_drawdown_pct:.2f}%")
    print(f"  Profit Factor:{blind_metrics.profit_factor:.2f}")
    print(f"  Trail Exits:  {blind_metrics.trail_exit_pct:.1f}%")

    print()
    print("=" * 80)

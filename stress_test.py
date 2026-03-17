#!/usr/bin/env python3
"""
stress_test.py — Rigorous stress testing of blind test results.

1. Slippage & Commission Injection (6 bps per trade)
2. Monte Carlo Simulation (5,000 iterations, sequence risk)
3. Sensitivity Check (remove top 5% outlier winners)
4. Export stress_test_results.csv with P5/P50/P95 equity curves
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

np.random.seed(42)

SLIPPAGE_BPS = 6  # basis points per trade (0.06%)
MC_ITERATIONS = 5000
INITIAL_CAPITAL = 10_000.0
OUTLIER_PCT = 0.05  # top 5%


# ═══════════════════════════════════════════════════════════════════════════════
# Load trades
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("VARANUS NEO-FLOW — STRESS TEST")
print("=" * 80)
print()

df = pd.read_csv("blind_test_trades.csv")
print(f"Loaded {len(df)} trades from blind_test_trades.csv")
print(f"Date range: {df['entry_ts'].min()} → {df['exit_ts'].max()}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Slippage & Commission Injection
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("1. SLIPPAGE & COMMISSION INJECTION (6 bps per trade)")
print("=" * 80)
print()

# Original metrics
orig_pnl_pct = df["pnl_pct"].values
orig_pnl_usd = df["pnl_usd"].values
orig_total_pnl = orig_pnl_usd.sum()
orig_winners = (orig_pnl_usd > 0).sum()
orig_losers = (orig_pnl_usd <= 0).sum()
orig_wr = orig_winners / len(df) * 100
orig_gross_profit = orig_pnl_usd[orig_pnl_usd > 0].sum()
orig_gross_loss = abs(orig_pnl_usd[orig_pnl_usd <= 0].sum())
orig_pf = orig_gross_profit / orig_gross_loss if orig_gross_loss > 0 else float("inf")

# Apply slippage: subtract 0.06% from each trade's pnl_pct
# This accounts for entry + exit combined (taker fee + slippage)
slippage_pct = SLIPPAGE_BPS / 100.0  # 0.06%

# Adjusted PnL: reduce pnl_pct by slippage, recalculate pnl_usd
adj_pnl_pct = orig_pnl_pct - slippage_pct
adj_pnl_usd = adj_pnl_pct / 100.0 * df["position_usd"].values

adj_total_pnl = adj_pnl_usd.sum()
adj_winners = (adj_pnl_usd > 0).sum()
adj_losers = (adj_pnl_usd <= 0).sum()
adj_wr = adj_winners / len(df) * 100
adj_gross_profit = adj_pnl_usd[adj_pnl_usd > 0].sum()
adj_gross_loss = abs(adj_pnl_usd[adj_pnl_usd <= 0].sum())
adj_pf = adj_gross_profit / adj_gross_loss if adj_gross_loss > 0 else float("inf")

# Sharpe ratio (annualized from trade returns)
# Assume avg ~3 bars per trade at 1h = 3h avg holding
# Trades per year estimate: 1097 trades in ~4.5 months → ~2,927/yr
trades_per_year = len(df) / 4.5 * 12
orig_sharpe = (orig_pnl_pct.mean() / orig_pnl_pct.std()) * np.sqrt(trades_per_year) if orig_pnl_pct.std() > 0 else 0
adj_sharpe = (adj_pnl_pct.mean() / adj_pnl_pct.std()) * np.sqrt(trades_per_year) if adj_pnl_pct.std() > 0 else 0

# Max drawdown on equity curve
def compute_equity_and_dd(pnl_usd_arr, capital=INITIAL_CAPITAL):
    equity = np.empty(len(pnl_usd_arr) + 1)
    equity[0] = capital
    equity[1:] = capital + np.cumsum(pnl_usd_arr)
    peak = np.maximum.accumulate(equity)
    dd_pct = (equity - peak) / peak * 100
    max_dd = dd_pct.min()
    return equity, max_dd

orig_equity, orig_max_dd = compute_equity_and_dd(orig_pnl_usd)
adj_equity, adj_max_dd = compute_equity_and_dd(adj_pnl_usd)

print(f"{'Metric':<25} {'Original':>15} {'Adjusted (6bps)':>15} {'Delta':>12}")
print("-" * 70)
print(f"{'Total Trades':<25} {len(df):>15} {len(df):>15} {'—':>12}")
print(f"{'Winners / Losers':<25} {f'{orig_winners}W / {orig_losers}L':>15} {f'{adj_winners}W / {adj_losers}L':>15} {f'{adj_winners - orig_winners:+d}W':>12}")
print(f"{'Win Rate':<25} {f'{orig_wr:.1f}%':>15} {f'{adj_wr:.1f}%':>15} {f'{adj_wr - orig_wr:+.1f}%':>12}")
print(f"{'Net PnL ($)':<25} {f'${orig_total_pnl:,.0f}':>15} {f'${adj_total_pnl:,.0f}':>15} {f'${adj_total_pnl - orig_total_pnl:+,.0f}':>12}")
print(f"{'Net PnL (%)':<25} {f'{orig_total_pnl/INITIAL_CAPITAL*100:+.1f}%':>15} {f'{adj_total_pnl/INITIAL_CAPITAL*100:+.1f}%':>15} {f'{(adj_total_pnl-orig_total_pnl)/INITIAL_CAPITAL*100:+.1f}%':>12}")
print(f"{'Profit Factor':<25} {f'{orig_pf:.2f}':>15} {f'{adj_pf:.2f}':>15} {f'{adj_pf - orig_pf:+.2f}':>12}")
print(f"{'Sharpe Ratio':<25} {f'{orig_sharpe:.2f}':>15} {f'{adj_sharpe:.2f}':>15} {f'{adj_sharpe - orig_sharpe:+.2f}':>12}")
print(f"{'Max Drawdown':<25} {f'{orig_max_dd:.2f}%':>15} {f'{adj_max_dd:.2f}%':>15} {f'{adj_max_dd - orig_max_dd:+.2f}%':>12}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Monte Carlo Simulation (Sequence Risk)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print(f"2. MONTE CARLO SIMULATION ({MC_ITERATIONS:,} iterations)")
print("=" * 80)
print()

n_trades = len(adj_pnl_usd)
mc_max_dd = np.empty(MC_ITERATIONS)
mc_final_equity = np.empty(MC_ITERATIONS)
mc_equity_curves = np.empty((MC_ITERATIONS, n_trades + 1))

print(f"Running {MC_ITERATIONS:,} simulations with {n_trades} slippage-adjusted trades...")

for i in range(MC_ITERATIONS):
    shuffled = np.random.permutation(adj_pnl_usd)
    equity = np.empty(n_trades + 1)
    equity[0] = INITIAL_CAPITAL
    equity[1:] = INITIAL_CAPITAL + np.cumsum(shuffled)
    mc_equity_curves[i] = equity

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    mc_max_dd[i] = dd.min()
    mc_final_equity[i] = equity[-1]

# Percentiles
dd_p5 = np.percentile(mc_max_dd, 5)     # 95% CI worst-case DD
dd_p25 = np.percentile(mc_max_dd, 25)
dd_p50 = np.percentile(mc_max_dd, 50)
dd_p75 = np.percentile(mc_max_dd, 75)
dd_p95 = np.percentile(mc_max_dd, 95)

eq_p5 = np.percentile(mc_final_equity, 5)
eq_p50 = np.percentile(mc_final_equity, 50)
eq_p95 = np.percentile(mc_final_equity, 95)

# Find the equity curves closest to each percentile for export
p5_idx = np.argmin(np.abs(mc_final_equity - np.percentile(mc_final_equity, 5)))
p50_idx = np.argmin(np.abs(mc_final_equity - np.percentile(mc_final_equity, 50)))
p95_idx = np.argmin(np.abs(mc_final_equity - np.percentile(mc_final_equity, 95)))

print(f"  Simulations complete.")
print()
print(f"  Max Drawdown Distribution:")
print(f"    {'5th percentile (worst):':<30} {dd_p5:.2f}%")
print(f"    {'25th percentile:':<30} {dd_p25:.2f}%")
print(f"    {'50th percentile (median):':<30} {dd_p50:.2f}%")
print(f"    {'75th percentile:':<30} {dd_p75:.2f}%")
print(f"    {'95th percentile (best):':<30} {dd_p95:.2f}%")
print()
print(f"  95% Confidence Interval MaxDD: {dd_p5:.2f}%")
print(f"  (95% of simulations had MaxDD better than {dd_p5:.2f}%)")
print()
print(f"  Final Equity Distribution:")
print(f"    {'5th percentile (worst):':<30} ${eq_p5:,.0f}  ({(eq_p5/INITIAL_CAPITAL - 1)*100:+.0f}%)")
print(f"    {'50th percentile (median):':<30} ${eq_p50:,.0f}  ({(eq_p50/INITIAL_CAPITAL - 1)*100:+.0f}%)")
print(f"    {'95th percentile (best):':<30} ${eq_p95:,.0f}  ({(eq_p95/INITIAL_CAPITAL - 1)*100:+.0f}%)")
print()

# Probability of losing money
prob_loss = (mc_final_equity < INITIAL_CAPITAL).sum() / MC_ITERATIONS * 100
print(f"  Probability of net loss: {prob_loss:.2f}%")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Sensitivity Check (Outlier Removal)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print(f"3. SENSITIVITY CHECK (Remove Top {OUTLIER_PCT*100:.0f}% Winners)")
print("=" * 80)
print()

# Use slippage-adjusted PnL
n_remove = int(np.ceil(len(df) * OUTLIER_PCT))
sorted_indices = np.argsort(adj_pnl_usd)[::-1]  # descending by PnL
outlier_indices = sorted_indices[:n_remove]
keep_indices = sorted_indices[n_remove:]

outlier_pnl = adj_pnl_usd[outlier_indices]
keep_pnl = adj_pnl_usd[keep_indices]

print(f"  Total trades: {len(df)}")
print(f"  Removed: {n_remove} trades (top {OUTLIER_PCT*100:.0f}% winners)")
print(f"  Remaining: {len(keep_pnl)} trades")
print()

# Outlier stats
print(f"  Removed outlier stats:")
print(f"    Total PnL removed:   ${outlier_pnl.sum():,.0f}")
print(f"    Avg PnL per outlier: ${outlier_pnl.mean():,.0f}")
print(f"    Min outlier PnL:     ${outlier_pnl.min():,.0f}")
print(f"    Max outlier PnL:     ${outlier_pnl.max():,.0f}")
print()

# Remaining trade metrics
keep_winners = (keep_pnl > 0).sum()
keep_losers = (keep_pnl <= 0).sum()
keep_wr = keep_winners / len(keep_pnl) * 100
keep_total = keep_pnl.sum()
keep_gross_profit = keep_pnl[keep_pnl > 0].sum()
keep_gross_loss = abs(keep_pnl[keep_pnl <= 0].sum())
keep_pf = keep_gross_profit / keep_gross_loss if keep_gross_loss > 0 else float("inf")

# Rebuild equity in original order minus outliers
keep_mask = np.ones(len(df), dtype=bool)
keep_mask[outlier_indices] = False
keep_pnl_ordered = adj_pnl_usd[keep_mask]
keep_equity, keep_max_dd = compute_equity_and_dd(keep_pnl_ordered)

print(f"  Remaining trade metrics:")
print(f"    Win Rate:       {keep_wr:.1f}%  ({keep_winners}W / {keep_losers}L)")
print(f"    Net PnL:        ${keep_total:,.0f}  ({keep_total/INITIAL_CAPITAL*100:+.1f}%)")
print(f"    Profit Factor:  {keep_pf:.2f}")
print(f"    Max Drawdown:   {keep_max_dd:.2f}%")
print()

profitable_without_outliers = keep_total > 0
print(f"  VERDICT: Strategy {'REMAINS PROFITABLE' if profitable_without_outliers else 'NOT PROFITABLE'} without top {OUTLIER_PCT*100:.0f}% outliers")
print(f"           PnL retention: {keep_total/adj_total_pnl*100:.1f}% of slippage-adjusted total")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("STRESS-ADJUSTED SUMMARY")
print("=" * 80)
print()
print(f"  {'Metric':<35} {'Value':>15}")
print(f"  {'-'*50}")
print(f"  {'Adjusted Win Rate (6bps slip)':<35} {adj_wr:>14.1f}%")
print(f"  {'Adjusted Net PnL':<35} {f'${adj_total_pnl:,.0f}':>15}")
print(f"  {'Adjusted Net PnL %':<35} {f'{adj_total_pnl/INITIAL_CAPITAL*100:+.1f}%':>15}")
print(f"  {'Adjusted Profit Factor':<35} {adj_pf:>15.2f}")
print(f"  {'Adjusted Sharpe Ratio':<35} {adj_sharpe:>15.2f}")
print(f"  {'Adjusted MaxDD (sequential)':<35} {f'{adj_max_dd:.2f}%':>15}")
print(f"  {'MC 95% CI MaxDD (sequence risk)':<35} {f'{dd_p5:.2f}%':>15}")
print(f"  {'MC Median Final Equity':<35} {f'${eq_p50:,.0f}':>15}")
print(f"  {'MC Worst Case Final Equity (P5)':<35} {f'${eq_p5:,.0f}':>15}")
print(f"  {'Probability of Loss':<35} {f'{prob_loss:.2f}%':>15}")
print(f"  {'Profitable w/o Top 5% Outliers':<35} {'YES' if profitable_without_outliers else 'NO':>15}")
print(f"  {'PnL Retention (no outliers)':<35} {f'{keep_total/adj_total_pnl*100:.1f}%':>15}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Export CSV
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("EXPORTING stress_test_results.csv")
print("=" * 80)
print()

# Build export dataframe with P5, P50, P95 equity curves
export_df = pd.DataFrame({
    "trade_index": range(n_trades + 1),
    "equity_p5_worst": mc_equity_curves[p5_idx],
    "equity_p50_median": mc_equity_curves[p50_idx],
    "equity_p95_best": mc_equity_curves[p95_idx],
    "equity_sequential_adjusted": adj_equity,
})

# Add summary rows as metadata at the end
summary_data = {
    "metric": [
        "slippage_bps", "mc_iterations", "initial_capital",
        "original_net_pnl", "adjusted_net_pnl", "pnl_impact",
        "original_win_rate", "adjusted_win_rate",
        "original_profit_factor", "adjusted_profit_factor",
        "original_sharpe", "adjusted_sharpe",
        "original_max_dd", "adjusted_max_dd_sequential",
        "mc_max_dd_p5_95ci", "mc_max_dd_p50_median",
        "mc_final_equity_p5", "mc_final_equity_p50", "mc_final_equity_p95",
        "probability_of_loss",
        "outlier_pct_removed", "pnl_without_outliers", "pnl_retention_pct",
        "profitable_without_outliers",
    ],
    "value": [
        SLIPPAGE_BPS, MC_ITERATIONS, INITIAL_CAPITAL,
        round(orig_total_pnl, 2), round(adj_total_pnl, 2), round(adj_total_pnl - orig_total_pnl, 2),
        round(orig_wr, 2), round(adj_wr, 2),
        round(orig_pf, 4), round(adj_pf, 4),
        round(orig_sharpe, 4), round(adj_sharpe, 4),
        round(orig_max_dd, 4), round(adj_max_dd, 4),
        round(dd_p5, 4), round(dd_p50, 4),
        round(eq_p5, 2), round(eq_p50, 2), round(eq_p95, 2),
        round(prob_loss, 4),
        OUTLIER_PCT * 100, round(keep_total, 2), round(keep_total / adj_total_pnl * 100, 2),
        1 if profitable_without_outliers else 0,
    ],
}
summary_df = pd.DataFrame(summary_data)

# Save equity curves
export_df.to_csv("stress_test_results.csv", index=False)
print(f"  Equity curves saved: stress_test_results.csv ({len(export_df)} rows)")

# Save summary separately for easy parsing
summary_df.to_csv("stress_test_summary.csv", index=False)
print(f"  Summary metrics saved: stress_test_summary.csv ({len(summary_df)} rows)")

print()
print("=" * 80)
print("STRESS TEST COMPLETE")
print("=" * 80)

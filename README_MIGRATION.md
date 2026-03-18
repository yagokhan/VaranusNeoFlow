# Varanus NeoFlow — Migration & Optimization Guide

## Overview

This guide explains how to set up the project on a new machine and run the
high-performance Walk-Forward Validation (WFV) optimization using the
bias-free engine.

## Prerequisites

- Python 3.10+
- 8+ CPU cores recommended (optimization parallelizes across folds)
- ~8 GB RAM (each fold loads its own copy of the dataset)
- ~20 GB disk for historical OHLCV data

## 1. Clone and Install

```bash
git clone git@github.com:yagokhan/VaranusNeoFlow.git
cd VaranusNeoFlow
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Fetch Historical Data

The data fetcher downloads Binance Futures klines for all 15 Tier-2 assets
across 4 timeframes (5m, 30m, 1h, 4h). Data is stored as Parquet files
under `data/`.

```bash
python data_fetcher.py
```

This takes 30–60 minutes depending on connection speed. The date range
covers 2023-01-01 to present.

**Assets:** ADA, AVAX, LINK, DOT, TRX, SOL, ATOM, NEAR, ALGO, UNI, ICP,
HBAR, SAND, MANA, THETA

**Timeframes:** 5m, 30m, 1h, 4h

## 3. Run Optimization

The optimizer runs 8-fold WFV with Optuna (TPE sampler). Each fold gets its
own SQLite database so runs are resumable.

```bash
# Full run — all cores, 200 trials per fold (recommended):
python run_optimize.py --n-trials 200

# Quick sanity check — 50 trials, 2 cores:
python run_optimize.py --n-trials 50 --n-jobs 2

# Resume an interrupted run:
python run_optimize.py --n-trials 200 --resume
```

### What it does

1. **Train** (40% of fold): Optuna searches the parameter space
2. **Validate** (30%): Best trial scored on unseen data
3. **Test** (30%): Final out-of-sample evaluation
4. 24-bar embargo between periods to prevent leakage
5. **Consensus**: Median of per-fold best params
6. **Blind test**: Runs 2025-11-01 → 2026-03-18 with consensus params

### Output

- `optuna_studies/fold_N.db` — Resumable Optuna databases
- `wfv_results_v2.json` — Full results: per-fold metrics, consensus params,
  blind test performance
- Console summary with per-fold table and blind test metrics

### Parameter Search Space

| Parameter | Range |
|-----------|-------|
| min_pearson_r | 0.70 – 0.95 |
| min_pvt_r | 0.60 – 0.95 |
| combined_gate | 0.65 – 0.90 |
| hard_sl_mult | 1.5 – 4.0 |
| trail_buffer | 0.3 – 1.0 |
| exhaust_r | 0.30 – 0.55 |
| pos_frac | 0.03 – 0.10 |

## 4. Apply New Consensus Params

After optimization completes, update the consensus params in:

- `wfv_results.json` (read by the live bot)
- `live_bot.py` fallback defaults (optional)

```bash
# Check results:
cat wfv_results_v2.json | python -m json.tool
```

## 5. Run the Live Bot

```bash
python live_bot.py --interval 60 --capital 1000 --pos-frac 0.05
```

The bot scans at HH:00:01 UTC using closed-bar logic to match backtest
semantics. SYNC_CHECK logs confirm data alignment every cycle.

## Key Architecture Notes

- **Closed-bar logic**: `scan_ns - 1` excludes the just-opened bar from
  indicator calculations. For 4h HTF: `scan_ns - 4h` ensures only fully
  closed 4h bars.
- **Entry at next bar open**: After a signal fires, entry price is the next
  bar's open (no look-ahead).
- **7-day rolling window**: Scan uses the last 7 days of bars per timeframe.
- **Sub-bar position updates**: Positions are checked on 5m bars for
  stop-loss and trail updates.

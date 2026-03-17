#!/usr/bin/env python3
"""
dashboard.py — Real-time Streamlit dashboard for Varanus Neo-Flow.

Reads from:
  - live_trades.csv (or blind_test_trades.csv as fallback)
  - Live scanner output for signal heatmap
  - wfv_results.json for consensus params

Run:
    cd /home/yagokhan/VaranusNeoFlow
    streamlit run dashboard.py --server.port 8501
"""

import sys
import os
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent
TRADES_FILE = BASE_DIR / "live_trades.csv"
TRADES_FALLBACK = BASE_DIR / "blind_test_trades.csv"
WFV_FILE = BASE_DIR / "wfv_results.json"
SCAN_LOG_FILE = BASE_DIR / "logs" / "scan_latest.json"
DATA_DIR = BASE_DIR / "data"
INITIAL_CAPITAL = 10_000.0

ASSETS = [
    "ADA", "AVAX", "LINK", "DOT", "TRX",
    "SOL", "ATOM", "NEAR", "ALGO", "UNI",
    "ICP", "HBAR", "SAND", "MANA", "THETA",
]
TF_LIST = ["5m", "30m", "1h"]

# ═══════════════════════════════════════════════════════════════════════════════
# Page setup
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Varanus Neo-Flow Dashboard",
    page_icon="🦎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Auto-refresh every 60 seconds
st.markdown(
    """<meta http-equiv="refresh" content="60">""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=55)
def load_trades():
    """Load trades from live CSV or fallback to blind test."""
    path = TRADES_FILE if TRADES_FILE.exists() else TRADES_FALLBACK
    if not path.exists():
        return pd.DataFrame(), str(path), False
    df = pd.read_csv(path)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")
    is_live = path == TRADES_FILE
    return df, str(path.name), is_live


@st.cache_data(ttl=55)
def load_consensus_params():
    """Load consensus params from WFV results."""
    if WFV_FILE.exists():
        with open(WFV_FILE) as f:
            data = json.load(f)
        return data.get("consensus_params", {}), data.get("blind_test", {})
    return {}, {}


@st.cache_data(ttl=55)
def load_scan_data():
    """Load latest scan results if available."""
    if SCAN_LOG_FILE.exists():
        with open(SCAN_LOG_FILE) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=55)
def run_live_scan():
    """Run a live scan of all 15 assets for signal heatmap data."""
    try:
        from backtest.data_loader import load_all_assets
        from neo_flow.adaptive_engine import (
            find_best_regression, compute_pvt_regression, get_htf_bias,
            trim_to_7d, SCAN_TIMEFRAMES, BARS_7D,
        )

        all_data = load_all_assets()

        rows = []
        for asset in ASSETS:
            asset_data = all_data.get(asset)
            if asset_data is None:
                continue

            # Build latest DataFrames for each TF
            for tf in TF_LIST:
                ad = asset_data.get(tf)
                if ad is None:
                    continue

                n = BARS_7D[tf]
                if len(ad.close) < 20:
                    continue

                close = ad.close[-n:] if len(ad.close) > n else ad.close
                log_src = np.log(close.astype(np.float64)[-min(200, len(close)):][::-1])

                # Find best regression for this TF
                from neo_flow.adaptive_engine import scan_all_periods, calc_log_regression
                results = scan_all_periods(close, tf)
                if not results:
                    continue

                best = max(results, key=lambda r: abs(r.pearson_r))
                abs_r = abs(best.pearson_r)
                direction = "LONG" if best.slope < 0 else "SHORT"

                # PVT
                pvt_arr = ad.pvt[-n:] if len(ad.pvt) > n else ad.pvt
                if len(pvt_arr) >= best.period:
                    from neo_flow.adaptive_engine import calc_linear_regression
                    _, pvt_r, pvt_slope, _ = calc_linear_regression(pvt_arr, best.period)
                    pvt_abs_r = abs(pvt_r)
                else:
                    pvt_abs_r = 0.0

                rows.append({
                    "asset": asset,
                    "tf": tf,
                    "pearson_r": round(abs_r, 4),
                    "pvt_r": round(pvt_abs_r, 4),
                    "direction": direction,
                    "period": best.period,
                    "midline": round(best.midline, 6),
                })

            # HTF bias
            ad_4h = asset_data.get("4h")
            if ad_4h is not None:
                from neo_flow.adaptive_engine import get_htf_bias as _htf
                n4h = BARS_7D["4h"]
                df_4h = pd.DataFrame({
                    "open": ad_4h.open_[-200:],
                    "high": ad_4h.high[-200:],
                    "low": ad_4h.low[-200:],
                    "close": ad_4h.close[-200:],
                    "volume": ad_4h.volume[-200:],
                })
                bias = _htf(df_4h)
                for r in rows:
                    if r["asset"] == asset and "htf_bias" not in r:
                        r["htf_bias"] = {1: "BULL", -1: "BEAR", 0: "NEUTRAL"}[bias]

        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        return None


def compute_metrics(df):
    """Compute trading metrics from a trades DataFrame."""
    if df.empty:
        return {}

    closed = df[df["exit_ts"].notna()].copy()
    if closed.empty:
        return {}

    total = len(closed)
    winners = (closed["pnl_usd"] > 0).sum()
    losers = (closed["pnl_usd"] <= 0).sum()
    wr = winners / total * 100 if total > 0 else 0

    total_pnl = closed["pnl_usd"].sum()
    gross_profit = closed.loc[closed["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss = abs(closed.loc[closed["pnl_usd"] <= 0, "pnl_usd"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Equity curve and drawdown
    equity = INITIAL_CAPITAL + closed["pnl_usd"].cumsum()
    equity_with_start = pd.concat([pd.Series([INITIAL_CAPITAL]), equity]).reset_index(drop=True)
    peak = equity_with_start.cummax()
    dd = (equity_with_start - peak) / peak * 100
    max_dd = dd.min()

    # Sharpe
    returns = closed["pnl_pct"].values
    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        trades_per_year = total / max(1, (closed["exit_ts"].max() - closed["entry_ts"].min()).days) * 365
        sharpe = returns.mean() / returns.std() * np.sqrt(trades_per_year)

    return {
        "total_trades": total,
        "winners": winners,
        "losers": losers,
        "win_rate": wr,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl / INITIAL_CAPITAL * 100,
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "avg_winner": closed.loc[closed["pnl_usd"] > 0, "pnl_pct"].mean() if winners > 0 else 0,
        "avg_loser": closed.loc[closed["pnl_usd"] <= 0, "pnl_pct"].mean() if losers > 0 else 0,
        "avg_duration": closed["duration_hours"].mean(),
        "equity": equity,
        "equity_with_start": equity_with_start,
        "drawdown": dd,
        "closed": closed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Load all data
# ═══════════════════════════════════════════════════════════════════════════════

df_trades, source_file, is_live = load_trades()
consensus, blind_summary = load_consensus_params()
m = compute_metrics(df_trades)

# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════

col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("🦎 Varanus Neo-Flow Dashboard")
with col_status:
    mode = "🟢 LIVE" if is_live else "🔵 BACKTEST"
    st.markdown(f"### {mode}")
    st.caption(f"Source: `{source_file}` | Refresh: 60s")
    st.caption(f"Last update: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Executive Summary — KPI Cards
# ═══════════════════════════════════════════════════════════════════════════════

if m:
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric(
            "Total PnL",
            f"${m['total_pnl']:,.0f}",
            f"{m['total_pnl_pct']:+.1f}%",
        )
    with c2:
        st.metric(
            "Win Rate",
            f"{m['win_rate']:.1f}%",
            f"{m['winners']}W / {m['losers']}L",
        )
    with c3:
        st.metric("Profit Factor", f"{m['profit_factor']:.2f}")
    with c4:
        st.metric(
            "Max Drawdown",
            f"{m['max_drawdown']:.2f}%",
            delta_color="inverse",
        )
    with c5:
        st.metric("Sharpe Ratio", f"{m['sharpe']:.2f}")
    with c6:
        st.metric(
            "Total Trades",
            f"{m['total_trades']}",
            f"Avg {m['avg_duration']:.1f}h",
        )

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Equity Curve + Drawdown
    # ═══════════════════════════════════════════════════════════════════════════

    st.subheader("Equity Curve & Drawdown")

    closed = m["closed"]
    eq_timestamps = closed["exit_ts"].values
    eq_values = m["equity"].values

    fig_eq = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    # Equity line
    fig_eq.add_trace(
        go.Scatter(
            x=eq_timestamps, y=eq_values,
            mode="lines", name="Equity",
            line=dict(color="#00C853", width=2),
            fill="tozeroy", fillcolor="rgba(0,200,83,0.1)",
        ),
        row=1, col=1,
    )
    fig_eq.add_hline(
        y=INITIAL_CAPITAL, line_dash="dash", line_color="gray",
        annotation_text=f"Start ${INITIAL_CAPITAL:,.0f}",
        row=1, col=1,
    )

    # Drawdown
    dd_values = m["drawdown"].values[1:]  # skip initial 0
    fig_eq.add_trace(
        go.Scatter(
            x=eq_timestamps, y=dd_values,
            mode="lines", name="Drawdown %",
            line=dict(color="#FF1744", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,23,68,0.15)",
        ),
        row=2, col=1,
    )

    fig_eq.update_layout(
        height=500, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis2_title="Date",
    )
    fig_eq.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig_eq.update_yaxes(title_text="DD %", row=2, col=1)

    st.plotly_chart(fig_eq, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Active Positions + Recent Trades
    # ═══════════════════════════════════════════════════════════════════════════

    col_active, col_recent = st.columns(2)

    with col_active:
        st.subheader("Open Positions")
        open_trades = df_trades[df_trades["exit_ts"].isna()]
        if not open_trades.empty:
            display_cols = ["asset", "direction", "entry_price", "best_tf",
                           "best_period", "confidence", "leverage", "position_usd", "hard_sl"]
            st.dataframe(
                open_trades[display_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No open positions" + (" (backtest mode — all trades closed)" if not is_live else ""))

    with col_recent:
        st.subheader("Last 15 Trades")
        recent = closed.tail(15).sort_values("exit_ts", ascending=False)
        display_df = recent[["asset", "direction", "best_tf", "entry_price",
                            "exit_price", "pnl_pct", "pnl_usd", "exit_reason",
                            "duration_hours"]].copy()
        display_df["pnl_pct"] = display_df["pnl_pct"].apply(lambda x: f"{x:+.2f}%")
        display_df["pnl_usd"] = display_df["pnl_usd"].apply(lambda x: f"${x:+,.0f}")
        display_df["duration_hours"] = display_df["duration_hours"].apply(lambda x: f"{x:.1f}h")
        display_df.columns = ["Asset", "Dir", "TF", "Entry", "Exit", "PnL%", "PnL$", "Exit Reason", "Dur"]
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Signal Analysis Heatmap
    # ═══════════════════════════════════════════════════════════════════════════

    st.subheader("Signal Analysis — Pearson R & PVT Heatmap (Latest Data)")

    with st.spinner("Scanning 15 assets across 3 timeframes..."):
        scan_df = run_live_scan()

    if scan_df is not None and not scan_df.empty:
        tab_r, tab_pvt = st.tabs(["Pearson |R| Heatmap", "PVT |R| Heatmap"])

        with tab_r:
            pivot_r = scan_df.pivot_table(
                index="asset", columns="tf", values="pearson_r", aggfunc="max"
            ).reindex(columns=TF_LIST)

            fig_r = go.Figure(data=go.Heatmap(
                z=pivot_r.values,
                x=pivot_r.columns.tolist(),
                y=pivot_r.index.tolist(),
                colorscale=[[0, "#1a1a2e"], [0.5, "#e94560"], [0.8, "#f5a623"], [1, "#00C853"]],
                zmin=0.5, zmax=1.0,
                text=np.round(pivot_r.values, 3),
                texttemplate="%{text}",
                textfont=dict(size=12),
                colorbar=dict(title="|R|"),
            ))
            fig_r.update_layout(
                height=450, margin=dict(l=0, r=0, t=30, b=0),
                title="Best |Pearson R| per Asset × Timeframe",
                yaxis=dict(categoryorder="array", categoryarray=ASSETS[::-1]),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        with tab_pvt:
            pivot_pvt = scan_df.pivot_table(
                index="asset", columns="tf", values="pvt_r", aggfunc="max"
            ).reindex(columns=TF_LIST)

            fig_pvt = go.Figure(data=go.Heatmap(
                z=pivot_pvt.values,
                x=pivot_pvt.columns.tolist(),
                y=pivot_pvt.index.tolist(),
                colorscale=[[0, "#1a1a2e"], [0.5, "#e94560"], [0.8, "#f5a623"], [1, "#0077b6"]],
                zmin=0.3, zmax=1.0,
                text=np.round(pivot_pvt.values, 3),
                texttemplate="%{text}",
                textfont=dict(size=12),
                colorbar=dict(title="PVT |R|"),
            ))
            fig_pvt.update_layout(
                height=450, margin=dict(l=0, r=0, t=30, b=0),
                title="Best PVT |R| per Asset × Timeframe",
                yaxis=dict(categoryorder="array", categoryarray=ASSETS[::-1]),
            )
            st.plotly_chart(fig_pvt, use_container_width=True)

        # Direction + HTF bias table
        with st.expander("Detailed Signal Table"):
            st.dataframe(
                scan_df.sort_values(["asset", "tf"]).reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )
    else:
        st.warning("Signal scan unavailable — data files may not be loaded.")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Exit Distribution + Per-TF Breakdown
    # ═══════════════════════════════════════════════════════════════════════════

    col_exit, col_tf = st.columns(2)

    with col_exit:
        st.subheader("Exit Distribution")

        exit_counts = closed["exit_reason"].value_counts()
        exit_labels = {
            "ADAPTIVE_TRAIL_HIT": "Adaptive Trail",
            "HARD_SL_HIT": "Hard SL",
            "TREND_EXHAUSTION": "Trend Exhaustion",
            "TIME_BARRIER": "Time Barrier",
            "END_OF_PERIOD": "End of Period",
        }
        exit_df = pd.DataFrame({
            "reason": [exit_labels.get(k, k) for k in exit_counts.index],
            "count": exit_counts.values,
            "pct": (exit_counts.values / exit_counts.sum() * 100),
        })

        colors = {
            "Adaptive Trail": "#00C853",
            "Hard SL": "#FF1744",
            "Trend Exhaustion": "#FF9100",
            "Time Barrier": "#651FFF",
            "End of Period": "#78909C",
        }
        exit_df["color"] = exit_df["reason"].map(colors).fillna("#999999")

        fig_exit = go.Figure(data=[go.Pie(
            labels=exit_df["reason"],
            values=exit_df["count"],
            hole=0.45,
            marker=dict(colors=exit_df["color"].tolist()),
            textinfo="label+percent",
            textfont=dict(size=13),
        )])
        fig_exit.update_layout(
            height=350, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_exit, use_container_width=True)

        # Exit stats table
        exit_df["avg_pnl"] = [
            closed.loc[closed["exit_reason"] == k, "pnl_pct"].mean()
            for k in exit_counts.index
        ]
        exit_df["avg_pnl"] = exit_df["avg_pnl"].apply(lambda x: f"{x:+.2f}%")
        exit_df["pct"] = exit_df["pct"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(exit_df[["reason", "count", "pct", "avg_pnl"]].rename(
            columns={"reason": "Exit Type", "count": "Count", "pct": "Share", "avg_pnl": "Avg PnL"}
        ), use_container_width=True, hide_index=True)

    with col_tf:
        st.subheader("Performance by Timeframe")

        tf_groups = closed.groupby("best_tf").agg(
            trades=("pnl_usd", "count"),
            win_rate=("pnl_usd", lambda x: (x > 0).mean() * 100),
            total_pnl=("pnl_usd", "sum"),
            avg_pnl=("pnl_pct", "mean"),
            avg_conf=("confidence", "mean"),
        ).round(2)

        fig_tf = go.Figure()
        for tf in tf_groups.index:
            row = tf_groups.loc[tf]
            fig_tf.add_trace(go.Bar(
                x=[tf], y=[row["total_pnl"]],
                name=tf,
                text=f"WR:{row['win_rate']:.0f}% | {int(row['trades'])} trades",
                textposition="outside",
                marker_color={"5m": "#00C853", "30m": "#2196F3", "1h": "#FF9100"}.get(tf, "#999"),
            ))

        fig_tf.update_layout(
            height=300, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Total PnL ($)",
        )
        st.plotly_chart(fig_tf, use_container_width=True)

        # Direction breakdown
        st.markdown("**Long vs Short**")
        dir_groups = closed.groupby("direction").agg(
            trades=("pnl_usd", "count"),
            win_rate=("pnl_usd", lambda x: (x > 0).mean() * 100),
            total_pnl=("pnl_usd", "sum"),
            avg_pnl=("pnl_pct", "mean"),
        ).round(2)
        dir_groups.index.name = "Direction"
        dir_groups.columns = ["Trades", "Win Rate %", "Total PnL $", "Avg PnL %"]
        st.dataframe(dir_groups, use_container_width=True)

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. Monthly Performance + Per-Asset Breakdown
    # ═══════════════════════════════════════════════════════════════════════════

    col_monthly, col_asset = st.columns(2)

    with col_monthly:
        st.subheader("Monthly Returns")
        closed_m = closed.copy()
        closed_m["month"] = closed_m["exit_ts"].dt.to_period("M").astype(str)
        monthly = closed_m.groupby("month").agg(
            trades=("pnl_usd", "count"),
            pnl=("pnl_usd", "sum"),
            wr=("pnl_usd", lambda x: (x > 0).mean() * 100),
        ).round(2)

        fig_monthly = go.Figure(data=[go.Bar(
            x=monthly.index,
            y=monthly["pnl"],
            marker_color=["#00C853" if v > 0 else "#FF1744" for v in monthly["pnl"]],
            text=[f"${v:,.0f}<br>WR:{wr:.0f}%" for v, wr in zip(monthly["pnl"], monthly["wr"])],
            textposition="outside",
        )])
        fig_monthly.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="PnL ($)",
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col_asset:
        st.subheader("Top Assets by PnL")
        asset_perf = closed.groupby("asset").agg(
            trades=("pnl_usd", "count"),
            pnl=("pnl_usd", "sum"),
            wr=("pnl_usd", lambda x: (x > 0).mean() * 100),
        ).sort_values("pnl", ascending=True).round(2)

        fig_asset = go.Figure(data=[go.Bar(
            x=asset_perf["pnl"],
            y=asset_perf.index,
            orientation="h",
            marker_color=["#00C853" if v > 0 else "#FF1744" for v in asset_perf["pnl"]],
            text=[f"${v:,.0f} | WR:{wr:.0f}%" for v, wr in zip(asset_perf["pnl"], asset_perf["wr"])],
            textposition="outside",
        )])
        fig_asset.update_layout(
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="PnL ($)",
        )
        st.plotly_chart(fig_asset, use_container_width=True)

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. System Health
    # ═══════════════════════════════════════════════════════════════════════════

    st.subheader("System Health")

    h1, h2, h3, h4 = st.columns(4)

    with h1:
        last_trade = closed["exit_ts"].max()
        st.metric("Last Trade", last_trade.strftime("%Y-%m-%d %H:%M") if pd.notna(last_trade) else "N/A")

    with h2:
        # Check data freshness
        data_files = list(DATA_DIR.glob("*.parquet"))
        if data_files:
            newest = max(f.stat().st_mtime for f in data_files)
            age_hours = (datetime.now().timestamp() - newest) / 3600
            status = "🟢 Fresh" if age_hours < 24 else "🟡 Stale" if age_hours < 72 else "🔴 Old"
            st.metric("Data Window", status, f"{age_hours:.0f}h ago")
        else:
            st.metric("Data Window", "🔴 No data")

    with h3:
        if consensus:
            st.metric("Consensus min_R", f"{consensus.get('min_pearson_r', 'N/A')}")
        else:
            st.metric("Consensus Params", "Not loaded")

    with h4:
        n_parquet = len(list(DATA_DIR.glob("*.parquet"))) if DATA_DIR.exists() else 0
        expected = 60  # 15 assets × 4 TFs
        status = "🟢 Complete" if n_parquet >= expected else f"🟡 {n_parquet}/{expected}"
        st.metric("Parquet Files", status)

    # Consensus params expander
    if consensus:
        with st.expander("Consensus Parameters (from WFV)"):
            param_df = pd.DataFrame([
                {"Parameter": k, "Value": v} for k, v in consensus.items()
            ])
            st.dataframe(param_df, use_container_width=True, hide_index=True)

else:
    st.error("No trade data found. Place `live_trades.csv` or `blind_test_trades.csv` in the project directory.")

# ═══════════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
st.caption("Varanus Neo-Flow v1.0 | Auto-refresh: 60s | Dashboard powered by Streamlit + Plotly")

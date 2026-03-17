#!/usr/bin/env python3
"""
live_bot.py — Varanus Neo-Flow Live Trading Bot

Scans 15 Tier-2 assets every hour, opens/manages positions on Binance Futures,
sends Telegram notifications for all actions.

Usage:
    # Dry-run (scan + alerts, no real orders):
    python live_bot.py

    # Live trading (requires BINANCE_API_KEY/SECRET in .env):
    python live_bot.py --live

    # Single scan + exit:
    python live_bot.py --once

    # Custom scan interval:
    python live_bot.py --interval 60
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

STATE_FILE = BASE_DIR / "live_state.json"
TRADES_FILE = BASE_DIR / "live_trades.csv"
WFV_FILE = BASE_DIR / "wfv_results.json"

# Load .env
def _load_env():
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

# Load consensus params from WFV
def _load_consensus():
    if WFV_FILE.exists():
        with open(WFV_FILE) as f:
            data = json.load(f)
        return data.get("consensus_params", {})
    return {}

CONSENSUS = _load_consensus()

ASSETS = [
    "ADA", "AVAX", "LINK", "DOT", "TRX",
    "SOL", "ATOM", "NEAR", "ALGO", "UNI",
    "ICP", "HBAR", "SAND", "MANA", "THETA",
]

SYMBOLS = [f"{a}USDT" for a in ASSETS]

HIGH_VOL_ASSETS = {"ICP"}

MAX_CONCURRENT = 4
MAX_DRAWDOWN_PCT = -15.0

# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════

import logging.handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            LOGS_DIR / "live_bot.log", maxBytes=10_000_000, backupCount=5,
        ),
    ],
)
logger = logging.getLogger("live_bot")

# Silence noisy loggers
for name in ["urllib3", "ccxt", "httpx", "httpcore", "telegram"]:
    logging.getLogger(name).setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# Telegram
# ═══════════════════════════════════════════════════════════════════════════════

import requests as _requests

def tg_send(text: str, parse_mode: str = "HTML"):
    """Send a Telegram message. Non-blocking, swallows errors."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured — skipping notification")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        _requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": parse_mode,
        }, timeout=10)
    except Exception as e:
        logger.error("Telegram send failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# Live Data: Parquet Cache + Binance API for new bars
# ═══════════════════════════════════════════════════════════════════════════════

KLINES_URL = "https://api.binance.com/api/v3/klines"

# Global cache — loaded once from parquet, updated each cycle with new bars
_cached_data: dict | None = None


def fetch_recent_klines(symbol: str, interval: str, start_ms: int = 0, limit: int = 1000) -> pd.DataFrame:
    """Fetch klines from Binance public API, optionally from a start time."""
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms > 0:
            params["startTime"] = start_ms
        resp = _requests.get(KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.error("Failed to fetch %s %s: %s", symbol, interval, e)
        return pd.DataFrame()


def _load_cache() -> dict:
    """Load full historical data from parquet files (same as backtest engine)."""
    from backtest.data_loader import load_all_assets
    logger.info("Loading parquet cache (15 assets × 4 TFs)...")
    t0 = time.perf_counter()
    all_data = load_all_assets()
    elapsed = time.perf_counter() - t0
    total = sum(len(ad.close) for asset in all_data.values() for ad in asset.values())
    logger.info("Cache loaded in %.1fs — %d total bars", elapsed, total)
    return all_data


def _append_new_bars(all_data: dict) -> dict:
    """
    Fetch new bars from Binance since the last cached timestamp and append.
    Recomputes PVT on the extended arrays.
    """
    from backtest.data_loader import AssetData, _compute_pvt, TIMEFRAMES

    updated = 0
    for asset in ASSETS:
        asset_data = all_data.get(asset, {})
        symbol = f"{asset}USDT"

        for tf in TIMEFRAMES:
            ad = asset_data.get(tf)
            if ad is None:
                continue

            # Last cached timestamp → fetch from next bar
            last_ns = ad.timestamps[-1]
            last_ms = last_ns // 1_000_000  # ns to ms
            start_ms = last_ms + 1

            df_new = fetch_recent_klines(symbol, tf, start_ms=start_ms, limit=1000)
            if df_new.empty or len(df_new) == 0:
                continue

            # Convert new data to arrays
            new_ts = df_new.index.values.astype("datetime64[ns]").astype("int64")

            # Filter only bars strictly after last cached
            mask = new_ts > last_ns
            if not mask.any():
                continue

            new_close = df_new["close"].values[mask].astype(np.float64)
            new_high = df_new["high"].values[mask].astype(np.float64)
            new_low = df_new["low"].values[mask].astype(np.float64)
            new_open = df_new["open"].values[mask].astype(np.float64)
            new_vol = df_new["volume"].values[mask].astype(np.float64)
            new_ts = new_ts[mask]

            # Append to existing arrays
            combined_close = np.concatenate([ad.close, new_close])
            combined_high = np.concatenate([ad.high, new_high])
            combined_low = np.concatenate([ad.low, new_low])
            combined_open = np.concatenate([ad.open_, new_open])
            combined_vol = np.concatenate([ad.volume, new_vol])
            combined_ts = np.concatenate([ad.timestamps, new_ts])

            # Recompute PVT on full array
            combined_pvt = _compute_pvt(combined_close, combined_vol)

            asset_data[tf] = AssetData(
                close=combined_close,
                high=combined_high,
                low=combined_low,
                open_=combined_open,
                volume=combined_vol,
                timestamps=combined_ts,
                pvt=combined_pvt,
            )
            updated += len(new_close)

        all_data[asset] = asset_data
        time.sleep(0.05)  # Rate limit

    return all_data, updated


def get_live_data() -> dict:
    """
    Get full dataset: parquet cache + latest bars from Binance.
    Cache is loaded once on first call, then updated each cycle.
    """
    global _cached_data

    if _cached_data is None:
        _cached_data = _load_cache()

    _cached_data, new_bars = _append_new_bars(_cached_data)
    if new_bars > 0:
        logger.info("Appended %d new bars from Binance", new_bars)
    else:
        logger.info("Cache up to date — no new bars")

    return _cached_data


def fetch_ticker_price(symbol: str) -> float | None:
    """Get current price for a symbol."""
    try:
        resp = _requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": symbol}, timeout=10,
        )
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Binance Order Execution (via ccxt)
# ═══════════════════════════════════════════════════════════════════════════════

_exchange = None

def _get_exchange():
    """Lazy-init ccxt Binance Futures client."""
    global _exchange
    if _exchange is None:
        import ccxt
        _exchange = ccxt.binanceusdm({
            "apiKey": BINANCE_API_KEY,
            "secret": BINANCE_API_SECRET,
            "options": {"defaultType": "future"},
            "enableRateLimit": True,
        })
    return _exchange


def place_market_order(symbol: str, side: str, amount_usd: float, leverage: int) -> dict | None:
    """
    Place a market order on Binance Futures.

    Parameters
    ----------
    symbol : e.g. "ADAUSDT"
    side : "buy" or "sell"
    amount_usd : Notional value in USD
    leverage : 1-5

    Returns order info dict or None on failure.
    """
    try:
        ex = _get_exchange()
        # Set leverage
        ex.set_leverage(leverage, symbol)

        # Get current price to compute quantity
        ticker = ex.fetch_ticker(symbol)
        price = ticker["last"]
        quantity = amount_usd / price

        # Get market info for precision
        market = ex.market(symbol)
        quantity = ex.amount_to_precision(symbol, quantity)

        order = ex.create_market_order(symbol, side, float(quantity))
        logger.info("ORDER PLACED: %s %s %s qty=%s @ ~%.4f", side, symbol, leverage, quantity, price)
        return order
    except Exception as e:
        logger.error("ORDER FAILED: %s %s — %s", side, symbol, e)
        tg_send(f"⚠️ <b>ORDER FAILED</b>\n{side.upper()} {symbol}\nError: {e}")
        return None


def place_stop_loss(symbol: str, side: str, quantity: float, stop_price: float) -> dict | None:
    """Place a stop-market order as hard SL."""
    try:
        ex = _get_exchange()
        stop_price = float(ex.price_to_precision(symbol, stop_price))
        order = ex.create_order(
            symbol, "STOP_MARKET", side, quantity,
            params={"stopPrice": stop_price, "closePosition": True},
        )
        logger.info("STOP SET: %s %s @ %.4f", side, symbol, stop_price)
        return order
    except Exception as e:
        logger.error("STOP FAILED: %s %s — %s", side, symbol, e)
        return None


def close_position(symbol: str, side: str, quantity: float) -> dict | None:
    """Close a position with a market order."""
    try:
        ex = _get_exchange()
        close_side = "sell" if side == "buy" else "buy"
        quantity = float(ex.amount_to_precision(symbol, quantity))
        order = ex.create_market_order(symbol, close_side, quantity, params={"reduceOnly": True})
        logger.info("POSITION CLOSED: %s %s qty=%s", close_side, symbol, quantity)
        return order
    except Exception as e:
        logger.error("CLOSE FAILED: %s — %s", symbol, e)
        return None


def cancel_all_orders(symbol: str):
    """Cancel all open orders for a symbol."""
    try:
        ex = _get_exchange()
        ex.cancel_all_orders(symbol)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Position State Management
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LivePosition:
    """State of an open live position."""
    trade_id: int
    asset: str
    symbol: str
    direction: int       # +1 LONG, -1 SHORT
    entry_price: float
    entry_ts: str
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
    quantity: float = 0.0
    bars_held: int = 0
    peak_r: float = 0.0
    order_id: str = ""
    sl_order_id: str = ""


@dataclass
class BotState:
    """Persistent bot state."""
    positions: dict = field(default_factory=dict)  # asset -> LivePosition dict
    trade_counter: int = 0
    realized_pnl: float = 0.0
    initial_capital: float = 10_000.0
    peak_equity: float = 10_000.0
    circuit_breaker: bool = False
    last_scan_ts: str = ""
    total_scans: int = 0


def save_state(state: BotState):
    """Persist state to JSON."""
    data = {
        "trade_counter": state.trade_counter,
        "realized_pnl": state.realized_pnl,
        "initial_capital": state.initial_capital,
        "peak_equity": state.peak_equity,
        "circuit_breaker": state.circuit_breaker,
        "last_scan_ts": state.last_scan_ts,
        "total_scans": state.total_scans,
        "positions": {k: asdict(v) for k, v in state.positions.items()},
    }
    STATE_FILE.write_text(json.dumps(data, indent=2, default=str))


def load_state() -> BotState:
    """Load state from JSON."""
    state = BotState()
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            state.trade_counter = data.get("trade_counter", 0)
            state.realized_pnl = data.get("realized_pnl", 0.0)
            state.initial_capital = data.get("initial_capital", 10_000.0)
            state.peak_equity = data.get("peak_equity", 10_000.0)
            state.circuit_breaker = data.get("circuit_breaker", False)
            state.last_scan_ts = data.get("last_scan_ts", "")
            state.total_scans = data.get("total_scans", 0)
            for k, v in data.get("positions", {}).items():
                state.positions[k] = LivePosition(**v)
        except Exception as e:
            logger.error("Failed to load state: %s", e)
    return state


def append_trade_csv(trade: dict):
    """Append a closed trade to live_trades.csv."""
    file_exists = TRADES_FILE.exists()
    fieldnames = [
        "trade_id", "asset", "direction", "entry_ts", "exit_ts",
        "entry_price", "exit_price", "best_tf", "best_period",
        "confidence", "pvt_r", "leverage", "position_usd",
        "hard_sl", "exit_reason", "bars_held", "duration_hours",
        "pnl_pct", "pnl_usd", "peak_r",
    ]
    with open(TRADES_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade)


# ═══════════════════════════════════════════════════════════════════════════════
# Core Bot Logic
# ═══════════════════════════════════════════════════════════════════════════════

class LiveBot:
    """Main live trading bot."""

    def __init__(self, live_mode: bool = False, capital: float = 10_000.0):
        self.live_mode = live_mode
        self.state = load_state()
        self.state.initial_capital = capital

        # Apply consensus params
        self.params = {
            "min_pearson_r": CONSENSUS.get("min_pearson_r", 0.83),
            "min_pvt_r": CONSENSUS.get("min_pvt_r", 0.80),
            "combined_gate": CONSENSUS.get("combined_gate", 0.80),
            "hard_sl_mult": CONSENSUS.get("hard_sl_mult", 2.5),
            "trail_buffer": CONSENSUS.get("trail_buffer", 0.5),
            "exhaust_r": CONSENSUS.get("exhaust_r", 0.425),
            "pos_frac": CONSENSUS.get("pos_frac", 0.05),
        }

        logger.info("Bot initialized — mode=%s, capital=$%.0f",
                     "LIVE" if live_mode else "DRY-RUN", capital)
        logger.info("Consensus params: %s", self.params)

    def _apply_params(self):
        """Patch adaptive_engine globals with consensus params."""
        import neo_flow.adaptive_engine as ae
        ae.MIN_PEARSON_R = self.params["min_pearson_r"]
        ae.MIN_PVT_PEARSON_R = self.params["min_pvt_r"]
        ae.COMBINED_GATE_THRESHOLD = self.params["combined_gate"]
        ae.HARD_SL_ATR_MULT = self.params["hard_sl_mult"]
        ae.TRAIL_BUFFER_STD = self.params["trail_buffer"]
        ae.TREND_EXHAUST_R = self.params["exhaust_r"]

    def _check_circuit_breaker(self) -> bool:
        """Check if drawdown exceeds max. Returns True if tripped."""
        equity = self.state.initial_capital + self.state.realized_pnl
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        dd_pct = (equity - self.state.peak_equity) / self.state.peak_equity * 100
        if dd_pct < MAX_DRAWDOWN_PCT:
            self.state.circuit_breaker = True
            save_state(self.state)
            msg = (
                f"🔴 <b>CIRCUIT BREAKER TRIPPED</b>\n"
                f"Drawdown: {dd_pct:.2f}% (limit: {MAX_DRAWDOWN_PCT}%)\n"
                f"Equity: ${equity:,.0f}\n"
                f"All scanning halted. Manual reset required."
            )
            logger.critical(msg)
            tg_send(msg)
            return True
        return False

    def _check_exits(self, all_data: dict):
        """
        Check all open positions for exit conditions.

        Walks through ALL bars on the position's native TF since entry
        (or since last check), exactly like the backtest engine.
        A 5m position gets every 5m bar checked, not just the latest.
        """
        from neo_flow.adaptive_engine import (
            calc_log_regression, compute_trail_sl,
        )

        to_close = []

        for asset, pos in list(self.state.positions.items()):
            ad = all_data.get(asset, {}).get(pos.best_tf)
            if ad is None:
                continue

            # Find bars since entry
            entry_ns = int(pd.Timestamp(pos.entry_ts).value)
            # Find first bar index after entry
            start_idx = np.searchsorted(ad.timestamps, entry_ns, side="right")
            # Skip bars we already checked (bars_held tracks how many we processed)
            start_idx = start_idx + pos.bars_held

            if start_idx >= len(ad.timestamps):
                continue

            exit_reason = None
            exit_price = None
            exit_idx = None

            # Walk through every bar since last check
            for idx in range(start_idx, len(ad.timestamps)):
                pos.bars_held += 1

                bar_high = float(ad.high[idx])
                bar_low = float(ad.low[idx])
                bar_close = float(ad.close[idx])

                # Update regression + trailing stop
                close_arr = ad.close[:idx + 1]
                current_r = 0.0
                if len(close_arr) >= pos.best_period:
                    std_dev, pearson_r, slope, intercept = calc_log_regression(
                        close_arr, pos.best_period,
                    )
                    midline = np.exp(intercept)
                    pos.midline = midline
                    pos.std_dev = std_dev
                    pos.peak_r = max(pos.peak_r, abs(pearson_r))

                    pos.trail_sl = compute_trail_sl(
                        pos.direction, midline, std_dev,
                        pos.trail_sl, self.params["trail_buffer"],
                    )
                    current_r = abs(pearson_r)

                # Hard SL (touch-based)
                if pos.direction == 1 and bar_low <= pos.hard_sl:
                    exit_reason = "HARD_SL_HIT"
                    exit_price = pos.hard_sl
                elif pos.direction == -1 and bar_high >= pos.hard_sl:
                    exit_reason = "HARD_SL_HIT"
                    exit_price = pos.hard_sl

                # Adaptive trail (touch-based)
                if exit_reason is None:
                    if pos.direction == 1 and bar_low <= pos.trail_sl:
                        exit_reason = "ADAPTIVE_TRAIL_HIT"
                        exit_price = pos.trail_sl
                    elif pos.direction == -1 and bar_high >= pos.trail_sl:
                        exit_reason = "ADAPTIVE_TRAIL_HIT"
                        exit_price = pos.trail_sl

                # Trend exhaustion
                if exit_reason is None and current_r > 0 and current_r < self.params["exhaust_r"]:
                    exit_reason = "TREND_EXHAUSTION"
                    exit_price = bar_close

                # Time barrier
                if exit_reason is None and pos.bars_held >= 200:
                    exit_reason = "TIME_BARRIER"
                    exit_price = bar_close

                if exit_reason:
                    exit_idx = idx
                    break

            if exit_reason and exit_price is not None:
                logger.info(
                    "EXIT %s at bar %d/%d: %s @ $%.4f (trail_sl=$%.4f)",
                    asset, exit_idx, len(ad.timestamps), exit_reason, exit_price, pos.trail_sl,
                )
                to_close.append((asset, pos, exit_price, exit_reason))

        for asset, pos, exit_price, reason in to_close:
            self._close_trade(pos, exit_price, reason)

    def _close_trade(self, pos: LivePosition, exit_price: float, reason: str):
        """Close a position and record the trade."""
        now = datetime.now(timezone.utc)

        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * pos.direction * 100.0
        pnl_pct_lev = pnl_pct * pos.leverage
        pnl_usd = pnl_pct_lev / 100.0 * pos.position_usd

        self.state.realized_pnl += pnl_usd

        # Execute on exchange
        if self.live_mode and pos.quantity > 0:
            side = "buy" if pos.direction == 1 else "sell"
            cancel_all_orders(pos.symbol)
            close_position(pos.symbol, side, pos.quantity)

        # Calculate duration
        entry_dt = pd.Timestamp(pos.entry_ts)
        duration_hours = (now - entry_dt).total_seconds() / 3600

        # Record trade
        trade = {
            "trade_id": pos.trade_id,
            "asset": pos.asset,
            "direction": "LONG" if pos.direction == 1 else "SHORT",
            "entry_ts": pos.entry_ts,
            "exit_ts": now.isoformat(),
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "best_tf": pos.best_tf,
            "best_period": pos.best_period,
            "confidence": pos.confidence,
            "pvt_r": pos.pvt_r,
            "leverage": pos.leverage,
            "position_usd": pos.position_usd,
            "hard_sl": pos.hard_sl,
            "exit_reason": reason,
            "bars_held": pos.bars_held,
            "duration_hours": round(duration_hours, 1),
            "pnl_pct": round(pnl_pct_lev, 4),
            "pnl_usd": round(pnl_usd, 2),
            "peak_r": round(pos.peak_r, 4),
        }
        append_trade_csv(trade)

        # Remove position
        del self.state.positions[pos.asset]

        # Telegram notification
        emoji = "🟢" if pnl_usd >= 0 else "🔴"
        dir_str = "LONG" if pos.direction == 1 else "SHORT"
        msg = (
            f"{emoji} <b>TRADE CLOSED — {pos.asset}</b>\n"
            f"Direction: {dir_str} | TF: {pos.best_tf}\n"
            f"Entry: ${pos.entry_price:.4f} → Exit: ${exit_price:.4f}\n"
            f"PnL: <b>{pnl_pct_lev:+.2f}%</b> (${pnl_usd:+,.2f})\n"
            f"Reason: {reason}\n"
            f"Duration: {duration_hours:.1f}h | Bars: {pos.bars_held}"
        )
        logger.info("CLOSED %s %s: %s %.2f%% ($%.2f)", dir_str, pos.asset, reason, pnl_pct_lev, pnl_usd)
        tg_send(msg)

        save_state(self.state)

    def _scan_and_enter(self, all_data: dict, skip_assets: set = None):
        """Scan all assets and open positions for qualifying signals."""
        from neo_flow.adaptive_engine import (
            scan_asset, compute_atr, compute_hard_sl, get_leverage,
        )
        from backtest.data_loader import build_scan_dataframes, build_htf_dataframe

        skip_assets = skip_assets or set()

        if len(self.state.positions) >= MAX_CONCURRENT:
            logger.info("Max concurrent positions reached (%d) — skipping scan", MAX_CONCURRENT)
            return

        signals_found = []
        now = datetime.now(timezone.utc)

        for asset in ASSETS:
            if asset in self.state.positions:
                continue
            if asset in skip_assets:
                logger.info("Skipping %s — closed this cycle (cooldown)", asset)
                continue
            if len(self.state.positions) >= MAX_CONCURRENT:
                break

            asset_data = all_data.get(asset)
            if asset_data is None:
                continue

            # Build scan DataFrames using latest 1h timestamp
            ad_1h = asset_data.get("1h")
            if ad_1h is None or len(ad_1h.timestamps) == 0:
                continue

            current_ns = ad_1h.timestamps[-1]
            scan_dfs = build_scan_dataframes(asset_data, current_ns)
            df_4h = build_htf_dataframe(asset_data, current_ns)

            if not scan_dfs or df_4h is None:
                continue

            signal = scan_asset(asset, scan_dfs, df_4h)
            if signal is None:
                continue

            signals_found.append(signal)

            # Get current price for entry
            price = fetch_ticker_price(f"{asset}USDT")
            if price is None:
                price = signal.entry_price

            # Position sizing
            lev = get_leverage(signal.confidence)
            vol_scalar = 0.75 if asset in HIGH_VOL_ASSETS else 1.0
            equity = self.state.initial_capital + self.state.realized_pnl
            pos_usd = equity * self.params["pos_frac"] * lev * vol_scalar

            if pos_usd <= 0 or lev == 0:
                continue

            # Hard SL
            hard_sl = compute_hard_sl(price, signal.atr, signal.direction, self.params["hard_sl_mult"])

            # Initial trailing stop
            std_dev_price = signal.midline * signal.std_dev
            if signal.direction == 1:
                trail_sl = signal.midline - self.params["trail_buffer"] * std_dev_price
            else:
                trail_sl = signal.midline + self.params["trail_buffer"] * std_dev_price

            # Execute on exchange
            quantity = 0.0
            order_id = ""
            sl_order_id = ""
            symbol = f"{asset}USDT"

            if self.live_mode:
                side = "buy" if signal.direction == 1 else "sell"
                order = place_market_order(symbol, side, pos_usd, lev)
                if order is None:
                    continue
                order_id = str(order.get("id", ""))
                quantity = float(order.get("filled", 0) or order.get("amount", 0))
                price = float(order.get("average", price) or price)

                # Place hard SL
                sl_side = "sell" if signal.direction == 1 else "buy"
                sl_order = place_stop_loss(symbol, sl_side, quantity, hard_sl)
                if sl_order:
                    sl_order_id = str(sl_order.get("id", ""))

            # Record position
            self.state.trade_counter += 1
            dir_str = "LONG" if signal.direction == 1 else "SHORT"

            pos = LivePosition(
                trade_id=self.state.trade_counter,
                asset=asset,
                symbol=symbol,
                direction=signal.direction,
                entry_price=price,
                entry_ts=now.isoformat(),
                hard_sl=hard_sl,
                trail_sl=trail_sl,
                midline=signal.midline,
                std_dev=signal.std_dev,
                best_tf=signal.best_tf,
                best_period=signal.best_period,
                confidence=signal.confidence,
                pvt_r=signal.pvt_r,
                leverage=lev,
                position_usd=pos_usd,
                quantity=quantity,
                peak_r=signal.confidence,
                order_id=order_id,
                sl_order_id=sl_order_id,
            )
            self.state.positions[asset] = pos
            save_state(self.state)

            # Telegram notification
            mode_tag = "LIVE" if self.live_mode else "DRY-RUN"
            msg = (
                f"🚀 <b>NEW {dir_str} — {asset}</b> [{mode_tag}]\n"
                f"TF: {signal.best_tf} | Period: {signal.best_period}\n"
                f"Entry: ${price:.4f}\n"
                f"Confidence: {signal.confidence:.4f} | PVT R: {signal.pvt_r:.4f}\n"
                f"Leverage: {lev}x | Size: ${pos_usd:,.0f}\n"
                f"Hard SL: ${hard_sl:.4f}\n"
                f"Trail SL: ${trail_sl:.4f}"
            )
            logger.info("OPENED %s %s @ $%.4f | Lev=%dx | Size=$%.0f | SL=$%.4f",
                        dir_str, asset, price, lev, pos_usd, hard_sl)
            tg_send(msg)

        if not signals_found:
            logger.info("No qualifying signals found this scan")

    def run_cycle(self):
        """Run one full scan cycle: check exits → scan entries."""
        now = datetime.now(timezone.utc)
        self.state.last_scan_ts = now.isoformat()
        self.state.total_scans += 1

        logger.info("=" * 60)
        logger.info("SCAN CYCLE #%d — %s", self.state.total_scans, now.strftime("%Y-%m-%d %H:%M UTC"))
        logger.info("Open positions: %d/%d", len(self.state.positions), MAX_CONCURRENT)

        # Circuit breaker check
        if self.state.circuit_breaker:
            logger.warning("Circuit breaker is TRIPPED — no scanning. Reset with --reset-breaker")
            tg_send("🔴 Circuit breaker active — bot idle. Use --reset-breaker to resume.")
            return

        if self._check_circuit_breaker():
            return

        # Apply consensus params
        self._apply_params()

        # Load cache + fetch new bars
        logger.info("Updating data (parquet cache + new bars)...")
        t0 = time.perf_counter()
        all_data = get_live_data()
        elapsed = time.perf_counter() - t0
        logger.info("Data ready in %.1fs (%d assets)", elapsed, len(all_data))

        # Check exits on open positions
        closed_this_cycle = set()
        if self.state.positions:
            logger.info("Checking %d open positions for exits...", len(self.state.positions))
            before = set(self.state.positions.keys())
            self._check_exits(all_data)
            closed_this_cycle = before - set(self.state.positions.keys())
            if closed_this_cycle:
                logger.info("Closed this cycle: %s — cooldown, no re-entry", closed_this_cycle)

        # Scan for new entries (skip assets closed this cycle)
        self._scan_and_enter(all_data, skip_assets=closed_this_cycle)

        # Summary
        equity = self.state.initial_capital + self.state.realized_pnl
        logger.info("Cycle complete — Equity: $%.0f | Realized PnL: $%.2f | Open: %d",
                     equity, self.state.realized_pnl, len(self.state.positions))

        save_state(self.state)

    def send_status(self):
        """Send a comprehensive status summary to Telegram."""
        equity = self.state.initial_capital + self.state.realized_pnl
        dd = (equity - self.state.peak_equity) / self.state.peak_equity * 100 if self.state.peak_equity > 0 else 0
        mode = "LIVE" if self.live_mode else "DRY-RUN"
        breaker = "🔴 TRIPPED" if self.state.circuit_breaker else "🟢 OK"
        roi = (equity - self.state.initial_capital) / self.state.initial_capital * 100

        # Unrealized PnL from open positions
        total_upnl = 0.0
        pos_lines = ""
        for asset, pos in self.state.positions.items():
            price = fetch_ticker_price(pos.symbol)
            if price:
                upnl_pct = (price - pos.entry_price) / pos.entry_price * pos.direction * 100 * pos.leverage
                upnl_usd = upnl_pct / 100.0 * pos.position_usd
                total_upnl += upnl_usd
                emoji = "🟢" if upnl_usd >= 0 else "🔴"
                dir_s = "L" if pos.direction == 1 else "S"
                pos_lines += (
                    f"\n  {emoji} <b>{asset}</b> {dir_s} {pos.best_tf} "
                    f"${pos.entry_price:.4f}→${price:.4f} "
                    f"<b>{upnl_pct:+.2f}%</b> (${upnl_usd:+,.1f}) "
                    f"{pos.leverage}x"
                )
            else:
                dir_s = "L" if pos.direction == 1 else "S"
                pos_lines += f"\n  ⚪ <b>{asset}</b> {dir_s} {pos.best_tf} (price N/A)"

        total_equity = equity + total_upnl

        # Trade history stats from CSV
        wins, losses, total_closed = 0, 0, 0
        best_trade, worst_trade = 0.0, 0.0
        today_pnl = 0.0
        if TRADES_FILE.exists():
            try:
                tdf = pd.read_csv(TRADES_FILE)
                total_closed = len(tdf)
                wins = int((tdf["pnl_usd"] > 0).sum())
                losses = int((tdf["pnl_usd"] <= 0).sum())
                if total_closed > 0:
                    best_trade = tdf["pnl_pct"].max()
                    worst_trade = tdf["pnl_pct"].min()
                # Today's PnL
                tdf["exit_ts"] = pd.to_datetime(tdf["exit_ts"], utc=True)
                today = pd.Timestamp.now(tz="UTC").normalize()
                today_trades = tdf[tdf["exit_ts"] >= today]
                today_pnl = today_trades["pnl_usd"].sum() if not today_trades.empty else 0.0
            except Exception:
                pass

        wr = wins / total_closed * 100 if total_closed > 0 else 0

        # Last scan time
        if self.state.last_scan_ts:
            try:
                last_dt = pd.Timestamp(self.state.last_scan_ts)
                ago = (pd.Timestamp.now(tz="UTC") - last_dt).total_seconds() / 60
                last_scan_str = f"{last_dt.strftime('%H:%M UTC')} ({ago:.0f}m ago)"
            except Exception:
                last_scan_str = self.state.last_scan_ts
        else:
            last_scan_str = "Never"

        msg = (
            f"🦎 <b>Varanus Neo-Flow Status</b> [{mode}]\n"
            f"{'━' * 30}\n\n"
            f"💰 <b>Portfolio</b>\n"
            f"  Equity: <b>${total_equity:,.2f}</b>\n"
            f"  Initial: ${self.state.initial_capital:,.0f}\n"
            f"  ROI: <b>{roi:+.2f}%</b>\n"
            f"  Realized PnL: ${self.state.realized_pnl:+,.2f}\n"
            f"  Unrealized PnL: ${total_upnl:+,.2f}\n"
            f"  Today PnL: ${today_pnl:+,.2f}\n"
            f"  Drawdown: {dd:.2f}%\n\n"
            f"📊 <b>Performance</b>\n"
            f"  Closed Trades: {total_closed}\n"
            f"  Win/Loss: {wins}W / {losses}L\n"
            f"  Win Rate: {wr:.1f}%\n"
            f"  Best Trade: {best_trade:+.2f}%\n"
            f"  Worst Trade: {worst_trade:+.2f}%\n\n"
            f"📡 <b>System</b>\n"
            f"  Circuit Breaker: {breaker}\n"
            f"  Total Scans: {self.state.total_scans}\n"
            f"  Last Scan: {last_scan_str}\n"
            f"  Open: {len(self.state.positions)}/{MAX_CONCURRENT}\n"
        )

        if pos_lines:
            msg += f"\n📌 <b>Open Positions</b>{pos_lines}\n"

        tg_send(msg)

    def send_positions(self):
        """Send detailed open positions to Telegram."""
        if not self.state.positions:
            tg_send("📌 No open positions.")
            return

        lines = ["📌 <b>Open Positions</b>\n"]
        total_upnl = 0.0
        for asset, pos in self.state.positions.items():
            price = fetch_ticker_price(pos.symbol)
            dir_str = "LONG" if pos.direction == 1 else "SHORT"
            entry_dt = pd.Timestamp(pos.entry_ts)
            dur = (pd.Timestamp.now(tz="UTC") - entry_dt).total_seconds() / 3600

            if price:
                upnl_pct = (price - pos.entry_price) / pos.entry_price * pos.direction * 100 * pos.leverage
                upnl_usd = upnl_pct / 100.0 * pos.position_usd
                total_upnl += upnl_usd
                dist_sl = abs(price - pos.hard_sl) / price * 100
                dist_trail = abs(price - pos.trail_sl) / price * 100
                emoji = "🟢" if upnl_usd >= 0 else "🔴"
                lines.append(
                    f"{emoji} <b>{asset}</b> — {dir_str}\n"
                    f"  Entry: ${pos.entry_price:.4f} | Now: ${price:.4f}\n"
                    f"  PnL: <b>{upnl_pct:+.2f}%</b> (${upnl_usd:+,.2f})\n"
                    f"  TF: {pos.best_tf} | Period: {pos.best_period}\n"
                    f"  Leverage: {pos.leverage}x | Size: ${pos.position_usd:,.0f}\n"
                    f"  Hard SL: ${pos.hard_sl:.4f} ({dist_sl:.2f}% away)\n"
                    f"  Trail SL: ${pos.trail_sl:.4f} ({dist_trail:.2f}% away)\n"
                    f"  Confidence: {pos.confidence:.4f} | Peak R: {pos.peak_r:.4f}\n"
                    f"  Duration: {dur:.1f}h | Bars: {pos.bars_held}\n"
                )
            else:
                lines.append(f"⚪ <b>{asset}</b> — {dir_str} (price unavailable)\n")

        lines.append(f"\n💵 Total Unrealized: <b>${total_upnl:+,.2f}</b>")
        tg_send("\n".join(lines))

    def send_pnl(self):
        """Send P&L breakdown to Telegram."""
        if not TRADES_FILE.exists():
            tg_send("📈 No trade history yet.")
            return

        try:
            df = pd.read_csv(TRADES_FILE)
            df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
        except Exception:
            tg_send("⚠️ Could not read trade history.")
            return

        total = len(df)
        if total == 0:
            tg_send("📈 No closed trades yet.")
            return

        wins = int((df["pnl_usd"] > 0).sum())
        losses = total - wins
        wr = wins / total * 100
        total_pnl = df["pnl_usd"].sum()
        total_pnl_pct = df["pnl_pct"].sum()
        avg_win = df.loc[df["pnl_usd"] > 0, "pnl_pct"].mean() if wins > 0 else 0
        avg_loss = df.loc[df["pnl_usd"] <= 0, "pnl_pct"].mean() if losses > 0 else 0
        best = df.loc[df["pnl_pct"].idxmax()]
        worst = df.loc[df["pnl_pct"].idxmin()]
        gross_profit = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
        gross_loss = abs(df.loc[df["pnl_usd"] <= 0, "pnl_usd"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Per-day
        df["date"] = df["exit_ts"].dt.date
        daily = df.groupby("date")["pnl_usd"].sum()
        best_day = daily.max()
        worst_day = daily.min()
        best_day_date = daily.idxmax()
        worst_day_date = daily.idxmin()

        # Today
        today = pd.Timestamp.now(tz="UTC").normalize()
        today_trades = df[df["exit_ts"] >= today]
        today_pnl = today_trades["pnl_usd"].sum() if not today_trades.empty else 0.0
        today_count = len(today_trades)

        # Per asset
        asset_pnl = df.groupby("asset")["pnl_usd"].sum().sort_values(ascending=False)
        top3 = asset_pnl.head(3)
        bot3 = asset_pnl.tail(3)

        # Exit reasons
        exits = df["exit_reason"].value_counts()
        exit_lines = "\n".join(f"  {k}: {v} ({v/total*100:.0f}%)" for k, v in exits.items())

        msg = (
            f"📈 <b>P&L Report</b>\n"
            f"{'━' * 30}\n\n"
            f"💰 <b>Overall</b>\n"
            f"  Total PnL: <b>${total_pnl:+,.2f}</b> ({total_pnl_pct:+.1f}%)\n"
            f"  Profit Factor: {pf:.2f}\n"
            f"  Trades: {total} ({wins}W / {losses}L)\n"
            f"  Win Rate: {wr:.1f}%\n"
            f"  Avg Winner: {avg_win:+.2f}%\n"
            f"  Avg Loser: {avg_loss:+.2f}%\n\n"
            f"📅 <b>Today</b>\n"
            f"  PnL: ${today_pnl:+,.2f} ({today_count} trades)\n\n"
            f"🏆 <b>Best/Worst</b>\n"
            f"  Best Trade: {best['asset']} {best['pnl_pct']:+.2f}% (${best['pnl_usd']:+,.2f})\n"
            f"  Worst Trade: {worst['asset']} {worst['pnl_pct']:+.2f}% (${worst['pnl_usd']:+,.2f})\n"
            f"  Best Day: {best_day_date} ${best_day:+,.2f}\n"
            f"  Worst Day: {worst_day_date} ${worst_day:+,.2f}\n\n"
            f"🏅 <b>Top Assets</b>\n"
        )
        for asset, pnl in top3.items():
            msg += f"  🟢 {asset}: ${pnl:+,.2f}\n"
        msg += f"\n😔 <b>Worst Assets</b>\n"
        for asset, pnl in bot3.items():
            msg += f"  🔴 {asset}: ${pnl:+,.2f}\n"
        msg += f"\n🚪 <b>Exit Reasons</b>\n{exit_lines}"

        tg_send(msg)

    def send_help(self):
        """Send command list to Telegram."""
        msg = (
            f"🦎 <b>Varanus Neo-Flow Commands</b>\n\n"
            f"/status — Full portfolio status\n"
            f"/positions — Detailed open positions\n"
            f"/pnl — P&L breakdown & stats\n"
            f"/today — Today's trades & PnL\n"
            f"/params — Current consensus params\n"
            f"/help — Show this menu"
        )
        tg_send(msg)

    def send_today(self):
        """Send today's trade summary."""
        if not TRADES_FILE.exists():
            tg_send("📅 No trades today.")
            return
        try:
            df = pd.read_csv(TRADES_FILE)
            df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
            today = pd.Timestamp.now(tz="UTC").normalize()
            today_df = df[df["exit_ts"] >= today]
        except Exception:
            tg_send("⚠️ Could not read trades.")
            return

        if today_df.empty:
            tg_send("📅 No trades closed today yet.")
            return

        total_pnl = today_df["pnl_usd"].sum()
        wins = int((today_df["pnl_usd"] > 0).sum())
        losses = len(today_df) - wins

        lines = [
            f"📅 <b>Today's Trades</b> ({len(today_df)} total)\n",
            f"PnL: <b>${total_pnl:+,.2f}</b> | {wins}W/{losses}L\n",
        ]
        for _, row in today_df.iterrows():
            emoji = "🟢" if row["pnl_usd"] > 0 else "🔴"
            lines.append(
                f"{emoji} {row['asset']} {row['direction']} {row['best_tf']} "
                f"→ {row['pnl_pct']:+.2f}% (${row['pnl_usd']:+,.2f}) "
                f"[{row['exit_reason']}]"
            )

        tg_send("\n".join(lines))

    def send_params(self):
        """Send current consensus params."""
        msg = (
            f"⚙️ <b>Consensus Parameters</b>\n\n"
            f"  min_pearson_r: {self.params['min_pearson_r']}\n"
            f"  min_pvt_r: {self.params['min_pvt_r']}\n"
            f"  combined_gate: {self.params['combined_gate']}\n"
            f"  hard_sl_mult: {self.params['hard_sl_mult']}x ATR\n"
            f"  trail_buffer: {self.params['trail_buffer']}\n"
            f"  exhaust_r: {self.params['exhaust_r']}\n"
            f"  pos_frac: {self.params['pos_frac']}\n\n"
            f"  max_concurrent: {MAX_CONCURRENT}\n"
            f"  max_drawdown: {MAX_DRAWDOWN_PCT}%"
        )
        tg_send(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# Telegram Command Listener (background thread)
# ═══════════════════════════════════════════════════════════════════════════════

import threading

def start_telegram_listener(bot: LiveBot):
    """Poll Telegram for incoming commands in a background thread."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured — command listener disabled")
        return

    def _poll():
        offset = 0
        logger.info("Telegram command listener started")
        while True:
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
                resp = _requests.get(url, params={
                    "offset": offset, "timeout": 30, "allowed_updates": '["message"]',
                }, timeout=40)
                data = resp.json()

                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    text = msg.get("text", "").strip().lower()

                    # Only respond to our chat
                    if chat_id != TELEGRAM_CHAT_ID:
                        continue

                    if text in ("/status", "status"):
                        bot.send_status()
                    elif text in ("/positions", "/pos", "positions"):
                        bot.send_positions()
                    elif text in ("/pnl", "pnl", "/profit", "profit"):
                        bot.send_pnl()
                    elif text in ("/today", "today"):
                        bot.send_today()
                    elif text in ("/params", "params", "/parameters"):
                        bot.send_params()
                    elif text in ("/help", "help", "/start"):
                        bot.send_help()
                    elif text:
                        tg_send(
                            "❓ Unknown command. Send /help for available commands."
                        )

            except _requests.exceptions.ReadTimeout:
                continue
            except Exception as e:
                logger.error("Telegram listener error: %s", e)
                time.sleep(5)

    thread = threading.Thread(target=_poll, daemon=True, name="tg-listener")
    thread.start()
    return thread


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Varanus Neo-Flow — Live Trading Bot")
    parser.add_argument("--live", action="store_true", help="Enable live order execution (requires Binance keys)")
    parser.add_argument("--once", action="store_true", help="Run a single scan cycle and exit")
    parser.add_argument("--interval", type=int, default=60, help="Scan interval in minutes (default: 60)")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Starting capital (default: 10000)")
    parser.add_argument("--reset-breaker", action="store_true", help="Reset circuit breaker")
    parser.add_argument("--status", action="store_true", help="Send status to Telegram and exit")
    args = parser.parse_args()

    bot = LiveBot(live_mode=args.live, capital=args.capital)

    if args.reset_breaker:
        bot.state.circuit_breaker = False
        save_state(bot.state)
        logger.info("Circuit breaker reset")
        tg_send("🟢 Circuit breaker has been reset. Bot will resume scanning.")
        return

    if args.status:
        bot.send_status()
        return

    # Startup message
    mode = "🔴 LIVE TRADING" if args.live else "🔵 DRY-RUN (no orders)"
    startup_msg = (
        f"🦎 <b>Varanus Neo-Flow Bot Started</b>\n\n"
        f"Mode: {mode}\n"
        f"Capital: ${args.capital:,.0f}\n"
        f"Scan Interval: {args.interval} min\n"
        f"Assets: {len(ASSETS)}\n"
        f"Max Positions: {MAX_CONCURRENT}\n\n"
        f"Consensus Params:\n"
        f"  min_R: {bot.params['min_pearson_r']}\n"
        f"  min_PVT_R: {bot.params['min_pvt_r']}\n"
        f"  gate: {bot.params['combined_gate']}\n"
        f"  hard_SL: {bot.params['hard_sl_mult']}x ATR\n"
        f"  trail_buffer: {bot.params['trail_buffer']}\n"
        f"  exhaust_R: {bot.params['exhaust_r']}\n"
        f"  pos_frac: {bot.params['pos_frac']}"
    )
    tg_send(startup_msg)

    # Start Telegram command listener
    start_telegram_listener(bot)

    # Graceful shutdown
    running = True

    def _shutdown(signum, frame):
        nonlocal running
        running = False
        logger.info("Shutdown signal received — finishing current cycle...")
        tg_send("🛑 <b>Bot shutting down</b> — graceful exit.")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if args.once:
        bot.run_cycle()
        return

    # Main loop
    logger.info("Starting scan loop — interval=%d min", args.interval)
    while running:
        try:
            bot.run_cycle()
        except Exception as e:
            logger.error("Cycle error: %s\n%s", e, traceback.format_exc())
            tg_send(f"⚠️ <b>CYCLE ERROR</b>\n{e}")

        if not running:
            break

        # Sleep until next scan
        next_scan = datetime.now(timezone.utc) + timedelta(minutes=args.interval)
        logger.info("Next scan: %s", next_scan.strftime("%H:%M UTC"))
        sleep_secs = args.interval * 60
        for _ in range(sleep_secs):
            if not running:
                break
            time.sleep(1)

    save_state(bot.state)
    logger.info("Bot stopped.")


if __name__ == "__main__":
    main()

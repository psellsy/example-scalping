"""
Scalping Bot — VWAP + RSI + Volume Signal with Bracket Orders

Supports two modes:
  STOCKS:  Trade equities via Alpaca (original params optimized on 1-min bars)
  FUTURES: Trade futures via Alpaca (params optimized on 2 years of hourly bars)

Usage:
  # Stocks mode (default)
  python main.py AAPL MSFT NVDA --lot 10000

  # Futures mode
  python main.py ES NQ --mode futures --contracts 1

  # Futures with micros
  python main.py MES MNQ --mode futures --contracts 10
"""

import os
import sys
import json
import yaml
import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from pathlib import Path

from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus
from alpaca.data.enums import DataFeed

FEED_MAP = {
    "iex": DataFeed.IEX,
    "sip": DataFeed.SIP,
}

logger = logging.getLogger()


# ─── Audio Alert ─────────────────────────────────────────────────────────────

def _play_alert(sound: str = "error"):
    """Play a macOS system sound asynchronously. Non-blocking, never raises."""
    import subprocess, threading
    sounds = {
        "error": "/System/Library/Sounds/Sosumi.aiff",
        "trade": "/System/Library/Sounds/Glass.aiff",
        "warning": "/System/Library/Sounds/Basso.aiff",
    }
    path = sounds.get(sound, sounds["error"])
    def _play():
        try:
            subprocess.run(["afplay", path], timeout=5, capture_output=True)
        except Exception:
            pass
    threading.Thread(target=_play, daemon=True).start()


class AudioAlertHandler(logging.Handler):
    """Logging handler that plays audio alerts on ERROR-level messages.
    Rate-limited to avoid rapid-fire beeping on repeated errors."""
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self._last_alert = 0.0

    def emit(self, record):
        import time as _time
        now = _time.time()
        if now - self._last_alert > 30:  # Max one alert per 30 seconds
            self._last_alert = now
            _play_alert("error")


# ─── Trade Logger ────────────────────────────────────────────────────────────

TRADE_LOG_DIR = Path(__file__).parent / "data"

class TradeLogger:
    """Appends completed trades to a JSONL file for comparison with backtests."""

    def __init__(self, broker: str = "alpaca"):
        self._path = TRADE_LOG_DIR / f"paper_trades_{broker}.jsonl"
        TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    def log_trade(self, trade: dict):
        with open(self._path, "a") as f:
            f.write(json.dumps(trade) + "\n")
        logger.info(f"[TRADE LOG] → {self._path.name}: {trade['symbol']} {trade['direction']} P&L=${trade.get('pnl', 0):+.2f}")

    def read_trades(self) -> list[dict]:
        if not self._path.exists():
            return []
        trades = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    trades.append(json.loads(line))
        return trades

_trade_logger = TradeLogger("alpaca")


# ─── Slot Tracker (shared across all symbols) ──────────────────────────────

class SlotTracker:
    """Tracks concurrent open positions across the fleet to enforce max_slots."""

    def __init__(self, max_slots: int = 4):
        self.max_slots = max_slots
        self._open: set[str] = set()  # symbols with open positions

    @property
    def open_count(self) -> int:
        return len(self._open)

    def has_free_slot(self) -> bool:
        if self.max_slots <= 0:
            return True  # unlimited
        return self.open_count < self.max_slots

    def acquire(self, symbol: str):
        self._open.add(symbol)

    def release(self, symbol: str):
        self._open.discard(symbol)


_slot_tracker = SlotTracker()


# ─── Contract Specs ───────────────────────────────────────────────────────────

FUTURES_CONTRACTS = {
    # E-minis
    "ES":  {"full": "ES=F",  "name": "E-mini S&P 500",     "point_value": 50,  "tick": 0.25, "margin": 15_000},
    "NQ":  {"full": "NQ=F",  "name": "E-mini Nasdaq 100",  "point_value": 20,  "tick": 0.25, "margin": 20_000},
    "YM":  {"full": "YM=F",  "name": "E-mini Dow",         "point_value": 5,   "tick": 1.00, "margin": 10_000},
    "RTY": {"full": "RTY=F", "name": "E-mini Russell 2000", "point_value": 50, "tick": 0.10, "margin": 7_500},
    # Micros
    "MES": {"full": "MES=F", "name": "Micro E-mini S&P",   "point_value": 5,   "tick": 0.25, "margin": 1_500},
    "MNQ": {"full": "MNQ=F", "name": "Micro E-mini Nasdaq", "point_value": 2,  "tick": 0.25, "margin": 2_000},
    "MYM": {"full": "MYM=F", "name": "Micro E-mini Dow",   "point_value": 0.5, "tick": 1.00, "margin": 1_000},
    "M2K": {"full": "M2K=F", "name": "Micro E-mini Russell", "point_value": 5, "tick": 0.10, "margin": 750},
}


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class Config:
    symbols: list[str]
    mode: str = "stocks"        # "stocks", "futures", or "etf"
    # Position sizing
    lot: float = 10000.0        # $ per trade (stocks mode, or fixed ETF mode)
    contracts: int = 1          # Contracts per trade (futures mode)
    compound_pct: float = 0.0   # If > 0, lot = compound_pct × account equity (overrides lot)
    max_slots: int = 4          # Max concurrent positions across all symbols (0 = unlimited)
    # Strategy params — defaults are stock-optimized
    tp_pct: float = 0.003       # +0.3% take profit
    sl_pct: float = 0.002       # -0.2% stop loss
    vwap_lower: float = -0.004  # VWAP band lower
    vwap_upper: float = 0.004   # VWAP band upper
    rsi_cross_level: float = 40.0   # Long: RSI crossing UP above this
    rsi_short_level: float = 60.0   # Short: RSI crossing DOWN below this
    vol_threshold: float = 1.5  # Volume > Nx average
    # Trailing stop
    trail_activate_pct: float = 0.0  # 0 = disabled. e.g. 0.003 = activate after +0.3%
    trail_distance_pct: float = 0.003  # Trail distance from best price (e.g. 0.003 = 0.3%)
    # Risk management
    daily_loss_pct: float = 0.01  # Stop after -1% daily loss
    max_daily_trades: int = 5
    # Indicator periods
    rsi_period: int = 14
    vol_ma_period: int = 20
    # Time filters
    no_trade_before: dtime = dtime(9, 40)
    no_trade_after: dtime = dtime(15, 30)
    eod_liquidation: dtime = dtime(15, 55)
    # Connection
    data_feed: str = "iex"      # "iex" for free, "sip" for pro
    paper: bool = True


# Futures-optimized params (from 2-year hourly optimization)
FUTURES_PARAMS = {
    "tp_pct": 0.012,            # +1.2% take profit (wider for hourly moves)
    "sl_pct": 0.001,            # -0.1% stop loss (tight cut)
    "vwap_lower": -0.011,       # -1.1% VWAP band (wide)
    "vwap_upper": 0.010,        # +1.0% VWAP band
    "rsi_cross_level": 30.0,    # Long: RSI crossing above 30
    "rsi_short_level": 55.0,    # Short: RSI crossing below 55
    "vol_threshold": 1.0,       # No volume filter (deep liquidity)
}

# Stock-optimized params (from 3-month 1-min optimization)
STOCK_PARAMS = {
    "tp_pct": 0.003,
    "sl_pct": 0.002,
    "vwap_lower": -0.004,
    "vwap_upper": 0.004,
    "rsi_cross_level": 40.0,
    "rsi_short_level": 60.0,
    "vol_threshold": 1.5,
}

# Futures ETF params (same as futures — backtested on 5 years of hourly data)
# Top 25 from 130-ETF screening — all profitable, 97% win rate across universe
ETF_PARAMS = FUTURES_PARAMS.copy()  # Same params, different symbols
ETF_PARAMS["trail_activate_pct"] = 0.003  # Activate trailing stop after +0.3%
ETF_PARAMS["trail_distance_pct"] = 0.003  # Trail 0.3% behind best price

ETF_SYMBOLS = {
    # Top 10 (original)
    "UVXY": {"tracks": "VIX (Volatility)", "name": "ProShares Ultra VIX"},
    "USO":  {"tracks": "Crude Oil", "name": "United States Oil Fund"},
    "SMH":  {"tracks": "Semiconductors", "name": "VanEck Semiconductor ETF"},
    "QQQ":  {"tracks": "NQ (Nasdaq 100)", "name": "Invesco QQQ Trust"},
    "XLK":  {"tracks": "Technology", "name": "Technology Select SPDR"},
    "SOXX": {"tracks": "Semiconductors", "name": "iShares Semiconductor ETF"},
    "IWM":  {"tracks": "RTY (Russell 2000)", "name": "iShares Russell 2000 ETF"},
    "XLF":  {"tracks": "Financials", "name": "Financial Select SPDR"},
    "XLRE": {"tracks": "Real Estate", "name": "Real Estate Select SPDR"},
    "VTI":  {"tracks": "Total US Market", "name": "Vanguard Total Stock Market"},
    # 11-15
    "SPY":  {"tracks": "S&P 500", "name": "SPDR S&P 500 ETF"},
    "VIXY": {"tracks": "VIX Short-Term", "name": "ProShares VIX Short-Term"},
    "EWZ":  {"tracks": "Brazil", "name": "iShares MSCI Brazil"},
    "TQQQ": {"tracks": "3x Nasdaq 100", "name": "ProShares UltraPro QQQ"},
    "ARKK": {"tracks": "Innovation", "name": "ARK Innovation ETF"},
    # 16-20
    "VXUS": {"tracks": "Intl ex-US", "name": "Vanguard Total Intl Stock"},
    "XLE":  {"tracks": "Energy", "name": "Energy Select SPDR"},
    "XLI":  {"tracks": "Industrials", "name": "Industrial Select SPDR"},
    "TNA":  {"tracks": "3x Small Cap", "name": "Direxion Small Cap Bull 3X"},
    "XLU":  {"tracks": "Utilities", "name": "Utilities Select SPDR"},
    # 21-25
    "XME":  {"tracks": "Metals & Mining", "name": "SPDR Metals & Mining ETF"},
    "KRE":  {"tracks": "Regional Banks", "name": "SPDR Regional Banking ETF"},
    "SQQQ": {"tracks": "3x Inverse Nasdaq", "name": "ProShares UltraPro Short QQQ"},
    "UNG":  {"tracks": "Natural Gas", "name": "United States Natural Gas Fund"},
    "UDOW": {"tracks": "3x Dow", "name": "ProShares UltraPro Dow30"},
    # 26-50 (expanded from backtest)
    "DIA":  {"tracks": "Dow Jones", "name": "SPDR Dow Jones Industrial Avg"},
    "BITO": {"tracks": "Bitcoin Futures", "name": "ProShares Bitcoin Strategy ETF"},
    "KWEB": {"tracks": "China Internet", "name": "KraneShares CSI China Internet"},
    "EEM":  {"tracks": "Emerging Markets", "name": "iShares MSCI Emerging Markets"},
    "KBE":  {"tracks": "Banks", "name": "SPDR S&P Bank ETF"},
    "EFA":  {"tracks": "EAFE Developed", "name": "iShares MSCI EAFE ETF"},
    "VEA":  {"tracks": "Developed Markets", "name": "Vanguard FTSE Developed Markets"},
    "XLY":  {"tracks": "Consumer Disc.", "name": "Consumer Discretionary SPDR"},
    "XLP":  {"tracks": "Consumer Staples", "name": "Consumer Staples SPDR"},
    "FAZ":  {"tracks": "3x Inverse Financials", "name": "Direxion Financial Bear 3X"},
    "SPXS": {"tracks": "3x Inverse S&P", "name": "Direxion S&P 500 Bear 3X"},
    "XLV":  {"tracks": "Health Care", "name": "Health Care Select SPDR"},
    "SDOW": {"tracks": "3x Inverse Dow", "name": "ProShares UltraPro Short Dow30"},
    "XBI":  {"tracks": "Biotech", "name": "SPDR S&P Biotech ETF"},
    "GDXJ": {"tracks": "Junior Gold Miners", "name": "VanEck Junior Gold Miners"},
    "GDX":  {"tracks": "Gold Miners", "name": "VanEck Gold Miners ETF"},
    "ICLN": {"tracks": "Clean Energy", "name": "iShares Global Clean Energy"},
    "SOXL": {"tracks": "3x Semiconductors", "name": "Direxion Semiconductor Bull 3X"},
    "SVXY": {"tracks": "Short VIX", "name": "ProShares Short VIX Short-Term"},
    "SLV":  {"tracks": "Silver", "name": "iShares Silver Trust"},
    "MCHI": {"tracks": "China Large Cap", "name": "iShares MSCI China ETF"},
    "TZA":  {"tracks": "3x Inverse Small Cap", "name": "Direxion Small Cap Bear 3X"},
    "EWG":  {"tracks": "Germany", "name": "iShares MSCI Germany ETF"},
    "IGV":  {"tracks": "Software", "name": "iShares Expanded Tech-Software"},
    "MDY":  {"tracks": "S&P MidCap 400", "name": "SPDR S&P MidCap 400 ETF"},
    # 51-75
    "UPRO": {"tracks": "3x S&P 500", "name": "ProShares UltraPro S&P 500"},
    "XOP":  {"tracks": "Oil & Gas E&P", "name": "SPDR Oil & Gas Exploration"},
    "XHB":  {"tracks": "Homebuilders", "name": "SPDR Homebuilders ETF"},
    "IYR":  {"tracks": "US Real Estate", "name": "iShares US Real Estate ETF"},
    "IJH":  {"tracks": "S&P MidCap 400", "name": "iShares Core S&P Mid-Cap"},
    "TLT":  {"tracks": "20+ Year Treasuries", "name": "iShares 20+ Year Treasury Bond"},
    "SPXL": {"tracks": "3x S&P 500", "name": "Direxion S&P 500 Bull 3X"},
    "JETS": {"tracks": "Airlines", "name": "US Global Jets ETF"},
    "EWJ":  {"tracks": "Japan", "name": "iShares MSCI Japan ETF"},
    "QID":  {"tracks": "2x Inverse Nasdaq", "name": "ProShares UltraShort QQQ"},
    "TMF":  {"tracks": "3x 20Y Treasuries", "name": "Direxion 20Y Treasury Bull 3X"},
    "TBT":  {"tracks": "2x Inverse Treasuries", "name": "ProShares UltraShort 20Y Treasury"},
    "NUGT": {"tracks": "3x Gold Miners", "name": "Direxion Gold Miners Bull 2X"},
    "EWC":  {"tracks": "Canada", "name": "iShares MSCI Canada ETF"},
    "SOXS": {"tracks": "3x Inverse Semis", "name": "Direxion Semiconductor Bear 3X"},
    "PAVE": {"tracks": "Infrastructure", "name": "Global X US Infrastructure Dev"},
    "DUST": {"tracks": "2x Inverse Gold Miners", "name": "Direxion Gold Miners Bear 2X"},
    "FXI":  {"tracks": "China Large Cap", "name": "iShares China Large-Cap ETF"},
    "IVV":  {"tracks": "S&P 500", "name": "iShares Core S&P 500 ETF"},
    "MSOS": {"tracks": "Cannabis", "name": "AdvisorShares Pure US Cannabis"},
    "XLB":  {"tracks": "Materials", "name": "Materials Select SPDR"},
    "HDV":  {"tracks": "High Dividend", "name": "iShares Core High Dividend"},
    "LABU": {"tracks": "3x Biotech", "name": "Direxion Biotech Bull 3X"},
    "VB":   {"tracks": "Small Cap", "name": "Vanguard Small-Cap ETF"},
    "INDA": {"tracks": "India", "name": "iShares MSCI India ETF"},
}


def load_config_from_yaml(path: str) -> tuple[str, str]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["api_keys"]["alpaca_key"], cfg["api_keys"]["alpaca_secret"]


# ─── Indicators ──────────────────────────────────────────────────────────────

def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def compute_vwap(bars_df: pd.DataFrame) -> float:
    if bars_df.empty:
        return 0.0
    typical = (bars_df["high"] + bars_df["low"] + bars_df["close"]) / 3.0
    cum_tv = (typical * bars_df["volume"]).sum()
    cum_v = bars_df["volume"].sum()
    return cum_tv / cum_v if cum_v > 0 else 0.0


# ─── Scalping Algorithm ─────────────────────────────────────────────────────

class ImprovedScalper:
    def __init__(self, trading_client: TradingClient, symbol: str, cfg: Config):
        self._client = trading_client
        self._symbol = symbol
        self._cfg = cfg
        self._l = logger.getChild(symbol)

        # Futures contract info
        self._is_futures = cfg.mode == "futures"
        if self._is_futures:
            spec = FUTURES_CONTRACTS.get(symbol, {})
            self._point_value = spec.get("point_value", 1)
            self._contracts = cfg.contracts
            self._l.info(f"Futures mode: {spec.get('name', symbol)} | "
                         f"{self._contracts} contracts × ${self._point_value}/pt")
        else:
            self._point_value = 1
            self._contracts = 0

        # Bar storage (today only) — try loading cached bars first
        self._bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self._prev_rsi = 50.0
        self._load_bars()  # Restore from disk if today's cache exists

        # State: IDLE, BUY_PENDING, LONG, SHORT_PENDING, SHORT, CLOSING
        self._state = "IDLE"
        self._direction = None
        self._entry_price = 0.0
        self._shares = 0          # Stocks: share count. Futures: contract count.
        self._tp_price = 0.0
        self._sl_price = 0.0
        self._pending_order_id = None
        self._close_order_id = None
        self._sl_order_id = None     # Server-side stop loss order
        self._tp_order_id = None     # Server-side take profit order
        self._closing_since: pd.Timestamp | None = None  # When CLOSING state began
        self._best_price: float = 0.0  # Best price since entry (for trailing stop)

        # Daily tracking
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._total_pnl = 0.0
        self._total_trades = 0
        self._current_date = None
        self._entry_time: str | None = None   # ISO timestamp of entry fill
        self._exit_reason: str | None = None  # Why we closed

        self._init_state()

    def _init_state(self):
        broker_pos = self._get_broker_position()
        if broker_pos is not None:
            self._adopt_position(broker_pos[0], broker_pos[1], "startup")
            self._ensure_server_orders()
        else:
            self._state = "IDLE"
            self._l.info("No position on broker, starting IDLE")

    def _now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="America/New_York")

    def _reset_daily(self):
        today = self._now().date()
        if today != self._current_date:
            self._current_date = today
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            self._prev_rsi = 50.0
            self._l.info(f"New trading day: {today}")

    # ─── Bar Cache (persist across restarts) ────────────────────────────────

    def _bar_cache_path(self) -> Path:
        """Path to this symbol's bar cache file."""
        cache_dir = TRADE_LOG_DIR / "bar_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{self._symbol}.json"

    def _save_bars(self):
        """Save current bars to disk. Called after each new bar."""
        if self._bars.empty:
            return
        try:
            records = []
            for ts, row in self._bars.iterrows():
                records.append({
                    "ts": str(ts),
                    "o": float(row["open"]),
                    "h": float(row["high"]),
                    "l": float(row["low"]),
                    "c": float(row["close"]),
                    "v": float(row["volume"]),
                })
            data = {"date": str(self._now().date()), "bars": records}
            self._bar_cache_path().write_text(json.dumps(data))
        except Exception as e:
            self._l.debug(f"Bar cache save failed: {e}")

    def _load_bars(self):
        """Load cached bars from disk if they're from today."""
        path = self._bar_cache_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            today = str(self._now().date())
            if data.get("date") != today:
                self._l.debug(f"Bar cache stale (cached={data.get('date')}, today={today}), ignoring")
                return
            records = data.get("bars", [])
            if not records:
                return
            rows = []
            timestamps = []
            for r in records:
                timestamps.append(pd.Timestamp(r["ts"]))
                rows.append({
                    "open": r["o"], "high": r["h"], "low": r["l"],
                    "close": r["c"], "volume": r["v"],
                })
            self._bars = pd.DataFrame(rows, index=timestamps)
            needed = max(self._cfg.rsi_period + 2, self._cfg.vol_ma_period + 1)
            self._l.info(f"Loaded {len(self._bars)} cached bars ({len(self._bars)}/{needed} needed)")
        except Exception as e:
            self._l.debug(f"Bar cache load failed: {e}")

    def _calc_pnl(self, exit_price: float) -> float:
        """Calculate P&L based on mode (stocks vs futures)."""
        price_diff = exit_price - self._entry_price
        if self._direction == "short":
            price_diff = -price_diff

        if self._is_futures:
            return price_diff * self._point_value * self._shares
        else:
            return price_diff * self._shares

    def on_bar(self, bar):
        """Process incoming bar (1-min for stocks, could be any TF for futures)."""
        self._reset_daily()

        ts = pd.Timestamp(bar.timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ts = ts.tz_convert("America/New_York")
        new_row = pd.DataFrame({
            "open": [bar.open],
            "high": [bar.high],
            "low": [bar.low],
            "close": [bar.close],
            "volume": [bar.volume],
        }, index=[ts])
        if self._bars.empty:
            self._bars = new_row
        elif ts in self._bars.index:
            pass  # Duplicate bar (already cached), skip
        else:
            self._bars = pd.concat([self._bars, new_row])

        # Persist bars to disk for fast restart (every bar, ~25 symbols × 1/min = trivial I/O)
        self._save_bars()

        t = ts.time()
        cfg = self._cfg

        self._l.debug(f"bar {ts} close={bar.close} vol={bar.volume} bars={len(self._bars)}")

        # ── LONG: check bracket exits (BEFORE warmup — must always run) ──
        if self._state == "LONG":
            if t >= cfg.eod_liquidation:
                self._close_position("long_eod")
                return
            # Update trailing stop before checking SL
            self._update_trailing_stop(bar.high)
            if bar.low <= self._sl_price:
                self._close_position("long_stop_loss")
                return
            if bar.high >= self._tp_price:
                self._close_position_limit(self._tp_price, "long_take_profit")
                return
            return

        # ── SHORT: check bracket exits (BEFORE warmup — must always run) ──
        if self._state == "SHORT":
            if t >= cfg.eod_liquidation:
                self._close_position("short_eod")
                return
            # Update trailing stop before checking SL
            self._update_trailing_stop(bar.low)
            if bar.high >= self._sl_price:
                self._close_position("short_stop_loss")
                return
            if bar.low <= self._tp_price:
                self._close_position_limit(self._tp_price, "short_take_profit")
                return
            return

        needed = max(cfg.rsi_period + 2, cfg.vol_ma_period + 1)
        if len(self._bars) < needed:
            if len(self._bars) % 5 == 0 or len(self._bars) == needed - 1:
                self._l.info(f"Warming up: {len(self._bars)}/{needed} bars")
            return
        if len(self._bars) == needed:
            self._l.info(f"Warmup complete — now scanning for signals")

        # ── Pending: Alpaca handles fill ──
        if self._state in ("BUY_PENDING", "SHORT_PENDING", "CLOSING"):
            return

        # ── IDLE: scan for signals ──
        if self._state != "IDLE":
            return

        if t < cfg.no_trade_before or t >= cfg.no_trade_after:
            return
        if self._daily_pnl <= -(cfg.daily_loss_pct * 100_000):
            return
        if self._daily_trades >= cfg.max_daily_trades:
            return

        closes = self._bars["close"].values.astype(float)
        volumes = self._bars["volume"].values.astype(float)

        rsi_now = compute_rsi(closes, cfg.rsi_period)
        vwap_now = compute_vwap(self._bars)
        vol_ma = volumes[-cfg.vol_ma_period:].mean()

        if vwap_now > 0:
            vwap_dist = (closes[-1] - vwap_now) / vwap_now
        else:
            vwap_dist = 999

        near_vwap = cfg.vwap_lower <= vwap_dist <= cfg.vwap_upper
        vol_ok = volumes[-1] > cfg.vol_threshold * vol_ma if vol_ma > 0 else False

        if near_vwap and vol_ok:
            # LONG signal: RSI crossing UP above threshold
            if self._prev_rsi < cfg.rsi_cross_level and rsi_now >= cfg.rsi_cross_level:
                self._submit_entry(closes[-1], "long")
            # SHORT signal: RSI crossing DOWN below threshold
            elif self._prev_rsi > cfg.rsi_short_level and rsi_now <= cfg.rsi_short_level:
                self._submit_entry(closes[-1], "short")

        self._prev_rsi = rsi_now

    def _submit_entry(self, price: float, direction: str):
        """Submit limit order to open a position."""
        if not _slot_tracker.has_free_slot():
            self._l.info(f"Skipping {direction}: {_slot_tracker.open_count}/{_slot_tracker.max_slots} slots full")
            return
        # SAFETY: check Alpaca for existing position on this symbol before opening
        # This prevents stacking if internal state gets out of sync with broker
        try:
            existing = self._client.get_open_position(self._symbol)
            if existing and abs(int(existing.qty)) > 0:
                self._l.warning(
                    f"BLOCKED entry: broker already has {existing.qty} {self._symbol} "
                    f"(side={existing.side}) — refusing to stack"
                )
                return
        except Exception:
            pass  # No position exists — safe to proceed
        if self._is_futures:
            qty = self._contracts
        else:
            # Determine lot size: compounding or fixed
            if self._cfg.compound_pct > 0:
                try:
                    acct = self._client.get_account()
                    equity = float(acct.equity)
                    buying_power = float(acct.buying_power)
                    # Divide compound allocation evenly across max_slots
                    slots = max(_slot_tracker.max_slots, 1)
                    lot = (equity * self._cfg.compound_pct) / slots
                    # Cap at available buying power (leave 5% buffer)
                    max_lot = buying_power * 0.95
                    if lot > max_lot:
                        self._l.info(f"Compound: ${lot:,.0f} capped to ${max_lot:,.0f} (buying power ${buying_power:,.0f})")
                        lot = max_lot
                    if lot < price:
                        self._l.info(f"Skipping: lot ${lot:,.0f} < price ${price:.2f}")
                        return
                    self._l.info(f"Compound: {self._cfg.compound_pct*100:.0f}% of ${equity:,.0f} / {slots} slots = ${lot:,.0f} lot (BP: ${buying_power:,.0f})")
                except Exception as e:
                    lot = self._cfg.lot
                    self._l.warning(f"Compound equity fetch failed ({e}), using fixed ${lot:,.0f}")
            else:
                lot = self._cfg.lot
            qty = max(1, int(lot / price))

        side = OrderSide.BUY if direction == "long" else OrderSide.SELL
        try:
            order = self._client.submit_order(
                LimitOrderRequest(
                    symbol=self._symbol,
                    qty=qty,
                    side=side,
                    type=OrderType.LIMIT,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(price, 2),
                )
            )
            self._pending_order_id = order.id
            self._shares = qty
            self._direction = direction
            self._state = "BUY_PENDING" if direction == "long" else "SHORT_PENDING"
            _slot_tracker.acquire(self._symbol)  # Reserve slot immediately on order submit

            unit = "contracts" if self._is_futures else "shares"
            self._l.info(
                f"{direction.upper()} LIMIT: {qty} {unit} @ ${price:.2f} (order {order.id}) | "
                f"Slots: {_slot_tracker.open_count}/{_slot_tracker.max_slots}"
            )
        except Exception as e:
            self._l.error(f"{direction} entry failed: {e}")

    def _update_trailing_stop(self, extreme_price: float):
        """Ratchet the stop loss toward price when trailing stop is active.
        For longs: extreme_price = bar.high. For shorts: extreme_price = bar.low.
        Only moves the SL in the favorable direction (never widens the stop)."""
        cfg = self._cfg
        if cfg.trail_activate_pct <= 0:
            return  # Trailing stop disabled

        # Update best price
        if self._direction == "long":
            if extreme_price > self._best_price:
                self._best_price = extreme_price
        else:  # short
            if self._best_price == 0 or extreme_price < self._best_price:
                self._best_price = extreme_price

        # Check if trailing stop has been activated
        if self._direction == "long":
            move_pct = (self._best_price - self._entry_price) / self._entry_price
        else:
            move_pct = (self._entry_price - self._best_price) / self._entry_price

        if move_pct < cfg.trail_activate_pct:
            return  # Not yet activated

        # Compute the trailing SL
        if self._direction == "long":
            trail_sl = self._best_price * (1 - cfg.trail_distance_pct)
        else:
            trail_sl = self._best_price * (1 + cfg.trail_distance_pct)

        # Only ratchet in the protective direction (never widen)
        if self._direction == "long":
            if trail_sl <= self._sl_price:
                return  # New trail SL is worse than current — skip
            old_sl = self._sl_price
            self._sl_price = trail_sl
        else:
            if trail_sl >= self._sl_price:
                return  # New trail SL is worse than current — skip
            old_sl = self._sl_price
            self._sl_price = trail_sl

        self._l.info(
            f"TRAIL SL: {self._direction.upper()} best=${self._best_price:.2f} "
            f"SL ${old_sl:.2f} → ${self._sl_price:.2f} "
            f"(+{move_pct*100:.2f}% from entry)"
        )

        # Update the server-side stop order on Alpaca
        self._replace_server_stop()

    def _replace_server_stop(self):
        """Cancel existing server SL and resubmit at the new (trailed) price."""
        if self._sl_order_id:
            try:
                self._client.cancel_order_by_id(str(self._sl_order_id))
            except Exception:
                pass
            self._sl_order_id = None
        self._submit_server_stop()

    def _submit_server_stop(self):
        """Submit a server-side stop loss order to Alpaca after entry fill.
        This protects the position even if the bot disconnects or market closes.
        Uses DAY for fractional qty (Alpaca requires it), GTC for whole shares."""
        if not self._shares or not self._sl_price:
            return
        side = OrderSide.SELL if self._direction == "long" else OrderSide.BUY
        qty = self._shares
        # Alpaca: fractional orders must be DAY, whole shares can be GTC
        is_fractional = qty != int(qty)
        tif = TimeInForce.DAY if is_fractional else TimeInForce.GTC
        try:
            order = self._client.submit_order(
                StopOrderRequest(
                    symbol=self._symbol,
                    qty=qty,
                    side=side,
                    time_in_force=tif,
                    stop_price=round(self._sl_price, 2),
                )
            )
            self._sl_order_id = order.id
            tif_label = "DAY" if is_fractional else "GTC"
            self._l.info(f"SERVER SL: {side.name} {qty} @ ${self._sl_price:.2f} [{tif_label}] (order {order.id})")
        except Exception as e:
            avail = self._parse_available_qty(e)
            if avail is not None and avail > 0 and avail != qty:
                self._l.warning(f"Server SL failed ({qty} requested, {avail} available) — retrying with actual qty")
                self._shares = avail
                is_frac = avail != int(avail)
                tif2 = TimeInForce.DAY if is_frac else TimeInForce.GTC
                try:
                    order = self._client.submit_order(
                        StopOrderRequest(
                            symbol=self._symbol,
                            qty=avail,
                            side=side,
                            time_in_force=tif2,
                            stop_price=round(self._sl_price, 2),
                        )
                    )
                    self._sl_order_id = order.id
                    self._l.info(f"SERVER SL: {side.name} {avail} [corrected] @ ${self._sl_price:.2f} (order {order.id})")
                    return
                except Exception as e2:
                    self._l.error(f"Server SL retry also failed: {e2} — software SL only!")
            elif avail == 0:
                self._l.warning(f"Server SL: 0 available — position gone, logging as external_close")
                self._log_exit(self._entry_price, "external_close")
                self._reset_position()
                return
            else:
                self._l.error(f"Server SL failed: {e} — software SL only!")

    def _submit_server_tp(self):
        """Submit a server-side take profit (limit) order to Alpaca after entry fill.
        This protects the position even if the bot disconnects.
        Uses DAY for fractional qty (Alpaca requires it), GTC for whole shares."""
        if not self._shares or not self._tp_price:
            return
        side = OrderSide.SELL if self._direction == "long" else OrderSide.BUY
        qty = self._shares
        is_fractional = qty != int(qty)
        tif = TimeInForce.DAY if is_fractional else TimeInForce.GTC
        try:
            order = self._client.submit_order(
                LimitOrderRequest(
                    symbol=self._symbol,
                    qty=qty,
                    side=side,
                    type=OrderType.LIMIT,
                    time_in_force=tif,
                    limit_price=round(self._tp_price, 2),
                )
            )
            self._tp_order_id = order.id
            tif_label = "DAY" if is_fractional else "GTC"
            self._l.info(f"SERVER TP: {side.name} {qty} @ ${self._tp_price:.2f} [{tif_label}] (order {order.id})")
        except Exception as e:
            self._l.error(f"Server TP failed: {e} — software TP only!")

    def _cancel_server_orders(self):
        """Cancel server-side SL and TP orders before we close via our own logic."""
        if self._sl_order_id:
            try:
                self._client.cancel_order_by_id(str(self._sl_order_id))
                self._l.debug(f"Cancelled server SL {self._sl_order_id}")
            except Exception:
                pass  # May already be filled/cancelled
            self._sl_order_id = None
        if self._tp_order_id:
            try:
                self._client.cancel_order_by_id(str(self._tp_order_id))
                self._l.debug(f"Cancelled server TP {self._tp_order_id}")
            except Exception:
                pass
            self._tp_order_id = None

    def _parse_available_qty(self, error_msg: str) -> int | None:
        """Extract actual available qty from Alpaca insufficient-qty error."""
        try:
            import re
            m = re.search(r'"available"\s*:\s*"(\d+)"', str(error_msg))
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None

    def _close_position(self, reason: str):
        """Close position at market."""
        self._cancel_server_orders()
        # Wait for Alpaca to release shares held by cancelled SL/TP orders
        import time
        time.sleep(2)
        self._exit_reason = reason
        side = OrderSide.SELL if self._direction == "long" else OrderSide.BUY
        qty = self._shares
        try:
            order = self._client.submit_order(
                MarketOrderRequest(
                    symbol=self._symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            )
            self._l.info(f"CLOSE MARKET ({reason}): {qty} (order {order.id})")
            self._close_order_id = order.id
            self._state = "CLOSING"
            self._closing_since = self._now()
        except Exception as e:
            avail = self._parse_available_qty(e)
            if avail is not None and avail > 0 and avail != qty:
                self._l.warning(f"Close failed ({qty} requested, {avail} available) — retrying with actual qty")
                self._shares = avail
                try:
                    order = self._client.submit_order(
                        MarketOrderRequest(
                            symbol=self._symbol,
                            qty=avail,
                            side=side,
                            time_in_force=TimeInForce.DAY,
                        )
                    )
                    self._l.info(f"CLOSE MARKET ({reason}): {avail} [corrected] (order {order.id})")
                    self._close_order_id = order.id
                    self._state = "CLOSING"
                    self._closing_since = self._now()
                    return
                except Exception as e2:
                    self._l.error(f"Close retry also failed: {e2}")
            elif avail == 0:
                self._l.warning(f"Position gone (0 available) — logging as external_close")
                self._log_exit(self._entry_price, "external_close")
                self._reset_position()
                return
            self._l.error(f"Close failed: {e}")

    def _close_position_limit(self, price: float, reason: str):
        """Close position at limit price."""
        self._cancel_server_orders()
        import time
        time.sleep(2)
        self._exit_reason = reason
        side = OrderSide.SELL if self._direction == "long" else OrderSide.BUY
        qty = self._shares
        try:
            order = self._client.submit_order(
                LimitOrderRequest(
                    symbol=self._symbol,
                    qty=qty,
                    side=side,
                    type=OrderType.LIMIT,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(price, 2),
                )
            )
            self._l.info(f"CLOSE LIMIT ({reason}): {qty} @ ${price:.2f} (order {order.id})")
            self._close_order_id = order.id
            self._state = "CLOSING"
            self._closing_since = self._now()
        except Exception as e:
            avail = self._parse_available_qty(e)
            if avail is not None and avail != qty:
                self._l.warning(f"Close limit failed ({qty} requested, {avail} available) — falling back to market with actual qty")
                self._shares = avail
            else:
                self._l.error(f"Close limit failed: {e}")
            self._close_position(reason)

    def on_trade_update(self, event: str, order_data: dict):
        """Handle order fill/cancel/reject events from Alpaca websocket.
        This is the fast path — reconcile() is the safety net if these are missed."""
        order_id = order_data.get("id")
        filled_price = float(order_data.get("filled_avg_price", 0) or 0)

        self._l.info(f"Trade update: {event} order={order_id} price={filled_price}")

        if event == "fill":
            # Entry filled
            if self._state in ("BUY_PENDING", "SHORT_PENDING") and order_id == str(self._pending_order_id):
                self._entry_price = filled_price
                self._entry_time = datetime.now().isoformat()
                if self._direction == "long":
                    self._tp_price = filled_price * (1 + self._cfg.tp_pct)
                    self._sl_price = filled_price * (1 - self._cfg.sl_pct)
                    # Ensure SL is at least 1 tick below entry (prevents rounding to same price)
                    if round(self._sl_price, 2) >= round(filled_price, 2):
                        self._sl_price = filled_price - 0.01
                    self._state = "LONG"
                else:
                    self._tp_price = filled_price * (1 - self._cfg.tp_pct)
                    self._sl_price = filled_price * (1 + self._cfg.sl_pct)
                    # Ensure SL is at least 1 tick above entry (prevents rounding to same price)
                    if round(self._sl_price, 2) <= round(filled_price, 2):
                        self._sl_price = filled_price + 0.01
                    self._state = "SHORT"
                self._pending_order_id = None
                self._best_price = filled_price  # Initialize trailing stop tracker

                unit = "contracts" if self._is_futures else "shares"
                self._l.info(
                    f"FILLED {self._direction.upper()} {self._shares} {unit} @ ${filled_price:.2f} | "
                    f"TP=${self._tp_price:.2f} SL=${self._sl_price:.2f} | "
                    f"Slots: {_slot_tracker.open_count}/{_slot_tracker.max_slots}"
                )
                self._submit_server_stop()
                self._submit_server_tp()

            # Server-side SL filled — cancel TP counterpart
            elif self._state in ("LONG", "SHORT") and self._sl_order_id and order_id == str(self._sl_order_id):
                self._sl_order_id = None
                if self._tp_order_id:
                    try:
                        self._client.cancel_order_by_id(str(self._tp_order_id))
                    except Exception:
                        pass
                    self._tp_order_id = None
                self._log_exit(filled_price, "server_stop_loss")
                self._reset_position()

            # Server-side TP filled — cancel SL counterpart
            elif self._state in ("LONG", "SHORT") and self._tp_order_id and order_id == str(self._tp_order_id):
                self._tp_order_id = None
                if self._sl_order_id:
                    try:
                        self._client.cancel_order_by_id(str(self._sl_order_id))
                    except Exception:
                        pass
                    self._sl_order_id = None
                self._log_exit(filled_price, "server_take_profit")
                self._reset_position()

            # Exit filled
            elif self._state == "CLOSING":
                self._log_exit(filled_price, self._exit_reason or "unknown")
                self._reset_position()

        elif event in ("canceled", "rejected"):
            if self._state in ("BUY_PENDING", "SHORT_PENDING"):
                _slot_tracker.release(self._symbol)
                self._l.info(f"Entry {event}, returning to IDLE | Slots: {_slot_tracker.open_count}/{_slot_tracker.max_slots}")
                self._state = "IDLE"
                self._pending_order_id = None
            elif self._state == "CLOSING":
                self._l.warning(f"Close order {event}, retrying market")
                self._state = "LONG" if self._direction == "long" else "SHORT"
                self._close_position("retry_after_" + event)

        elif event == "partial_fill":
            self._l.info("Partial fill, waiting for complete fill")

    def _reset_position(self):
        _slot_tracker.release(self._symbol)
        self._cancel_server_orders()
        self._state = "IDLE"
        self._direction = None
        self._entry_price = 0.0
        self._shares = 0
        self._tp_price = 0.0
        self._sl_price = 0.0
        self._pending_order_id = None
        self._close_order_id = None
        self._sl_order_id = None
        self._tp_order_id = None
        self._closing_since = None
        self._best_price = 0.0
        self._entry_time = None
        self._exit_reason = None
        self._close_retries = 0

    # ── Reconciliation helpers ──────────────────────────────────────────

    def _log_exit(self, exit_price: float, reason: str):
        """Record a completed trade — single place for P&L accounting + trade log."""
        pnl = self._calc_pnl(exit_price)
        self._daily_pnl += pnl
        self._daily_trades += 1
        self._total_pnl += pnl
        self._total_trades += 1
        if self._entry_price > 0:
            pnl_pct = round((exit_price / self._entry_price - 1) * 100, 4) if self._direction == "long" \
                      else round((1 - exit_price / self._entry_price) * 100, 4)
        else:
            pnl_pct = 0.0
        self._l.info(
            f"EXIT {self._direction.upper()} @ ${exit_price:.2f} | "
            f"P&L=${pnl:+.2f} ({pnl_pct:+.2f}%) | reason={reason} | "
            f"Day=${self._daily_pnl:+.2f} ({self._daily_trades}/{self._cfg.max_daily_trades}) | "
            f"Total=${self._total_pnl:+.2f} ({self._total_trades} trades)"
        )
        _trade_logger.log_trade({
            "broker": "alpaca",
            "symbol": self._symbol,
            "direction": self._direction,
            "shares": self._shares,
            "entry_price": self._entry_price,
            "exit_price": exit_price,
            "entry_time": self._entry_time,
            "exit_time": datetime.now().isoformat(),
            "pnl": round(pnl, 2),
            "pnl_pct": pnl_pct,
            "exit_reason": reason,
            "mode": self._cfg.mode,
            "cumulative_pnl": round(self._total_pnl, 2),
            "total_trades": self._total_trades,
        })

    def _adopt_position(self, qty: int, avg_entry: float, source: str):
        """Sync internal state to match a broker-confirmed position."""
        self._entry_price = avg_entry
        self._shares = abs(qty)
        self._best_price = avg_entry  # Conservative: assume no movement yet
        self._entry_time = self._entry_time or datetime.now().isoformat()
        if qty > 0:
            self._direction = "long"
            self._tp_price = avg_entry * (1 + self._cfg.tp_pct)
            self._sl_price = avg_entry * (1 - self._cfg.sl_pct)
            if round(self._sl_price, 2) >= round(avg_entry, 2):
                self._sl_price = avg_entry - 0.01
            self._state = "LONG"
        else:
            self._direction = "short"
            self._tp_price = avg_entry * (1 - self._cfg.tp_pct)
            self._sl_price = avg_entry * (1 + self._cfg.sl_pct)
            if round(self._sl_price, 2) <= round(avg_entry, 2):
                self._sl_price = avg_entry + 0.01
            self._state = "SHORT"
        _slot_tracker.acquire(self._symbol)
        self._pending_order_id = None
        self._close_order_id = None
        self._l.info(
            f"[RECONCILE:{source}] {self._direction.upper()} {self._shares} @ ${avg_entry:.2f} | "
            f"TP=${self._tp_price:.2f} SL=${self._sl_price:.2f} | "
            f"Slots: {_slot_tracker.open_count}/{_slot_tracker.max_slots}"
        )

    def _get_broker_position(self):
        """Query Alpaca for current position. Returns (signed_qty, avg_entry) or None.
        Uses pos.side to determine sign — pos.qty is always positive in alpaca-py."""
        try:
            pos = self._client.get_open_position(self._symbol)
            qty = abs(int(pos.qty))
            side = str(pos.side).lower()
            if "short" in side:
                qty = -qty
            return qty, float(pos.avg_entry_price)
        except Exception:
            return None  # No position on broker

    def _ensure_server_orders(self):
        """Guarantee server-side SL and TP orders exist. Re-submit if missing."""
        # ── Check / resubmit SL ──
        sl_ok = False
        if self._sl_order_id:
            try:
                order = self._client.get_order_by_id(str(self._sl_order_id))
                status = str(order.status).lower()
                if status in ("filled",):
                    filled_price = float(order.filled_avg_price or 0)
                    self._sl_order_id = None
                    self._log_exit(filled_price, "server_stop_loss")
                    self._reset_position()
                    return
                if status in ("canceled", "cancelled", "expired", "rejected"):
                    self._l.warning(f"Server SL {status} — resubmitting")
                    self._sl_order_id = None
                else:
                    sl_ok = True
            except Exception as e:
                self._l.warning(f"Server SL check failed ({e}) — resubmitting")
                self._sl_order_id = None

        # Before submitting a new SL, check if one already exists on Alpaca
        if not sl_ok and not self._sl_order_id:
            try:
                open_orders = self._client.get_orders(GetOrdersRequest(
                    status=QueryOrderStatus.OPEN,
                    symbols=[self._symbol],
                ))
                for o in open_orders:
                    otype = str(getattr(o, 'order_type', None) or getattr(o, 'type', '')).lower()
                    ostatus = str(o.status).lower()
                    is_stop = 'stop' in otype
                    is_active = any(s in ostatus for s in ("new", "accepted", "held", "pending"))
                    if is_stop and is_active:
                        self._sl_order_id = o.id
                        stop_px = float(getattr(o, 'stop_price', 0) or 0)
                        self._l.info(
                            f"[RECONCILE] Adopted existing server SL: {o.side} {o.qty} @ ${stop_px:.2f} (order {o.id})"
                        )
                        sl_ok = True
                        break
            except Exception as e:
                self._l.warning(f"Failed to discover existing SL orders: {e}")

        if not sl_ok:
            self._submit_server_stop()

        # ── Check / resubmit TP ──
        tp_ok = False
        if self._tp_order_id:
            try:
                order = self._client.get_order_by_id(str(self._tp_order_id))
                status = str(order.status).lower()
                if status in ("filled",):
                    filled_price = float(order.filled_avg_price or 0)
                    self._tp_order_id = None
                    # Cancel SL counterpart
                    if self._sl_order_id:
                        try:
                            self._client.cancel_order_by_id(str(self._sl_order_id))
                        except Exception:
                            pass
                        self._sl_order_id = None
                    self._log_exit(filled_price, "server_take_profit")
                    self._reset_position()
                    return
                if status in ("canceled", "cancelled", "expired", "rejected"):
                    self._l.warning(f"Server TP {status} — resubmitting")
                    self._tp_order_id = None
                else:
                    tp_ok = True
            except Exception as e:
                self._l.warning(f"Server TP check failed ({e}) — resubmitting")
                self._tp_order_id = None

        # Discover existing TP limit orders on Alpaca
        if not tp_ok and not self._tp_order_id:
            try:
                open_orders = self._client.get_orders(GetOrdersRequest(
                    status=QueryOrderStatus.OPEN,
                    symbols=[self._symbol],
                ))
                close_side = "sell" if self._direction == "long" else "buy"
                for o in open_orders:
                    otype = str(getattr(o, 'order_type', None) or getattr(o, 'type', '')).lower()
                    ostatus = str(o.status).lower()
                    oside = str(o.side).lower()
                    is_limit = 'limit' in otype and 'stop' not in otype
                    is_active = any(s in ostatus for s in ("new", "accepted", "held", "pending"))
                    if is_limit and is_active and oside == close_side and str(o.id) != str(self._sl_order_id):
                        self._tp_order_id = o.id
                        limit_px = float(getattr(o, 'limit_price', 0) or 0)
                        self._l.info(
                            f"[RECONCILE] Adopted existing server TP: {o.side} {o.qty} @ ${limit_px:.2f} (order {o.id})"
                        )
                        tp_ok = True
                        break
            except Exception as e:
                self._l.warning(f"Failed to discover existing TP orders: {e}")

        if not tp_ok:
            self._submit_server_tp()

    # ── Main reconciliation loop ─────────────────────────────────────

    def reconcile(self):
        """Authoritative reconciliation: Alpaca position state is the source of truth.

        Called every 30s. Handles every state mismatch between bot and broker:
          - PENDING order filled/cancelled without websocket event
          - CLOSING order filled/cancelled without websocket event
          - Position vanished (external close, margin call, etc.)
          - Orphan position (IDLE but broker has position)
          - Server-side SL missing or expired
          - EOD liquidation
        """
        now = self._now()
        broker_pos = self._get_broker_position()  # (qty, avg_entry) or None

        # ── PENDING states: check order status ──
        if self._state in ("BUY_PENDING", "SHORT_PENDING"):
            if broker_pos is not None:
                # Position exists — order filled, we missed the event
                self._adopt_position(broker_pos[0], broker_pos[1], "missed_fill")
                self._ensure_server_orders()
                return

            # No position yet — check the order
            if self._pending_order_id:
                try:
                    order = self._client.get_order_by_id(str(self._pending_order_id))
                    status = str(order.status).lower()
                    if status == "filled":
                        # Order filled but position query might have raced — adopt from order
                        filled_price = float(order.filled_avg_price or 0)
                        qty = int(order.filled_qty or order.qty or self._shares)
                        if self._direction == "short":
                            qty = -qty
                        self._adopt_position(qty, filled_price, "order_filled")
                        self._ensure_server_orders()
                        return
                    if status in ("canceled", "cancelled", "expired", "rejected"):
                        _slot_tracker.release(self._symbol)
                        self._l.info(f"Entry order {status} | Slots: {_slot_tracker.open_count}/{_slot_tracker.max_slots}")
                        self._state = "IDLE"
                        self._pending_order_id = None
                        return
                    # Still pending — cancel if stale (> 2 min)
                    submitted = pd.Timestamp(order.submitted_at).tz_convert("America/New_York")
                    if now - submitted > pd.Timedelta("2 min"):
                        self._client.cancel_order_by_id(str(self._pending_order_id))
                        _slot_tracker.release(self._symbol)
                        self._l.info(f"Canceled stale entry {self._pending_order_id} | Slots: {_slot_tracker.open_count}/{_slot_tracker.max_slots}")
                        self._state = "IDLE"
                        self._pending_order_id = None
                except Exception as e:
                    self._l.warning(f"Pending order check failed: {e}")
            return

        # ── CLOSING state: check if close order completed ──
        if self._state == "CLOSING":
            if broker_pos is None:
                # Position gone — close completed
                reason = self._exit_reason or "close_confirmed"
                if self._close_order_id:
                    try:
                        order = self._client.get_order_by_id(str(self._close_order_id))
                        if str(order.status).lower() == "filled":
                            exit_price = float(order.filled_avg_price or 0)
                            self._log_exit(exit_price, reason)
                            self._reset_position()
                            return
                    except Exception:
                        pass
                # Couldn't get fill price — estimate from last known
                self._l.info(f"[RECONCILE] Position gone during CLOSING (reason={reason})")
                self._reset_position()
                return

            # Position still exists — close order may have failed
            if self._close_order_id:
                try:
                    order = self._client.get_order_by_id(str(self._close_order_id))
                    status = str(order.status).lower()
                    if status in ("canceled", "cancelled", "expired", "rejected"):
                        self._l.warning(f"Close order {status} — retrying market close")
                        self._close_order_id = None
                        self._state = "LONG" if self._direction == "long" else "SHORT"
                        self._close_position(self._exit_reason or "retry_close")
                        return
                except Exception as e:
                    self._l.warning(f"Close order check failed: {e}")

            # TIMEOUT: stuck in CLOSING for > 2 minutes — cancel and force market close
            if self._closing_since and (now - self._closing_since) > pd.Timedelta("2 min"):
                elapsed = (now - self._closing_since).total_seconds()
                self._close_retries = getattr(self, "_close_retries", 0) + 1
                if self._close_retries > 3:
                    self._l.error(
                        f"[RECONCILE] CLOSING stuck after {self._close_retries} retries — giving up, logging as external_close"
                    )
                    _play_alert("error")
                    self._log_exit(self._entry_price, "stuck_close_abandoned")
                    self._reset_position()
                    return
                self._l.warning(
                    f"[RECONCILE] CLOSING stuck for {elapsed:.0f}s (retry {self._close_retries}/3) — cancelling order and forcing market close"
                )
                _play_alert("warning")
                # Cancel the stale close order
                if self._close_order_id:
                    try:
                        self._client.cancel_order_by_id(str(self._close_order_id))
                        self._l.info(f"Cancelled stuck close order {self._close_order_id}")
                    except Exception:
                        pass  # May already be gone
                    self._close_order_id = None
                # Cancel ALL open orders for THIS symbol to release held shares
                try:
                    from alpaca.trading.requests import GetOrdersRequest
                    from alpaca.trading.enums import QueryOrderStatus
                    open_orders = self._client.get_orders(
                        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[self._symbol])
                    )
                    for o in open_orders:
                        try:
                            self._client.cancel_order_by_id(str(o.id))
                        except Exception:
                            pass
                    if open_orders:
                        self._l.info(f"Cancelled {len(open_orders)} open orders for {self._symbol}")
                except Exception as e:
                    self._l.warning(f"Failed to cancel orders for {self._symbol}: {e}")
                # Revert to active state and force a new market close
                self._state = "LONG" if self._direction == "long" else "SHORT"
                self._close_position(self._exit_reason or "timeout_retry")
            return

        # ── LONG / SHORT: verify position still exists ──
        if self._state in ("LONG", "SHORT"):
            if broker_pos is None:
                # Position vanished — external close, margin call, or server SL
                self._l.warning(f"Position GONE on broker! State was {self._state}")
                # Check if server SL filled
                if self._sl_order_id:
                    try:
                        order = self._client.get_order_by_id(str(self._sl_order_id))
                        if str(order.status).lower() == "filled":
                            exit_price = float(order.filled_avg_price or 0)
                            self._sl_order_id = None
                            self._log_exit(exit_price, "server_stop_loss")
                            self._reset_position()
                            return
                    except Exception:
                        pass
                # Guard: on_trade_update may have already handled this while we were checking
                if self._state not in ("LONG", "SHORT") or self._entry_price == 0.0:
                    self._l.info(f"[RECONCILE] Already handled by trade update — skipping duplicate log")
                    return
                # Unknown exit — log with entry price as exit (P&L=0 since we don't know the price)
                self._l.warning(f"[RECONCILE] External close detected — logging as external_close")
                self._log_exit(self._entry_price, "external_close")
                self._reset_position()
                return

            # Position exists — check for qty mismatch (partial fill/external partial close)
            broker_qty = abs(broker_pos[0])  # broker_pos = (signed_qty, avg_entry_price)
            if broker_qty != self._shares:
                self._l.warning(
                    f"[RECONCILE] Qty mismatch! Bot thinks {self._shares}, broker has {broker_qty} — correcting"
                )
                self._shares = broker_qty
                # Cancel stale SL and resubmit with correct qty
                self._cancel_server_orders()

            # Ensure server SL is active
            self._ensure_server_orders()

            # EOD liquidation check
            if now.time() >= self._cfg.eod_liquidation:
                self._close_position("eod_reconcile")
            return

        # ── IDLE: check for orphan positions ──
        if self._state == "IDLE":
            if broker_pos is not None:
                self._adopt_position(broker_pos[0], broker_pos[1], "orphan")
                self._ensure_server_orders()
            return


# ─── Main Entry Point ───────────────────────────────────────────────────────

def main(args):
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "trading-bot", "config", "config.yaml"
    )
    api_key, api_secret = load_config_from_yaml(config_path)

    # Build config with mode-appropriate defaults
    is_futures = args.mode == "futures"
    is_etf = args.mode == "etf"
    if is_futures:
        params = FUTURES_PARAMS
    elif is_etf:
        params = ETF_PARAMS
    else:
        params = STOCK_PARAMS

    cfg = Config(
        symbols=args.symbols,
        mode=args.mode,
        lot=args.lot,
        compound_pct=args.compound,
        max_slots=args.max_slots,
        contracts=args.contracts,
        data_feed=args.feed,
        paper=not args.live,
        **params,
    )

    # Configure slot tracker
    _slot_tracker.max_slots = cfg.max_slots

    # Validate symbols
    if is_futures:
        for sym in cfg.symbols:
            if sym not in FUTURES_CONTRACTS:
                logger.warning(
                    f"Unknown futures symbol '{sym}'. "
                    f"Known: {', '.join(FUTURES_CONTRACTS.keys())}"
                )
    elif is_etf:
        for sym in cfg.symbols:
            if sym not in ETF_SYMBOLS:
                logger.warning(
                    f"Unknown ETF symbol '{sym}'. "
                    f"Known: {', '.join(ETF_SYMBOLS.keys())}"
                )

    # Initialize clients
    trading_client = TradingClient(api_key, api_secret, paper=cfg.paper)
    stream = StockDataStream(api_key, api_secret, feed=FEED_MAP[cfg.data_feed])

    # Build fleet
    fleet: dict[str, ImprovedScalper] = {}
    for symbol in cfg.symbols:
        fleet[symbol] = ImprovedScalper(trading_client, symbol, cfg)

    # Bar handler
    async def on_bar(bar):
        symbol = bar.symbol
        if symbol in fleet:
            fleet[symbol].on_bar(bar)

    stream.subscribe_bars(on_bar, *cfg.symbols)

    # Track whether we've ever seen the market open
    _saw_market_open = False

    # Periodic checkup
    _heartbeat_tick = 0
    async def periodic():
        nonlocal _saw_market_open, _heartbeat_tick
        while True:
            await asyncio.sleep(30)
            try:
                clock = trading_client.get_clock()
                if clock.is_open:
                    if not _saw_market_open:
                        logger.info("Market is OPEN — bot active")
                        _saw_market_open = True
                elif not clock.is_open:
                    if is_futures:
                        # Futures trade Sun 6pm–Fri 5pm ET; don't exit on stock clock
                        from datetime import datetime as dt
                        now = dt.now()
                        if now.weekday() == 5:  # Saturday
                            logger.info("Saturday — futures market closed, exiting")
                            sys.exit(0)
                        else:
                            logger.debug("Stock market closed but futures may be active, staying alive")
                    elif _saw_market_open:
                        # Market was open and now closed — done for today
                        logger.info("Market closed after trading session, exiting")
                        sys.exit(0)
                    else:
                        # Market hasn't opened yet — wait for it
                        next_open = clock.next_open
                        logger.info(f"Market not yet open. Next open: {next_open}. Waiting...")
            except Exception:
                pass
            for algo in fleet.values():
                algo.reconcile()
            # Periodic heartbeat: show bar counts and states (every 60s)
            _heartbeat_tick += 1
            if _heartbeat_tick % 2 == 0:
                summary_parts = []
                for sym, algo in fleet.items():
                    n = len(algo._bars) if hasattr(algo, '_bars') and algo._bars is not None else 0
                    needed = max(algo._cfg.rsi_period + 2, algo._cfg.vol_ma_period + 1)
                    if algo._state not in ("IDLE",):
                        summary_parts.append(f"{sym}:{algo._state}")
                    elif n < needed:
                        summary_parts.append(f"{sym}:{n}/{needed}")
                    else:
                        summary_parts.append(f"{sym}:IDLE")
                if summary_parts:
                    logger.info(f"Heartbeat: {' | '.join(summary_parts)}")

    # Print startup config
    mode_label = "FUTURES" if is_futures else ("FUTURES ETF" if is_etf else "STOCKS")
    logger.info(f"{'='*60}")
    logger.info(f"  {mode_label} SCALPER — {', '.join(cfg.symbols)}")
    logger.info(f"{'='*60}")
    if is_futures:
        for sym in cfg.symbols:
            spec = FUTURES_CONTRACTS.get(sym, {})
            logger.info(f"  {sym}: {spec.get('name', '?')} | "
                        f"{cfg.contracts} contracts × ${spec.get('point_value', '?')}/pt | "
                        f"Margin: ${spec.get('margin', '?'):,}")
    elif is_etf:
        for sym in cfg.symbols:
            info = ETF_SYMBOLS.get(sym, {})
            logger.info(f"  {sym}: {info.get('name', '?')} (tracks {info.get('tracks', '?')})")
        if cfg.compound_pct > 0:
            logger.info(f"  COMPOUNDING: {cfg.compound_pct*100:.0f}% of equity per trade")
        else:
            logger.info(f"  Lot: ${cfg.lot:,.0f} per trade")
        slots_label = "unlimited" if cfg.max_slots <= 0 else str(cfg.max_slots)
        logger.info(f"  Max concurrent slots: {slots_label}")
        logger.info(f"  Futures-optimized params")
    else:
        logger.info(f"  Lot: ${cfg.lot:,.0f} per trade")
    logger.info(f"  TP: {cfg.tp_pct*100:.2f}% | SL: {cfg.sl_pct*100:.2f}%")
    logger.info(f"  VWAP: {cfg.vwap_lower*100:+.2f}%/{cfg.vwap_upper*100:+.2f}%")
    logger.info(f"  RSI Long: {cfg.rsi_cross_level} | RSI Short: {cfg.rsi_short_level}")
    logger.info(f"  Volume: {cfg.vol_threshold}x | Feed: {cfg.data_feed} | Paper: {cfg.paper}")
    et_now = pd.Timestamp.now(tz="America/New_York")
    logger.info(f"  Clock: {et_now.strftime('%Y-%m-%d %H:%M ET')} | Trade window: {cfg.no_trade_before}–{cfg.no_trade_after} ET | EOD: {cfg.eod_liquidation} ET")
    # Show cached bar status
    cached_count = sum(1 for algo in fleet.values() if len(algo._bars) > 0)
    needed = max(cfg.rsi_period + 2, cfg.vol_ma_period + 1)
    if cached_count > 0:
        ready = sum(1 for algo in fleet.values() if len(algo._bars) >= needed)
        logger.info(f"  Bar cache: {cached_count}/{len(fleet)} symbols loaded, {ready} ready (no warmup needed)")
    else:
        logger.info(f"  Bar cache: empty, {needed}-bar warmup required")
    logger.info(f"{'='*60}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(asyncio.gather(
        stream._run_forever(),
        periodic(),
    ))


if __name__ == "__main__":
    import argparse
    from datetime import timezone as _tz, timedelta as _td

    # Use US Eastern timezone for all log timestamps
    class ETFormatter(logging.Formatter):
        """Formatter that outputs timestamps in US Eastern time."""
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            import zoneinfo
            self._et = zoneinfo.ZoneInfo("America/New_York")

        def formatTime(self, record, datefmt=None):
            from datetime import datetime
            dt = datetime.fromtimestamp(record.created, tz=self._et)
            if datefmt:
                return dt.strftime(datefmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S") + f",{int(record.msecs):03d}"

    fmt = "%(asctime)s ET:%(filename)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s"
    et_formatter = ETFormatter(fmt)

    handler = logging.StreamHandler()
    handler.setFormatter(et_formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    fh = logging.FileHandler("console.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(et_formatter)
    logger.addHandler(fh)

    # Audio alert on errors (rate-limited to 1 per 30s)
    logger.addHandler(AudioAlertHandler())

    parser = argparse.ArgumentParser(
        description="VWAP+RSI scalping bot (stocks & futures)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL MSFT NVDA              # Stocks, default params
  python main.py AAPL --lot 5000 --feed sip  # Stocks, custom lot, SIP feed
  python main.py ES NQ --mode futures         # Futures, 1 contract each
  python main.py MES MNQ --mode futures --contracts 10  # Micros, 10 contracts
  python main.py ES --mode futures --contracts 2 --live # LIVE futures, 2 contracts
        """,
    )
    parser.add_argument("symbols", nargs="+",
                        help="Symbols to trade (stocks: AAPL MSFT | futures: ES NQ | etf: SPY QQQ IWM)")
    parser.add_argument("--mode", default="stocks", choices=["stocks", "futures", "etf"],
                        help="Trading mode (default: stocks)")
    parser.add_argument("--lot", type=float, default=10000,
                        help="Lot size in $ for stocks mode (default: 10000)")
    parser.add_argument("--compound", type=float, default=0.0,
                        help="Compound pct: lot = X%% of equity (0.50 = 50%%). Overrides --lot")
    parser.add_argument("--max-slots", type=int, default=4,
                        help="Max concurrent positions across all symbols (0=unlimited, default: 4)")
    parser.add_argument("--contracts", type=int, default=1,
                        help="Contracts per trade for futures mode (default: 1)")
    parser.add_argument("--feed", default="iex", choices=["iex", "sip"],
                        help="Data feed (default: iex)")
    parser.add_argument("--live", action="store_true",
                        help="Use live trading (default: paper)")

    main(parser.parse_args())

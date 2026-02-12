"""
Trading 212 ISA Scalping Bot — UCITS ETFs on the London Stock Exchange.

Runs the VWAP+RSI+Volume strategy on ISA-eligible UCITS ETFs using:
  - yfinance for hourly price data (signal generation)
  - Trading 212 API for order execution (market/limit/stop orders)

ISA CONSTRAINTS:
  - Long only (no short selling in an ISA)
  - No leverage (lot ≤ available cash)
  - £20,000 annual contribution limit
  - UCITS ETFs on LSE only

Usage:
  # Demo account (default)
  python t212_isa_scalper.py

  # Specific ETFs only
  python t212_isa_scalper.py --symbols VUAG EQQQ SEMI

  # Custom lot size (50% of equity)
  python t212_isa_scalper.py --compound 0.50

  # Live account (REAL MONEY)
  python t212_isa_scalper.py --live
"""

import os
import sys
import json
import time
import logging
import threading
import base64
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, date
from pathlib import Path

try:
    import requests as _requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# ─── Logging ─────────────────────────────────────────────────────────────────

LOG_DIR = Path(__file__).parent / "data"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("t212_isa")

# ─── Trade Logger ────────────────────────────────────────────────────────────

class TradeLogger:
    """Appends completed trades to a JSONL file."""

    def __init__(self):
        self._path = LOG_DIR / "paper_trades_t212_isa.jsonl"

    def log_trade(self, trade: dict):
        with open(self._path, "a") as f:
            f.write(json.dumps(trade, default=str) + "\n")
        logger.info(
            f"[TRADE LOG] {trade['symbol']} {trade['direction']} "
            f"P&L=£{trade.get('pnl', 0):+.2f}"
        )

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

_trade_logger = TradeLogger()


# ─── T212 API Client ────────────────────────────────────────────────────────

class RateLimiter:
    """Per-endpoint rate limiter using token bucket from T212 headers."""

    def __init__(self):
        self._limits: dict[str, dict] = {}
        self._lock = threading.Lock()

    def wait(self, endpoint: str, default_period: float = 5.0):
        """Block until we can make a request to this endpoint."""
        with self._lock:
            if endpoint not in self._limits:
                self._limits[endpoint] = {"next_allowed": 0.0, "period": default_period}

            info = self._limits[endpoint]
            now = time.time()
            if now < info["next_allowed"]:
                wait_time = info["next_allowed"] - now
                logger.debug(f"Rate limit: waiting {wait_time:.1f}s for {endpoint}")
                time.sleep(wait_time)

    def update(self, endpoint: str, headers: dict):
        """Update rate limits from response headers."""
        with self._lock:
            remaining = int(headers.get("x-ratelimit-remaining", "1"))
            period = float(headers.get("x-ratelimit-period", "5"))
            limit = int(headers.get("x-ratelimit-limit", "1"))
            reset_ts = float(headers.get("x-ratelimit-reset", "0"))

            if endpoint not in self._limits:
                self._limits[endpoint] = {}

            self._limits[endpoint]["period"] = period

            if remaining <= 0 and reset_ts > 0:
                self._limits[endpoint]["next_allowed"] = reset_ts
            elif limit == 1:
                # Single-request endpoints: space by period
                self._limits[endpoint]["next_allowed"] = time.time() + period
            else:
                # Burst-capable: no wait unless depleted
                self._limits[endpoint]["next_allowed"] = 0.0


class T212Client:
    """Trading 212 API v0 client with built-in rate limiting."""

    DEMO_BASE = "https://demo.trading212.com/api/v0"
    LIVE_BASE = "https://live.trading212.com/api/v0"

    def __init__(self, api_key: str, api_secret: str, live: bool = False):
        self._base = self.LIVE_BASE if live else self.DEMO_BASE
        self._auth = "Basic " + base64.b64encode(
            f"{api_key}:{api_secret}".encode()
        ).decode()
        self._rate = RateLimiter()
        self._live = live

    def _request(self, method: str, path: str, body: dict | None = None,
                 default_period: float = 5.0) -> dict | list | None:
        """Make an API request with rate limiting and error handling."""
        endpoint = f"{method} {path.split('?')[0]}"
        self._rate.wait(endpoint, default_period)

        url = f"{self._base}{path}"
        headers = {"Authorization": self._auth}

        try:
            if method == "GET":
                resp = _requests.get(url, headers=headers, timeout=15)
            elif method == "POST":
                resp = _requests.post(url, headers=headers, json=body, timeout=15)
            elif method == "DELETE":
                resp = _requests.delete(url, headers=headers, timeout=15)
            else:
                resp = _requests.request(method, url, headers=headers, json=body, timeout=15)

            # Update rate limits from response headers
            self._rate.update(endpoint, dict(resp.headers))

            if resp.status_code == 429:
                logger.warning(f"Rate limited on {endpoint}, backing off...")
                time.sleep(5)
                return self._request(method, path, body, default_period)
            elif resp.status_code == 404:
                return None
            elif resp.status_code >= 400:
                # Known T212 demo ISA limitation — don't spam ERROR
                if "selling-equity-not-owned" in (resp.text or ""):
                    logger.debug(f"T212 API {resp.status_code}: selling-equity-not-owned (expected on demo ISA)")
                elif "insufficient-free" in (resp.text or ""):
                    logger.warning(f"T212 API {resp.status_code}: insufficient funds")
                else:
                    logger.error(f"T212 API {resp.status_code}: {resp.text}")
                return None

            if not resp.text:
                return None
            return resp.json()

        except _requests.exceptions.RequestException as e:
            logger.error(f"T212 connection error: {e}")
            return None

    # ── Account ──

    def account_cash(self) -> dict:
        return self._request("GET", "/equity/account/cash", default_period=2.0)

    def account_info(self) -> dict:
        return self._request("GET", "/equity/account/info", default_period=30.0)

    # ── Portfolio ──

    def positions(self) -> list:
        return self._request("GET", "/equity/portfolio", default_period=5.0) or []

    def position(self, ticker: str) -> dict | None:
        return self._request("GET", f"/equity/portfolio/{ticker}", default_period=1.0)

    # ── Orders ──

    def open_orders(self) -> list:
        return self._request("GET", "/equity/orders", default_period=5.0) or []

    def place_market_order(self, ticker: str, quantity: float) -> dict:
        return self._request("POST", "/equity/orders/market", {
            "ticker": ticker,
            "quantity": quantity,
        }, default_period=1.2)

    def place_limit_order(self, ticker: str, quantity: float,
                          limit_price: float, time_validity: str = "DAY") -> dict:
        return self._request("POST", "/equity/orders/limit", {
            "ticker": ticker,
            "quantity": quantity,
            "limitPrice": limit_price,
            "timeValidity": time_validity,
        }, default_period=2.0)

    def place_stop_order(self, ticker: str, quantity: float,
                         stop_price: float, time_validity: str = "DAY") -> dict:
        return self._request("POST", "/equity/orders/stop", {
            "ticker": ticker,
            "quantity": quantity,
            "stopPrice": stop_price,
            "timeValidity": time_validity,
        }, default_period=2.0)

    def cancel_order(self, order_id: int) -> dict | None:
        return self._request("DELETE", f"/equity/orders/{order_id}", default_period=1.2)


# ─── UCITS ETF Universe ─────────────────────────────────────────────────────

# T212 ticker → Yahoo Finance ticker mapping
# Top 15 ETFs ranked by 2-year hourly backtest P&L (0.3% SL)
# GBP/GBX = no FX fee, USD = 0.15% T212 FX fee
UCITS_UNIVERSE = {
    # GBP instruments
    "VUAGl_EQ":  {"yf": "VUAG.L", "short": "VUAG", "ccy": "GBP", "name": "Vanguard S&P 500 (Acc)"},       # #1  +£4,645
    "VMIDl_EQ":  {"yf": "VMID.L", "short": "VMID", "ccy": "GBP", "name": "Vanguard FTSE 250 (Dist)"},     # #8  +£1,800
    "VEURl_EQ":  {"yf": "VEUR.L", "short": "VEUR", "ccy": "GBP", "name": "Vanguard FTSE Europe (Dist)"},  # #11   +£900
    # GBX instruments (trade in pence — divide by 100 for £)
    "EQQQl_EQ":  {"yf": "EQQQ.L", "short": "EQQQ", "ccy": "GBX", "name": "Invesco NASDAQ-100 (Dist)"},   # #5  +£2,822
    "SGLNl_EQ":  {"yf": "SGLN.L", "short": "SGLN", "ccy": "GBX", "name": "iShares Physical Gold"},        # #4  +£2,828
    "MIDDl_EQ":  {"yf": "MIDD.L", "short": "MIDD", "ccy": "GBX", "name": "iShares FTSE 250 (Dist)"},      # #13   +£772
    "IUKPl_EQ":  {"yf": "IUKP.L", "short": "IUKP", "ccy": "GBX", "name": "iShares UK Property (Dist)"},   # #10 +£1,041
    "HMEFl_EQ":  {"yf": "HMEF.L", "short": "HMEF", "ccy": "GBX", "name": "HSBC MSCI EM (Dist)"},          # #12   +£870
    "HMWOl_EQ":  {"yf": "HMWO.L", "short": "HMWO", "ccy": "GBX", "name": "HSBC MSCI World (Acc)"},        # #14   +£718
    # USD instruments (0.15% T212 FX fee applies)
    "IUITl_EQ":  {"yf": "IUIT.L", "short": "IUIT", "ccy": "USD", "name": "iShares S&P 500 IT Sector"},    # #2  +£2,999
    "LQDEl_EQ":  {"yf": "LQDE.L", "short": "LQDE", "ccy": "USD", "name": "iShares USD Corp Bond"},        # #3  +£2,848
    "ECARl_EQ":  {"yf": "ECAR.L", "short": "ECAR", "ccy": "USD", "name": "iShares Electric Vehicles"},     # #6  +£2,250
    "IUFSl_EQ":  {"yf": "IUFS.L", "short": "IUFS", "ccy": "USD", "name": "iShares S&P 500 Financials"},   # #7  +£2,250
    "ISLNl_EQ":  {"yf": "ISLN.L", "short": "ISLN", "ccy": "USD", "name": "iShares Physical Silver"},       # #9  +£1,199
    "IUESl_EQ":  {"yf": "IUES.L", "short": "IUES", "ccy": "USD", "name": "iShares S&P 500 Energy"},        # #15   +£600
}


# ─── Strategy Parameters ────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    # Same params as UCITS backtest (futures-optimized)
    tp_pct: float = 0.012        # +1.2% take profit
    sl_pct: float = 0.001        # -0.1% stop loss (matches Alpaca ETF params)
    vwap_lower: float = -0.011   # VWAP band lower
    vwap_upper: float = 0.010    # VWAP band upper
    rsi_long_level: float = 30.0 # RSI crossing UP above this → BUY
    vol_threshold: float = 1.0   # Volume > 1x average
    rsi_period: int = 14
    vol_ma_period: int = 20
    max_daily_trades: int = 5
    daily_loss_pct: float = 0.01 # Stop after -1% daily loss
    # Time filters (London time)
    no_trade_before: dtime = dtime(8, 10)
    no_trade_after: dtime = dtime(16, 0)
    eod_liquidation: dtime = dtime(16, 25)
    # Position sizing
    compound_pct: float = 0.50   # 50% of equity per trade
    fixed_lot: float = 0.0       # If > 0, overrides compound_pct (in £)
    max_slots: int = 4           # Max concurrent positions


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
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def compute_vwap(highs, lows, closes, volumes) -> float:
    if len(closes) == 0:
        return 0.0
    typical = (highs + lows + closes) / 3.0
    cum_tv = (typical * volumes).sum()
    cum_v = volumes.sum()
    return cum_tv / cum_v if cum_v > 0 else 0.0


# ─── Slot Tracker ────────────────────────────────────────────────────────────

class SlotTracker:
    def __init__(self, max_slots: int = 4):
        self.max_slots = max_slots
        self._open: set[str] = set()

    @property
    def count(self) -> int:
        return len(self._open)

    def has_free(self) -> bool:
        return self.max_slots <= 0 or self.count < self.max_slots

    def acquire(self, symbol: str):
        self._open.add(symbol)

    def release(self, symbol: str):
        self._open.discard(symbol)

    def is_held(self, symbol: str) -> bool:
        return symbol in self._open


# ─── Per-Symbol Scalper ──────────────────────────────────────────────────────

class UCITSScalper:
    """Manages signal generation + T212 order execution for one ETF."""

    def __init__(self, t212_ticker: str, info: dict, client: T212Client,
                 cfg: StrategyConfig, slots: SlotTracker):
        self.t212_ticker = t212_ticker
        self.yf_ticker = info["yf"]
        self.short_name = info["short"]
        self.ccy = info["ccy"]
        self.name = info["name"]
        self._client = client
        self._cfg = cfg
        self._slots = slots
        self._l = logger.getChild(self.short_name)

        # State: IDLE, ENTERING, LONG, EXITING
        self._state = "IDLE"
        self._entry_price = 0.0
        self._quantity = 0.0
        self._tp_price = 0.0
        self._sl_price = 0.0
        self._entry_time: str | None = None
        self._tp_order_id: int | None = None
        self._sl_order_id: int | None = None
        self._entry_order_id: int | None = None
        self._bracket_placed = False
        self._bracket_attempts = 0
        self._bracket_max_attempts = 1

        # Indicator state
        self._prev_rsi = 50.0
        self._bars: pd.DataFrame | None = None

        # Daily tracking
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._total_pnl = 0.0
        self._total_trades = 0
        self._current_date: date | None = None

    def _price_in_gbp(self, price: float) -> float:
        """Convert price to GBP (GBX instruments are in pence)."""
        if self.ccy == "GBX":
            return price / 100.0
        return price

    def _reset_daily(self):
        today = datetime.now().date()
        if today != self._current_date:
            self._current_date = today
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._l.info(f"New trading day: {today}")

    def fetch_bars(self) -> pd.DataFrame | None:
        """Download recent 1-minute bars from yfinance.
        Matches Alpaca ETF bot behavior — signals checked on every 1-min bar.
        Uses 5d period to ensure enough bars for RSI-14 at market open."""
        try:
            data = yf.download(
                self.yf_ticker,
                period="5d",
                interval="1m",
                auto_adjust=True,
                progress=False,
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if len(data) < 5:
                return None
            if data.index.tz is None:
                data.index = data.index.tz_localize("UTC")
            data.index = data.index.tz_convert("Europe/London")
            # Filter to non-zero volume (trading hours only)
            data = data[data["Volume"] > 0]
            self._bars = data
            return data
        except Exception as e:
            self._l.error(f"yfinance fetch failed: {e}")
            return None

    def check_signal(self) -> str | None:
        """Check for entry signal on latest hourly bar. Returns 'long' or None.
        ISA = long only, no shorts."""
        self._reset_daily()

        if self._bars is None or len(self._bars) < self._cfg.rsi_period + 2:
            return None

        now = datetime.now()
        t = now.time()

        # Time filters
        if t < self._cfg.no_trade_before or t >= self._cfg.no_trade_after:
            return None

        # Daily limits
        if self._daily_trades >= self._cfg.max_daily_trades:
            return None

        # Slot check
        if not self._slots.has_free():
            return None

        # Already in position
        if self._state != "IDLE":
            return None

        closes = self._bars["Close"].values.astype(float)
        highs = self._bars["High"].values.astype(float)
        lows = self._bars["Low"].values.astype(float)
        volumes = self._bars["Volume"].values.astype(float)

        rsi_now = compute_rsi(closes, self._cfg.rsi_period)
        vwap_now = compute_vwap(highs, lows, closes, volumes)

        vol_ma = volumes[-self._cfg.vol_ma_period:].mean()

        if vwap_now > 0:
            vwap_dist = (closes[-1] - vwap_now) / vwap_now
        else:
            vwap_dist = 999

        near_vwap = self._cfg.vwap_lower <= vwap_dist <= self._cfg.vwap_upper
        vol_ok = volumes[-1] > self._cfg.vol_threshold * vol_ma if vol_ma > 0 else False

        signal = None
        if near_vwap and vol_ok:
            # LONG signal: RSI crossing UP above threshold
            if self._prev_rsi < self._cfg.rsi_long_level and rsi_now >= self._cfg.rsi_long_level:
                signal = "long"
                self._l.info(
                    f"SIGNAL: RSI {self._prev_rsi:.1f}→{rsi_now:.1f} | "
                    f"VWAP dist {vwap_dist*100:+.2f}% | Vol {volumes[-1]/vol_ma:.1f}x"
                )

        self._prev_rsi = rsi_now
        return signal

    def enter_position(self, equity: float, cash: float = 0):
        """Place market buy order on T212."""
        if self._state != "IDLE":
            return

        # Get approximate price from last yfinance close
        if self._bars is None or len(self._bars) == 0:
            return
        approx_price = float(self._bars["Close"].iloc[-1])

        # Calculate lot size, capped by available cash
        if self._cfg.fixed_lot > 0:
            lot_gbp = self._cfg.fixed_lot
        else:
            lot_gbp = equity * self._cfg.compound_pct

        # Don't exceed 95% of available cash (T212 holds reserves for spread/fees)
        if cash > 0:
            lot_gbp = min(lot_gbp, cash * 0.95)
            if lot_gbp < 10:
                self._l.warning(f"Insufficient cash: £{cash:.2f} free")
                return

        # Convert price to GBP for lot calculation
        price_gbp = self._price_in_gbp(approx_price)
        if price_gbp <= 0:
            return

        quantity = lot_gbp / price_gbp
        # Round to 2 decimal places (T212 supports fractional)
        quantity = round(quantity, 2)

        if quantity < 0.01:
            self._l.warning(f"Quantity too small: {quantity} (lot £{lot_gbp:.0f}, price £{price_gbp:.2f})")
            return

        self._l.info(
            f"ENTERING: {quantity:.2f} shares @ ~£{price_gbp:.2f} "
            f"(lot £{lot_gbp:,.0f})"
        )

        try:
            result = self._client.place_market_order(self.t212_ticker, quantity)
            if result and "id" in result:
                self._entry_order_id = result["id"]
                self._state = "ENTERING"
                self._quantity = quantity
                self._slots.acquire(self.short_name)
                self._l.info(f"Market order placed: #{result['id']} for {quantity:.2f} shares")
            else:
                self._l.error(f"Order failed: {result}")
        except Exception as e:
            self._l.error(f"Entry order failed: {e}")

    def check_entry_fill(self):
        """Check if our entry market order has filled by looking at positions."""
        if self._state != "ENTERING":
            return

        pos = self._client.position(self.t212_ticker)
        if pos and "quantity" in pos:
            qty = float(pos["quantity"])
            avg_price = float(pos["averagePrice"])

            self._entry_price = avg_price
            self._quantity = qty
            self._entry_time = datetime.now().isoformat()

            # Set TP/SL prices
            self._tp_price = avg_price * (1 + self._cfg.tp_pct)
            self._sl_price = avg_price * (1 - self._cfg.sl_pct)
            self._state = "LONG"

            price_gbp = self._price_in_gbp(avg_price)
            tp_gbp = self._price_in_gbp(self._tp_price)
            sl_gbp = self._price_in_gbp(self._sl_price)

            self._l.info(
                f"FILLED LONG: {qty:.2f} @ £{price_gbp:.2f} | "
                f"TP=£{tp_gbp:.2f} (+{self._cfg.tp_pct*100:.1f}%) | "
                f"SL=£{sl_gbp:.2f} (-{self._cfg.sl_pct*100:.1f}%)"
            )

            # Best-effort bracket orders (T212 demo ISA rejects limit/stop
            # sells — "selling-equity-not-owned").  Try once, then rely on
            # polling TP/SL which is always active.
            self._bracket_attempts = 0
            self._bracket_max_attempts = 1
            self._bracket_placed = False
            self._l.info("Polling TP/SL active — will attempt bracket orders on next scan")

    def _try_place_bracket_orders(self):
        """Best-effort bracket order placement with retry across scan cycles.
        Called from monitor_position(). If all attempts fail, polling handles exits."""
        if self._bracket_placed or self._bracket_attempts >= self._bracket_max_attempts:
            return

        self._bracket_attempts += 1
        self._l.info(f"Bracket order attempt {self._bracket_attempts}/{self._bracket_max_attempts}")

        tp_ok = False
        sl_ok = False

        try:
            tp_result = self._client.place_limit_order(
                self.t212_ticker,
                -self._quantity,
                round(self._tp_price, 4),
                "GOOD_TILL_CANCEL",
            )
            if tp_result and "id" in tp_result:
                self._tp_order_id = tp_result["id"]
                self._l.info(f"TP limit sell placed: #{tp_result['id']} @ {self._tp_price:.4f}")
                tp_ok = True
            else:
                self._l.warning(f"TP order rejected: {tp_result}")
        except Exception as e:
            self._l.warning(f"TP order failed: {e}")

        time.sleep(2.5)

        try:
            sl_result = self._client.place_stop_order(
                self.t212_ticker,
                -self._quantity,
                round(self._sl_price, 4),
                "GOOD_TILL_CANCEL",
            )
            if sl_result and "id" in sl_result:
                self._sl_order_id = sl_result["id"]
                self._l.info(f"SL stop sell placed: #{sl_result['id']} @ {self._sl_price:.4f}")
                sl_ok = True
            else:
                self._l.warning(f"SL order rejected: {sl_result}")
        except Exception as e:
            self._l.warning(f"SL order failed: {e}")

        if tp_ok and sl_ok:
            self._bracket_placed = True
            self._l.info("Both bracket orders placed successfully")
        elif self._bracket_attempts >= self._bracket_max_attempts:
            self._l.info("Bracket orders exhausted retries — polling TP/SL is active")

    def monitor_position(self):
        """Check if TP or SL has been hit.
        Primary strategy: polling-based TP/SL (always active).
        Secondary: bracket orders (best-effort, retried across scan cycles).
        """
        if self._state != "LONG":
            return

        # Try to place bracket orders if not yet placed (deferred retry)
        if not self._bracket_placed and self._bracket_attempts < self._bracket_max_attempts:
            self._try_place_bracket_orders()

        # Check if position still exists
        pos = self._client.position(self.t212_ticker)

        if pos is None:
            # Position closed — bracket order filled or manual close
            exit_reason = "unknown"
            exit_price = self._entry_price  # Fallback

            # Determine which bracket order filled
            if self._tp_order_id or self._sl_order_id:
                time.sleep(2)
                orders = self._client.open_orders()
                open_ids = {o["id"] for o in orders} if orders else set()

                if self._tp_order_id and self._tp_order_id not in open_ids:
                    exit_reason = "take_profit"
                    exit_price = self._tp_price
                    if self._sl_order_id and self._sl_order_id in open_ids:
                        self._client.cancel_order(self._sl_order_id)
                elif self._sl_order_id and self._sl_order_id not in open_ids:
                    exit_reason = "stop_loss"
                    exit_price = self._sl_price
                    if self._tp_order_id and self._tp_order_id in open_ids:
                        self._client.cancel_order(self._tp_order_id)

            self._record_exit(exit_price, exit_reason)
            return

        # Position still open — ALWAYS check P&L for polling-based TP/SL
        # This is the primary exit mechanism (bracket orders are best-effort)
        current_price = float(pos.get("currentPrice", 0))
        ppl = float(pos.get("ppl", 0))  # Unrealized P&L in account currency

        if current_price > 0 and self._entry_price > 0:
            pnl_pct = (current_price / self._entry_price - 1)

            if pnl_pct >= self._cfg.tp_pct:
                self._l.info(f"TP hit via polling: {pnl_pct*100:+.2f}% >= {self._cfg.tp_pct*100:.1f}%")
                self._close_position("take_profit_poll")
                return
            if pnl_pct <= -self._cfg.sl_pct:
                self._l.info(f"SL hit via polling: {pnl_pct*100:+.2f}% <= -{self._cfg.sl_pct*100:.1f}%")
                self._close_position("stop_loss_poll")
                return

        # EOD flatten check
        now = datetime.now()
        if now.time() >= self._cfg.eod_liquidation:
            self._close_position("eod_flatten")

    def _record_exit(self, exit_price: float, reason: str):
        """Record a closed trade and reset state."""
        price_gbp = self._price_in_gbp(exit_price)
        entry_gbp = self._price_in_gbp(self._entry_price)
        pnl = (price_gbp - entry_gbp) * self._quantity

        self._daily_pnl += pnl
        self._daily_trades += 1
        self._total_pnl += pnl
        self._total_trades += 1

        self._l.info(
            f"CLOSED ({reason}): {self._quantity:.2f} @ £{price_gbp:.2f} | "
            f"P&L=£{pnl:+.2f} | Day=£{self._daily_pnl:+.2f} | "
            f"Total=£{self._total_pnl:+.2f} ({self._total_trades} trades)"
        )

        _trade_logger.log_trade({
            "broker": "t212_isa",
            "symbol": self.short_name,
            "t212_ticker": self.t212_ticker,
            "direction": "long",
            "quantity": self._quantity,
            "entry_price": self._entry_price,
            "exit_price": exit_price,
            "entry_time": self._entry_time,
            "exit_time": datetime.now().isoformat(),
            "pnl": round(pnl, 2),
            "pnl_pct": round((exit_price / self._entry_price - 1) * 100, 4),
            "exit_reason": reason,
            "ccy": self.ccy,
            "daily_trade_num": self._daily_trades,
            "cumulative_pnl": round(self._total_pnl, 2),
        })

        self._reset_position()

    def _close_position(self, reason: str):
        """Close position at market (EOD flatten or manual)."""
        if self._state not in ("LONG", "ENTERING"):
            return

        self._l.info(f"CLOSING ({reason}): selling {self._quantity:.2f} shares")
        self._state = "EXITING"

        # Cancel TP and SL orders first
        for oid in [self._tp_order_id, self._sl_order_id]:
            if oid:
                try:
                    self._client.cancel_order(oid)
                    time.sleep(1.5)
                except Exception:
                    pass

        # Sell at market — T212 uses negative qty for sell direction
        try:
            result = self._client.place_market_order(self.t212_ticker, -self._quantity)
            if result:
                self._l.info(f"Sell market order placed: {result.get('id', '?')}")
                # Wait for fill, then check
                time.sleep(3)
                pos = self._client.position(self.t212_ticker)
                if pos is None:
                    # Sold — get exit price estimate
                    if self._bars is not None and len(self._bars) > 0:
                        exit_price = float(self._bars["Close"].iloc[-1])
                    else:
                        exit_price = self._entry_price
                    self._record_exit(exit_price, reason)
                else:
                    self._l.warning("Position still open after sell — retrying")
                    self._state = "LONG"
            else:
                self._l.error("Sell order returned None")
                self._state = "LONG"
        except Exception as e:
            self._l.error(f"Close failed: {e}")
            self._state = "LONG"

    def _reset_position(self):
        self._slots.release(self.short_name)
        self._state = "IDLE"
        self._entry_price = 0.0
        self._quantity = 0.0
        self._tp_price = 0.0
        self._sl_price = 0.0
        self._entry_time = None
        self._tp_order_id = None
        self._sl_order_id = None
        self._entry_order_id = None
        self._bracket_placed = False
        self._bracket_attempts = 0
        self._bracket_max_attempts = 1

    def recover_position(self):
        """Check T212 for any existing position on startup."""
        pos = self._client.position(self.t212_ticker)
        if pos and "quantity" in pos:
            qty = float(pos["quantity"])
            if qty > 0:
                avg_price = float(pos["averagePrice"])
                self._entry_price = avg_price
                self._quantity = qty
                self._tp_price = avg_price * (1 + self._cfg.tp_pct)
                self._sl_price = avg_price * (1 - self._cfg.sl_pct)
                self._state = "LONG"
                self._slots.acquire(self.short_name)

                price_gbp = self._price_in_gbp(avg_price)
                self._l.info(
                    f"Recovered LONG: {qty:.2f} @ £{price_gbp:.2f} | "
                    f"TP=£{self._price_in_gbp(self._tp_price):.2f} | "
                    f"SL=£{self._price_in_gbp(self._sl_price):.2f}"
                )

                # Bracket orders will be attempted on next scan cycles
                # (deferred retry — T212 demo ISA has position sync delays)
                self._bracket_attempts = 0
                self._bracket_placed = False
                return True
        return False


# ─── Main Loop ───────────────────────────────────────────────────────────────

def run_bot(api_key: str, api_secret: str, symbols: list[str] | None = None,
            live: bool = False, compound: float = 0.50, fixed_lot: float = 0.0,
            max_slots: int = 4):
    """Main bot loop."""

    client = T212Client(api_key, api_secret, live=live)
    cfg = StrategyConfig(
        compound_pct=compound,
        fixed_lot=fixed_lot,
        max_slots=max_slots,
    )
    slots = SlotTracker(max_slots)

    # Determine which ETFs to trade
    if symbols:
        # Map short names to T212 tickers
        universe = {}
        for t212, info in UCITS_UNIVERSE.items():
            if info["short"] in symbols:
                universe[t212] = info
    else:
        universe = UCITS_UNIVERSE.copy()

    if not universe:
        logger.error("No valid symbols found in T212 universe")
        return

    # Verify account
    account = client.account_cash()
    if not account:
        logger.error("Cannot connect to T212 API")
        return

    cash = float(account.get("free", 0))
    total = float(account.get("total", 0))
    invested = float(account.get("invested", 0))
    mode = "LIVE" if live else "DEMO"

    logger.info("=" * 70)
    logger.info(f"  T212 ISA SCALPER — {mode}")
    logger.info("=" * 70)
    logger.info(f"  Account: £{total:,.2f} total | £{cash:,.2f} cash | £{invested:,.2f} invested")
    logger.info(f"  Universe: {len(universe)} UCITS ETFs")
    logger.info(f"  Strategy: LONG ONLY (ISA constraint)")
    logger.info(f"  TP: {cfg.tp_pct*100:.1f}% | SL: {cfg.sl_pct*100:.1f}%")
    logger.info(f"  VWAP: {cfg.vwap_lower*100:+.1f}%/{cfg.vwap_upper*100:+.1f}%")
    logger.info(f"  RSI Long: {cfg.rsi_long_level}")
    if cfg.fixed_lot > 0:
        logger.info(f"  Lot: £{cfg.fixed_lot:,.0f} fixed")
    else:
        logger.info(f"  Lot: {cfg.compound_pct*100:.0f}% of equity (£{total * cfg.compound_pct:,.0f})")
    logger.info(f"  Max slots: {cfg.max_slots}")
    logger.info(f"  Trade window: {cfg.no_trade_before}–{cfg.no_trade_after} London")
    logger.info(f"  EOD flatten: {cfg.eod_liquidation} London")
    for t212, info in universe.items():
        logger.info(f"    {info['short']:<6} {info['name']:<40} ({info['ccy']})")
    logger.info("=" * 70)

    # Build fleet
    fleet: dict[str, UCITSScalper] = {}
    for t212_ticker, info in universe.items():
        scalper = UCITSScalper(t212_ticker, info, client, cfg, slots)
        fleet[info["short"]] = scalper
        time.sleep(1.5)  # Rate limit on position checks

    # Recover any existing positions
    logger.info("Checking for existing positions...")
    for name, scalper in fleet.items():
        scalper.recover_position()
        time.sleep(3)  # Individual position endpoint: 1 req/1s, but pad for safety

    # ── Main loop ──
    logger.info("Bot running. Fetching 1-min bars every 60s (same as Alpaca ETF bot)...")
    scan_count = 0

    while True:
        try:
            now = datetime.now()
            t = now.time()

            # ── Before market: sleep ──
            if t < dtime(7, 55):
                next_wake = now.replace(hour=7, minute=55, second=0)
                sleep_secs = (next_wake - now).total_seconds()
                if sleep_secs > 0:
                    logger.info(f"Pre-market. Sleeping until 07:55 ({sleep_secs/60:.0f}min)...")
                    time.sleep(min(sleep_secs, 300))  # Wake every 5min max
                continue

            # ── After market: done for today ──
            if t >= dtime(16, 35):
                # Make sure everything is flat
                for name, scalper in fleet.items():
                    if scalper._state == "LONG":
                        scalper._close_position("eod_final")
                        time.sleep(5)

                logger.info(f"Market closed. Day summary:")
                for name, scalper in fleet.items():
                    if scalper._daily_trades > 0:
                        logger.info(
                            f"  {name}: {scalper._daily_trades} trades, "
                            f"£{scalper._daily_pnl:+.2f}"
                        )
                total_day = sum(s._daily_pnl for s in fleet.values())
                total_trades = sum(s._daily_trades for s in fleet.values())
                logger.info(f"  TOTAL: {total_trades} trades, £{total_day:+.2f}")

                # Sleep until next trading day
                logger.info("Sleeping until next market open...")
                time.sleep(3600)  # Check again in 1h
                continue

            # ── Fetch 1-min bars for IDLE symbols (every scan = every 60s) ──
            for name, scalper in fleet.items():
                if scalper._state != "IDLE":
                    continue
                scalper.fetch_bars()
                time.sleep(0.5)  # Don't hammer yfinance

            # ── Check signals on latest 1-min bar ──
            for name, scalper in fleet.items():
                signal = scalper.check_signal()
                if signal == "long":
                    # Get current equity + cash for lot sizing
                    try:
                        acct = client.account_cash()
                        if acct is None:
                            logger.warning("account_cash() returned None (rate limited?) — skipping entry")
                            time.sleep(3)
                            continue
                        equity = float(acct.get("total", 5000))
                        cash = float(acct.get("free", 0))
                        logger.info(f"Account: £{equity:.2f} total, £{cash:.2f} free")
                        if cash < 50:
                            logger.warning(f"Insufficient free cash: £{cash:.2f}")
                            continue
                    except Exception as e:
                        logger.warning(f"Failed to get account cash: {e} — skipping entry")
                        continue
                    time.sleep(3)
                    scalper.enter_position(equity, cash=cash)
                    time.sleep(3)

            # ── Check entry fills ──
            for name, scalper in fleet.items():
                if scalper._state == "ENTERING":
                    scalper.check_entry_fill()
                    time.sleep(2)

            # ── Monitor open positions ──
            for name, scalper in fleet.items():
                if scalper._state == "LONG":
                    scalper.monitor_position()
                    time.sleep(2)

            # ── EOD flatten check ──
            if t >= cfg.eod_liquidation:
                for name, scalper in fleet.items():
                    if scalper._state == "LONG":
                        scalper._close_position("eod_flatten")
                        time.sleep(5)

            scan_count += 1
            if scan_count % 10 == 0:
                open_positions = [n for n, s in fleet.items() if s._state != "IDLE"]
                if open_positions:
                    logger.info(f"Scan #{scan_count} | Open: {', '.join(open_positions)} | Slots: {slots.count}/{slots.max_slots}")
                else:
                    logger.debug(f"Scan #{scan_count} | All idle | Slots: {slots.count}/{slots.max_slots}")

            # Sleep between scans — 60s matches the 1-min bar granularity
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            time.sleep(30)

    # Final summary
    logger.info("=" * 70)
    logger.info("SESSION SUMMARY")
    logger.info("=" * 70)
    for name, scalper in fleet.items():
        if scalper._total_trades > 0:
            logger.info(
                f"  {name}: {scalper._total_trades} trades, "
                f"£{scalper._total_pnl:+.2f}"
            )
    total_pnl = sum(s._total_pnl for s in fleet.values())
    total_trades = sum(s._total_trades for s in fleet.values())
    logger.info(f"  TOTAL: {total_trades} trades, £{total_pnl:+.2f}")
    logger.info("=" * 70)


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    # Logging setup
    class LondonFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            import zoneinfo
            dt = datetime.fromtimestamp(record.created, tz=zoneinfo.ZoneInfo("Europe/London"))
            if datefmt:
                return dt.strftime(datefmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = LondonFormatter(fmt)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    # File handler
    fh = logging.FileHandler(LOG_DIR / "t212_isa.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    parser = argparse.ArgumentParser(
        description="T212 ISA UCITS ETF Scalping Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python t212_isa_scalper.py                            # All 15 UCITS ETFs, demo
  python t212_isa_scalper.py --symbols VUAG EQQQ SEMI   # Specific ETFs only
  python t212_isa_scalper.py --compound 0.75             # 75% of equity per trade
  python t212_isa_scalper.py --lot 2000                  # Fixed £2k per trade
  python t212_isa_scalper.py --live                      # REAL MONEY (be careful!)
        """,
    )
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Short names to trade (e.g., VUAG EQQQ SEMI)")
    parser.add_argument("--compound", type=float, default=0.50,
                        help="Compound pct: lot = X%% of equity (default: 0.50 = 50%%)")
    parser.add_argument("--lot", type=float, default=0.0,
                        help="Fixed lot in £ (overrides --compound)")
    parser.add_argument("--max-slots", type=int, default=4,
                        help="Max concurrent positions (default: 4)")
    parser.add_argument("--live", action="store_true",
                        help="Use LIVE account (REAL MONEY)")
    parser.add_argument("--key", default=None,
                        help="T212 API key (or set T212_API_KEY env var)")
    parser.add_argument("--secret", default=None,
                        help="T212 API secret (or set T212_API_SECRET env var)")

    args = parser.parse_args()

    # Get credentials
    api_key = args.key or os.environ.get("T212_API_KEY", "45522990ZZTQMECLiJqefhFLieZHyvNKVxXlA")
    api_secret = args.secret or os.environ.get("T212_API_SECRET", "Ra-rSfv9KlqVTRCXt-JSSb-WbfMB3gMNMnL9yS0On2Y")

    if not api_key or not api_secret:
        print("ERROR: Set T212_API_KEY and T212_API_SECRET environment variables")
        print("  or pass --key and --secret arguments")
        sys.exit(1)

    if args.live:
        logger.warning("=" * 70)
        logger.warning("  *** LIVE MODE — REAL MONEY ***")
        logger.warning("  This will trade with real money in your ISA!")
        logger.warning("  Press Ctrl+C within 10 seconds to abort...")
        logger.warning("=" * 70)
        time.sleep(10)

    run_bot(
        api_key=api_key,
        api_secret=api_secret,
        symbols=args.symbols,
        live=args.live,
        compound=args.compound,
        fixed_lot=args.lot,
        max_slots=args.max_slots,
    )

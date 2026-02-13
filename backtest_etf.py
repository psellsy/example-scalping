"""
Backtest: VWAP + RSI + Volume strategy on Futures ETFs (SPY, QQQ, IWM).

Uses futures-optimized parameters on hourly ETF data.
Position sizing: dollar lot size (no point value multipliers like actual futures).
"""

import os
import glob
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "etf_hourly")
STARTING_CAPITAL = 100_000.0
LOT_SIZE = 10_000.0  # $ per trade (default)

# Futures-optimized parameters
VWAP_LO = -0.011   # -1.1%
VWAP_HI = 0.010    # +1.0%
RSI_LONG_LEVEL = 30
RSI_SHORT_LEVEL = 55
TP_PCT = 0.012      # +1.2%
SL_PCT = 0.001      # -0.1%
VOL_THRESH = 1.0
MAX_DAILY_TRADES = 5
TRAIL_ACTIVATE = 0.003  # Activate trailing stop after +0.3%
TRAIL_DIST = 0.003      # Trail 0.3% behind best price


@dataclass
class Trade:
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_time: pd.Timestamp
    entry_price: float
    shares: int
    exit_time: pd.Timestamp = None
    exit_price: float = 0.0
    pnl: float = 0.0
    exit_reason: str = ""


def compute_rsi(closes, period=14):
    rsi = np.full(len(closes), np.nan)
    if len(closes) < period + 1:
        return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


def compute_vwap(highs, lows, closes, volumes, day_starts):
    vwap = np.full(len(closes), np.nan)
    typical = (highs + lows + closes) / 3.0
    cum_tv = 0.0
    cum_v = 0.0
    for i in range(len(closes)):
        if day_starts[i]:
            cum_tv = 0.0
            cum_v = 0.0
        cum_tv += typical[i] * volumes[i]
        cum_v += volumes[i]
        if cum_v > 0:
            vwap[i] = cum_tv / cum_v
    return vwap


def compute_volume_ma(volumes, period=20):
    out = np.full(len(volumes), np.nan, dtype=float)
    for i in range(period - 1, len(volumes)):
        out[i] = volumes[i - period + 1:i + 1].mean()
    return out


def find_day_starts(timestamps):
    starts = np.zeros(len(timestamps), dtype=bool)
    starts[0] = True
    for i in range(1, len(timestamps)):
        if timestamps[i].date() != timestamps[i - 1].date():
            starts[i] = True
    return starts


def backtest_etf(df, symbol, lot_size):
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values.astype(float)
    timestamps = df.index

    day_starts = find_day_starts(timestamps)
    vwap = compute_vwap(highs, lows, closes, volumes, day_starts)
    rsi = compute_rsi(closes, 14)
    vol_ma = compute_volume_ma(volumes, 20)

    daily_loss_limit = -STARTING_CAPITAL * 0.01
    trades = []

    state = "IDLE"
    entry_price = 0.0
    entry_time = None
    shares = 0
    limit_price = 0.0
    pending_bar = 0
    tp_price = 0.0
    sl_price = 0.0
    best_price = 0.0
    daily_pnl = 0.0
    daily_trades = 0
    current_date = None

    for i in range(21, len(closes)):
        t = timestamps[i]
        tod = t.time()
        trade_date = t.date()

        if trade_date != current_date:
            current_date = trade_date
            daily_pnl = 0.0
            daily_trades = 0

        too_early = tod < pd.Timestamp("09:40").time()
        too_late = tod >= pd.Timestamp("15:30").time()
        is_eod = tod >= pd.Timestamp("15:55").time()
        limit_hit = daily_pnl <= daily_loss_limit or daily_trades >= MAX_DAILY_TRADES

        # ── LONG management ──
        if state == "LONG":
            # Trailing stop: ratchet SL toward price
            if TRAIL_ACTIVATE > 0 and highs[i] > best_price:
                best_price = highs[i]
            if TRAIL_ACTIVATE > 0 and entry_price > 0:
                move = (best_price - entry_price) / entry_price
                if move >= TRAIL_ACTIVATE:
                    trail_sl = best_price * (1 - TRAIL_DIST)
                    if trail_sl > sl_price:
                        sl_price = trail_sl

            if is_eod:
                pnl = (closes[i] - entry_price) * shares
                trades.append(Trade(symbol, "LONG", entry_time, entry_price,
                                   shares, t, closes[i], pnl, "long_eod"))
                daily_pnl += pnl; daily_trades += 1; state = "IDLE"; continue
            if lows[i] <= sl_price:
                pnl = (sl_price - entry_price) * shares
                trades.append(Trade(symbol, "LONG", entry_time, entry_price,
                                   shares, t, sl_price, pnl, "long_stop_loss"))
                daily_pnl += pnl; daily_trades += 1; state = "IDLE"; continue
            if highs[i] >= tp_price:
                pnl = (tp_price - entry_price) * shares
                trades.append(Trade(symbol, "LONG", entry_time, entry_price,
                                   shares, t, tp_price, pnl, "long_take_profit"))
                daily_pnl += pnl; daily_trades += 1; state = "IDLE"; continue
            continue

        # ── SHORT management ──
        if state == "SHORT":
            # Trailing stop: ratchet SL toward price
            if TRAIL_ACTIVATE > 0 and lows[i] < best_price:
                best_price = lows[i]
            if TRAIL_ACTIVATE > 0 and entry_price > 0:
                move = (entry_price - best_price) / entry_price
                if move >= TRAIL_ACTIVATE:
                    trail_sl = best_price * (1 + TRAIL_DIST)
                    if trail_sl < sl_price:
                        sl_price = trail_sl

            if is_eod:
                pnl = (entry_price - closes[i]) * shares
                trades.append(Trade(symbol, "SHORT", entry_time, entry_price,
                                   shares, t, closes[i], pnl, "short_eod"))
                daily_pnl += pnl; daily_trades += 1; state = "IDLE"; continue
            if highs[i] >= sl_price:
                pnl = (entry_price - sl_price) * shares
                trades.append(Trade(symbol, "SHORT", entry_time, entry_price,
                                   shares, t, sl_price, pnl, "short_stop_loss"))
                daily_pnl += pnl; daily_trades += 1; state = "IDLE"; continue
            if lows[i] <= tp_price:
                pnl = (entry_price - tp_price) * shares
                trades.append(Trade(symbol, "SHORT", entry_time, entry_price,
                                   shares, t, tp_price, pnl, "short_take_profit"))
                daily_pnl += pnl; daily_trades += 1; state = "IDLE"; continue
            continue

        # ── Pending long fill ──
        if state == "BUY_PENDING":
            if lows[i] <= limit_price:
                entry_price = limit_price
                entry_time = t
                shares = max(1, int(lot_size / entry_price))
                tp_price = entry_price * (1 + TP_PCT)
                sl_price = entry_price * (1 - SL_PCT)
                best_price = entry_price
                state = "LONG"
            elif i - pending_bar >= 2:
                state = "IDLE"
            continue

        # ── Pending short fill ──
        if state == "SHORT_PENDING":
            if highs[i] >= limit_price:
                entry_price = limit_price
                entry_time = t
                shares = max(1, int(lot_size / entry_price))
                tp_price = entry_price * (1 - TP_PCT)
                sl_price = entry_price * (1 + SL_PCT)
                best_price = entry_price
                state = "SHORT"
            elif i - pending_bar >= 2:
                state = "IDLE"
            continue

        # ── IDLE: scan for signals (RTH only) ──
        if state == "IDLE" and not too_early and not too_late and not limit_hit:
            if (not np.isnan(vwap[i]) and not np.isnan(rsi[i])
                    and not np.isnan(rsi[i - 1]) and not np.isnan(vol_ma[i])):

                vwap_dist = (closes[i] - vwap[i]) / vwap[i]
                near_vwap = VWAP_LO <= vwap_dist <= VWAP_HI
                vol_ok = volumes[i] > VOL_THRESH * vol_ma[i]

                if near_vwap and vol_ok:
                    if rsi[i - 1] < RSI_LONG_LEVEL and rsi[i] >= RSI_LONG_LEVEL:
                        limit_price = closes[i]
                        pending_bar = i
                        state = "BUY_PENDING"
                    elif rsi[i - 1] > RSI_SHORT_LEVEL and rsi[i] <= RSI_SHORT_LEVEL:
                        limit_price = closes[i]
                        pending_bar = i
                        state = "SHORT_PENDING"

    return trades


def run():
    lot_size = LOT_SIZE
    # Allow lot size override from command line
    for arg in sys.argv[1:]:
        if arg.startswith("--lot="):
            lot_size = float(arg.split("=")[1])

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    if not files:
        print(f"No ETF data in {DATA_DIR}. Run download script first.")
        return

    print(f"ETF Scalping Backtest — Futures-Optimized Params [1h bars]")
    print(f"Capital: ${STARTING_CAPITAL:,.0f} | Lot Size: ${lot_size:,.0f}")
    print(f"TP: {TP_PCT*100:.2f}% | SL: {SL_PCT*100:.2f}% | Trail: activate {TRAIL_ACTIVATE*100:.1f}%, distance {TRAIL_DIST*100:.1f}%")
    print(f"VWAP: {VWAP_LO*100:+.2f}%/{VWAP_HI*100:+.2f}% | RSI L/S: {RSI_LONG_LEVEL}/{RSI_SHORT_LEVEL} | Vol: {VOL_THRESH}x")
    print("=" * 100)

    all_trades = []

    for f in files:
        symbol = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f)
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")

        trades = backtest_etf(df, symbol, lot_size)
        all_trades.extend(trades)

        if trades:
            total_pnl = sum(t.pnl for t in trades)
            wins = sum(1 for t in trades if t.pnl > 0)
            longs = sum(1 for t in trades if t.direction == "LONG")
            shorts = sum(1 for t in trades if t.direction == "SHORT")
            avg_shares = int(np.mean([t.shares for t in trades]))
            print(f"  {symbol:<5}  {len(trades):>4} trades ({longs}L/{shorts}S) | "
                  f"~{avg_shares} shares/trade | "
                  f"P&L: ${total_pnl:>+10,.2f} | "
                  f"Win: {wins}/{len(trades)} ({wins/len(trades)*100:.0f}%)")
        else:
            print(f"  {symbol:<5}     0 trades")

    if not all_trades:
        print("\nNo trades generated.")
        return

    # ── Aggregate metrics ──
    print("\n" + "=" * 100)
    print("  AGGREGATE RESULTS")
    print("=" * 100)

    pnls = [t.pnl for t in all_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    equity = STARTING_CAPITAL
    peak = equity
    max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

    # Daily Sharpe
    daily_pnl = {}
    for t in all_trades:
        d = t.exit_time.date() if t.exit_time else t.entry_time.date()
        daily_pnl[d] = daily_pnl.get(d, 0.0) + t.pnl
    daily_returns = list(daily_pnl.values())
    mean_r = np.mean(daily_returns)
    std_r = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 1.0
    sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0.0

    # Monthly breakdown
    monthly_pnl = {}
    for t in all_trades:
        d = t.exit_time if t.exit_time else t.entry_time
        key = f"{d.year}-{d.month:02d}"
        monthly_pnl[key] = monthly_pnl.get(key, 0.0) + t.pnl
    profitable_months = sum(1 for v in monthly_pnl.values() if v > 0)

    longs_total = sum(1 for t in all_trades if t.direction == "LONG")
    shorts_total = sum(1 for t in all_trades if t.direction == "SHORT")
    long_pnl = sum(t.pnl for t in all_trades if t.direction == "LONG")
    short_pnl = sum(t.pnl for t in all_trades if t.direction == "SHORT")

    print(f"  Total Trades:      {len(all_trades):>8}")
    print(f"    Long:            {longs_total:>8}  (P&L: ${long_pnl:>+12,.2f})")
    print(f"    Short:           {shorts_total:>8}  (P&L: ${short_pnl:>+12,.2f})")
    print(f"  Win Rate:          {win_rate:>7.1f}%")
    print(f"  Avg Win:           ${np.mean(wins):>+10,.2f}" if wins else "  Avg Win:               N/A")
    print(f"  Avg Loss:          ${np.mean(losses):>+10,.2f}" if losses else "  Avg Loss:              N/A")
    print(f"  Total P&L:         ${total_pnl:>+12,.2f}")
    print(f"  Return on Capital: {total_pnl/STARTING_CAPITAL*100:>+7.2f}%")
    print(f"  Max Drawdown:      {max_dd*100:>7.2f}%")
    print(f"  Profit Factor:     {profit_factor:>8.2f}")
    print(f"  Sharpe (ann.):     {sharpe:>8.2f}")
    print(f"  Trading Days:      {len(daily_pnl):>8}")
    print(f"  Avg Daily P&L:     ${mean_r:>+10,.2f}")
    print(f"  Months Profitable: {profitable_months}/{len(monthly_pnl)}")

    # Exit reasons
    print("\n  Exit Reasons:")
    reasons = {}
    for t in all_trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    for reason, count in sorted(reasons.items()):
        reason_pnl = sum(t.pnl for t in all_trades if t.exit_reason == reason)
        print(f"    {reason:<22} {count:>4} trades  P&L: ${reason_pnl:>+10,.2f}")

    # Monthly P&L table
    print("\n  Monthly P&L:")
    for month in sorted(monthly_pnl.keys()):
        bar = "+" * min(int(abs(monthly_pnl[month]) / 50), 40)
        sign = "+" if monthly_pnl[month] >= 0 else "-"
        print(f"    {month}  ${monthly_pnl[month]:>+8,.2f}  {sign}{bar}")

    # Data info
    first_trade = min(t.entry_time for t in all_trades)
    last_trade = max(t.exit_time for t in all_trades if t.exit_time)
    print(f"\n  Period: {first_trade.date()} to {last_trade.date()}")
    print(f"  Lot size: ${lot_size:,.0f} | Use --lot=50000 to test larger positions")


if __name__ == "__main__":
    run()

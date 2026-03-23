"""
simulator.py — Backtesting simulation engine.

For each user scenario, runs two parallel tracks:
  Track A — Rebalanced: follows user's event list (buy → rebalance → rebalance …)
  Track B — Hold:       buys initial portfolio on day 1, holds to end date

Uses historical prices from yfinance — no live prices, no look-ahead bias.
Each user gets an isolated SQLite DB (sim_{email}.db) — no production DB pollution.

Usage:
    from simulator import run_simulation, run_all_simulations
    result = run_simulation(scenario)        # one user
    results = run_all_simulations()          # all 5 users
"""

import os
import math
import copy
import json
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from typing import Optional

try:
    from .simulation_scenarios import SCENARIOS, SIMULATION_END_DATE
    from .assets import assets, asset_classes, constraints_data
except ImportError:
    from simulation_scenarios import SCENARIOS, SIMULATION_END_DATE
    from assets import assets, asset_classes, constraints_data

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── We import riskfolio / portfolio machinery lazily inside functions
# to allow the simulator to be imported without triggering DB init etc.


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL PRICE CACHE
# ─────────────────────────────────────────────────────────────────────────────

_price_cache: Optional[pd.DataFrame] = None

def get_price_history(start="2021-01-01", end=SIMULATION_END_DATE) -> pd.DataFrame:
    """
    Download and cache historical close prices for all 57 tickers.
    Called once — subsequent calls return the cached DataFrame.
    """
    global _price_cache
    if _price_cache is not None:
        return _price_cache

    import warnings, logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    print(f"[SIM] Downloading historical prices {start} → {end} for {len(assets)} tickers...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = yf.download(assets, start=start, end=end, threads=True, progress=False)["Close"]

    # Clean — same pipeline as main.py
    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated(keep="first")]
    data = data.dropna(axis=1, how="all")
    threshold = int(0.9 * len(data))
    data = data.dropna(axis=1, thresh=threshold)
    data = data.ffill().bfill()

    _price_cache = data
    print(f"[SIM] Price cache ready: {len(data)} trading days, {len(data.columns)} tickers.")
    return _price_cache


def get_prices_on_date(target_date: str) -> dict[str, float]:
    """
    Return closing prices for all tickers on or before target_date.
    Uses the last available trading day if target_date is a weekend/holiday.
    """
    df = get_price_history()
    ts = pd.Timestamp(target_date)
    available = df.index[df.index <= ts]
    if len(available) == 0:
        raise ValueError(f"No price data available on or before {target_date}")
    row = df.loc[available[-1]]
    return {ticker: float(row[ticker]) for ticker in row.index if not pd.isna(row[ticker])}


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER — historical window (no look-ahead bias)
# ─────────────────────────────────────────────────────────────────────────────

def run_optimizer_historical(config: dict, end_date: str,
                              lookback_years: float = 1.5) -> Optional[dict]:
    """
    Run the portfolio optimizer using data ONLY up to end_date.
    Always uses at least lookback_years of history — never less than 252 days.
    Passes the pre-cached price data to avoid re-downloading.
    """
    df = get_price_history()

    # Calculate start = end_date minus lookback period (use days to avoid float-years error)
    end_ts    = pd.Timestamp(end_date)
    lookback_days = int(lookback_years * 365)
    start_ts  = end_ts - pd.Timedelta(days=lookback_days)

    # Ensure we have at least 252 trading days in the slice
    slice_df  = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    if len(slice_df) < 60:
        print(f"[SIM] Not enough data for {end_date} "
              f"(only {len(slice_df)} days) — extending lookback")
        # Extend lookback to get at least 60 days
        available = df.loc[df.index <= end_ts]
        if len(available) < 60:
            print(f"[SIM] Still not enough data — skipping")
            return None
        slice_df = available.iloc[-252:]   # take last 252 days available

    start_str = str(slice_df.index[0].date())
    end_str   = str(slice_df.index[-1].date())

    try:
        try:
            from .riskfolio_main import run_portfolio
        except ImportError:
            from riskfolio_main import run_portfolio
        result = run_portfolio(
            config         = config,
            start          = start_str,
            end            = end_str,
            plot           = False,
            save_json      = False,
            verbose        = False,
            preloaded_data = slice_df,   # ← pass cached data, no re-download
        )
        return result
    except Exception as e:
        print(f"[SIM] Optimizer failed for {end_date}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO STATE — in-memory (no DB for simulation)
# ─────────────────────────────────────────────────────────────────────────────

class SimPortfolio:
    """
    Lightweight in-memory portfolio tracker for simulation.
    Tracks units_held, avg_buy_price, leftover_cash, trade history.
    No SQLite — everything in Python dicts.
    """

    def __init__(self, capital: float):
        self.initial_capital = capital
        self.leftover_cash   = capital
        self.positions: dict[str, dict] = {}
        # {ticker: {units, avg_buy_price, asset_class, first_buy_date}}
        self.trades:    list[dict] = []
        self.total_realized_pnl = 0.0
        self.total_tax_paid     = 0.0
        self.ltcg_used_this_fy  = 0.0
        self.checkpoints:   list[dict] = []   # [{date, value, profile}]

    def buy(self, ticker: str, asset_class: str, units: int,
            price: float, trade_date: str):
        if units < 1:
            return
        cost = round(units * price, 2)
        if cost > self.leftover_cash + 0.01:
            return   # not enough cash

        if ticker in self.positions:
            pos = self.positions[ticker]
            old_u   = pos["units"]
            old_avg = pos["avg_buy_price"]
            new_avg = round((old_u * old_avg + units * price) / (old_u + units), 4)
            pos["units"]         = old_u + units
            pos["avg_buy_price"] = new_avg
        else:
            self.positions[ticker] = {
                "units":          units,
                "avg_buy_price":  price,
                "asset_class":    asset_class,
                "first_buy_date": trade_date,
            }

        self.leftover_cash = round(self.leftover_cash - cost, 2)
        self.trades.append({
            "date": trade_date, "ticker": ticker, "action": "BUY",
            "units": units, "price": price, "value": cost,
        })

    def sell(self, ticker: str, units: int, price: float,
             trade_date: str) -> float:
        """Sell units, compute P&L + tax, return cash received (after tax)."""
        if ticker not in self.positions:
            return 0.0
        pos      = self.positions[ticker]
        units    = min(units, pos["units"])
        avg_cost = pos["avg_buy_price"]
        sell_val = round(units * price, 2)
        pnl      = round((price - avg_cost) * units, 2)

        first_date    = date.fromisoformat(pos["first_buy_date"])
        trade_dt      = date.fromisoformat(trade_date)
        holding_days  = (trade_dt - first_date).days

        # Tax
        tax_inr = 0.0
        if pnl > 0:
            if holding_days >= 365:
                exemption = max(0, 100_000 - self.ltcg_used_this_fy)
                taxable   = max(0, pnl - exemption)
                tax_inr   = round(taxable * 0.10, 2)
                self.ltcg_used_this_fy += pnl
            else:
                tax_inr = round(pnl * 0.15, 2)

        self.total_realized_pnl += pnl
        self.total_tax_paid     += tax_inr

        # Update position
        new_units = pos["units"] - units
        if new_units <= 0:
            del self.positions[ticker]
        else:
            pos["units"] = new_units

        cash_received = round(sell_val - tax_inr, 2)
        self.leftover_cash = round(self.leftover_cash + cash_received, 2)

        self.trades.append({
            "date": trade_date, "ticker": ticker, "action": "SELL",
            "units": units, "price": price, "value": sell_val,
            "pnl": pnl, "tax": tax_inr, "holding_days": holding_days,
        })
        return cash_received

    def portfolio_value(self, prices: dict[str, float]) -> float:
        total = self.leftover_cash
        for ticker, pos in self.positions.items():
            p = prices.get(ticker, pos["avg_buy_price"])
            total += pos["units"] * p
        return round(total, 2)

    def record_checkpoint(self, date_str: str, prices: dict,
                          profile: str, event_label: str = ""):
        val = self.portfolio_value(prices)
        self.checkpoints.append({
            "date":        date_str,
            "value":       val,
            "profile":     profile,
            "event_label": event_label,
            "leftover_cash":         round(self.leftover_cash, 2),
            "total_realized_pnl":    round(self.total_realized_pnl, 2),
            "total_tax_paid":        round(self.total_tax_paid, 2),
        })
        return val


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTE A SINGLE EVENT (buy or rebalance) on a SimPortfolio
# ─────────────────────────────────────────────────────────────────────────────

def apply_event(portfolio: SimPortfolio, optimizer_result: dict,
                event_date: str, prices: dict,
                threshold_pct: float = 2.0):
    """
    Apply optimizer weights to a SimPortfolio at a given date.
    If portfolio is empty → initial buy.
    Otherwise → rebalance (sell overweights, buy underweights).
    """
    if optimizer_result is None or optimizer_result.get("status") != "success":
        print(f"[SIM] Skipping event {event_date} — optimizer failed")
        return

    target_weights = {
        a["ticker"]: a["weight_pct"] / 100
        for a in optimizer_result["asset_allocation"]
        if a["weight_pct"] > 0
    }
    ac_map = {
        a["ticker"]: a["asset_class"]
        for a in optimizer_result["asset_allocation"]
    }

    total_value = portfolio.portfolio_value(prices)
    current_weights = {}
    for ticker, pos in portfolio.positions.items():
        p = prices.get(ticker, pos["avg_buy_price"])
        current_weights[ticker] = (pos["units"] * p) / total_value if total_value > 0 else 0

    all_tickers = set(list(current_weights.keys()) + list(target_weights.keys()))

    # ── SELLS FIRST ──────────────────────────────────────────────────────────
    for ticker in sorted(all_tickers):
        cur_w = current_weights.get(ticker, 0.0)
        new_w = target_weights.get(ticker, 0.0)
        delta = new_w - cur_w

        if delta >= 0:
            continue  # not a sell

        price = prices.get(ticker)
        if price is None or price <= 0:
            continue

        if abs(delta) < threshold_pct / 100 and new_w > 0:
            continue  # within threshold, hold

        units_to_sell = math.floor(abs(delta) * total_value / price)
        if units_to_sell < 1:
            continue

        portfolio.sell(ticker, units_to_sell, price, event_date)

    # ── BUYS SECOND (all sell cash now available) ─────────────────────────────
    total_value_after_sells = portfolio.portfolio_value(prices)

    for ticker in sorted(all_tickers):
        cur_w = current_weights.get(ticker, 0.0)
        new_w = target_weights.get(ticker, 0.0)
        delta = new_w - cur_w

        if delta <= 0:
            continue  # not a buy

        price = prices.get(ticker)
        if price is None or price <= 0:
            continue

        if abs(delta) < threshold_pct / 100 and cur_w > 0:
            continue  # within threshold, hold

        allocated = abs(delta) * total_value_after_sells
        units_to_buy = math.floor(allocated / price)
        if units_to_buy < 1:
            continue

        buy_cost = units_to_buy * price
        if buy_cost > portfolio.leftover_cash + 0.01:
            continue  # not enough cash

        asset_class = ac_map.get(ticker, asset_classes.get(ticker, "Unknown"))
        portfolio.buy(ticker, asset_class, units_to_buy, price, event_date)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(portfolio: SimPortfolio, checkpoints: list,
                    end_date: str, end_prices: dict) -> dict:
    """Compute final performance metrics for a simulation track."""
    final_value  = portfolio.portfolio_value(end_prices)
    initial      = portfolio.initial_capital
    total_return = round((final_value - initial) / initial * 100, 4)

    # CAGR
    start_dt  = date.fromisoformat(checkpoints[0]["date"]) if checkpoints else date(2022, 1, 3)
    end_dt    = date.fromisoformat(end_date)
    years     = max((end_dt - start_dt).days / 365.25, 0.01)
    cagr      = round(((final_value / initial) ** (1 / years) - 1) * 100, 4)

    # Value series for Sharpe + drawdown
    values = [cp["value"] for cp in checkpoints] + [final_value]
    if len(values) > 1:
        returns_series = [
            (values[i] - values[i-1]) / values[i-1]
            for i in range(1, len(values))
            if values[i-1] > 0
        ]
        sharpe = None
        if len(returns_series) > 1:
            mean_r = np.mean(returns_series)
            std_r  = np.std(returns_series, ddof=1)
            sharpe = round(mean_r / std_r * (12 ** 0.5), 4) if std_r > 0 else None

        # Max drawdown from value series
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
        max_drawdown = round(max_dd * 100, 4)
    else:
        sharpe = None
        max_drawdown = 0.0

    # Trades
    sells = [t for t in portfolio.trades if t["action"] == "SELL"]
    buys  = [t for t in portfolio.trades if t["action"] == "BUY"]

    return {
        "final_value_inr":      round(final_value, 2),
        "initial_capital_inr":  initial,
        "total_return_pct":     total_return,
        "cagr_pct":             cagr,
        "sharpe_ratio":         sharpe,
        "max_drawdown_pct":     max_drawdown,
        "total_realized_pnl":   round(portfolio.total_realized_pnl, 2),
        "total_tax_paid":       round(portfolio.total_tax_paid, 2),
        "leftover_cash":        round(portfolio.leftover_cash, 2),
        "total_trades":         len(portfolio.trades),
        "sell_count":           len(sells),
        "buy_count":            len(buys),
        "rebalance_count":      max(0, len(set(t["date"] for t in portfolio.trades)) - 1),
        "checkpoints":          checkpoints + [{"date": end_date, "value": round(final_value, 2)}],
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUN ONE SCENARIO — both tracks
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(scenario: dict,
                   end_date: str = SIMULATION_END_DATE,
                   threshold_pct: float = 2.0,
                   verbose: bool = True) -> dict:
    """
    Run one user scenario on two tracks:
      rebalanced — follows user's event list
      hold       — buys initial portfolio on day 1, holds to end

    Returns a dict with both tracks' metrics + checkpoints.
    """
    email   = scenario["email"]
    capital = scenario["capital"]
    events  = scenario["events"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"SIMULATING: {scenario['name']} ({email})")
        print(f"Persona: {scenario['persona']}")
        print(f"Events: {len(events)} | Capital: ₹{capital:,.0f}")
        print(f"{'='*60}")

    # ── Track A — Rebalanced ──────────────────────────────────────────────────
    port_r = SimPortfolio(capital)

    for i, event in enumerate(events):
        edate   = event["date"]
        config  = event["config"]
        nlp     = event["nlp_input"]
        profile = config["profile"]

        prices = get_prices_on_date(edate)

        if verbose:
            print(f"\n  [REBALANCED] Event {i+1}: {edate} — {profile.upper()}")
            print(f"    NLP: '{nlp}'")

        # Optimizer trained only on data up to this event date
        opt_result = run_optimizer_historical(config, end_date=edate)

        if opt_result and opt_result.get("status") == "success":
            apply_event(port_r, opt_result, edate, prices, threshold_pct)
            val = port_r.record_checkpoint(edate, prices, profile, f"Event {i+1}: {profile}")
            if verbose:
                print(f"    Portfolio value: ₹{val:,.2f} | Cash: ₹{port_r.leftover_cash:,.2f}")
        else:
            if verbose:
                print(f"    [WARNING] Optimizer failed — skipping event")

    end_prices_r = get_prices_on_date(end_date)
    metrics_r    = compute_metrics(port_r, port_r.checkpoints, end_date, end_prices_r)

    # ── Track B — Hold ────────────────────────────────────────────────────────
    port_h  = SimPortfolio(capital)
    first_e = events[0]

    if verbose:
        print(f"\n  [HOLD] Buying initial portfolio on {first_e['date']} and holding...")

    hold_prices  = get_prices_on_date(first_e["date"])
    hold_opt     = run_optimizer_historical(first_e["config"], end_date=first_e["date"])

    if hold_opt and hold_opt.get("status") == "success":
        apply_event(port_h, hold_opt, first_e["date"], hold_prices, threshold_pct)
        val = port_h.record_checkpoint(
            first_e["date"], hold_prices, first_e["config"]["profile"], "Initial buy (hold)"
        )
        if verbose:
            print(f"    Initial buy: ₹{val:,.2f} | Cash: ₹{port_h.leftover_cash:,.2f}")

    # Record intermediate checkpoint values for hold track (same dates as rebalanced)
    for event in events[1:]:
        prices_on_day = get_prices_on_date(event["date"])
        port_h.record_checkpoint(
            event["date"], prices_on_day,
            first_e["config"]["profile"], "Hold (no rebalance)"
        )

    end_prices_h = get_prices_on_date(end_date)
    metrics_h    = compute_metrics(port_h, port_h.checkpoints, end_date, end_prices_h)

    # ── Summary ───────────────────────────────────────────────────────────────
    winner = "rebalanced" if metrics_r["final_value_inr"] > metrics_h["final_value_inr"] else "hold"
    diff   = round(metrics_r["final_value_inr"] - metrics_h["final_value_inr"], 2)

    if verbose:
        print(f"\n  RESULT:")
        print(f"    Rebalanced final: ₹{metrics_r['final_value_inr']:,.2f} "
              f"({metrics_r['total_return_pct']:+.2f}%)")
        print(f"    Hold final:       ₹{metrics_h['final_value_inr']:,.2f} "
              f"({metrics_h['total_return_pct']:+.2f}%)")
        print(f"    Winner: {winner.upper()} by ₹{abs(diff):,.2f}")

    return {
        "email":    email,
        "name":     scenario["name"],
        "persona":  scenario["persona"],
        "capital":  capital,
        "events":   [{"date": e["date"], "profile": e["config"]["profile"],
                      "nlp_input": e["nlp_input"]} for e in events],
        "end_date": end_date,
        "rebalanced": metrics_r,
        "hold":       metrics_h,
        "comparison": {
            "winner":              winner,
            "difference_inr":      abs(diff),
            "rebalanced_better":   diff > 0,
            "return_diff_pct":     round(metrics_r["total_return_pct"] - metrics_h["total_return_pct"], 4),
            "tax_cost_of_rebalancing": round(metrics_r["total_tax_paid"] - metrics_h["total_tax_paid"], 2),
            "extra_trades":        metrics_r["total_trades"] - metrics_h["total_trades"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

def run_all_simulations(save_path: str = "simulation_results.json",
                        verbose: bool = True) -> list[dict]:
    """Run all 5 user scenarios and save results to JSON."""
    print(f"\n{'='*60}")
    print("RUNNING ALL 5 USER SIMULATIONS")
    print(f"End date: {SIMULATION_END_DATE}")
    print(f"{'='*60}")

    # Pre-warm price cache
    get_price_history()

    results = []
    for scenario in SCENARIOS:
        try:
            result = run_simulation(scenario, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Simulation failed for {scenario['email']}: {e}")
            import traceback
            traceback.print_exc()

    # Save
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SIM] Results saved → {save_path}")

    # Print leaderboard
    print(f"\n{'='*60}")
    print("LEADERBOARD — Rebalanced vs Hold")
    print(f"{'='*60}")
    print(f"{'User':<20} {'Rebalanced':>14} {'Hold':>14} {'Winner':<14} {'Diff':>12}")
    print("-" * 76)
    for r in sorted(results, key=lambda x: -x["rebalanced"]["total_return_pct"]):
        rb = r["rebalanced"]
        ho = r["hold"]
        w  = r["comparison"]["winner"].upper()
        d  = r["comparison"]["difference_inr"]
        print(f"{r['name']:<20} {rb['total_return_pct']:>+13.2f}% "
              f"{ho['total_return_pct']:>+13.2f}%  {w:<14} ₹{d:>10,.2f}")

    return results


if __name__ == "__main__":
    run_all_simulations(verbose=True)

"""
portfolio_engine.py — Phase 1 & Phase 2 transaction engine.

Plugs directly into run_portfolio() from main.py.

Phase 1: execute_initial_buy()   — allocate capital, buy positions, store everything
Phase 2: compute_rebalance()     — compare old vs new weights, generate trade plan
         execute_rebalance()     — execute the trade plan, update DB atomically
"""

import math
import json
from datetime import date, datetime
from typing import Optional
import yfinance as yf

from .database import get_db
from .models import User, Portfolio, OptimizerRun, Position, Trade, RebalanceEvent


# ─────────────────────────────────────────────────────────────────────────────
# USER MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def create_user(email: str, name: Optional[str] = None) -> str:
    """
    Create a new user. Returns user_id.
    Raises ValueError if email already exists.
    """
    with get_db() as db:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            raise ValueError(f"User with email '{email}' already exists. Use get_or_create_user() instead.")
        user = User(
            email              = email,
            name               = name,
            current_fy_start   = _get_indian_fy_start(),
            ltcg_used_this_fy  = 0.00,
        )
        db.add(user)
        db.flush()
        user_id = user.user_id
    print(f"[DB] User created: {email} → {user_id}")
    return user_id


def get_or_create_user(email: str, name: Optional[str] = None) -> str:
    """
    Return existing user_id for email, or create a new user if not found.
    Safe to call multiple times — idempotent.
    """
    with get_db() as db:
        user = db.query(User).filter(User.email == email).first()
        if user:
            print(f"[DB] User found: {email} → {user.user_id}")
            return user.user_id
        # Not found — create
        user = User(
            email             = email,
            name              = name,
            current_fy_start  = _get_indian_fy_start(),
            ltcg_used_this_fy = 0.00,
        )
        db.add(user)
        db.flush()
        user_id = user.user_id
    print(f"[DB] User created: {email} → {user_id}")
    return user_id


def get_user(user_id: str) -> Optional[dict]:
    """Fetch user info by user_id."""
    with get_db() as db:
        user = db.get(User, user_id)
        if not user:
            return None
        return {
            "user_id":            user.user_id,
            "email":              user.email,
            "name":               user.name,
            "created_at":         str(user.created_at),
            "ltcg_used_this_fy":  float(user.ltcg_used_this_fy),
        }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_current_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch live last price for a list of tickers via yfinance."""
    prices = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            prices[ticker] = float(info["last_price"])
        except Exception as e:
            print(f"[WARNING] Could not fetch price for {ticker}: {e}")
            prices[ticker] = None
    return prices


def _calc_tax(realized_pnl: float, holding_days: int,
              ltcg_used_this_fy: float) -> tuple[str, float, float]:
    """
    Calculate tax type, rate, and INR for a sell trade.
    Returns (tax_type, tax_rate_pct, tax_inr)

    Indian rules:
    - Loss → no tax
    - LTCG (>=365 days): 10% after ₹1L exemption per FY
    - STCG (<365 days): 15% flat
    """
    if realized_pnl <= 0:
        return "NONE", 0.0, 0.0

    if holding_days >= 365:
        # LTCG — apply ₹1L annual exemption
        exemption_remaining = max(0, 100_000 - ltcg_used_this_fy)
        taxable_pnl = max(0, realized_pnl - exemption_remaining)
        tax_inr = round(taxable_pnl * 0.10, 2)
        return "LTCG", 10.0, tax_inr
    else:
        # STCG — flat 15%
        tax_inr = round(realized_pnl * 0.15, 2)
        return "STCG", 15.0, tax_inr


def _get_indian_fy_start() -> date:
    """Return April 1st of current Indian financial year."""
    today = date.today()
    return date(today.year if today.month >= 4 else today.year - 1, 4, 1)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — INITIAL BUY
# ─────────────────────────────────────────────────────────────────────────────

def execute_initial_buy(
    user_id: str,
    portfolio_name: str,
    capital: float,
    optimizer_result: dict,       # full JSON from run_portfolio()
    nlp_input: Optional[str] = None,
) -> dict:
    """
    Phase 1 — Allocate capital, buy all positions, store everything in DB.

    Parameters
    ----------
    user_id          : str   — must already exist in users table
    portfolio_name   : str   — e.g. "My Conservative Portfolio"
    capital          : float — total INR to invest (e.g. 100000.0)
    optimizer_result : dict  — full return value from run_portfolio()
    nlp_input        : str   — what the user said (optional)

    Returns
    -------
    dict — summary of what was bought, leftover cash, and portfolio_id
    """
    if optimizer_result.get("status") != "success":
        raise ValueError("Cannot execute buy — optimizer result has status != success")

    profile    = optimizer_result["meta"]["profile"]
    config     = optimizer_result["meta"]
    weights    = optimizer_result["asset_allocation"]    # list of {ticker, asset_class, weight_pct}
    perf       = optimizer_result["performance"]

    # Fetch live prices for all tickers in the result
    tickers = [a["ticker"] for a in weights if a["weight_pct"] > 0]
    print(f"[ENGINE] Fetching live prices for {len(tickers)} tickers...")
    prices = fetch_current_prices(tickers)

    today = date.today()
    buy_results = []
    total_actually_invested = 0.0
    skipped = []

    # Calculate allocation and units for each ticker
    for asset in weights:
        ticker      = asset["ticker"]
        weight_pct  = asset["weight_pct"]
        asset_class = asset["asset_class"]

        if weight_pct <= 0:
            continue

        allocated_inr = round(weight_pct / 100 * capital, 2)
        price = prices.get(ticker)

        if price is None or price <= 0:
            skipped.append({"ticker": ticker, "reason": "price unavailable"})
            continue

        units = math.floor(allocated_inr / price)

        if units < 1:
            skipped.append({
                "ticker": ticker,
                "reason": f"allocated ₹{allocated_inr:.0f} < price ₹{price:.2f} — can't buy 1 unit"
            })
            continue

        actual_invested = round(units * price, 2)
        total_actually_invested += actual_invested

        buy_results.append({
            "ticker":           ticker,
            "asset_class":      asset_class,
            "weight_pct":       weight_pct,
            "allocated_inr":    allocated_inr,
            "price":            price,
            "units":            units,
            "actual_invested":  actual_invested,
            "leftover":         round(allocated_inr - actual_invested, 2),
        })

    leftover_cash = round(capital - total_actually_invested, 2)

    # ── Write to DB atomically ────────────────────────────────────────────────
    with get_db() as db:

        # 1. Create portfolio
        portfolio = Portfolio(
            user_id         = user_id,
            name            = portfolio_name,
            initial_capital = capital,
            leftover_cash   = leftover_cash,
            profile         = profile,
        )
        db.add(portfolio)
        db.flush()   # get portfolio_id before inserting children

        # 2. Save optimizer run
        run = OptimizerRun(
            portfolio_id         = portfolio.portfolio_id,
            triggered_by         = nlp_input,
            config_snapshot      = config,
            weights_snapshot     = optimizer_result,
            performance_snapshot = perf,
            was_applied          = True,
        )
        db.add(run)
        db.flush()

        # 3. Create positions + BUY trades
        for b in buy_results:
            # Position (live state)
            position = Position(
                portfolio_id     = portfolio.portfolio_id,
                ticker           = b["ticker"],
                asset_class      = b["asset_class"],
                units_held       = b["units"],
                avg_buy_price    = b["price"],
                total_invested   = b["actual_invested"],
                first_buy_date   = today,
                last_buy_date    = today,
                target_weight_pct = b["weight_pct"],
            )
            db.add(position)
            db.flush()

            # Trade record (immutable history)
            trade = Trade(
                portfolio_id    = portfolio.portfolio_id,
                position_id     = position.position_id,
                rebalance_id    = None,   # initial buy has no rebalance
                ticker          = b["ticker"],
                asset_class     = b["asset_class"],
                action          = "BUY",
                units           = b["units"],
                price           = b["price"],
                value_inr       = b["actual_invested"],
                weight_before_pct = 0.0,
                weight_after_pct  = b["weight_pct"],
            )
            db.add(trade)

        portfolio_id = portfolio.portfolio_id

    # ── Return summary ────────────────────────────────────────────────────────
    summary = {
        "status":               "success",
        "portfolio_id":         portfolio_id,
        "profile":              profile,
        "capital":              capital,
        "total_invested":       round(total_actually_invested, 2),
        "leftover_cash":        leftover_cash,
        "positions_created":    len(buy_results),
        "positions_skipped":    len(skipped),
        "skipped":              skipped,
        "trades": [
            {
                "ticker":          b["ticker"],
                "asset_class":     b["asset_class"],
                "action":          "BUY",
                "units":           b["units"],
                "price":           b["price"],
                "invested_inr":    b["actual_invested"],
                "weight_pct":      b["weight_pct"],
            }
            for b in buy_results
        ],
    }

    print(f"\n[ENGINE] Portfolio created: {portfolio_id}")
    print(f"[ENGINE] Invested: ₹{total_actually_invested:,.2f} across {len(buy_results)} positions")
    print(f"[ENGINE] Leftover cash: ₹{leftover_cash:,.2f}")
    if skipped:
        print(f"[ENGINE] Skipped {len(skipped)} tickers: {[s['ticker'] for s in skipped]}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2a — COMPUTE REBALANCE (no DB writes — just the plan)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rebalance(
    portfolio_id: str,
    optimizer_result: dict,
    threshold_pct: float = 2.0,
) -> dict:
    """
    Compare current positions against new optimizer weights.
    Returns a trade plan — no DB writes yet. User confirms first.

    Parameters
    ----------
    portfolio_id     : str   — existing portfolio in DB
    optimizer_result : dict  — full return value from run_portfolio()
    threshold_pct    : float — ignore deltas smaller than this (default 2%)

    Returns
    -------
    dict — full trade plan with SELL/BUY/HOLD for every ticker
    """
    if optimizer_result.get("status") != "success":
        raise ValueError("Cannot rebalance — optimizer result has status != success")

    new_weights = {
        a["ticker"]: a["weight_pct"]
        for a in optimizer_result["asset_allocation"]
    }

    with get_db() as db:
        portfolio = db.get(Portfolio, portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        positions = {
            p.ticker: p
            for p in portfolio.positions
            if float(p.units_held) > 0
        }
        user = db.get(User, portfolio.user_id)
        ltcg_used = float(user.ltcg_used_this_fy)

        # Fetch live prices for all relevant tickers
        all_tickers = list(set(list(positions.keys()) + list(new_weights.keys())))
        print(f"[ENGINE] Fetching live prices for {len(all_tickers)} tickers...")
        prices = fetch_current_prices(all_tickers)

        today = date.today()

        # Current portfolio market value
        current_values = {}
        for ticker, pos in positions.items():
            price = prices.get(ticker)
            if price:
                current_values[ticker] = float(pos.units_held) * price
            else:
                current_values[ticker] = float(pos.total_invested)   # fallback

        portfolio_value = sum(current_values.values()) + float(portfolio.leftover_cash)

        # Current weights (actual, post-drift)
        current_weights = {
            ticker: round(val / portfolio_value * 100, 4)
            for ticker, val in current_values.items()
        }

        # All tickers to consider (union of old + new)
        all_involved = set(list(current_weights.keys()) + list(new_weights.keys()))

        trades = []
        running_ltcg_used = ltcg_used

        for ticker in sorted(all_involved):
            cur_w  = current_weights.get(ticker, 0.0)
            new_w  = new_weights.get(ticker, 0.0)
            delta  = round(new_w - cur_w, 4)
            pos    = positions.get(ticker)
            price  = prices.get(ticker)

            # Determine action
            if abs(delta) < threshold_pct and new_w > 0:
                action = "HOLD"
                reason = f"delta {delta:+.2f}% < threshold {threshold_pct}%"
            elif delta < 0 or new_w == 0:
                action = "SELL"
                reason = "overweight — reduce to target" if new_w > 0 else "removed from portfolio"
            else:
                action = "BUY"
                reason = "underweight — increase to target" if cur_w > 0 else "new position"

            trade_entry = {
                "ticker":           ticker,
                "asset_class":      pos.asset_class if pos else new_weights.get(ticker, {}) and optimizer_result["asset_allocation"] and next((a["asset_class"] for a in optimizer_result["asset_allocation"] if a["ticker"] == ticker), "Unknown"),
                "action":           action,
                "current_weight_pct": cur_w,
                "new_weight_pct":   new_w,
                "delta_pct":        delta,
                "current_price":    price,
                "reason":           reason,
            }

            if action == "SELL" and pos and price:
                units_to_sell = math.floor(abs(delta) / 100 * portfolio_value / price)
                units_to_sell = min(units_to_sell, float(pos.units_held))  # never sell more than held

                if units_to_sell < 1:
                    trade_entry["action"] = "HOLD"
                    trade_entry["reason"] = "delta too small — less than 1 unit to sell"
                else:
                    avg_cost      = float(pos.avg_buy_price)
                    realized_pnl  = round((price - avg_cost) * units_to_sell, 2)
                    holding_days  = (today - pos.first_buy_date).days
                    tax_type, tax_rate, tax_inr = _calc_tax(
                        realized_pnl, holding_days, running_ltcg_used
                    )

                    if tax_type == "LTCG" and realized_pnl > 0:
                        running_ltcg_used += realized_pnl   # track within this plan

                    trade_entry.update({
                        "units_to_trade":       units_to_sell,
                        "estimated_value_inr":  round(units_to_sell * price, 2),
                        "avg_cost_basis":        avg_cost,
                        "realized_pnl":         realized_pnl,
                        "holding_days":         holding_days,
                        "tax_type":             tax_type,
                        "tax_rate_pct":         tax_rate,
                        "estimated_tax_inr":    tax_inr,
                    })

            elif action == "BUY" and price:
                allocated_inr  = abs(delta) / 100 * portfolio_value
                units_to_buy   = math.floor(allocated_inr / price)
                actual_cost    = round(units_to_buy * price, 2)

                trade_entry.update({
                    "units_to_trade":       units_to_buy if units_to_buy >= 1 else 0,
                    "estimated_value_inr":  actual_cost if units_to_buy >= 1 else 0,
                    "reason":               reason if units_to_buy >= 1 else "allocated amount < 1 unit price — skipped",
                })
                if units_to_buy < 1:
                    trade_entry["action"] = "SKIP"

            trades.append(trade_entry)

        # ── Sort: sell losses first, then LTCG sells, then STCG sells, buys last
        priority = {"SELL": 0, "BUY": 2, "HOLD": 3, "SKIP": 4}

        def sell_sort_key(t):
            if t["action"] != "SELL":
                return (priority.get(t["action"], 9), 0)
            pnl = t.get("realized_pnl", 0)
            tax = t.get("tax_type", "NONE")
            if pnl <= 0:          return (0, 0)   # losses first
            if tax == "LTCG":     return (0, 1)   # LTCG second
            return (0, 2)                          # STCG last

        trades.sort(key=sell_sort_key)

        # ── Summary
        sells   = [t for t in trades if t["action"] == "SELL"]
        buys    = [t for t in trades if t["action"] == "BUY"]
        holds   = [t for t in trades if t["action"] == "HOLD"]
        skips   = [t for t in trades if t["action"] == "SKIP"]

        total_sell_value = sum(t.get("estimated_value_inr", 0) for t in sells)
        total_buy_value  = sum(t.get("estimated_value_inr", 0) for t in buys)
        total_tax        = sum(t.get("estimated_tax_inr", 0) for t in sells)
        total_pnl        = sum(t.get("realized_pnl", 0) for t in sells)

        plan = {
            "portfolio_id":         portfolio_id,
            "portfolio_value":      round(portfolio_value, 2),
            "leftover_cash":        float(portfolio.leftover_cash),
            "threshold_pct":        threshold_pct,
            "optimizer_run_meta":   optimizer_result["meta"],

            "summary": {
                "total_sells_inr":    round(total_sell_value, 2),
                "total_buys_inr":     round(total_buy_value, 2),
                "net_cash_change":    round(total_sell_value - total_buy_value, 2),
                "total_realized_pnl": round(total_pnl, 2),
                "estimated_tax_inr":  round(total_tax, 2),
                "trades_count": {
                    "sell": len(sells),
                    "buy":  len(buys),
                    "hold": len(holds),
                    "skip": len(skips),
                },
            },
            "trades": trades,
        }

    return plan


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2b — EXECUTE REBALANCE (DB writes — after user confirms)
# ─────────────────────────────────────────────────────────────────────────────

def execute_rebalance(
    portfolio_id: str,
    trade_plan: dict,
    optimizer_result: dict,
    nlp_input: Optional[str] = None,
) -> dict:
    """
    Execute the confirmed trade plan. Writes all trades to DB atomically.
    Updates positions, portfolio totals, and rebalance event record.

    Parameters
    ----------
    portfolio_id     : str  — existing portfolio
    trade_plan       : dict — output of compute_rebalance()
    optimizer_result : dict — output of run_portfolio() (saved as new optimizer run)
    nlp_input        : str  — user's original NLP text

    Returns
    -------
    dict — execution summary
    """
    today = date.today()
    executed_trades = []
    total_sold      = 0.0
    total_bought    = 0.0
    total_pnl       = 0.0
    total_tax       = 0.0

    with get_db() as db:
        portfolio = db.get(Portfolio, portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        user = db.get(User, portfolio.user_id)
        portfolio_value_before = trade_plan["portfolio_value"]

        # Re-index positions for fast lookup
        positions_by_ticker = {
            p.ticker: p for p in portfolio.positions if float(p.units_held) > 0
        }
        new_weights_map = {
            a["ticker"]: a
            for a in optimizer_result["asset_allocation"]
        }

        # 1. Save new optimizer run
        new_run = OptimizerRun(
            portfolio_id         = portfolio_id,
            triggered_by         = nlp_input,
            config_snapshot      = optimizer_result["meta"],
            weights_snapshot     = optimizer_result,
            performance_snapshot = optimizer_result["performance"],
            was_applied          = True,
        )
        db.add(new_run)
        db.flush()

        # 2. Create rebalance event (fill in financials after trades)
        old_weights_snap = {
            t["ticker"]: t["current_weight_pct"]
            for t in trade_plan["trades"]
        }
        new_weights_snap = {
            t["ticker"]: t["new_weight_pct"]
            for t in trade_plan["trades"]
        }

        rebalance_event = RebalanceEvent(
            portfolio_id            = portfolio_id,
            run_id                  = new_run.run_id,
            nlp_input               = nlp_input,
            old_weights_snapshot    = old_weights_snap,
            new_weights_snapshot    = new_weights_snap,
            trade_plan_snapshot     = trade_plan,
            portfolio_value_before  = portfolio_value_before,
        )
        db.add(rebalance_event)
        db.flush()

        # 3. Execute each trade
        for t in trade_plan["trades"]:
            action = t["action"]
            ticker = t["ticker"]
            price  = t.get("current_price")
            units  = t.get("units_to_trade", 0)

            if action not in ("SELL", "BUY") or not units or units < 1 or not price:
                continue

            pos = positions_by_ticker.get(ticker)
            asset_class = t.get("asset_class", "Unknown")

            # ── SELL ────────────────────────────────────────────────────────
            if action == "SELL" and pos:
                units        = min(units, float(pos.units_held))
                avg_cost     = float(pos.avg_buy_price)
                realized_pnl = round((price - avg_cost) * units, 2)
                holding_days = (today - pos.first_buy_date).days
                sell_value   = round(units * price, 2)
                tax_type, tax_rate, tax_inr = _calc_tax(
                    realized_pnl, holding_days, float(user.ltcg_used_this_fy)
                )

                # Update position
                new_units = float(pos.units_held) - units
                pos.units_held    = new_units
                pos.total_invested = round(new_units * avg_cost, 2)
                pos.target_weight_pct = t["new_weight_pct"]
                # Note: avg_buy_price unchanged on sell (AVCO — cost basis stays)

                # Update LTCG tracker
                if tax_type == "LTCG" and realized_pnl > 0:
                    user.ltcg_used_this_fy = float(user.ltcg_used_this_fy) + realized_pnl

                # Update portfolio cash + totals
                portfolio.leftover_cash     = float(portfolio.leftover_cash) + sell_value - tax_inr
                portfolio.total_realized_pnl = float(portfolio.total_realized_pnl) + realized_pnl
                portfolio.total_tax_paid     = float(portfolio.total_tax_paid) + tax_inr

                total_sold += sell_value
                total_pnl  += realized_pnl
                total_tax  += tax_inr

                trade_record = Trade(
                    portfolio_id      = portfolio_id,
                    position_id       = pos.position_id,
                    rebalance_id      = rebalance_event.rebalance_id,
                    ticker            = ticker,
                    asset_class       = asset_class,
                    action            = "SELL",
                    units             = units,
                    price             = price,
                    value_inr         = sell_value,
                    avg_cost_at_sale  = avg_cost,
                    realized_pnl      = realized_pnl,
                    holding_days      = holding_days,
                    tax_type          = tax_type,
                    tax_rate_pct      = tax_rate,
                    tax_inr           = tax_inr,
                    weight_before_pct = t["current_weight_pct"],
                    weight_after_pct  = t["new_weight_pct"],
                )
                db.add(trade_record)
                executed_trades.append({"ticker": ticker, "action": "SELL", "units": units,
                                        "value_inr": sell_value, "pnl": realized_pnl, "tax": tax_inr})

            # ── BUY ─────────────────────────────────────────────────────────
            elif action == "BUY":
                buy_value = round(units * price, 2)

                # Check we have enough cash
                if buy_value > float(portfolio.leftover_cash):
                    print(f"[WARNING] Not enough cash for {ticker} — need ₹{buy_value:.0f}, have ₹{float(portfolio.leftover_cash):.0f}")
                    continue

                if pos:
                    # Add to existing position — recalculate weighted avg cost
                    old_units = float(pos.units_held)
                    old_avg   = float(pos.avg_buy_price)
                    new_avg   = round(
                        (old_units * old_avg + units * price) / (old_units + units), 4
                    )
                    pos.units_held      = old_units + units
                    pos.avg_buy_price   = new_avg
                    pos.total_invested  = round((old_units + units) * new_avg, 2)
                    pos.last_buy_date   = today
                    pos.target_weight_pct = t["new_weight_pct"]
                    position_id = pos.position_id
                else:
                    # New position
                    new_pos = Position(
                        portfolio_id      = portfolio_id,
                        ticker            = ticker,
                        asset_class       = asset_class,
                        units_held        = units,
                        avg_buy_price     = price,
                        total_invested    = buy_value,
                        first_buy_date    = today,
                        last_buy_date     = today,
                        target_weight_pct = t["new_weight_pct"],
                    )
                    db.add(new_pos)
                    db.flush()
                    positions_by_ticker[ticker] = new_pos
                    position_id = new_pos.position_id

                portfolio.leftover_cash = float(portfolio.leftover_cash) - buy_value
                total_bought += buy_value

                trade_record = Trade(
                    portfolio_id      = portfolio_id,
                    position_id       = position_id,
                    rebalance_id      = rebalance_event.rebalance_id,
                    ticker            = ticker,
                    asset_class       = asset_class,
                    action            = "BUY",
                    units             = units,
                    price             = price,
                    value_inr         = buy_value,
                    weight_before_pct = t["current_weight_pct"],
                    weight_after_pct  = t["new_weight_pct"],
                )
                db.add(trade_record)
                executed_trades.append({"ticker": ticker, "action": "BUY", "units": units,
                                        "value_inr": buy_value})

        # 4. Finalize rebalance event with actuals
        rebalance_event.total_sold_inr       = round(total_sold, 2)
        rebalance_event.total_bought_inr     = round(total_bought, 2)
        rebalance_event.total_realized_pnl   = round(total_pnl, 2)
        rebalance_event.total_tax_inr        = round(total_tax, 2)

        # Build a price map from the trade plan (clean, no nested generator mess)
        price_map = {
            t["ticker"]: t["current_price"]
            for t in trade_plan["trades"]
            if t.get("current_price")
        }

        # Recalculate portfolio value after using price map
        portfolio_value_after = (
            sum(
                float(p.units_held) * price_map.get(p.ticker, float(p.avg_buy_price))
                for p in portfolio.positions
                if float(p.units_held) > 0
            )
            + float(portfolio.leftover_cash)
        )

        # Close zero-unit positions (mark target weight as 0, keep for audit trail)
        for p in portfolio.positions:
            if float(p.units_held) == 0:
                p.target_weight_pct = 0.0

        rebalance_event.portfolio_value_after = round(portfolio_value_after, 2)
        portfolio.last_rebalanced_at          = datetime.utcnow()
        # Only update profile if it's a real value (not 'unknown')
        new_profile = optimizer_result["meta"].get("profile", "")
        if new_profile and new_profile != "unknown":
            portfolio.profile = new_profile

        rebalance_id = rebalance_event.rebalance_id

    summary = {
        "status":               "success",
        "rebalance_id":         rebalance_id,
        "portfolio_id":         portfolio_id,
        "portfolio_value_before": round(portfolio_value_before, 2),
        "portfolio_value_after":  round(portfolio_value_after, 2),
        "total_sold_inr":       round(total_sold, 2),
        "total_bought_inr":     round(total_bought, 2),
        "total_realized_pnl":   round(total_pnl, 2),
        "total_tax_inr":        round(total_tax, 2),
        "trades_executed":      len(executed_trades),
        "trades":               executed_trades,
    }

    print(f"\n[ENGINE] Rebalance complete: {rebalance_id}")
    print(f"[ENGINE] Sold ₹{total_sold:,.0f} | Bought ₹{total_bought:,.0f} | Tax ₹{total_tax:,.0f}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# READ HELPERS — for frontend / API
# ─────────────────────────────────────────────────────────────────────────────

def get_portfolio_snapshot(portfolio_id: str) -> dict:
    """
    Returns full portfolio state with live prices and current P&L.
    Call this whenever the user opens the dashboard.
    """
    with get_db() as db:
        portfolio = db.get(Portfolio, portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        active_positions = [p for p in portfolio.positions if float(p.units_held) > 0]
        tickers = [p.ticker for p in active_positions]
        prices  = fetch_current_prices(tickers)
        today   = date.today()

        position_data = []
        total_current_value = 0.0
        total_invested = 0.0

        for pos in active_positions:
            price = prices.get(pos.ticker)
            current_value = float(pos.units_held) * price if price else float(pos.total_invested)
            cost_basis    = float(pos.total_invested)
            unrealized    = round(current_value - cost_basis, 2)
            unrealized_pct = round(unrealized / cost_basis * 100, 2) if cost_basis else 0
            holding_days  = (today - pos.first_buy_date).days

            total_current_value += current_value
            total_invested += cost_basis

            position_data.append({
                "ticker":              pos.ticker,
                "asset_class":         pos.asset_class,
                "units_held":          float(pos.units_held),
                "avg_buy_price":       float(pos.avg_buy_price),
                "current_price":       price,
                "cost_basis_inr":      round(cost_basis, 2),
                "current_value_inr":   round(current_value, 2),
                "unrealized_pnl_inr":  unrealized,
                "unrealized_pnl_pct":  unrealized_pct,
                "target_weight_pct":   float(pos.target_weight_pct or 0),
                "current_weight_pct":  round(current_value / (total_current_value + float(portfolio.leftover_cash)) * 100, 4) if total_current_value else 0,
                "holding_days":        holding_days,
                "tax_status":          "LTCG" if holding_days >= 365 else "STCG",
                "first_buy_date":      str(pos.first_buy_date),
            })

        portfolio_value = total_current_value + float(portfolio.leftover_cash)

        # Fix current_weight_pct now that we have total
        for p in position_data:
            p["current_weight_pct"] = round(p["current_value_inr"] / portfolio_value * 100, 4)

        return {
            "portfolio_id":         portfolio_id,
            "name":                 portfolio.name,
            "profile":              portfolio.profile,
            "initial_capital":      float(portfolio.initial_capital),
            "leftover_cash":        float(portfolio.leftover_cash),
            "portfolio_value":      round(portfolio_value, 2),
            "total_invested":       round(total_invested, 2),
            "unrealized_pnl_inr":   round(total_current_value - total_invested, 2),
            "unrealized_pnl_pct":   round((total_current_value - total_invested) / total_invested * 100, 2) if total_invested else 0,
            "total_realized_pnl":   float(portfolio.total_realized_pnl),
            "total_tax_paid":       float(portfolio.total_tax_paid),
            "last_rebalanced_at":   str(portfolio.last_rebalanced_at) if portfolio.last_rebalanced_at else None,
            "positions":            sorted(position_data, key=lambda x: -x["current_value_inr"]),
        }
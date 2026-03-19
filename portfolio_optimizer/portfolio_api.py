"""
portfolio_api.py — Public API for BuildMyPortfolio.AI

Single importable module. Two entry points:

  1. initialize_portfolio(email, capital, config)
     → First-time setup: runs optimizer, buys positions, stores everything.
     → Returns full result JSON.

  2. rebalance_portfolio(email, config, portfolio_id=None)
     → Subsequent runs: new config in, trade plan computed, executed, full result returned.
     → Returns full result JSON with before/after comparison.

NLP team passes in a CONFIG dict and email. That's it.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from datetime import date, datetime
from typing import Optional

from database import init_db
from riskfolio_main import run_portfolio
from portfolio_engine import (
    get_or_create_user,
    execute_initial_buy,
    compute_rebalance,
    execute_rebalance,
    get_portfolio_snapshot,
)
from models import Portfolio, OptimizerRun
from database import SessionLocal

# Initialise DB tables on import — safe to call multiple times
init_db()


# ─────────────────────────────────────────────────────────────────────────────
# JSON SERIALIZATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

class _SafeEncoder(json.JSONEncoder):
    """
    Handles types that standard json.dumps can't serialize:
    - datetime / date  → ISO string
    - Decimal          → float
    - float nan/inf    → null  (JSON has no NaN/Infinity)
    - bytes            → base64 string
    """
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat(timespec="seconds") + "Z"
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, bytes):
            import base64
            return base64.b64encode(obj).decode()
        return super().default(obj)

    def encode(self, obj):
        # Replace nan/inf floats with None before encoding
        return super().encode(self._sanitize(obj))

    def _sanitize(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj


def to_json(result: dict, indent: int = 2) -> str:
    """
    Convert any portfolio_api result dict to a JSON string.
    Safe for all types — datetime, Decimal, NaN, etc.

    Usage:
        result = initialize_portfolio(...)
        json_str = to_json(result)

        # Or pretty-print:
        print(to_json(result, indent=2))

        # Or write to file:
        with open("result.json", "w") as f:
            f.write(to_json(result))
    """
    return json.dumps(result, cls=_SafeEncoder, indent=indent, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_active_portfolio(user_id: str) -> Optional[str]:
    """Return the most recent active portfolio_id for a user, or None."""
    db = SessionLocal()
    try:
        portfolio = (
            db.query(Portfolio)
            .filter(Portfolio.user_id == user_id, Portfolio.is_active == True)
            .order_by(Portfolio.created_at.desc())
            .first()
        )
        return portfolio.portfolio_id if portfolio else None
    finally:
        db.close()


def _positions_pnl(snapshot: dict) -> dict:
    """
    Split positions from snapshot into winners, losers, and flat.
    Adds a summary block for easy frontend rendering.
    """
    positions = snapshot.get("positions", [])

    winners = []
    losers  = []
    flat    = []

    for p in positions:
        pnl = p.get("unrealized_pnl_inr", 0)
        if pnl > 0.5:
            winners.append(p)
        elif pnl < -0.5:
            losers.append(p)
        else:
            flat.append(p)

    # Sort each group by absolute P&L descending
    winners.sort(key=lambda x: x["unrealized_pnl_inr"], reverse=True)
    losers.sort(key=lambda x: x["unrealized_pnl_inr"])

    total_unrealized = sum(p.get("unrealized_pnl_inr", 0) for p in positions)
    total_invested   = sum(p.get("cost_basis_inr", 0) for p in positions)

    return {
        "winners": winners,
        "losers":  losers,
        "flat":    flat,
        "summary": {
            "total_positions":     len(positions),
            "profitable_count":    len(winners),
            "loss_count":          len(losers),
            "flat_count":          len(flat),
            "total_unrealized_pnl_inr": round(total_unrealized, 2),
            "total_unrealized_pnl_pct": round(
                total_unrealized / total_invested * 100, 2
            ) if total_invested else 0,
        },
    }


def _class_breakdown(snapshot: dict) -> list[dict]:
    """Group positions by asset class with totals."""
    from collections import defaultdict
    buckets = defaultdict(lambda: {"value_inr": 0, "cost_inr": 0, "count": 0})

    total_value = snapshot.get("portfolio_value", 1)

    for p in snapshot.get("positions", []):
        cls = p.get("asset_class", "Unknown")
        buckets[cls]["value_inr"] += p.get("current_value_inr", 0)
        buckets[cls]["cost_inr"]  += p.get("cost_basis_inr", 0)
        buckets[cls]["count"]     += 1

    result = []
    for cls, data in sorted(buckets.items(), key=lambda x: -x[1]["value_inr"]):
        pnl = data["value_inr"] - data["cost_inr"]
        result.append({
            "asset_class":        cls,
            "current_value_inr":  round(data["value_inr"], 2),
            "cost_basis_inr":     round(data["cost_inr"], 2),
            "unrealized_pnl_inr": round(pnl, 2),
            "weight_pct":         round(data["value_inr"] / total_value * 100, 2),
            "position_count":     data["count"],
        })
    return result


def _build_response(
    event_type:        str,
    user_id:           str,
    email:             str,
    portfolio_id:      str,
    optimizer_result:  dict,
    snapshot:          dict,
    execution_summary: Optional[dict] = None,
    rebalance_summary: Optional[dict] = None,
    trade_plan:        Optional[dict] = None,
) -> dict:
    """Assemble the full JSON response payload."""

    pnl_breakdown   = _positions_pnl(snapshot)
    class_breakdown = _class_breakdown(snapshot)

    response = {

        # ── Meta ──────────────────────────────────────────────────────────────
        "event_type":    event_type,          # "initial_buy" | "rebalance"
        "timestamp":     datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "user": {
            "user_id": user_id,
            "email":   email,
        },

        # ── Portfolio identity ────────────────────────────────────────────────
        "portfolio": {
            "portfolio_id":       portfolio_id,
            "name":               snapshot.get("name"),
            "profile":            snapshot.get("profile"),
            "initial_capital":    snapshot.get("initial_capital"),
        },

        # ── Optimizer run details ─────────────────────────────────────────────
        "optimizer": {
            "config": optimizer_result.get("meta", {}),
            "performance": optimizer_result.get("performance", {}),
            "target_weights":    optimizer_result.get("asset_allocation", []),
            "target_class_weights": optimizer_result.get("class_allocation", {}),
            "efficient_frontier": optimizer_result.get("efficient_frontier", {}),
            "diversification":   optimizer_result.get("diversification", {}),
        },

        # ── Current portfolio state (live prices) ─────────────────────────────
        "portfolio_state": {
            "portfolio_value_inr":    snapshot.get("portfolio_value"),
            "total_invested_inr":     snapshot.get("total_invested"),
            "leftover_cash_inr":      snapshot.get("leftover_cash"),
            "unrealized_pnl_inr":     snapshot.get("unrealized_pnl_inr"),
            "unrealized_pnl_pct":     snapshot.get("unrealized_pnl_pct"),
            "realized_pnl_inr":       snapshot.get("total_realized_pnl"),
            "total_tax_paid_inr":     snapshot.get("total_tax_paid"),
            "last_rebalanced_at":     snapshot.get("last_rebalanced_at"),
        },

        # ── Positions (full detail) ────────────────────────────────────────────
        "positions": {
            "all":     snapshot.get("positions", []),
            "winners": pnl_breakdown["winners"],
            "losers":  pnl_breakdown["losers"],
            "flat":    pnl_breakdown["flat"],
            "pnl_summary": pnl_breakdown["summary"],
        },

        # ── Asset class breakdown ─────────────────────────────────────────────
        "class_breakdown": class_breakdown,
    }

    # ── Initial buy extras ────────────────────────────────────────────────────
    if event_type == "initial_buy" and execution_summary:
        response["initial_buy"] = {
            "capital_deployed_inr":   execution_summary.get("total_invested"),
            "leftover_cash_inr":      execution_summary.get("leftover_cash"),
            "positions_created":      execution_summary.get("positions_created"),
            "positions_skipped":      execution_summary.get("positions_skipped"),
            "skipped_tickers":        execution_summary.get("skipped", []),
            "trades":                 execution_summary.get("trades", []),
        }

    # ── Rebalance extras ──────────────────────────────────────────────────────
    if event_type == "rebalance" and rebalance_summary and trade_plan:
        response["rebalance"] = {
            "rebalance_id":           rebalance_summary.get("rebalance_id"),
            "portfolio_value_before": rebalance_summary.get("portfolio_value_before"),
            "portfolio_value_after":  rebalance_summary.get("portfolio_value_after"),
            "value_change_inr":       round(
                (rebalance_summary.get("portfolio_value_after", 0) or 0) -
                (rebalance_summary.get("portfolio_value_before", 0) or 0), 2
            ),
            "total_sold_inr":         rebalance_summary.get("total_sold_inr"),
            "total_bought_inr":       rebalance_summary.get("total_bought_inr"),
            "total_realized_pnl_inr": rebalance_summary.get("total_realized_pnl"),
            "total_tax_inr":          rebalance_summary.get("total_tax_inr"),
            "trades_executed": {
                "sell":  len([t for t in rebalance_summary.get("trades", []) if t["action"] == "SELL"]),
                "buy":   len([t for t in rebalance_summary.get("trades", []) if t["action"] == "BUY"]),
            },
            "full_trade_plan": trade_plan.get("trades", []),
            "plan_summary":    trade_plan.get("summary", {}),
        }

    return response


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — ENTRY POINT 1: INITIALIZE PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

def initialize_portfolio(
    email:          str,
    capital:        float,
    config:         dict,
    portfolio_name: Optional[str] = None,
    nlp_input:      Optional[str] = None,
    name:           Optional[str] = None,
    threshold_pct:  float = 2.0,
) -> dict:
    """
    First-time portfolio setup.

    Called when a user creates their portfolio for the first time.
    Runs the optimizer, buys all positions, stores everything in DB.

    Parameters
    ----------
    email          : str   — user email (creates user if not exists)
    capital        : float — total INR to invest  e.g. 100_000.0
    config         : dict  — optimizer CONFIG dict from NLP team
    portfolio_name : str   — optional display name (defaults to profile name)
    nlp_input      : str   — raw NLP text that produced the config (optional)
    name           : str   — user's display name (optional)
    threshold_pct  : float — rebalance threshold for future use (default 2%)

    Returns
    -------
    dict — full JSON payload (see schema below)
    """
    if capital <= 0:
        raise ValueError(f"Capital must be positive, got {capital}")
    if not config.get("profile"):
        raise ValueError("CONFIG must have a 'profile' key (conservative/balanced/aggressive)")

    # 1. Get or create user
    user_id = get_or_create_user(email, name)

    # 2. Check — user shouldn't already have an active portfolio
    existing = _get_active_portfolio(user_id)
    if existing:
        raise ValueError(
            f"User {email} already has an active portfolio ({existing}). "
            f"Use rebalance_portfolio() instead."
        )

    # 3. Run optimizer
    optimizer_result = run_portfolio(
        config    = config,
        plot      = False,
        save_json = False,
        verbose   = False,
    )
    if optimizer_result.get("status") != "success":
        raise RuntimeError(f"Optimizer failed: {optimizer_result.get('error')}")

    # 4. Execute initial buy
    pname = portfolio_name or f"My {config['profile'].title()} Portfolio"
    execution = execute_initial_buy(
        user_id          = user_id,
        portfolio_name   = pname,
        capital          = capital,
        optimizer_result = optimizer_result,
        nlp_input        = nlp_input,
    )
    portfolio_id = execution["portfolio_id"]

    # 5. Fetch live snapshot (current state with live prices)
    snapshot = get_portfolio_snapshot(portfolio_id)

    # 6. Assemble and return response
    return _build_response(
        event_type       = "initial_buy",
        user_id          = user_id,
        email            = email,
        portfolio_id     = portfolio_id,
        optimizer_result = optimizer_result,
        snapshot         = snapshot,
        execution_summary = execution,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — ENTRY POINT 2: REBALANCE PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

def rebalance_portfolio(
    email:         str,
    config:        dict,
    portfolio_id:  Optional[str] = None,
    nlp_input:     Optional[str] = None,
    threshold_pct: float = 2.0,
    dry_run:       bool = False,
) -> dict:
    """
    Rebalance an existing portfolio with a new config.

    Called every time the NLP team produces a new CONFIG.
    Computes the trade plan, executes trades, updates DB, returns full result.

    Parameters
    ----------
    email          : str   — user email (must already exist)
    config         : dict  — new optimizer CONFIG dict from NLP team
    portfolio_id   : str   — specific portfolio to rebalance (optional,
                             defaults to user's most recent active portfolio)
    nlp_input      : str   — raw NLP text that produced the config (optional)
    threshold_pct  : float — min weight delta to trigger a trade (default 2%)
    dry_run        : bool  — if True, compute plan but don't execute trades

    Returns
    -------
    dict — full JSON payload with before/after comparison, trade list,
           winners/losers, new weights, new efficient frontier, everything
    """
    if not config.get("profile"):
        raise ValueError("CONFIG must have a 'profile' key")

    # 1. Resolve user and portfolio
    db = SessionLocal()
    try:
        from models import User
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise ValueError(f"User {email} not found. Call initialize_portfolio() first.")
        user_id = user.user_id
    finally:
        db.close()

    if not portfolio_id:
        portfolio_id = _get_active_portfolio(user_id)
    if not portfolio_id:
        raise ValueError(
            f"No active portfolio found for {email}. "
            f"Call initialize_portfolio() first."
        )

    # 2. Run optimizer with new config
    optimizer_result = run_portfolio(
        config    = config,
        plot      = False,
        save_json = False,
        verbose   = False,
    )
    if optimizer_result.get("status") != "success":
        raise RuntimeError(f"Optimizer failed: {optimizer_result.get('error')}")

    # 3. Compute trade plan
    trade_plan = compute_rebalance(
        portfolio_id     = portfolio_id,
        optimizer_result = optimizer_result,
        threshold_pct    = threshold_pct,
    )

    # 4. Execute (unless dry_run)
    if dry_run:
        # Return the plan without executing — useful for showing user a preview
        snapshot = get_portfolio_snapshot(portfolio_id)
        response = _build_response(
            event_type       = "rebalance",
            user_id          = user_id,
            email            = email,
            portfolio_id     = portfolio_id,
            optimizer_result = optimizer_result,
            snapshot         = snapshot,
            trade_plan       = trade_plan,
        )
        response["dry_run"] = True
        response["rebalance"] = {
            "dry_run":         True,
            "message":         "Trade plan computed but not executed — set dry_run=False to execute",
            "full_trade_plan": trade_plan.get("trades", []),
            "plan_summary":    trade_plan.get("summary", {}),
        }
        return response

    rebalance_summary = execute_rebalance(
        portfolio_id     = portfolio_id,
        trade_plan       = trade_plan,
        optimizer_result = optimizer_result,
        nlp_input        = nlp_input,
    )

    # 5. Fetch updated snapshot
    snapshot = get_portfolio_snapshot(portfolio_id)

    # 6. Assemble and return response
    return _build_response(
        event_type        = "rebalance",
        user_id           = user_id,
        email             = email,
        portfolio_id      = portfolio_id,
        optimizer_result  = optimizer_result,
        snapshot          = snapshot,
        rebalance_summary = rebalance_summary,
        trade_plan        = trade_plan,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — EXTRA: GET CURRENT STATE (no optimizer run)
# ─────────────────────────────────────────────────────────────────────────────

def get_portfolio_state(email: str, portfolio_id: Optional[str] = None) -> dict:
    """
    Get the current portfolio state with live prices and P&L.
    No optimizer run — just reads DB + fetches live prices.
    Call this for the dashboard refresh.

    Parameters
    ----------
    email        : str — user email
    portfolio_id : str — optional, defaults to most recent active portfolio

    Returns
    -------
    dict — portfolio state with positions, winners, losers, class breakdown
    """
    db = SessionLocal()
    try:
        from models import User
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise ValueError(f"User {email} not found.")
        user_id = user.user_id
    finally:
        db.close()

    if not portfolio_id:
        portfolio_id = _get_active_portfolio(user_id)
    if not portfolio_id:
        raise ValueError(f"No active portfolio found for {email}.")

    snapshot        = get_portfolio_snapshot(portfolio_id)
    pnl_breakdown   = _positions_pnl(snapshot)
    class_breakdown = _class_breakdown(snapshot)

    return {
        "timestamp":    datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "user":         {"user_id": user_id, "email": email},
        "portfolio": {
            "portfolio_id":    portfolio_id,
            "name":            snapshot.get("name"),
            "profile":         snapshot.get("profile"),
            "initial_capital": snapshot.get("initial_capital"),
        },
        "portfolio_state": {
            "portfolio_value_inr":  snapshot.get("portfolio_value"),
            "total_invested_inr":   snapshot.get("total_invested"),
            "leftover_cash_inr":    snapshot.get("leftover_cash"),
            "unrealized_pnl_inr":   snapshot.get("unrealized_pnl_inr"),
            "unrealized_pnl_pct":   snapshot.get("unrealized_pnl_pct"),
            "realized_pnl_inr":     snapshot.get("total_realized_pnl"),
            "total_tax_paid_inr":   snapshot.get("total_tax_paid"),
            "last_rebalanced_at":   snapshot.get("last_rebalanced_at"),
        },
        "positions": {
            "all":         snapshot.get("positions", []),
            "winners":     pnl_breakdown["winners"],
            "losers":      pnl_breakdown["losers"],
            "flat":        pnl_breakdown["flat"],
            "pnl_summary": pnl_breakdown["summary"],
        },
        "class_breakdown": class_breakdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — python portfolio_api.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG

    EMAIL   = "test@buildmyportfolio.ai"
    CAPITAL = 100_000.0

    print("\n" + "="*60)
    print("TEST — initialize_portfolio()")
    print("="*60)
    result = initialize_portfolio(
        email      = EMAIL,
        capital    = CAPITAL,
        config     = CONSERVATIVE_CONFIG,
        nlp_input  = "I want a conservative portfolio",
        name       = "Test User",
    )
    print(f"Event        : {result['event_type']}")
    print(f"Portfolio ID : {result['portfolio']['portfolio_id']}")
    print(f"Profile      : {result['portfolio']['profile']}")
    print(f"Value        : ₹{result['portfolio_state']['portfolio_value_inr']:,.2f}")
    print(f"Invested     : ₹{result['portfolio_state']['total_invested_inr']:,.2f}")
    print(f"Leftover     : ₹{result['portfolio_state']['leftover_cash_inr']:,.2f}")
    print(f"Return       : {result['optimizer']['performance']['expected_annual_return_pct']}%")
    print(f"Volatility   : {result['optimizer']['performance']['annual_volatility_pct']}%")
    print(f"Sharpe       : {result['optimizer']['performance']['sharpe_ratio']}")
    print(f"Winners      : {result['positions']['pnl_summary']['profitable_count']}")
    print(f"Losers       : {result['positions']['pnl_summary']['loss_count']}")
    print(f"Positions    : {result['initial_buy']['positions_created']} created, "
          f"{result['initial_buy']['positions_skipped']} skipped")
    print(f"Frontier pts : {len(result['optimizer']['efficient_frontier'].get('points', []))}")

    pid = result['portfolio']['portfolio_id']

    print("\n" + "="*60)
    print("TEST — rebalance_portfolio() dry_run=True")
    print("="*60)
    dry = rebalance_portfolio(
        email    = EMAIL,
        config   = AGGRESSIVE_CONFIG,
        nlp_input= "I think the market is going up, be more aggressive",
        dry_run  = True,
    )
    plan = dry.get("rebalance", {})
    summary = plan.get("plan_summary", {})
    trades  = summary.get("trades_count", {})
    print(f"Dry run plan : {trades.get('sell',0)} sells, {trades.get('buy',0)} buys, {trades.get('hold',0)} holds")
    print(f"Est. tax     : ₹{summary.get('estimated_tax_inr', 0):,.2f}")

    print("\n" + "="*60)
    print("TEST — rebalance_portfolio() execute")
    print("="*60)
    result2 = rebalance_portfolio(
        email     = EMAIL,
        config    = AGGRESSIVE_CONFIG,
        nlp_input = "I think the market is going up, be more aggressive",
        dry_run   = False,
    )
    rb = result2.get("rebalance", {})
    print(f"Event        : {result2['event_type']}")
    print(f"Before       : ₹{rb.get('portfolio_value_before', 0):,.2f}")
    print(f"After        : ₹{rb.get('portfolio_value_after', 0):,.2f}")
    print(f"Sold         : ₹{rb.get('total_sold_inr', 0):,.2f}")
    print(f"Bought       : ₹{rb.get('total_bought_inr', 0):,.2f}")
    print(f"PnL          : ₹{rb.get('total_realized_pnl_inr', 0):,.2f}")
    print(f"Tax          : ₹{rb.get('total_tax_inr', 0):,.2f}")
    print(f"Trades       : {rb.get('trades_executed', {})}")
    print(f"Winners now  : {result2['positions']['pnl_summary']['profitable_count']}")
    print(f"Losers now   : {result2['positions']['pnl_summary']['loss_count']}")

    print("\n" + "="*60)
    print("TEST — get_portfolio_state() (dashboard refresh)")
    print("="*60)
    state = get_portfolio_state(EMAIL)
    print(f"Value        : ₹{state['portfolio_state']['portfolio_value_inr']:,.2f}")
    print(f"Unrealized   : ₹{state['portfolio_state']['unrealized_pnl_inr']:,.2f} "
          f"({state['portfolio_state']['unrealized_pnl_pct']}%)")
    print("\nClass breakdown:")
    for cls in state["class_breakdown"]:
        print(f"  {cls['asset_class']:<15} ₹{cls['current_value_inr']:>10,.2f}  "
              f"({cls['weight_pct']}%)  PnL: ₹{cls['unrealized_pnl_inr']:,.2f}")


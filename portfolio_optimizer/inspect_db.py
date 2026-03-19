"""
inspect_db.py — Run this locally to see exactly what's in your portfolio.db
Usage: python inspect_db.py
"""

from database import init_db, SessionLocal
from models import Portfolio, Position, Trade, RebalanceEvent, OptimizerRun, User

init_db()
db = SessionLocal()

# ── USERS ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("USERS")
print("="*60)
users = db.query(User).all()
if not users:
    print("  (none)")
for u in users:
    print(f"  {u.user_id[:8]}... | {u.email} | ltcg_used=₹{u.ltcg_used_this_fy}")

# ── PORTFOLIOS ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PORTFOLIOS")
print("="*60)
portfolios = db.query(Portfolio).all()
if not portfolios:
    print("  (none)")
for p in portfolios:
    print(f"""
  ID       : {p.portfolio_id}
  Name     : {p.name}
  Profile  : {p.profile}
  Capital  : ₹{float(p.initial_capital):,.2f}
  Leftover : ₹{float(p.leftover_cash):,.2f}
  Real PnL : ₹{float(p.total_realized_pnl):,.2f}
  Tax Paid : ₹{float(p.total_tax_paid):,.2f}
  Rebal At : {p.last_rebalanced_at}""")

# ── POSITIONS ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("POSITIONS (current live state)")
print("="*60)
positions = db.query(Position).all()
if not positions:
    print("  (none)")
else:
    print(f"  {'Ticker':<22} {'Units':>8} {'AvgBuy':>10} {'Invested':>12} {'Target%':>8} {'1st Buy'}")
    print("  " + "-"*72)
    for pos in sorted(positions, key=lambda x: float(x.total_invested), reverse=True):
        print(f"  {pos.ticker:<22} {float(pos.units_held):>8.2f} "
              f"₹{float(pos.avg_buy_price):>9,.2f} "
              f"₹{float(pos.total_invested):>11,.2f} "
              f"{float(pos.target_weight_pct or 0):>7.2f}% "
              f"  {pos.first_buy_date}")

# ── TRADES ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRADES (full history)")
print("="*60)
trades = db.query(Trade).order_by(Trade.executed_at).all()
if not trades:
    print("  (none)")
else:
    for t in trades:
        pnl_part = ""
        if t.action == "SELL" and t.realized_pnl is not None:
            pnl_part = (f" | PnL=₹{float(t.realized_pnl):,.2f}"
                        f" | {t.tax_type} tax=₹{float(t.tax_inr or 0):,.2f}"
                        f" | held {t.holding_days}d")
        print(f"  {t.action:<4} | {t.ticker:<22} | "
              f"units={float(t.units):.2f} @₹{float(t.price):,.2f} "
              f"= ₹{float(t.value_inr):,.2f}{pnl_part}")

# ── OPTIMIZER RUNS ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("OPTIMIZER RUNS")
print("="*60)
runs = db.query(OptimizerRun).order_by(OptimizerRun.run_at).all()
if not runs:
    print("  (none)")
for r in runs:
    cfg = r.config_snapshot or {}
    perf = r.performance_snapshot or {}
    print(f"""
  Run ID   : {r.run_id[:8]}...
  Applied  : {r.was_applied}
  Trigger  : {r.triggered_by!r}
  Profile  : {cfg.get('profile', '?')} | Model: {cfg.get('model','?')} | RM: {cfg.get('risk_measure','?')}
  Return   : {perf.get('expected_annual_return_pct','?')}% | Vol: {perf.get('annual_volatility_pct','?')}% | Sharpe: {perf.get('sharpe_ratio','?')}
  Run at   : {r.run_at}""")

# ── REBALANCE EVENTS ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("REBALANCE EVENTS")
print("="*60)
events = db.query(RebalanceEvent).order_by(RebalanceEvent.rebalanced_at).all()
if not events:
    print("  (none)")
for ev in events:
    print(f"""
  ID       : {ev.rebalance_id[:8]}...
  NLP      : {ev.nlp_input!r}
  Val Bef  : ₹{float(ev.portfolio_value_before or 0):,.2f}
  Val Aft  : ₹{float(ev.portfolio_value_after or 0):,.2f}
  Sold     : ₹{float(ev.total_sold_inr or 0):,.2f}
  Bought   : ₹{float(ev.total_bought_inr or 0):,.2f}
  PnL      : ₹{float(ev.total_realized_pnl or 0):,.2f}
  Tax      : ₹{float(ev.total_tax_inr or 0):,.2f}""")

    # Show the trade plan that was applied
    plan = ev.trade_plan_snapshot or {}
    trade_list = plan.get("trades", [])
    sells = [t for t in trade_list if t.get("action") == "SELL"]
    buys  = [t for t in trade_list if t.get("action") == "BUY"]
    holds = [t for t in trade_list if t.get("action") == "HOLD"]
    print(f"  Plan     : {len(sells)} sells, {len(buys)} buys, {len(holds)} holds")
    for t in trade_list:
        if t.get("action") in ("SELL", "BUY"):
            print(f"    {t['action']:<4} {t['ticker']:<22} "
                  f"delta={t.get('delta_pct',0):+.2f}% "
                  f"units={t.get('units_to_trade','?')} "
                  f"val=₹{t.get('estimated_value_inr',0):,.0f}")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Users      : {db.query(User).count()}")
print(f"  Portfolios : {db.query(Portfolio).count()}")
print(f"  Positions  : {db.query(Position).count()}")
print(f"  Trades     : {db.query(Trade).count()}")
print(f"  Opt runs   : {db.query(OptimizerRun).count()}")
print(f"  Rebalances : {db.query(RebalanceEvent).count()}")
print()

db.close()

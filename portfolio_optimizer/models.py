"""
models.py — SQLAlchemy ORM models for BuildMyPortfolio.AI
All 6 tables: users, portfolios, optimizer_runs, positions, trades, rebalance_events
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Boolean, Numeric, Date,
    DateTime, Text, ForeignKey, Enum as SAEnum, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def new_uuid() -> str:
    return str(uuid.uuid4())


# ── 1. USERS ─────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    user_id             = Column(String(36), primary_key=True, default=new_uuid)
    email               = Column(String(255), unique=True, nullable=False)
    name                = Column(String(255), nullable=True)
    created_at          = Column(DateTime, default=datetime.utcnow)

    # Indian tax — track cumulative LTCG this FY (₹1L exemption)
    ltcg_used_this_fy   = Column(Numeric(14, 2), default=0.00)
    current_fy_start    = Column(Date, nullable=True)   # e.g. 2024-04-01

    # Relationships
    portfolios          = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email}>"


# ── 2. PORTFOLIOS ─────────────────────────────────────────────────────────────
class Portfolio(Base):
    __tablename__ = "portfolios"

    portfolio_id        = Column(String(36), primary_key=True, default=new_uuid)
    user_id             = Column(String(36), ForeignKey("users.user_id"), nullable=False)
    name                = Column(String(255), nullable=False)   # "My Conservative Portfolio"

    # Capital
    initial_capital     = Column(Numeric(14, 2), nullable=False)  # original user input
    leftover_cash       = Column(Numeric(14, 2), default=0.00)    # uninvested (from rounding)

    # Profile
    profile             = Column(String(50), nullable=False)      # conservative/balanced/aggressive

    # Timestamps
    created_at          = Column(DateTime, default=datetime.utcnow)
    last_rebalanced_at  = Column(DateTime, nullable=True)

    # Running totals (updated after every trade)
    total_realized_pnl  = Column(Numeric(14, 2), default=0.00)
    total_tax_paid      = Column(Numeric(14, 2), default=0.00)

    # Soft delete
    is_active           = Column(Boolean, default=True)

    # Relationships
    user                = relationship("User", back_populates="portfolios")
    positions           = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    trades              = relationship("Trade", back_populates="portfolio", cascade="all, delete-orphan")
    optimizer_runs      = relationship("OptimizerRun", back_populates="portfolio", cascade="all, delete-orphan")
    rebalance_events    = relationship("RebalanceEvent", back_populates="portfolio", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Portfolio {self.name} [{self.profile}]>"


# ── 3. OPTIMIZER RUNS ────────────────────────────────────────────────────────
class OptimizerRun(Base):
    __tablename__ = "optimizer_runs"

    run_id              = Column(String(36), primary_key=True, default=new_uuid)
    portfolio_id        = Column(String(36), ForeignKey("portfolios.portfolio_id"), nullable=False)

    # What triggered this run
    triggered_by        = Column(Text, nullable=True)   # raw NLP input

    # Full snapshots as JSON (SQLite stores as TEXT, Postgres uses JSONB)
    config_snapshot     = Column(JSON, nullable=False)  # full CONFIG dict
    weights_snapshot    = Column(JSON, nullable=False)  # full run_portfolio() output
    performance_snapshot = Column(JSON, nullable=True)  # {sharpe, vol, return, cvar}

    run_at              = Column(DateTime, default=datetime.utcnow)
    was_applied         = Column(Boolean, default=False)  # did user accept and execute?

    # Relationships
    portfolio           = relationship("Portfolio", back_populates="optimizer_runs")
    rebalance_events    = relationship("RebalanceEvent", back_populates="optimizer_run")

    def __repr__(self):
        return f"<OptimizerRun {self.run_id[:8]} applied={self.was_applied}>"


# ── 4. POSITIONS ──────────────────────────────────────────────────────────────
# Live state — one row per ticker per portfolio. Updated after every trade.
class Position(Base):
    __tablename__ = "positions"

    position_id         = Column(String(36), primary_key=True, default=new_uuid)
    portfolio_id        = Column(String(36), ForeignKey("portfolios.portfolio_id"), nullable=False)

    ticker              = Column(String(20), nullable=False)    # e.g. RELIANCE.NS
    asset_class         = Column(String(20), nullable=False)    # Equity/Debt/Gold/etc.

    # Holdings
    units_held          = Column(Numeric(14, 4), nullable=False, default=0)
    avg_buy_price       = Column(Numeric(14, 4), nullable=False)  # weighted average cost basis
    total_invested      = Column(Numeric(14, 2), nullable=False)  # units_held × avg_buy_price

    # Tax tracking — needed for LTCG vs STCG determination
    first_buy_date      = Column(Date, nullable=False)   # oldest lot — determines LTCG eligibility
    last_buy_date       = Column(Date, nullable=False)   # most recent purchase

    # Optimizer target
    target_weight_pct   = Column(Numeric(8, 4), nullable=True)  # last optimizer output weight

    updated_at          = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    portfolio           = relationship("Portfolio", back_populates="positions")
    trades              = relationship("Trade", back_populates="position")

    def __repr__(self):
        return f"<Position {self.ticker} units={self.units_held} avg={self.avg_buy_price}>"


# ── 5. TRADES ─────────────────────────────────────────────────────────────────
# Immutable audit trail — never edit, only append.
class Trade(Base):
    __tablename__ = "trades"

    trade_id            = Column(String(36), primary_key=True, default=new_uuid)
    portfolio_id        = Column(String(36), ForeignKey("portfolios.portfolio_id"), nullable=False)
    position_id         = Column(String(36), ForeignKey("positions.position_id"), nullable=False)
    rebalance_id        = Column(String(36), ForeignKey("rebalance_events.rebalance_id"), nullable=True)
    # NULL for initial buy — only set for rebalance trades

    # Denormalized for easy querying without joins
    ticker              = Column(String(20), nullable=False)
    asset_class         = Column(String(20), nullable=False)
    action              = Column(SAEnum("BUY", "SELL", name="trade_action"), nullable=False)

    # Trade details
    units               = Column(Numeric(14, 4), nullable=False)
    price               = Column(Numeric(14, 4), nullable=False)   # execution price
    value_inr           = Column(Numeric(14, 2), nullable=False)   # units × price

    # SELL-only fields (NULL for BUY)
    avg_cost_at_sale    = Column(Numeric(14, 4), nullable=True)    # avg_buy_price at time of sale
    realized_pnl        = Column(Numeric(14, 2), nullable=True)    # (price - avg_cost) × units
    holding_days        = Column(Integer, nullable=True)           # days from first_buy_date
    tax_type            = Column(String(10), nullable=True)        # STCG / LTCG / NONE (loss)
    tax_rate_pct        = Column(Numeric(5, 2), nullable=True)     # 15.0 or 10.0
    tax_inr             = Column(Numeric(12, 2), nullable=True)    # estimated tax

    # Weight context (for analysis)
    weight_before_pct   = Column(Numeric(8, 4), nullable=True)    # position weight before trade
    weight_after_pct    = Column(Numeric(8, 4), nullable=True)    # target weight after trade

    executed_at         = Column(DateTime, default=datetime.utcnow)

    # Relationships
    portfolio           = relationship("Portfolio", back_populates="trades")
    position            = relationship("Position", back_populates="trades")
    rebalance_event     = relationship("RebalanceEvent", back_populates="trades")

    def __repr__(self):
        return f"<Trade {self.action} {self.ticker} units={self.units} @{self.price}>"


# ── 6. REBALANCE EVENTS ───────────────────────────────────────────────────────
class RebalanceEvent(Base):
    __tablename__ = "rebalance_events"

    rebalance_id            = Column(String(36), primary_key=True, default=new_uuid)
    portfolio_id            = Column(String(36), ForeignKey("portfolios.portfolio_id"), nullable=False)
    run_id                  = Column(String(36), ForeignKey("optimizer_runs.run_id"), nullable=False)

    nlp_input               = Column(Text, nullable=True)           # what user said

    # Snapshots (JSON)
    old_weights_snapshot    = Column(JSON, nullable=True)           # weights before
    new_weights_snapshot    = Column(JSON, nullable=True)           # weights after
    trade_plan_snapshot     = Column(JSON, nullable=True)           # full trade list shown to user

    # Financial summary
    portfolio_value_before  = Column(Numeric(14, 2), nullable=True)
    portfolio_value_after   = Column(Numeric(14, 2), nullable=True)
    total_sold_inr          = Column(Numeric(14, 2), default=0.00)
    total_bought_inr        = Column(Numeric(14, 2), default=0.00)
    total_realized_pnl      = Column(Numeric(14, 2), default=0.00)
    total_tax_inr           = Column(Numeric(14, 2), default=0.00)

    rebalanced_at           = Column(DateTime, default=datetime.utcnow)

    # Relationships
    portfolio               = relationship("Portfolio", back_populates="rebalance_events")
    optimizer_run           = relationship("OptimizerRun", back_populates="rebalance_events")
    trades                  = relationship("Trade", back_populates="rebalance_event")

    def __repr__(self):
        return f"<RebalanceEvent {self.rebalance_id[:8]} portfolio={self.portfolio_id[:8]}>"

"""
database.py — SQLite engine, session factory, and DB initialisation
Switch to Postgres later by changing DATABASE_URL only.
"""

import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from models import Base

# ── CONFIG ────────────────────────────────────────────────────────────────────
# SQLite for local dev. Switch to Postgres with:
# DATABASE_URL = "postgresql://user:password@localhost:5432/portfolio_db"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///portfolio.db")

# ── ENGINE ────────────────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,   # set True to log all SQL (useful for debugging)
)

# SQLite-specific: enforce foreign keys (disabled by default in SQLite)
if "sqlite" in DATABASE_URL:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# ── SESSION FACTORY ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_db() -> Session:
    """
    Context manager for DB sessions.
    Automatically commits on success, rolls back on exception.

    Usage:
        with get_db() as db:
            db.add(some_model)
            # commit happens automatically on exit
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── INIT ──────────────────────────────────────────────────────────────────────
def init_db():
    """Create all tables if they don't exist. Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)
    print("[DB] All tables created (or already exist).")


def drop_all():
    """Drop all tables. DANGEROUS — only use in dev/testing."""
    Base.metadata.drop_all(bind=engine)
    print("[DB] All tables dropped.")


if __name__ == "__main__":
    init_db()

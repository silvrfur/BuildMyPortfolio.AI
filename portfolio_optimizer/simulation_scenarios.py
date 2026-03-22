"""
simulation_scenarios.py — 5 user simulation scenarios.

Each scenario defines:
  - email, name, capital
  - events: list of (date_str, config_name, nlp_input)
    First event = initial buy date
    Subsequent events = rebalance dates

The simulator runs each scenario on two parallel tracks:
  Track A — Rebalanced: follows the event list exactly
  Track B — Hold:       only buys on first event, holds to end
"""

try:
    from .config import CONSERVATIVE_CONFIG, BALANCED_CONFIG, AGGRESSIVE_CONFIG
except ImportError:
    from config import CONSERVATIVE_CONFIG, BALANCED_CONFIG, AGGRESSIVE_CONFIG

# Shorthand aliases
C = CONSERVATIVE_CONFIG
B = BALANCED_CONFIG
A = AGGRESSIVE_CONFIG

SIMULATION_END_DATE = "2025-01-01"   # all scenarios evaluated up to this date

SCENARIOS = [

    # ── USER 1 — The Reactive Investor ────────────────────────────────────────
    # Starts conservative, goes aggressive on optimism, de-risks on war fears
    {
        "email":   "user1@gmail.com",
        "name":    "Arjun Sharma",
        "capital": 100_000.0,
        "persona": "Reactive Investor — chases sentiment, rebalances often",
        "events": [
            {
                "date":      "2022-01-03",
                "config":    C,
                "nlp_input": "Market uncertain after COVID rally, stay safe",
            },
            {
                "date":      "2022-02-01",
                "config":    A,
                "nlp_input": "Market feels strong, I want to go aggressive",
            },
            {
                "date":      "2022-04-01",
                "config":    B,
                "nlp_input": "Things seem okay now, balanced approach",
            },
            {
                "date":      "2022-05-01",
                "config":    C,
                "nlp_input": "War and inflation fears, make it conservative",
            },
            {
                "date":      "2022-12-01",
                "config":    B,
                "nlp_input": "Stability is returning, go balanced",
            },
        ],
    },

    # ── USER 2 — The Dip Buyer ────────────────────────────────────────────────
    # Starts balanced, turns aggressive on dips, normalises on recovery
    {
        "email":   "user2@gmail.com",
        "name":    "Priya Nair",
        "capital": 100_000.0,
        "persona": "Dip Buyer — contrarian, buys when others fear",
        "events": [
            {
                "date":      "2022-01-03",
                "config":    B,
                "nlp_input": "Steady investor, balanced approach to start",
            },
            {
                "date":      "2022-06-01",
                "config":    A,
                "nlp_input": "Market is down a lot, great buying opportunity, go aggressive",
            },
            {
                "date":      "2023-01-02",
                "config":    B,
                "nlp_input": "Took profits from the rally, back to balanced",
            },
            {
                "date":      "2023-06-01",
                "config":    A,
                "nlp_input": "Bull run starting, ride the momentum aggressively",
            },
            {
                "date":      "2024-01-02",
                "config":    B,
                "nlp_input": "Election year, be cautious, go balanced",
            },
        ],
    },

    # ── USER 3 — The Panic Seller ─────────────────────────────────────────────
    # Starts aggressive, panics on crashes, recovers slowly
    {
        "email":   "user3@gmail.com",
        "name":    "Rahul Mehta",
        "capital": 100_000.0,
        "persona": "Panic Seller — high initial risk, sells on bad news",
        "events": [
            {
                "date":      "2022-01-03",
                "config":    A,
                "nlp_input": "Young investor, high risk appetite, go all in",
            },
            {
                "date":      "2022-05-01",
                "config":    C,
                "nlp_input": "Market crashed badly, I am scared, make it very safe",
            },
            {
                "date":      "2023-01-02",
                "config":    A,
                "nlp_input": "Recovered my confidence, going aggressive again",
            },
            {
                "date":      "2023-10-01",
                "config":    C,
                "nlp_input": "Middle East conflict, too risky, go conservative",
            },
            {
                "date":      "2024-04-01",
                "config":    A,
                "nlp_input": "Rally resumed, back to aggressive",
            },
        ],
    },

    # ── USER 4 — The Set and Forget ───────────────────────────────────────────
    # Stays conservative throughout, minimal rebalancing
    {
        "email":   "user4@gmail.com",
        "name":    "Sunita Iyer",
        "capital": 100_000.0,
        "persona": "Set and Forget — conservative throughout, rebalances once a year",
        "events": [
            {
                "date":      "2022-01-03",
                "config":    C,
                "nlp_input": "Retiree, capital preservation is priority",
            },
            {
                "date":      "2023-01-02",
                "config":    C,
                "nlp_input": "Still conservative, annual rebalance only",
            },
            {
                "date":      "2024-01-02",
                "config":    C,
                "nlp_input": "Staying conservative for another year",
            },
        ],
    },

    # ── USER 5 — The News Follower ────────────────────────────────────────────
    # Reacts to geopolitical events, frequent but moderate changes
    {
        "email":   "user5@gmail.com",
        "name":    "Kiran Desai",
        "capital": 100_000.0,
        "persona": "News Follower — reacts to geopolitical events frequently",
        "events": [
            {
                "date":      "2022-01-03",
                "config":    B,
                "nlp_input": "Moderate investor, balanced start",
            },
            {
                "date":      "2022-03-01",
                "config":    C,
                "nlp_input": "Russia Ukraine war started, go conservative immediately",
            },
            {
                "date":      "2022-07-01",
                "config":    B,
                "nlp_input": "Ceasefire hopes, back to balanced",
            },
            {
                "date":      "2022-12-01",
                "config":    A,
                "nlp_input": "Santa rally, year end optimism, go aggressive",
            },
            {
                "date":      "2023-06-01",
                "config":    B,
                "nlp_input": "Good returns, normalise to balanced",
            },
            {
                "date":      "2023-10-01",
                "config":    C,
                "nlp_input": "Israel Gaza conflict, go conservative again",
            },
        ],
    },
]

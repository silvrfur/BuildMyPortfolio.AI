from datetime import date

import pandas as pd

from portfolio_optimizer.latent_state_simulator import generate_latent_scenario


def _mock_price_history() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", "2025-01-01", freq="B")
    prices = [100 + idx * 0.1 for idx in range(len(dates))]
    return pd.DataFrame(
        {
            "NIFTYBEES.NS": prices,
            "RELIANCE.NS": [value * 1.05 for value in prices],
        },
        index=dates,
    )


def test_generate_latent_scenario_creates_multiple_chats_per_state():
    scenario = {
        "email": "demo@test.com",
        "name": "Demo User",
        "capital": 100_000.0,
        "persona": "Reactive Investor",
        "events": [{"date": "2022-01-03"}],
    }

    generated = generate_latent_scenario(
        scenario,
        seed=7,
        price_history=_mock_price_history(),
    )

    assert generated["latent_timeline"]
    assert all(state["chats"] for state in generated["latent_timeline"])
    assert any(len(state["chats"]) > 1 for state in generated["latent_timeline"])


def test_generate_latent_scenario_respects_minimum_gap_of_three_months():
    scenario = {
        "email": "gap@test.com",
        "name": "Gap User",
        "capital": 100_000.0,
        "persona": "Set and Forget",
        "events": [{"date": "2022-01-03"}],
    }

    generated = generate_latent_scenario(
        scenario,
        seed=11,
        price_history=_mock_price_history(),
    )

    starts = [date.fromisoformat(state["state_start"]) for state in generated["latent_timeline"]]
    for previous, current in zip(starts, starts[1:]):
        months_apart = (current.year - previous.year) * 12 + (current.month - previous.month)
        assert months_apart >= 3

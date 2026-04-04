from datetime import date

import pandas as pd

from simulation.latent_state_simulator import generate_latent_scenario, _rank_population_h2_representatives


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


def test_rank_population_h2_representatives_prefers_true_tracking_with_separation_from_static():
    results = [
        {
            "user_id": "sim_user_003",
            "static_theta": {"risk_sensitivity": 0.75, "patience_level": 0.75, "analytical_thinking": 0.75, "controlled_perception": 0.75},
            "events": [
                {
                    "theta_true": {"risk_sensitivity": 0.2, "patience_level": 0.3, "analytical_thinking": 0.4, "controlled_perception": 0.5},
                    "theta_inferred": {"risk_sensitivity": 0.4, "patience_level": 0.5, "analytical_thinking": 0.6, "controlled_perception": 0.7},
                },
                {
                    "theta_true": {"risk_sensitivity": 0.25, "patience_level": 0.35, "analytical_thinking": 0.45, "controlled_perception": 0.55},
                    "theta_inferred": {"risk_sensitivity": 0.42, "patience_level": 0.52, "analytical_thinking": 0.62, "controlled_perception": 0.72},
                },
            ],
            "h2_true_reference_improvement": {"dynamic_overall_rmse": 0.22},
        },
        {
            "user_id": "sim_user_001",
            "static_theta": {"risk_sensitivity": 0.21, "patience_level": 0.31, "analytical_thinking": 0.41, "controlled_perception": 0.51},
            "events": [
                {
                    "theta_true": {"risk_sensitivity": 0.2, "patience_level": 0.3, "analytical_thinking": 0.4, "controlled_perception": 0.5},
                    "theta_inferred": {"risk_sensitivity": 0.2, "patience_level": 0.3, "analytical_thinking": 0.4, "controlled_perception": 0.5},
                },
                {
                    "theta_true": {"risk_sensitivity": 0.25, "patience_level": 0.35, "analytical_thinking": 0.45, "controlled_perception": 0.55},
                    "theta_inferred": {"risk_sensitivity": 0.25, "patience_level": 0.35, "analytical_thinking": 0.45, "controlled_perception": 0.55},
                },
            ],
            "h2_true_reference_improvement": {"dynamic_overall_rmse": 0.04, "static_overall_rmse": 0.02},
        },
        {
            "user_id": "sim_user_002",
            "static_theta": {"risk_sensitivity": 0.50, "patience_level": 0.55, "analytical_thinking": 0.60, "controlled_perception": 0.65},
            "events": [
                {
                    "theta_true": {"risk_sensitivity": 0.2, "patience_level": 0.3, "analytical_thinking": 0.4, "controlled_perception": 0.5},
                    "theta_inferred": {"risk_sensitivity": 0.18, "patience_level": 0.28, "analytical_thinking": 0.38, "controlled_perception": 0.48},
                },
                {
                    "theta_true": {"risk_sensitivity": 0.25, "patience_level": 0.35, "analytical_thinking": 0.45, "controlled_perception": 0.55},
                    "theta_inferred": {"risk_sensitivity": 0.23, "patience_level": 0.33, "analytical_thinking": 0.43, "controlled_perception": 0.53},
                },
            ],
            "h2_true_reference_improvement": {"dynamic_overall_rmse": 0.04, "static_overall_rmse": 0.18},
        },
    ]

    ranked = _rank_population_h2_representatives(results, limit=3)

    assert [item["user_id"] for item in ranked] == ["sim_user_002", "sim_user_001", "sim_user_003"]

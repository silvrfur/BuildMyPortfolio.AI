import pytest

from simulation.h3_retention_simulator import (
    DEFAULT_LATENT_SIGNAL_THRESHOLD,
    STRATEGY_TIME_VOL,
    STRATEGY_TIME_VOL_LATENT,
    _latent_signal_distance,
    _strategy_rebalance_decision,
    simulate_h3_retention,
)


def test_latent_signal_distance_is_mean_absolute_shift():
    current = {
        "risk_sensitivity": 0.6,
        "patience_level": 0.2,
        "analytical_thinking": 0.5,
        "controlled_perception": 0.3,
    }
    reference = {
        "risk_sensitivity": 0.4,
        "patience_level": 0.3,
        "analytical_thinking": 0.4,
        "controlled_perception": 0.5,
    }

    assert _latent_signal_distance(current, reference) == pytest.approx(0.15)


def test_latent_threshold_strategy_requires_base_trigger_and_shift():
    state = {
        "last_rebalance_month": 1,
        "last_rebalance_theta": {
            "risk_sensitivity": 0.45,
            "patience_level": 0.55,
            "analytical_thinking": 0.55,
            "controlled_perception": 0.55,
        },
    }
    event = {
        "month_index": 1,
        "date": "2022-02-01",
        "market_volatility_score": 0.20,
        "theta_inferred": {
            "risk_sensitivity": 0.80,
            "patience_level": 0.20,
            "analytical_thinking": 0.60,
            "controlled_perception": 0.30,
        },
    }

    time_vol = _strategy_rebalance_decision(
        strategy=STRATEGY_TIME_VOL,
        event=event,
        state=state,
        time_rebalance_months=3,
        volatility_trigger_threshold=0.35,
        latent_signal_threshold=DEFAULT_LATENT_SIGNAL_THRESHOLD,
    )
    time_vol_latent = _strategy_rebalance_decision(
        strategy=STRATEGY_TIME_VOL_LATENT,
        event=event,
        state=state,
        time_rebalance_months=3,
        volatility_trigger_threshold=0.35,
        latent_signal_threshold=DEFAULT_LATENT_SIGNAL_THRESHOLD,
    )

    assert time_vol["base_trigger"] is False
    assert time_vol["latent_trigger"] is True
    assert time_vol["should_rebalance"] is False
    assert time_vol_latent["should_rebalance"] is False


def test_simulate_h3_retention_returns_two_strategy_comparison(monkeypatch):
    def fake_get_price_history():
        return None

    def fake_get_prices_on_date(_date):
        return {"ETF": 100.0}

    def fake_run_optimizer_historical(config, _date):
        return {
            "status": "success",
            "asset_allocation": [
                {"ticker": "ETF", "weight_pct": 100.0, "asset_class": config["profile"]},
            ],
        }

    monkeypatch.setattr("simulation.h3_retention_simulator.get_price_history", fake_get_price_history)
    monkeypatch.setattr("simulation.h3_retention_simulator.get_prices_on_date", fake_get_prices_on_date)
    monkeypatch.setattr("simulation.h3_retention_simulator.run_optimizer_historical", fake_run_optimizer_historical)

    population_payload = {
        "generated_at": "2026-04-03",
        "months": 4,
        "results": [
            {
                "user_id": "sim_user_001",
                "events": [
                    {
                        "date": "2022-01-01",
                        "month_index": 0,
                        "theta_true": {
                            "risk_sensitivity": 0.35,
                            "patience_level": 0.70,
                            "analytical_thinking": 0.60,
                            "controlled_perception": 0.68,
                        },
                        "theta_inferred": {
                            "risk_sensitivity": 0.35,
                            "patience_level": 0.70,
                            "analytical_thinking": 0.60,
                            "controlled_perception": 0.68,
                        },
                        "market_volatility_score": 0.20,
                        "life_event": "none",
                    },
                    {
                        "date": "2022-02-01",
                        "month_index": 1,
                        "theta_true": {
                            "risk_sensitivity": 0.38,
                            "patience_level": 0.67,
                            "analytical_thinking": 0.59,
                            "controlled_perception": 0.66,
                        },
                        "theta_inferred": {
                            "risk_sensitivity": 0.38,
                            "patience_level": 0.67,
                            "analytical_thinking": 0.59,
                            "controlled_perception": 0.66,
                        },
                        "market_volatility_score": 0.45,
                        "life_event": "job_uncertainty",
                    },
                    {
                        "date": "2022-03-01",
                        "month_index": 2,
                        "theta_true": {
                            "risk_sensitivity": 0.70,
                            "patience_level": 0.25,
                            "analytical_thinking": 0.55,
                            "controlled_perception": 0.35,
                        },
                        "theta_inferred": {
                            "risk_sensitivity": 0.70,
                            "patience_level": 0.25,
                            "analytical_thinking": 0.55,
                            "controlled_perception": 0.35,
                        },
                        "market_volatility_score": 0.48,
                        "life_event": "family_expense_shock",
                    },
                    {
                        "date": "2022-04-01",
                        "month_index": 3,
                        "theta_true": {
                            "risk_sensitivity": 0.72,
                            "patience_level": 0.22,
                            "analytical_thinking": 0.58,
                            "controlled_perception": 0.30,
                        },
                        "theta_inferred": {
                            "risk_sensitivity": 0.72,
                            "patience_level": 0.22,
                            "analytical_thinking": 0.58,
                            "controlled_perception": 0.30,
                        },
                        "market_volatility_score": 0.52,
                        "life_event": "none",
                    },
                ],
            },
            {
                "user_id": "sim_user_002",
                "events": [
                    {
                        "date": "2022-01-01",
                        "month_index": 0,
                        "theta_true": {
                            "risk_sensitivity": 0.55,
                            "patience_level": 0.45,
                            "analytical_thinking": 0.50,
                            "controlled_perception": 0.48,
                        },
                        "theta_inferred": {
                            "risk_sensitivity": 0.55,
                            "patience_level": 0.45,
                            "analytical_thinking": 0.50,
                            "controlled_perception": 0.48,
                        },
                        "market_volatility_score": 0.25,
                        "life_event": "none",
                    },
                    {
                        "date": "2022-02-01",
                        "month_index": 1,
                        "theta_true": {
                            "risk_sensitivity": 0.56,
                            "patience_level": 0.43,
                            "analytical_thinking": 0.51,
                            "controlled_perception": 0.46,
                        },
                        "theta_inferred": {
                            "risk_sensitivity": 0.56,
                            "patience_level": 0.43,
                            "analytical_thinking": 0.51,
                            "controlled_perception": 0.46,
                        },
                        "market_volatility_score": 0.38,
                        "life_event": "none",
                    },
                    {
                        "date": "2022-03-01",
                        "month_index": 2,
                        "theta_true": {
                            "risk_sensitivity": 0.60,
                            "patience_level": 0.40,
                            "analytical_thinking": 0.52,
                            "controlled_perception": 0.44,
                        },
                        "theta_inferred": {
                            "risk_sensitivity": 0.60,
                            "patience_level": 0.40,
                            "analytical_thinking": 0.52,
                            "controlled_perception": 0.44,
                        },
                        "market_volatility_score": 0.42,
                        "life_event": "health_stress",
                    },
                    {
                        "date": "2022-04-01",
                        "month_index": 3,
                        "theta_true": {
                            "risk_sensitivity": 0.62,
                            "patience_level": 0.38,
                            "analytical_thinking": 0.52,
                            "controlled_perception": 0.42,
                        },
                        "theta_inferred": {
                            "risk_sensitivity": 0.62,
                            "patience_level": 0.38,
                            "analytical_thinking": 0.52,
                            "controlled_perception": 0.42,
                        },
                        "market_volatility_score": 0.44,
                        "life_event": "none",
                    },
                ],
            },
        ],
    }

    payload = simulate_h3_retention(
        population_payload=population_payload,
        seed=7,
        verbose=False,
    )

    assert payload["population_source"] == "H1 population"
    assert payload["num_investors"] == 2
    assert set(payload["summary"]["kaplan_meier"].keys()) == {
        STRATEGY_TIME_VOL,
        STRATEGY_TIME_VOL_LATENT,
    }
    assert payload["group_summary"][STRATEGY_TIME_VOL]["avg_rebalances"] >= payload["group_summary"][STRATEGY_TIME_VOL_LATENT]["avg_rebalances"]
    assert payload["portfolio_outcomes_summary"][STRATEGY_TIME_VOL]["max_drawdown_pct"] is not None

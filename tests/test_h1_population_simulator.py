import pytest

from simulation.h1_population_simulator import (
    LATENT_KEYS,
    _extract_signals_with_fallback,
    generate_synthetic_users,
    run_population_h1_simulation,
    theta_scalar,
)


def test_generate_synthetic_users_creates_requested_population(tmp_path):
    users = generate_synthetic_users(
        num_users=12,
        seed=5,
        save_path=tmp_path / "users.json",
    )

    assert len(users) == 12
    assert users[0]["user_id"] == "sim_user_001"
    assert set(users[0]["baseline_true_theta"].keys()) == set(LATENT_KEYS)
    assert 0.0 <= users[0]["static_theta_scalar"] <= 1.0


def test_theta_scalar_is_equal_weight_average():
    theta = {
        "risk_sensitivity": 0.2,
        "patience_level": 0.4,
        "analytical_thinking": 0.6,
        "controlled_perception": 0.8,
    }

    assert theta_scalar(theta) == 0.5


def test_run_population_h1_simulation_returns_user_and_dimension_metrics(tmp_path):
    payload = run_population_h1_simulation(
        num_users=8,
        months=6,
        seed=11,
        save_users_path=tmp_path / "synthetic_users.json",
        save_results_path=tmp_path / "population_results.json",
    )

    assert payload["num_users"] == 8
    assert len(payload["results"]) == 8
    assert payload["average_error_over_time"]
    assert payload["dimension_average_error_over_time"]
    assert payload["final_error_cdf"]["num_investors"] == 8

    first = payload["results"][0]
    assert first["events"]
    assert set(first["dimension_error_series"].keys()) == set(LATENT_KEYS)
    assert len(first["scalar_error_series"]) == 6
    assert "static_misalignment" in first
    assert payload["include_inference"] is False
    assert "static_vs_dynamic" not in first
    assert "theta_inferred" not in first["events"][0]


def test_run_population_h1_simulation_can_include_h2_inference_fields(tmp_path):
    payload = run_population_h1_simulation(
        num_users=4,
        months=3,
        seed=17,
        save_users_path=tmp_path / "synthetic_users.json",
        save_results_path=tmp_path / "population_results.json",
        include_inference=True,
    )

    first = payload["results"][0]
    assert payload["include_inference"] is True
    assert "static_vs_dynamic" in first
    assert "theta_inferred" in first["events"][0]
    assert "signal_source_summary" in payload


def test_strict_real_nlp_mode_rejects_non_transformer_backend(monkeypatch):
    def fake_extract_with_metadata(_text):
        class FakeSignals:
            def model_dump(self):
                return {"fear_sentiment": 0.5}

        return FakeSignals(), {"backend_mode": "keyword_heuristics"}

    monkeypatch.setattr(
        "simulation.h1_population_simulator.extract_schema_variables_with_metadata",
        fake_extract_with_metadata,
    )

    with pytest.raises(RuntimeError):
        _extract_signals_with_fallback(
            "chat",
            {
                "risk_sensitivity": 0.5,
                "patience_level": 0.5,
                "analytical_thinking": 0.5,
                "controlled_perception": 0.5,
            },
            require_real_nlp=True,
        )

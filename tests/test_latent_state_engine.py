import pytest

from latent_state_engine.bayesian_update import BayesianLatentEngine, BetaState
from latent_state_engine.mapper import map_signals_to_latent
from latent_state_engine.run_bayesian import run_bayesian
from latent_state_engine.schemas import LatentStateSchema


def _sample_nlp_signals() -> dict[str, float]:
    return {
        "fear_sentiment": 0.8,
        "risk_language_density": 0.6,
        "time_horizon_bias": -0.2,
        "urgency_score": 0.7,
        "analytical_marker": 0.4,
        "herding_marker": 0.3,
        "internal_locus_score": 0.5,
        "external_locus_score": 0.6,
        "uncertainty_score": 0.2,
    }


def test_map_signals_to_latent_expected_values():
    latent = map_signals_to_latent(_sample_nlp_signals())

    assert isinstance(latent, LatentStateSchema)
    assert latent.risk_sensitivity == pytest.approx(0.6537)
    assert latent.patience_level == pytest.approx(0.3137)
    assert latent.analytical_thinking == pytest.approx(0.4384)
    assert latent.controlled_perception == pytest.approx(0.4913)


def test_beta_state_mean_and_variance():
    state = BetaState(alpha=2.0, beta=3.0)

    assert state.mean() == pytest.approx(0.4)
    assert state.variance() == pytest.approx(0.04)


def test_bayesian_update_single_latent():
    engine = BayesianLatentEngine()
    engine.update("risk_sensitivity", evidence=0.8, strength=0.5)

    means = engine.get_means()
    assert means["risk_sensitivity"] == pytest.approx(1.4 / 2.5)


def test_bayesian_update_batch_with_dict():
    engine = BayesianLatentEngine()
    evidence = {
        "risk_sensitivity": 0.9,
        "patience_level": 0.2,
        "analytical_thinking": 0.7,
        "controlled_perception": 0.4,
    }

    engine.update_batch(evidence, strength=1.0)
    means = engine.get_means()

    assert means["risk_sensitivity"] == pytest.approx((1.0 + 0.9) / 3.0)
    assert means["patience_level"] == pytest.approx((1.0 + 0.2) / 3.0)
    assert means["analytical_thinking"] == pytest.approx((1.0 + 0.7) / 3.0)
    assert means["controlled_perception"] == pytest.approx((1.0 + 0.4) / 3.0)


def test_apply_decay_reduces_accumulated_evidence():
    engine = BayesianLatentEngine()
    engine.update("risk_sensitivity", evidence=1.0, strength=2.0)
    before = engine.states["risk_sensitivity"]

    before_alpha = before.alpha
    before_beta = before.beta

    engine.apply_decay(decay_factor=0.5)
    after = engine.states["risk_sensitivity"]

    assert after.alpha == pytest.approx(before_alpha * 0.5)
    assert after.beta == pytest.approx(before_beta * 0.5)


def test_bayesian_update_batch_accepts_dimension_specific_strength():
    engine = BayesianLatentEngine()
    evidence = {
        "risk_sensitivity": 0.6,
        "patience_level": 0.6,
        "analytical_thinking": 0.6,
        "controlled_perception": 0.6,
    }

    engine.update_batch(
        evidence,
        strength={
            "risk_sensitivity": 1.0,
            "patience_level": 1.0,
            "analytical_thinking": 3.0,
            "controlled_perception": 2.0,
        },
    )
    means = engine.get_means()

    assert means["analytical_thinking"] > means["risk_sensitivity"]
    assert means["controlled_perception"] > means["risk_sensitivity"]


def test_apply_decay_accepts_dimension_specific_values():
    engine = BayesianLatentEngine()
    engine.update_batch(
        {
            "risk_sensitivity": 1.0,
            "patience_level": 1.0,
            "analytical_thinking": 1.0,
            "controlled_perception": 1.0,
        },
        strength=2.0,
    )

    before_risk = engine.states["risk_sensitivity"].alpha
    before_analytical = engine.states["analytical_thinking"].alpha
    engine.apply_decay(
        {
            "risk_sensitivity": 0.9,
            "patience_level": 0.9,
            "analytical_thinking": 0.7,
            "controlled_perception": 0.8,
        }
    )

    assert engine.states["risk_sensitivity"].alpha == pytest.approx(before_risk * 0.9)
    assert engine.states["analytical_thinking"].alpha == pytest.approx(before_analytical * 0.7)


def test_run_bayesian_returns_means_and_uncertainty():
    output = run_bayesian(_sample_nlp_signals(), include_uncertainty=True)

    assert set(output.keys()) == {"means", "variances"}
    assert set(output["means"].keys()) == {
        "risk_sensitivity",
        "patience_level",
        "analytical_thinking",
        "controlled_perception",
    }
    assert set(output["variances"].keys()) == {
        "risk_sensitivity",
        "patience_level",
        "analytical_thinking",
        "controlled_perception",
    }

"""
Behavior pipeline tests for portfolio integration.

What this test file proves:

- rebalance_portfolio_with_behavior() calls the expected steps
- if theta says "don't rebalance", it returns `no_rebalance`
- if theta says "rebalance", it passes a config into `rebalance_portfolio()`
- one test keeps the real latent-state and theta-to-config path and checks the
  config handed into the portfolio layer
"""

from integration.theta_adapter import select_config_from_theta
from latent_state_engine import run_bayesian
from portfolio_optimizer.portfolio_api import rebalance_portfolio_with_behavior


def test_rebalance_portfolio_with_behavior_skips_when_signal_is_not_strong(monkeypatch):
    def fake_extract_schema_variables(_user_text: str):
        return {"fear_sentiment": 0.1}

    def fake_run_bayesian(_signals):
        return {
            "risk_sensitivity": 0.2,
            "patience_level": 0.8,
            "analytical_thinking": 0.6,
            "controlled_perception": 0.8,
        }

    monkeypatch.setattr(
        "portfolio_optimizer.portfolio_api.extract_schema_variables",
        fake_extract_schema_variables,
    )
    monkeypatch.setattr(
        "portfolio_optimizer.portfolio_api.run_bayesian",
        fake_run_bayesian,
    )

    response = rebalance_portfolio_with_behavior(
        email="test@test.com",
        user_text="I am fine, no major change",
    )

    assert response["status"] == "no_rebalance"
    assert response["reason"] == "behavioral change not significant"
    assert response["theta"]["risk_sensitivity"] == 0.2


def test_rebalance_portfolio_with_behavior_triggers_rebalance(monkeypatch):
    captured = {}

    def fake_extract_schema_variables(_user_text: str):
        return {"fear_sentiment": 0.95}

    def fake_run_bayesian(_signals):
        return {
            "risk_sensitivity": 0.9,
            "patience_level": 0.2,
            "analytical_thinking": 0.4,
            "controlled_perception": 0.2,
        }

    def fake_select_config_from_theta(theta):
        captured["theta"] = theta
        return {"profile": "conservative"}

    def fake_rebalance_portfolio(**kwargs):
        captured["kwargs"] = kwargs
        return {
            "status": "success",
            "event_type": "rebalance",
            "portfolio": {"portfolio_id": kwargs["portfolio_id"]},
        }

    monkeypatch.setattr(
        "portfolio_optimizer.portfolio_api.extract_schema_variables",
        fake_extract_schema_variables,
    )
    monkeypatch.setattr(
        "portfolio_optimizer.portfolio_api.run_bayesian",
        fake_run_bayesian,
    )
    monkeypatch.setattr(
        "portfolio_optimizer.portfolio_api.select_config_from_theta",
        fake_select_config_from_theta,
    )
    monkeypatch.setattr(
        "portfolio_optimizer.portfolio_api.rebalance_portfolio",
        fake_rebalance_portfolio,
    )

    response = rebalance_portfolio_with_behavior(
        email="test@test.com",
        user_text="I'm panicking, markets are crashing badly",
        portfolio_id="portfolio-123",
        dry_run=True,
    )

    assert response["status"] == "success"
    assert captured["theta"]["risk_sensitivity"] == 0.9
    assert captured["kwargs"]["config"]["profile"] == "conservative"
    assert captured["kwargs"]["email"] == "test@test.com"
    assert captured["kwargs"]["portfolio_id"] == "portfolio-123"
    assert captured["kwargs"]["nlp_input"] == "I'm panicking, markets are crashing badly"
    assert captured["kwargs"]["dry_run"] is True


def test_real_latent_state_output_flows_into_portfolio_config(monkeypatch):
    captured = {}

    real_signals = {
        "fear_sentiment": 0.9,
        "risk_language_density": 0.8,
        "time_horizon_bias": -1.0,
        "urgency_score": 0.9,
        "analytical_marker": 0.2,
        "herding_marker": 0.7,
        "internal_locus_score": 0.2,
        "external_locus_score": 0.8,
        "uncertainty_score": 0.9,
    }

    expected_theta = run_bayesian(real_signals)
    expected_config = select_config_from_theta(expected_theta)

    def fake_extract_schema_variables(_user_text: str):
        return real_signals

    def fake_rebalance_portfolio(**kwargs):
        captured["kwargs"] = kwargs
        return {
            "status": "success",
            "event_type": "rebalance",
            "portfolio": {"portfolio_id": kwargs["portfolio_id"]},
        }

    monkeypatch.setattr(
        "portfolio_optimizer.portfolio_api.extract_schema_variables",
        fake_extract_schema_variables,
    )
    # Force the portfolio handoff path so this test focuses on real theta -> config flow.
    monkeypatch.setattr(
        "portfolio_optimizer.portfolio_api.should_trigger_rebalance",
        lambda theta: True,
    )
    monkeypatch.setattr(
        "portfolio_optimizer.portfolio_api.rebalance_portfolio",
        fake_rebalance_portfolio,
    )

    response = rebalance_portfolio_with_behavior(
        email="integration@test.com",
        user_text="panic text for integration path",
        portfolio_id="portfolio-integration",
        dry_run=True,
    )

    assert response["status"] == "success"
    assert captured["kwargs"]["email"] == "integration@test.com"
    assert captured["kwargs"]["portfolio_id"] == "portfolio-integration"
    assert captured["kwargs"]["nlp_input"] == "panic text for integration path"
    assert captured["kwargs"]["dry_run"] is True
    assert captured["kwargs"]["config"] == expected_config

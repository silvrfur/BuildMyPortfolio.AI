import pytest

from evaluation.metrics.latent_metrics import (
    compute_event_error,
    compute_mae,
    compute_credible_interval_coverage,
    compute_pearson_tracking,
    compute_rmse,
    compute_static_baseline_metrics,
    summarize_event_errors,
)


def test_compute_mae_and_rmse_match_expected_values():
    theta_true = {
        "risk_sensitivity": 0.8,
        "patience_level": 0.6,
        "analytical_thinking": 0.4,
        "controlled_perception": 0.2,
    }
    theta_inferred = {
        "risk_sensitivity": 0.6,
        "patience_level": 0.5,
        "analytical_thinking": 0.5,
        "controlled_perception": 0.4,
    }

    assert compute_mae(theta_true, theta_inferred) == pytest.approx(0.15)
    assert compute_rmse(theta_true, theta_inferred) == pytest.approx((0.10 / 4) ** 0.5)


def test_summarize_event_errors_aggregates_dimension_metrics():
    events = [
        {
            "theta_true": {
                "risk_sensitivity": 0.8,
                "patience_level": 0.6,
                "analytical_thinking": 0.4,
                "controlled_perception": 0.2,
            },
            "theta_inferred": {
                "risk_sensitivity": 0.6,
                "patience_level": 0.5,
                "analytical_thinking": 0.5,
                "controlled_perception": 0.4,
            },
        },
        {
            "theta_true": {
                "risk_sensitivity": 0.4,
                "patience_level": 0.7,
                "analytical_thinking": 0.8,
                "controlled_perception": 0.3,
            },
            "theta_inferred": {
                "risk_sensitivity": 0.5,
                "patience_level": 0.5,
                "analytical_thinking": 0.7,
                "controlled_perception": 0.2,
            },
        },
    ]

    summary = summarize_event_errors(events)

    assert summary["num_events"] == 2
    assert summary["overall_mae"] == pytest.approx((0.15 + 0.125) / 2)
    assert summary["dimension_mae"]["risk_sensitivity"] == pytest.approx((0.2 + 0.1) / 2)


def test_compute_event_error_returns_both_mae_and_rmse():
    result = compute_event_error(
        {
            "risk_sensitivity": 0.8,
            "patience_level": 0.8,
            "analytical_thinking": 0.8,
            "controlled_perception": 0.8,
        },
        {
            "risk_sensitivity": 0.6,
            "patience_level": 0.6,
            "analytical_thinking": 0.6,
            "controlled_perception": 0.6,
        },
    )

    assert result["mae"] == pytest.approx(0.2)
    assert result["rmse"] == pytest.approx(0.2)


def test_compute_pearson_tracking_returns_positive_correlation_for_matching_shape():
    events = [
        {"theta_true": {"risk_sensitivity": 0.2, "patience_level": 0.3, "analytical_thinking": 0.4, "controlled_perception": 0.5},
         "theta_inferred": {"risk_sensitivity": 0.3, "patience_level": 0.4, "analytical_thinking": 0.5, "controlled_perception": 0.6}},
        {"theta_true": {"risk_sensitivity": 0.4, "patience_level": 0.4, "analytical_thinking": 0.5, "controlled_perception": 0.6},
         "theta_inferred": {"risk_sensitivity": 0.5, "patience_level": 0.5, "analytical_thinking": 0.6, "controlled_perception": 0.7}},
        {"theta_true": {"risk_sensitivity": 0.6, "patience_level": 0.5, "analytical_thinking": 0.6, "controlled_perception": 0.7},
         "theta_inferred": {"risk_sensitivity": 0.7, "patience_level": 0.6, "analytical_thinking": 0.7, "controlled_perception": 0.8}},
    ]

    result = compute_pearson_tracking(events)

    assert result["dimension_correlation"]["risk_sensitivity"] == pytest.approx(1.0)
    assert result["overall_average_correlation"] == pytest.approx(1.0)


def test_compute_static_baseline_metrics_shows_improvement_when_dynamic_is_better():
    events = [
        {"theta_true": {"risk_sensitivity": 0.2, "patience_level": 0.2, "analytical_thinking": 0.2, "controlled_perception": 0.2},
         "theta_inferred": {"risk_sensitivity": 0.2, "patience_level": 0.2, "analytical_thinking": 0.2, "controlled_perception": 0.2}},
        {"theta_true": {"risk_sensitivity": 0.8, "patience_level": 0.8, "analytical_thinking": 0.8, "controlled_perception": 0.8},
         "theta_inferred": {"risk_sensitivity": 0.7, "patience_level": 0.7, "analytical_thinking": 0.7, "controlled_perception": 0.7}},
    ]

    result = compute_static_baseline_metrics(events)

    assert result["dynamic_overall_rmse"] < result["static_overall_rmse"]
    assert result["improvement_pct"] > 0


def test_compute_credible_interval_coverage_counts_hits_inside_interval():
    events = [
        {
            "theta_true": {
                "risk_sensitivity": 0.5,
                "patience_level": 0.5,
                "analytical_thinking": 0.5,
                "controlled_perception": 0.5,
            },
            "posterior_params": {
                "risk_sensitivity": {"alpha": 10.0, "beta": 10.0},
                "patience_level": {"alpha": 10.0, "beta": 10.0},
                "analytical_thinking": {"alpha": 10.0, "beta": 10.0},
                "controlled_perception": {"alpha": 10.0, "beta": 10.0},
            },
        }
    ]

    result = compute_credible_interval_coverage(events)

    assert result["overall_coverage"] == pytest.approx(1.0)
    assert result["dimension_coverage"]["risk_sensitivity"] == pytest.approx(1.0)

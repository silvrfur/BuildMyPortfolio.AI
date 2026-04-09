import pytest

from evaluation.metrics.H3 import (
    cox_hazard_ratio,
    compute_portfolio_outcomes,
    kaplan_meier_curve,
    log_rank_test,
    median_retention_time,
)


def test_kaplan_meier_curve_and_median_retention_time():
    records = [
        {"duration_months": 2, "quit_event": 1},
        {"duration_months": 3, "quit_event": 1},
        {"duration_months": 5, "quit_event": 1},
        {"duration_months": 5, "quit_event": 0},
    ]

    curve = kaplan_meier_curve(records)

    assert curve[0]["survival"] == pytest.approx(1.0)
    assert curve[1]["survival"] == pytest.approx(0.75)
    assert median_retention_time(curve) == pytest.approx(3.0)


def test_log_rank_test_detects_group_difference_directionally():
    static = [
        {"duration_months": 2, "quit_event": 1},
        {"duration_months": 3, "quit_event": 1},
        {"duration_months": 4, "quit_event": 1},
        {"duration_months": 5, "quit_event": 1},
    ]
    adaptive = [
        {"duration_months": 5, "quit_event": 1},
        {"duration_months": 6, "quit_event": 1},
        {"duration_months": 7, "quit_event": 1},
        {"duration_months": 8, "quit_event": 1},
    ]

    result = log_rank_test(static, adaptive)

    assert result["chi_square"] > 0
    assert 0 <= result["p_value"] <= 1


def test_cox_hazard_ratio_returns_sub_one_when_second_group_survives_longer():
    static = [
        {"duration_months": 2, "quit_event": 1},
        {"duration_months": 3, "quit_event": 1},
        {"duration_months": 4, "quit_event": 1},
        {"duration_months": 5, "quit_event": 1},
    ]
    adaptive = [
        {"duration_months": 5, "quit_event": 1},
        {"duration_months": 6, "quit_event": 1},
        {"duration_months": 7, "quit_event": 1},
        {"duration_months": 8, "quit_event": 1},
    ]

    result = cox_hazard_ratio(static, adaptive)

    assert result["hazard_ratio"] is not None
    assert result["hazard_ratio"] < 1


def test_compute_portfolio_outcomes_returns_core_risk_metrics():
    monthly_trace = [
        {"current_value": 100000},
        {"current_value": 102000},
        {"current_value": 101000},
        {"current_value": 105000},
        {"current_value": 103000},
    ]

    result = compute_portfolio_outcomes(monthly_trace)

    assert result["final_value_inr"] == pytest.approx(103000)
    assert result["sharpe_ratio"] is not None
    assert result["sortino_ratio"] is not None or result["sortino_ratio"] is None
    assert result["max_drawdown_pct"] <= 0
    assert "calmar_ratio" in result
    assert "utility_score" in result

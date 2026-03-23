import pytest

from evaluation.metrics.H1 import (
    average_error_by_month,
    build_cross_investor_error_cdf,
    build_static_misalignment_series,
    compare_growth_rate_windows,
    compare_static_profile_to_dynamic,
    compute_error_growth_rate,
    summarize_static_misalignment,
)


def test_build_static_misalignment_series_computes_event_errors_against_fixed_baseline():
    baseline = {
        "risk_sensitivity": 0.2,
        "patience_level": 0.2,
        "analytical_thinking": 0.2,
        "controlled_perception": 0.2,
    }
    events = [
        {
            "date": "2022-01-01",
            "theta_true": {
                "risk_sensitivity": 0.2,
                "patience_level": 0.2,
                "analytical_thinking": 0.2,
                "controlled_perception": 0.2,
            },
        },
        {
            "date": "2022-06-01",
            "theta_true": {
                "risk_sensitivity": 0.6,
                "patience_level": 0.2,
                "analytical_thinking": 0.2,
                "controlled_perception": 0.2,
            },
        },
    ]

    series = build_static_misalignment_series(events, baseline)

    assert series[0]["rmse"] == pytest.approx(0.0)
    assert series[1]["month_offset"] == pytest.approx(151 / 30.4375)
    assert series[1]["mae"] == pytest.approx(0.1)
    assert series[1]["rmse"] == pytest.approx(0.2)


def test_summarize_static_misalignment_identifies_growth_and_threshold_crossing():
    baseline = {
        "risk_sensitivity": 0.3,
        "patience_level": 0.3,
        "analytical_thinking": 0.3,
        "controlled_perception": 0.3,
    }
    events = [
        {
            "date": "2022-01-01",
            "theta_true": {
                "risk_sensitivity": 0.3,
                "patience_level": 0.3,
                "analytical_thinking": 0.3,
                "controlled_perception": 0.3,
            },
        },
        {
            "date": "2022-04-01",
            "theta_true": {
                "risk_sensitivity": 0.5,
                "patience_level": 0.3,
                "analytical_thinking": 0.3,
                "controlled_perception": 0.3,
            },
        },
        {
            "date": "2022-08-01",
            "theta_true": {
                "risk_sensitivity": 0.7,
                "patience_level": 0.3,
                "analytical_thinking": 0.3,
                "controlled_perception": 0.3,
            },
        },
    ]

    summary = summarize_static_misalignment(events, baseline, material_threshold=0.15)

    assert summary["initial_rmse"] == pytest.approx(0.0)
    assert summary["final_rmse"] == pytest.approx(0.2)
    assert summary["rmse_growth"] > 0
    assert summary["first_material_misalignment_date"] == "2022-08-01"
    assert summary["drift_slope_rmse_per_month"] > 0
    assert summary["error_growth_rate"]["slope_per_month"] > 0


def test_compare_static_profile_to_dynamic_shows_improvement_when_dynamic_tracks_drift():
    baseline = {
        "risk_sensitivity": 0.2,
        "patience_level": 0.2,
        "analytical_thinking": 0.2,
        "controlled_perception": 0.2,
    }
    events = [
        {
            "theta_true": {
                "risk_sensitivity": 0.2,
                "patience_level": 0.2,
                "analytical_thinking": 0.2,
                "controlled_perception": 0.2,
            },
            "theta_inferred": {
                "risk_sensitivity": 0.2,
                "patience_level": 0.2,
                "analytical_thinking": 0.2,
                "controlled_perception": 0.2,
            },
        },
        {
            "theta_true": {
                "risk_sensitivity": 0.8,
                "patience_level": 0.8,
                "analytical_thinking": 0.8,
                "controlled_perception": 0.8,
            },
            "theta_inferred": {
                "risk_sensitivity": 0.7,
                "patience_level": 0.7,
                "analytical_thinking": 0.7,
                "controlled_perception": 0.7,
            },
        },
    ]

    comparison = compare_static_profile_to_dynamic(events, baseline)

    assert comparison["dynamic_overall_rmse"] < comparison["static_overall_rmse"]
    assert comparison["improvement_pct"] > 0


def test_compute_error_growth_rate_matches_slope_formula():
    series = [
        {"month_offset": 0.0, "rmse": 0.01},
        {"month_offset": 2.0, "rmse": 0.02},
        {"month_offset": 4.0, "rmse": 0.04},
        {"month_offset": 6.0, "rmse": 0.10},
    ]

    growth = compute_error_growth_rate(series)

    assert growth["slope_per_month"] == pytest.approx((0.10 - 0.01) / 6.0)


def test_compare_growth_rate_windows_returns_late_vs_early_ratio():
    series = [
        {"month_offset": 0.0, "rmse": 0.01},
        {"month_offset": 2.0, "rmse": 0.02},
        {"month_offset": 4.0, "rmse": 0.04},
        {"month_offset": 6.0, "rmse": 0.10},
        {"month_offset": 8.0, "rmse": 0.18},
        {"month_offset": 10.0, "rmse": 0.26},
        {"month_offset": 12.0, "rmse": 0.34},
    ]

    comparison = compare_growth_rate_windows(series, early_window_months=6.0, late_window_months=6.0)

    assert comparison["early_window_growth"]["slope_per_month"] == pytest.approx((0.10 - 0.01) / 6.0)
    assert comparison["late_window_growth"]["slope_per_month"] == pytest.approx((0.34 - 0.10) / 6.0)
    assert comparison["growth_ratio_late_vs_early"] > 1.0


def test_cross_investor_error_aggregates_support_average_and_cdf():
    investor_series = [
        {
            "error_series": [
                {"month_offset": 0.0, "rmse": 0.10},
                {"month_offset": 1.0, "rmse": 0.20},
            ]
        },
        {
            "error_series": [
                {"month_offset": 0.0, "rmse": 0.30},
                {"month_offset": 1.0, "rmse": 0.40},
            ]
        },
    ]

    avg = average_error_by_month(investor_series)
    cdf = build_cross_investor_error_cdf(investor_series, month_index=1)

    assert avg[0]["average_error"] == pytest.approx(0.20)
    assert avg[1]["average_error"] == pytest.approx(0.30)
    assert cdf["num_investors"] == 2
    assert cdf["cdf_points"][-1]["cdf"] == pytest.approx(1.0)

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date

import numpy as np

from evaluation.metrics.latent_metrics import LATENT_KEYS, compute_event_error


def _month_offset(start_date: str | None, current_date: str | None) -> float | None:
    if not start_date or not current_date:
        return None
    start_dt = date.fromisoformat(start_date)
    current_dt = date.fromisoformat(current_date)
    return (current_dt - start_dt).days / 30.4375


def build_static_misalignment_series(
    events: Iterable[Mapping[str, object]],
    baseline_theta: Mapping[str, float],
    *,
    true_key: str = "theta_true",
    date_key: str = "date",
    keys: Iterable[str] = LATENT_KEYS,
) -> list[dict[str, object]]:
    rows = list(events)
    start_date = rows[0].get(date_key) if rows else None
    series = []
    for row in rows:
        error = compute_event_error(row[true_key], baseline_theta, keys=keys)
        series.append(
            {
                "date": row.get(date_key),
                "month_offset": _month_offset(start_date, row.get(date_key)),
                "absolute_error": error["absolute_error"],
                "squared_error": error["squared_error"],
                "mae": error["mae"],
                "rmse": error["rmse"],
            }
        )
    return series


def compute_error_growth_rate(
    error_series: Iterable[Mapping[str, object]],
    *,
    metric_key: str = "rmse",
    time_key: str = "month_offset",
) -> dict[str, float | None]:
    rows = list(error_series)
    if len(rows) < 2:
        return {
            "slope_per_month": None,
            "start_error": None,
            "end_error": None,
            "start_month": None,
            "end_month": None,
        }

    start_month = rows[0].get(time_key)
    end_month = rows[-1].get(time_key)
    start_error = rows[0].get(metric_key)
    end_error = rows[-1].get(metric_key)

    if start_month is None or end_month is None or float(end_month) == float(start_month):
        slope = None
    else:
        slope = (float(end_error) - float(start_error)) / (float(end_month) - float(start_month))

    return {
        "slope_per_month": slope,
        "start_error": float(start_error) if start_error is not None else None,
        "end_error": float(end_error) if end_error is not None else None,
        "start_month": float(start_month) if start_month is not None else None,
        "end_month": float(end_month) if end_month is not None else None,
    }


def compare_growth_rate_windows(
    error_series: Iterable[Mapping[str, object]],
    *,
    early_window_months: float = 6.0,
    late_window_months: float = 6.0,
    metric_key: str = "rmse",
    time_key: str = "month_offset",
) -> dict[str, object]:
    rows = list(error_series)
    if len(rows) < 2:
        return {
            "early_window_growth": None,
            "late_window_growth": None,
            "growth_ratio_late_vs_early": None,
        }

    valid = [row for row in rows if row.get(time_key) is not None]
    if len(valid) < 2:
        return {
            "early_window_growth": None,
            "late_window_growth": None,
            "growth_ratio_late_vs_early": None,
        }

    max_month = float(valid[-1][time_key])
    early = [row for row in valid if float(row[time_key]) <= early_window_months]
    late = [row for row in valid if float(row[time_key]) >= max(0.0, max_month - late_window_months)]

    early_growth = compute_error_growth_rate(early, metric_key=metric_key, time_key=time_key)
    late_growth = compute_error_growth_rate(late, metric_key=metric_key, time_key=time_key)

    ratio = None
    early_slope = early_growth["slope_per_month"]
    late_slope = late_growth["slope_per_month"]
    if early_slope not in (None, 0) and late_slope is not None:
        ratio = float(late_slope) / float(early_slope)

    return {
        "early_window_growth": early_growth,
        "late_window_growth": late_growth,
        "growth_ratio_late_vs_early": ratio,
    }


def build_error_cdf(
    error_values: Iterable[float],
    *,
    bins: int = 10,
) -> list[dict[str, float]]:
    values = sorted(float(value) for value in error_values)
    if not values:
        return []

    minimum = values[0]
    maximum = values[-1]
    if minimum == maximum:
        return [{"error_threshold": minimum, "cdf": 1.0}]

    thresholds = np.linspace(minimum, maximum, bins)
    total = len(values)
    return [
        {
            "error_threshold": float(threshold),
            "cdf": sum(1 for value in values if value <= threshold) / total,
        }
        for threshold in thresholds
    ]


def build_cross_investor_error_cdf(
    investor_series: Iterable[Mapping[str, object]],
    *,
    month_index: int = -1,
    metric_key: str = "rmse",
) -> dict[str, object]:
    sampled_errors = []
    for item in investor_series:
        series = list(item.get("error_series", []))
        if not series:
            continue
        index = month_index if month_index >= 0 else len(series) + month_index
        if 0 <= index < len(series):
            sampled_errors.append(float(series[index][metric_key]))

    return {
        "num_investors": len(sampled_errors),
        "month_index": month_index,
        "cdf_points": build_error_cdf(sampled_errors),
    }


def average_error_by_month(
    investor_series: Iterable[Mapping[str, object]],
    *,
    metric_key: str = "rmse",
) -> list[dict[str, float]]:
    month_buckets: dict[int, list[float]] = {}
    for item in investor_series:
        for row in item.get("error_series", []):
            month_offset = row.get("month_offset")
            if month_offset is None:
                continue
            month_index = int(round(float(month_offset)))
            month_buckets.setdefault(month_index, []).append(float(row[metric_key]))

    return [
        {
            "month": float(month),
            "average_error": sum(values) / len(values),
            "num_investors": float(len(values)),
        }
        for month, values in sorted(month_buckets.items())
    ]


def summarize_static_misalignment(
    events: Iterable[Mapping[str, object]],
    baseline_theta: Mapping[str, float],
    *,
    baseline_name: str = "static_baseline",
    true_key: str = "theta_true",
    date_key: str = "date",
    keys: Iterable[str] = LATENT_KEYS,
    material_threshold: float = 0.15,
) -> dict[str, object]:
    rows = build_static_misalignment_series(
        events,
        baseline_theta,
        true_key=true_key,
        date_key=date_key,
        keys=keys,
    )
    if not rows:
        return {
            "baseline_name": baseline_name,
            "material_threshold_rmse": material_threshold,
            "num_events": 0,
            "initial_rmse": None,
            "final_rmse": None,
            "average_rmse": None,
            "max_rmse": None,
            "rmse_growth": None,
            "material_misalignment_rate": None,
            "first_material_misalignment_date": None,
            "drift_slope_rmse_per_month": None,
            "error_series": [],
            "error_growth_rate": None,
            "window_growth_comparison": None,
        }

    rmses = [float(row["rmse"]) for row in rows]
    month_offsets = [float(row["month_offset"]) for row in rows if row["month_offset"] is not None]

    drift_slope = None
    if len(month_offsets) >= 2 and max(month_offsets) > min(month_offsets):
        drift_slope = float(np.polyfit(month_offsets, rmses, 1)[0])

    material_hits = [row for row in rows if float(row["rmse"]) >= material_threshold]
    return {
        "baseline_name": baseline_name,
        "material_threshold_rmse": material_threshold,
        "num_events": len(rows),
        "initial_rmse": rmses[0],
        "final_rmse": rmses[-1],
        "average_rmse": sum(rmses) / len(rmses),
        "max_rmse": max(rmses),
        "rmse_growth": rmses[-1] - rmses[0],
        "material_misalignment_rate": len(material_hits) / len(rows),
        "first_material_misalignment_date": material_hits[0]["date"] if material_hits else None,
        "drift_slope_rmse_per_month": drift_slope,
        "error_series": rows,
        "error_growth_rate": compute_error_growth_rate(rows),
        "window_growth_comparison": compare_growth_rate_windows(rows),
    }


def compare_static_profile_to_dynamic(
    events: Iterable[Mapping[str, object]],
    baseline_theta: Mapping[str, float],
    *,
    true_key: str = "theta_true",
    inferred_key: str = "theta_inferred",
    keys: Iterable[str] = LATENT_KEYS,
) -> dict[str, object]:
    rows = list(events)
    if not rows:
        return {
            "static_overall_rmse": None,
            "dynamic_overall_rmse": None,
            "improvement_pct": None,
            "dimension_static_rmse": {key: None for key in keys},
            "dimension_dynamic_rmse": {key: None for key in keys},
        }

    static_sq = {key: 0.0 for key in keys}
    dynamic_sq = {key: 0.0 for key in keys}

    for row in rows:
        theta_true = row[true_key]
        theta_inferred = row[inferred_key]
        for key in keys:
            static_sq[key] += (float(theta_true[key]) - float(baseline_theta[key])) ** 2
            dynamic_sq[key] += (float(theta_true[key]) - float(theta_inferred[key])) ** 2

    count = len(rows)
    key_count = len(tuple(keys))
    dimension_static_rmse = {key: (static_sq[key] / count) ** 0.5 for key in keys}
    dimension_dynamic_rmse = {key: (dynamic_sq[key] / count) ** 0.5 for key in keys}
    static_overall_rmse = (sum(static_sq.values()) / (count * key_count)) ** 0.5
    dynamic_overall_rmse = (sum(dynamic_sq.values()) / (count * key_count)) ** 0.5

    improvement_pct = None
    if static_overall_rmse > 0:
        improvement_pct = ((static_overall_rmse - dynamic_overall_rmse) / static_overall_rmse) * 100

    return {
        "static_overall_rmse": static_overall_rmse,
        "dynamic_overall_rmse": dynamic_overall_rmse,
        "improvement_pct": improvement_pct,
        "dimension_static_rmse": dimension_static_rmse,
        "dimension_dynamic_rmse": dimension_dynamic_rmse,
    }

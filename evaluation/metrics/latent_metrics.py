from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import sqrt

import numpy as np
from scipy.stats import beta, pearsonr


LATENT_KEYS = (
    "risk_sensitivity",
    "patience_level",
    "analytical_thinking",
    "controlled_perception",
)


def compute_mae(
    theta_true: Mapping[str, float],
    theta_inferred: Mapping[str, float],
    *,
    keys: Iterable[str] = LATENT_KEYS,
) -> float:
    active_keys = tuple(keys)
    return sum(abs(float(theta_true[key]) - float(theta_inferred[key])) for key in active_keys) / len(active_keys)


def compute_rmse(
    theta_true: Mapping[str, float],
    theta_inferred: Mapping[str, float],
    *,
    keys: Iterable[str] = LATENT_KEYS,
) -> float:
    active_keys = tuple(keys)
    mse = sum((float(theta_true[key]) - float(theta_inferred[key])) ** 2 for key in active_keys) / len(active_keys)
    return sqrt(mse)


def compute_event_error(
    theta_true: Mapping[str, float],
    theta_inferred: Mapping[str, float],
    *,
    keys: Iterable[str] = LATENT_KEYS,
) -> dict[str, float]:
    active_keys = tuple(keys)
    absolute_error = sum(abs(float(theta_true[key]) - float(theta_inferred[key])) for key in active_keys)
    squared_error = sum((float(theta_true[key]) - float(theta_inferred[key])) ** 2 for key in active_keys)
    return {
        "absolute_error": absolute_error,
        "squared_error": squared_error,
        "mae": absolute_error / len(active_keys),
        "rmse": sqrt(squared_error / len(active_keys)),
    }


def summarize_event_errors(
    events: Iterable[Mapping[str, object]],
    *,
    true_key: str = "theta_true",
    inferred_key: str = "theta_inferred",
    keys: Iterable[str] = LATENT_KEYS,
) -> dict[str, object]:
    rows = list(events)
    active_keys = tuple(keys)
    if not rows:
        return {
            "num_events": 0,
            "overall_mae": None,
            "overall_rmse": None,
            "dimension_mae": {key: None for key in active_keys},
            "dimension_rmse": {key: None for key in active_keys},
        }

    event_errors = [compute_event_error(row[true_key], row[inferred_key], keys=active_keys) for row in rows]
    dimension_abs = {key: [] for key in active_keys}
    dimension_sq = {key: [] for key in active_keys}

    for row in rows:
        theta_true = row[true_key]
        theta_inferred = row[inferred_key]
        for key in active_keys:
            diff = float(theta_true[key]) - float(theta_inferred[key])
            dimension_abs[key].append(abs(diff))
            dimension_sq[key].append(diff**2)

    return {
        "num_events": len(rows),
        "overall_mae": sum(error["mae"] for error in event_errors) / len(event_errors),
        "overall_rmse": sqrt(sum(error["squared_error"] for error in event_errors) / (len(rows) * len(active_keys))),
        "dimension_mae": {key: sum(values) / len(values) for key, values in dimension_abs.items()},
        "dimension_rmse": {key: sqrt(sum(values) / len(values)) for key, values in dimension_sq.items()},
    }


def compute_pearson_tracking(
    events: Iterable[Mapping[str, object]],
    *,
    true_key: str = "theta_true",
    inferred_key: str = "theta_inferred",
    keys: Iterable[str] = LATENT_KEYS,
) -> dict[str, object]:
    rows = list(events)
    active_keys = tuple(keys)
    correlations: dict[str, float | None] = {}

    for key in active_keys:
        true_values = [float(row[true_key][key]) for row in rows]
        inferred_values = [float(row[inferred_key][key]) for row in rows]
        if len(true_values) < 2 or len(set(true_values)) == 1 or len(set(inferred_values)) == 1:
            correlations[key] = None
            continue
        correlations[key] = float(pearsonr(true_values, inferred_values).statistic)

    valid = [value for value in correlations.values() if value is not None]
    return {
        "dimension_correlation": correlations,
        "overall_average_correlation": (sum(valid) / len(valid)) if valid else None,
    }


def compute_grouped_pearson_tracking(
    events: Iterable[Mapping[str, object]],
    *,
    group_key: str,
    true_key: str = "theta_true",
    inferred_key: str = "theta_inferred",
    keys: Iterable[str] = LATENT_KEYS,
) -> dict[str, object]:
    rows = list(events)
    active_keys = tuple(keys)
    grouped_rows: dict[object, list[Mapping[str, object]]] = {}

    for row in rows:
        group_value = row.get(group_key)
        if group_value is None:
            continue
        grouped_rows.setdefault(group_value, []).append(row)

    group_correlations: dict[str, dict[str, float | None]] = {}
    dimension_samples = {key: [] for key in active_keys}

    for group_value, items in grouped_rows.items():
        correlations: dict[str, float | None] = {}
        for key in active_keys:
            true_values = [float(item[true_key][key]) for item in items]
            inferred_values = [float(item[inferred_key][key]) for item in items]
            if len(true_values) < 2 or len(set(true_values)) == 1 or len(set(inferred_values)) == 1:
                correlations[key] = None
                continue
            corr = float(pearsonr(true_values, inferred_values).statistic)
            correlations[key] = corr
            dimension_samples[key].append(corr)
        group_correlations[str(group_value)] = correlations

    dimension_correlation = {
        key: (sum(values) / len(values)) if values else None
        for key, values in dimension_samples.items()
    }
    valid = [value for value in dimension_correlation.values() if value is not None]
    return {
        "group_key": group_key,
        "num_groups": len(group_correlations),
        "group_correlations": group_correlations,
        "dimension_correlation": dimension_correlation,
        "overall_average_correlation": (sum(valid) / len(valid)) if valid else None,
    }


def compute_credible_interval_coverage(
    events: Iterable[Mapping[str, object]],
    *,
    true_key: str = "theta_true",
    params_key: str = "posterior_params",
    keys: Iterable[str] = LATENT_KEYS,
    alpha: float = 0.05,
) -> dict[str, object]:
    rows = list(events)
    active_keys = tuple(keys)
    hits = {key: 0 for key in active_keys}
    totals = {key: 0 for key in active_keys}

    for row in rows:
        theta_true = row.get(true_key, {})
        posterior_params = row.get(params_key, {})
        for key in active_keys:
            params = posterior_params.get(key)
            if not params:
                continue
            lower = float(beta.ppf(alpha / 2, float(params["alpha"]), float(params["beta"])))
            upper = float(beta.ppf(1 - alpha / 2, float(params["alpha"]), float(params["beta"])))
            value = float(theta_true[key])
            totals[key] += 1
            if lower <= value <= upper:
                hits[key] += 1

    dimension_coverage = {
        key: (hits[key] / totals[key]) if totals[key] else None
        for key in active_keys
    }
    valid = [value for value in dimension_coverage.values() if value is not None]
    return {
        "dimension_coverage": dimension_coverage,
        "overall_coverage": (sum(valid) / len(valid)) if valid else None,
    }


def compute_static_baseline_metrics(
    events: Iterable[Mapping[str, object]],
    *,
    static_theta: Mapping[str, float] | None = None,
    static_theta_key: str = "theta_static",
    true_key: str = "theta_true",
    inferred_key: str = "theta_inferred",
    keys: Iterable[str] = LATENT_KEYS,
) -> dict[str, object]:
    rows = list(events)
    active_keys = tuple(keys)
    if not rows:
        return {
            "static_overall_rmse": None,
            "dynamic_overall_rmse": None,
            "improvement_pct": None,
            "dimension_static_rmse": {key: None for key in active_keys},
            "dimension_dynamic_rmse": {key: None for key in active_keys},
        }

    static_sq = {key: [] for key in active_keys}
    dynamic_sq = {key: [] for key in active_keys}

    for row in rows:
        baseline = static_theta or row.get(static_theta_key) or rows[0][true_key]
        for key in active_keys:
            static_sq[key].append((float(row[true_key][key]) - float(baseline[key])) ** 2)
            dynamic_sq[key].append((float(row[true_key][key]) - float(row[inferred_key][key])) ** 2)

    dimension_static_rmse = {key: sqrt(sum(values) / len(values)) for key, values in static_sq.items()}
    dimension_dynamic_rmse = {key: sqrt(sum(values) / len(values)) for key, values in dynamic_sq.items()}
    static_overall_rmse = sqrt(sum(sum(values) for values in static_sq.values()) / (len(rows) * len(active_keys)))
    dynamic_overall_rmse = sqrt(sum(sum(values) for values in dynamic_sq.values()) / (len(rows) * len(active_keys)))
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

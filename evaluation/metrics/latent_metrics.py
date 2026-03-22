from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from scipy.stats import beta as beta_dist


LATENT_KEYS = (
    "risk_sensitivity",
    "patience_level",
    "analytical_thinking",
    "controlled_perception",
)


def _vector(theta: Mapping[str, float], keys: Iterable[str] = LATENT_KEYS) -> list[float]:
    return [float(theta[key]) for key in keys]


def _safe_corr(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    if max(x) == min(x) or max(y) == min(y):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compute_absolute_errors(
    theta_true: Mapping[str, float],
    theta_inferred: Mapping[str, float],
    keys: Iterable[str] = LATENT_KEYS,
) -> dict[str, float]:
    return {
        key: abs(float(theta_true[key]) - float(theta_inferred[key]))
        for key in keys
    }


def compute_squared_errors(
    theta_true: Mapping[str, float],
    theta_inferred: Mapping[str, float],
    keys: Iterable[str] = LATENT_KEYS,
) -> dict[str, float]:
    return {
        key: (float(theta_true[key]) - float(theta_inferred[key])) ** 2
        for key in keys
    }


def compute_mae(
    theta_true: Mapping[str, float],
    theta_inferred: Mapping[str, float],
    keys: Iterable[str] = LATENT_KEYS,
) -> float:
    """
    Standard MAE, averaged across latent dimensions.

    MAE = (1 / K) * sum_k |theta_true_k - theta_inferred_k|
    """
    errors = compute_absolute_errors(theta_true, theta_inferred, keys=keys)
    return sum(errors.values()) / len(errors)


def compute_rmse(
    theta_true: Mapping[str, float],
    theta_inferred: Mapping[str, float],
    keys: Iterable[str] = LATENT_KEYS,
) -> float:
    """
    Standard RMSE, averaged across latent dimensions.

    RMSE = sqrt((1 / K) * sum_k (theta_true_k - theta_inferred_k)^2)
    """
    squared = compute_squared_errors(theta_true, theta_inferred, keys=keys)
    return (sum(squared.values()) / len(squared)) ** 0.5


def compute_event_error(
    theta_true: Mapping[str, float],
    theta_inferred: Mapping[str, float],
    keys: Iterable[str] = LATENT_KEYS,
) -> dict[str, object]:
    abs_errors = compute_absolute_errors(theta_true, theta_inferred, keys=keys)
    sq_errors = compute_squared_errors(theta_true, theta_inferred, keys=keys)
    return {
        "absolute_error": abs_errors,
        "squared_error": sq_errors,
        "mae": sum(abs_errors.values()) / len(abs_errors),
        "rmse": (sum(sq_errors.values()) / len(sq_errors)) ** 0.5,
    }


def summarize_event_errors(
    events: Iterable[Mapping[str, object]],
    *,
    true_key: str = "theta_true",
    inferred_key: str = "theta_inferred",
    keys: Iterable[str] = LATENT_KEYS,
) -> dict[str, object]:
    rows = list(events)
    if not rows:
        return {
            "num_events": 0,
            "overall_mae": None,
            "overall_rmse": None,
            "dimension_mae": {key: None for key in keys},
            "dimension_rmse": {key: None for key in keys},
        }

    abs_accumulator = {key: 0.0 for key in keys}
    sq_accumulator = {key: 0.0 for key in keys}
    mae_values = []
    rmse_values = []

    for row in rows:
        result = compute_event_error(
            row[true_key], row[inferred_key], keys=keys
        )
        mae_values.append(float(result["mae"]))
        rmse_values.append(float(result["rmse"]))
        for key, value in result["absolute_error"].items():
            abs_accumulator[key] += float(value)
        for key, value in result["squared_error"].items():
            sq_accumulator[key] += float(value)

    count = len(rows)
    return {
        "num_events": count,
        "overall_mae": sum(mae_values) / count,
        "overall_rmse": sum(rmse_values) / count,
        "dimension_mae": {
            key: abs_accumulator[key] / count
            for key in keys
        },
        "dimension_rmse": {
            key: (sq_accumulator[key] / count) ** 0.5
            for key in keys
        },
    }


def compute_static_baseline_metrics(
    events: Iterable[Mapping[str, object]],
    *,
    true_key: str = "theta_true",
    inferred_key: str = "theta_inferred",
    keys: Iterable[str] = LATENT_KEYS,
) -> dict[str, object]:
    rows = list(events)
    if not rows:
        return {
            "baseline_source": "initial_inferred",
            "static_overall_rmse": None,
            "dynamic_overall_rmse": None,
            "improvement_pct": None,
            "dimension_static_rmse": {key: None for key in keys},
            "dimension_dynamic_rmse": {key: None for key in keys},
        }

    baseline = {
        key: float(rows[0][inferred_key][key])
        for key in keys
    }

    static_sq = {key: 0.0 for key in keys}
    dynamic_sq = {key: 0.0 for key in keys}
    for row in rows:
        theta_true = row[true_key]
        theta_inferred = row[inferred_key]
        for key in keys:
            static_sq[key] += (float(theta_true[key]) - baseline[key]) ** 2
            dynamic_sq[key] += (float(theta_true[key]) - float(theta_inferred[key])) ** 2

    count = len(rows)
    dimension_static_rmse = {key: (static_sq[key] / count) ** 0.5 for key in keys}
    dimension_dynamic_rmse = {key: (dynamic_sq[key] / count) ** 0.5 for key in keys}
    static_overall_rmse = (sum(static_sq.values()) / (count * len(tuple(keys)))) ** 0.5
    dynamic_overall_rmse = (sum(dynamic_sq.values()) / (count * len(tuple(keys)))) ** 0.5
    improvement_pct = None
    if static_overall_rmse > 0:
        improvement_pct = ((static_overall_rmse - dynamic_overall_rmse) / static_overall_rmse) * 100

    return {
        "baseline_source": "initial_inferred",
        "static_theta": baseline,
        "static_overall_rmse": static_overall_rmse,
        "dynamic_overall_rmse": dynamic_overall_rmse,
        "improvement_pct": improvement_pct,
        "dimension_static_rmse": dimension_static_rmse,
        "dimension_dynamic_rmse": dimension_dynamic_rmse,
    }


def compute_pearson_tracking(
    events: Iterable[Mapping[str, object]],
    *,
    true_key: str = "theta_true",
    inferred_key: str = "theta_inferred",
    keys: Iterable[str] = LATENT_KEYS,
) -> dict[str, object]:
    rows = list(events)
    if not rows:
        return {
            "overall_average_correlation": None,
            "dimension_correlation": {key: None for key in keys},
        }

    dimension_correlation = {}
    valid = []
    for key in keys:
        x = [float(row[true_key][key]) for row in rows]
        y = [float(row[inferred_key][key]) for row in rows]
        corr = _safe_corr(x, y)
        dimension_correlation[key] = corr
        if corr is not None:
            valid.append(corr)

    return {
        "overall_average_correlation": (sum(valid) / len(valid)) if valid else None,
        "dimension_correlation": dimension_correlation,
    }


def compute_credible_interval_coverage(
    events: Iterable[Mapping[str, object]],
    *,
    true_key: str = "theta_true",
    params_key: str = "posterior_params",
    keys: Iterable[str] = LATENT_KEYS,
    level: float = 0.90,
) -> dict[str, object]:
    rows = list(events)
    if not rows:
        return {
            "interval_level": level,
            "overall_coverage": None,
            "dimension_coverage": {key: None for key in keys},
        }

    alpha_tail = (1.0 - level) / 2.0
    hits_by_dim = {key: 0 for key in keys}
    total_by_dim = {key: 0 for key in keys}

    for row in rows:
        theta_true = row[true_key]
        posterior_params = row.get(params_key) or {}
        for key in keys:
            params = posterior_params.get(key)
            if not params:
                continue
            lower = float(beta_dist.ppf(alpha_tail, params["alpha"], params["beta"]))
            upper = float(beta_dist.ppf(1.0 - alpha_tail, params["alpha"], params["beta"]))
            total_by_dim[key] += 1
            if lower <= float(theta_true[key]) <= upper:
                hits_by_dim[key] += 1

    dimension_coverage = {
        key: (hits_by_dim[key] / total_by_dim[key]) if total_by_dim[key] else None
        for key in keys
    }
    valid = [value for value in dimension_coverage.values() if value is not None]
    return {
        "interval_level": level,
        "overall_coverage": (sum(valid) / len(valid)) if valid else None,
        "dimension_coverage": dimension_coverage,
    }

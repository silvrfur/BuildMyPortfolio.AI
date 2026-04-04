from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite

import numpy as np
from scipy.stats import chi2
from statsmodels.duration.hazard_regression import PHReg


def kaplan_meier_curve(
    records: Iterable[Mapping[str, object]],
    *,
    duration_key: str = "duration_months",
    event_key: str = "quit_event",
) -> list[dict[str, float]]:
    rows = sorted(
        (
            {
                "duration": int(record[duration_key]),
                "event": int(record[event_key]),
            }
            for record in records
        ),
        key=lambda row: row["duration"],
    )
    if not rows:
        return []

    event_times = sorted({row["duration"] for row in rows if row["event"] == 1})
    curve = [{"time": 0, "at_risk": len(rows), "events": 0, "censored": 0, "survival": 1.0}]
    survival = 1.0

    for time in event_times:
        at_risk = sum(1 for row in rows if row["duration"] >= time)
        events = sum(1 for row in rows if row["duration"] == time and row["event"] == 1)
        censored = sum(1 for row in rows if row["duration"] == time and row["event"] == 0)
        if at_risk > 0:
            survival *= (1.0 - events / at_risk)
        curve.append(
            {
                "time": time,
                "at_risk": at_risk,
                "events": events,
                "censored": censored,
                "survival": survival,
            }
        )
    return curve


def median_retention_time(curve: Iterable[Mapping[str, float]]) -> float | None:
    for point in curve:
        if float(point["survival"]) <= 0.5:
            return float(point["time"])
    return None


def log_rank_test(
    group_a_records: Iterable[Mapping[str, object]],
    group_b_records: Iterable[Mapping[str, object]],
    *,
    duration_key: str = "duration_months",
    event_key: str = "quit_event",
) -> dict[str, float | None]:
    a = [{"duration": int(row[duration_key]), "event": int(row[event_key])} for row in group_a_records]
    b = [{"duration": int(row[duration_key]), "event": int(row[event_key])} for row in group_b_records]
    event_times = sorted({row["duration"] for row in a + b if row["event"] == 1})
    if not event_times:
        return {"chi_square": 0.0, "p_value": 1.0}

    observed_minus_expected = 0.0
    variance = 0.0

    for time in event_times:
        n1 = sum(1 for row in a if row["duration"] >= time)
        n2 = sum(1 for row in b if row["duration"] >= time)
        d1 = sum(1 for row in a if row["duration"] == time and row["event"] == 1)
        d2 = sum(1 for row in b if row["duration"] == time and row["event"] == 1)
        n = n1 + n2
        d = d1 + d2
        if n <= 1 or d == 0:
            continue

        expected_1 = d * (n1 / n)
        observed_minus_expected += d1 - expected_1
        variance += (n1 * n2 * d * (n - d)) / (n**2 * (n - 1))

    if variance <= 0:
        return {"chi_square": 0.0, "p_value": 1.0}

    chi_square = (observed_minus_expected**2) / variance
    p_value = float(chi2.sf(chi_square, df=1))
    return {"chi_square": float(chi_square), "p_value": p_value}


def cox_hazard_ratio(
    group_a_records: Iterable[Mapping[str, object]],
    group_b_records: Iterable[Mapping[str, object]],
    *,
    duration_key: str = "duration_months",
    event_key: str = "quit_event",
) -> dict[str, float | None]:
    rows = []
    for record in group_a_records:
        rows.append((int(record[duration_key]), int(record[event_key]), 0))
    for record in group_b_records:
        rows.append((int(record[duration_key]), int(record[event_key]), 1))

    if len(rows) < 3 or sum(event for _, event, _ in rows) == 0:
        return {"hazard_ratio": None, "p_value": None, "ci_lower": None, "ci_upper": None}

    durations = np.array([row[0] for row in rows], dtype=float)
    status = np.array([row[1] for row in rows], dtype=int)
    exog = np.array([[row[2]] for row in rows], dtype=float)

    try:
        model = PHReg(durations, exog, status=status)
        result = model.fit(disp=0)
        coef = float(result.params[0])
        se = float(result.bse[0])
        hr = float(np.exp(coef))
        ci_lower = float(np.exp(coef - 1.96 * se))
        ci_upper = float(np.exp(coef + 1.96 * se))
        p_value = float(result.pvalues[0])
        if not all(isfinite(value) for value in (hr, ci_lower, ci_upper, p_value)):
            raise ValueError("non-finite Cox result")
        return {
            "hazard_ratio": hr,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    except Exception:
        return {"hazard_ratio": None, "p_value": None, "ci_lower": None, "ci_upper": None}


def summarize_survival_by_group(
    records: Iterable[Mapping[str, object]],
    *,
    group_key: str = "strategy",
    duration_key: str = "duration_months",
    event_key: str = "quit_event",
) -> dict[str, object]:
    rows = list(records)
    groups = sorted({str(row[group_key]) for row in rows})
    curves = {}
    medians = {}

    for group in groups:
        group_rows = [row for row in rows if row[group_key] == group]
        curve = kaplan_meier_curve(group_rows, duration_key=duration_key, event_key=event_key)
        curves[group] = curve
        medians[group] = median_retention_time(curve)

    pairwise = {}
    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            left_rows = [row for row in rows if row[group_key] == left]
            right_rows = [row for row in rows if row[group_key] == right]
            pairwise[f"{left}__vs__{right}"] = {
                "log_rank": log_rank_test(left_rows, right_rows, duration_key=duration_key, event_key=event_key),
                "hazard_ratio": cox_hazard_ratio(left_rows, right_rows, duration_key=duration_key, event_key=event_key),
            }

    return {
        "kaplan_meier": curves,
        "median_retention_months": medians,
        "pairwise_comparisons": pairwise,
    }


def compute_portfolio_outcomes(
    monthly_trace: Iterable[Mapping[str, object]],
    *,
    risk_free_annual: float = 0.06,
    utility_lambda: float = 3.0,
) -> dict[str, float | None]:
    rows = list(monthly_trace)
    if not rows:
        return {
            "final_value_inr": None,
            "total_return_pct": None,
            "cagr_pct": None,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown_pct": None,
            "calmar_ratio": None,
            "utility_score": None,
        }

    values = [float(row["current_value"]) for row in rows]
    initial = values[0]
    final = values[-1]
    total_return_pct = ((final - initial) / initial) * 100 if initial > 0 else None
    years = max(len(values) / 12, 1 / 12)
    cagr_pct = (((final / initial) ** (1 / years)) - 1) * 100 if initial > 0 and final > 0 else None

    monthly_returns = []
    for previous, current in zip(values, values[1:]):
        if previous > 0:
            monthly_returns.append((current - previous) / previous)

    if monthly_returns:
        mean_monthly = float(np.mean(monthly_returns))
        std_monthly = float(np.std(monthly_returns, ddof=1)) if len(monthly_returns) > 1 else 0.0
        annual_return = mean_monthly * 12
        annual_vol = std_monthly * np.sqrt(12)
        sharpe = ((annual_return - risk_free_annual) / annual_vol) if annual_vol > 0 else None

        monthly_rf = risk_free_annual / 12
        downside = [ret for ret in monthly_returns if ret < monthly_rf]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else (
            float(np.std(downside, ddof=0)) if len(downside) == 1 else 0.0
        )
        annual_downside = downside_std * np.sqrt(12)
        sortino = ((annual_return - risk_free_annual) / annual_downside) if annual_downside > 0 else None
        utility_score = annual_return - utility_lambda * annual_vol
    else:
        annual_return = None
        annual_vol = None
        sharpe = None
        sortino = None
        utility_score = None

    peak = values[0]
    max_drawdown = 0.0
    for value in values:
        if value > peak:
            peak = value
        drawdown = (value - peak) / peak if peak > 0 else 0.0
        if drawdown < max_drawdown:
            max_drawdown = drawdown
    max_drawdown_pct = max_drawdown * 100
    calmar = (annual_return / abs(max_drawdown)) if annual_return is not None and max_drawdown < 0 else None

    return {
        "final_value_inr": round(final, 2),
        "total_return_pct": round(total_return_pct, 4) if total_return_pct is not None else None,
        "cagr_pct": round(cagr_pct, 4) if cagr_pct is not None else None,
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "sortino_ratio": round(sortino, 4) if sortino is not None else None,
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "calmar_ratio": round(calmar, 4) if calmar is not None else None,
        "utility_score": round(utility_score, 4) if utility_score is not None else None,
    }


def summarize_portfolio_outcomes_by_group(
    records: Iterable[Mapping[str, object]],
    *,
    group_key: str = "strategy",
    metrics_key: str = "portfolio_outcomes",
) -> dict[str, dict[str, float | None]]:
    rows = list(records)
    groups = sorted({str(row[group_key]) for row in rows})
    metric_names = (
        "final_value_inr",
        "total_return_pct",
        "cagr_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown_pct",
        "calmar_ratio",
        "utility_score",
    )
    summary: dict[str, dict[str, float | None]] = {}
    for group in groups:
        group_rows = [row for row in rows if row[group_key] == group]
        group_summary = {}
        for metric in metric_names:
            values = [
                float(row[metrics_key][metric])
                for row in group_rows
                if row.get(metrics_key) and row[metrics_key].get(metric) is not None
            ]
            group_summary[metric] = round(sum(values) / len(values), 4) if values else None
        summary[group] = group_summary
    return summary

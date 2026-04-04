from __future__ import annotations

import json
import math
import random
import statistics
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from evaluation.metrics import (
    LATENT_KEYS,
    average_error_by_month,
    build_cross_investor_error_cdf,
    summarize_static_misalignment,
)
from nlp.signal_extractor import extract_schema_variables_with_metadata
from simulation.simulator import get_price_history


ROOT_DIR = Path(__file__).resolve().parents[1]
SIMULATION_DIR = ROOT_DIR / "simulation" / "generated_h1_population"
EVALUATION_H1_DIR = ROOT_DIR / "evaluation" / "H1"
PLOTS_NEW_DIR = EVALUATION_H1_DIR / "plots_new"
DEFAULT_USERS_PATH = SIMULATION_DIR / "synthetic_users.json"
DEFAULT_RESULTS_PATH = SIMULATION_DIR / "population_h1_results.json"
DEFAULT_USER_PROFILES_DIR = SIMULATION_DIR / "users"
DEFAULT_EVALUATION_RESULTS_PATH = EVALUATION_H1_DIR / "population_result.json"


@dataclass(frozen=True)
class Archetype:
    name: str
    base_theta: dict[str, float]
    market_sensitivity: float
    life_event_sensitivity: float
    recovery_bias: float
    self_report_noise: float


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        name="reactive_growth_seeker",
        base_theta={
            "risk_sensitivity": 0.68,
            "patience_level": 0.38,
            "analytical_thinking": 0.46,
            "controlled_perception": 0.42,
        },
        market_sensitivity=0.90,
        life_event_sensitivity=0.60,
        recovery_bias=0.10,
        self_report_noise=0.08,
    ),
    Archetype(
        name="balanced_planner",
        base_theta={
            "risk_sensitivity": 0.44,
            "patience_level": 0.62,
            "analytical_thinking": 0.65,
            "controlled_perception": 0.66,
        },
        market_sensitivity=0.45,
        life_event_sensitivity=0.35,
        recovery_bias=0.22,
        self_report_noise=0.05,
    ),
    Archetype(
        name="anxious_capital_preserver",
        base_theta={
            "risk_sensitivity": 0.80,
            "patience_level": 0.36,
            "analytical_thinking": 0.42,
            "controlled_perception": 0.34,
        },
        market_sensitivity=0.82,
        life_event_sensitivity=0.78,
        recovery_bias=0.08,
        self_report_noise=0.06,
    ),
    Archetype(
        name="disciplined_long_horizon",
        base_theta={
            "risk_sensitivity": 0.28,
            "patience_level": 0.84,
            "analytical_thinking": 0.70,
            "controlled_perception": 0.78,
        },
        market_sensitivity=0.25,
        life_event_sensitivity=0.28,
        recovery_bias=0.30,
        self_report_noise=0.04,
    ),
    Archetype(
        name="analytical_but_stressed",
        base_theta={
            "risk_sensitivity": 0.58,
            "patience_level": 0.48,
            "analytical_thinking": 0.80,
            "controlled_perception": 0.46,
        },
        market_sensitivity=0.52,
        life_event_sensitivity=0.55,
        recovery_bias=0.18,
        self_report_noise=0.05,
    ),
    Archetype(
        name="family_goal_investor",
        base_theta={
            "risk_sensitivity": 0.40,
            "patience_level": 0.60,
            "analytical_thinking": 0.55,
            "controlled_perception": 0.58,
        },
        market_sensitivity=0.35,
        life_event_sensitivity=0.72,
        recovery_bias=0.20,
        self_report_noise=0.05,
    ),
)


LIFE_EVENTS: dict[str, dict[str, object]] = {
    "none": {
        "drift": {key: 0.0 for key in LATENT_KEYS},
        "description": "no material personal event",
    },
    "family_expense_shock": {
        "drift": {
            "risk_sensitivity": 0.14,
            "patience_level": -0.10,
            "analytical_thinking": -0.03,
            "controlled_perception": -0.10,
        },
        "description": "unexpected family financial pressure",
    },
    "job_uncertainty": {
        "drift": {
            "risk_sensitivity": 0.12,
            "patience_level": -0.08,
            "analytical_thinking": -0.02,
            "controlled_perception": -0.12,
        },
        "description": "job or income uncertainty",
    },
    "salary_growth": {
        "drift": {
            "risk_sensitivity": -0.05,
            "patience_level": 0.03,
            "analytical_thinking": 0.02,
            "controlled_perception": 0.07,
        },
        "description": "improved income stability",
    },
    "new_dependents": {
        "drift": {
            "risk_sensitivity": 0.08,
            "patience_level": 0.04,
            "analytical_thinking": 0.01,
            "controlled_perception": -0.05,
        },
        "description": "family responsibilities increased",
    },
    "health_stress": {
        "drift": {
            "risk_sensitivity": 0.16,
            "patience_level": -0.12,
            "analytical_thinking": -0.06,
            "controlled_perception": -0.16,
        },
        "description": "medical or health-related stress",
    },
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _add_months(dt: date, months: int) -> date:
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    day = min(dt.day, monthrange(year, month)[1])
    return date(year, month, day)


def _safe_json_dump(value):
    if isinstance(value, dict):
        return {key: _safe_json_dump(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json_dump(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_json_dump(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _save_json(path: Path | str, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(_safe_json_dump(payload), handle, indent=2)


def _ensure_population_output_dirs() -> None:
    SIMULATION_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_H1_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_NEW_DIR.mkdir(parents=True, exist_ok=True)


def theta_scalar(theta: dict[str, float]) -> float:
    return sum(float(theta[key]) for key in LATENT_KEYS) / len(LATENT_KEYS)


def _jitter_theta(base_theta: dict[str, float], rng: random.Random, scale: float = 0.08) -> dict[str, float]:
    return {
        key: round(_clamp(float(value) + rng.uniform(-scale, scale)), 4)
        for key, value in base_theta.items()
    }


def _simulate_self_reported_theta(
    theta_initial_true: dict[str, float],
    *,
    rng: random.Random,
    noise: float,
    midpoint_pull: float = 0.10,
) -> dict[str, float]:
    return {
        key: round(
            _clamp((1.0 - midpoint_pull) * float(value) + midpoint_pull * 0.5 + rng.uniform(-noise, noise)),
            4,
        )
        for key, value in theta_initial_true.items()
    }


def _signals_from_theta(theta: dict[str, float]) -> dict[str, float]:
    risk = float(theta["risk_sensitivity"])
    patience = float(theta["patience_level"])
    analytical = float(theta["analytical_thinking"])
    control = float(theta["controlled_perception"])
    return {
        "fear_sentiment": risk,
        "risk_language_density": risk,
        "time_horizon_bias": _clamp(2.0 * patience - 1.0, lower=-1.0, upper=1.0),
        "urgency_score": 1.0 - patience,
        "analytical_marker": analytical,
        "herding_marker": _clamp((risk + (1.0 - control)) / 2.0),
        "internal_locus_score": control,
        "external_locus_score": 1.0 - control,
        "uncertainty_score": _clamp(0.6 * risk + 0.4 * (1.0 - control)),
    }


def _extract_signals_with_fallback(
    chat_text: str,
    theta_true: dict[str, float],
    *,
    signal_mode: str = "auto",
    require_real_nlp: bool = False,
) -> tuple[dict[str, float], str, dict[str, object]]:
    if signal_mode == "synthetic":
        return _signals_from_theta(theta_true), "synthetic_oracle", {"signal_mode": signal_mode}
    try:
        signals, metadata = extract_schema_variables_with_metadata(chat_text)
        source = str(metadata.get("backend_mode") or "nlp")
        if require_real_nlp and source not in {"local_transformers", "remote_hf_inference"}:
            raise RuntimeError(
                "Strict NLP mode requires local or remote transformer inference, "
                f"but backend_mode={source!r}."
            )
        if hasattr(signals, "model_dump"):
            return signals.model_dump(), source, metadata
        return dict(signals), source, metadata
    except Exception as exc:
        if require_real_nlp:
            raise RuntimeError("Strict NLP mode failed; no fallback allowed.") from exc
        return _signals_from_theta(theta_true), "synthetic_fallback", {"error": str(exc)}


def _calibrate_posterior_params(
    posterior_params: dict[str, dict[str, float]],
    *,
    concentration_scale: float | dict[str, float] = 1.0,
) -> dict[str, dict[str, float]]:
    scaled: dict[str, dict[str, float]] = {}
    for key, params in posterior_params.items():
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        scale = (
            float(concentration_scale.get(key, 0.40))
            if isinstance(concentration_scale, dict)
            else float(concentration_scale)
        )
        scaled[key] = {
            "alpha": 1.0 + max(0.0, alpha - 1.0) * scale,
            "beta": 1.0 + max(0.0, beta - 1.0) * scale,
        }
    return scaled


def _chat_text_for_state(
    theta: dict[str, float],
    *,
    market_context: dict[str, object],
    life_event: str,
) -> str:
    risk = float(theta["risk_sensitivity"])
    patience = float(theta["patience_level"])
    analytical = float(theta["analytical_thinking"])
    control = float(theta["controlled_perception"])

    if risk >= 0.80:
        risk_text = (
            "I am very scared of losses and drawdowns, capital preservation matters most, "
            "and I want strong downside protection right now."
        )
    elif risk >= 0.60:
        risk_text = (
            "I am cautious about risk, I keep thinking about downside, "
            "and I do not want large losses."
        )
    elif risk >= 0.40:
        risk_text = (
            "I can tolerate balanced risk if the setup is sensible, "
            "but I still care about avoiding unnecessary drawdowns."
        )
    else:
        risk_text = (
            "I am comfortable taking risk for upside and I can live with volatility "
            "if long-term returns look attractive."
        )

    if patience <= 0.20:
        patience_text = (
            "I want an immediate move, I am focused on this week, "
            "and I do not want to wait for a long-term thesis."
        )
    elif patience <= 0.45:
        patience_text = (
            "I need flexibility soon and I am thinking in the near term more than in years."
        )
    elif patience <= 0.70:
        patience_text = (
            "I can stay patient for months if the thesis still makes sense."
        )
    else:
        patience_text = (
            "I am comfortable holding for years and letting compounding work over a long horizon."
        )

    if analytical >= 0.75:
        analytical_text = (
            "I am relying on research, earnings, valuation, cash-flow analysis, "
            "probability estimates, scenario analysis, and data before making the decision."
        )
    elif analytical >= 0.55:
        analytical_text = (
            "I am looking at data, structured reasoning, checklists, and evidence, "
            "not just reacting emotionally."
        )
    elif analytical >= 0.35:
        analytical_text = (
            "I am only doing light analysis and partly reacting to headlines, narratives, and market mood."
        )
    else:
        analytical_text = (
            "I am mostly reacting to price action, narratives, and gut feel rather than deep analysis."
        )

    if control <= 0.20:
        control_text = (
            "It feels like the market is deciding for me, events are pushing me around, "
            "and I do not feel in control of my decisions."
        )
    elif control <= 0.45:
        control_text = (
            "I feel somewhat pushed around by events, outside forces, and market noise, "
            "and my conviction is not stable."
        )
    elif control <= 0.70:
        control_text = (
            "I am trying to stay disciplined, follow my own plan, and stick to my process."
        )
    else:
        control_text = (
            "I trust my plan, I feel fully in control, and I want to follow my own strategy with discipline."
        )

    crowd_text = (
        "I keep noticing what everyone else is doing and it affects my conviction."
        if risk >= 0.60 and control <= 0.45
        else "I notice the crowd, but I still want to make my own call."
        if control <= 0.65
        else "I do not want to follow the crowd and I prefer my own plan."
    )

    uncertainty_text = (
        "I feel unsure and hesitant about what to do next."
        if risk >= 0.70 and control <= 0.40
        else "I have some uncertainty, but I can still make a decision."
        if risk >= 0.50 or control <= 0.55
        else "I feel confident about the decision."
    )

    evidence_style_text = (
        "My decision is based on data, valuation, balance sheet quality, and structured research."
        if analytical >= 0.70
        else "I am using some evidence, but I am also influenced by headlines and narratives."
        if analytical >= 0.40
        else "I am not relying on structured analysis, models, or research right now."
    )

    agency_text = (
        "I feel personal agency, I trust my own discipline, and I am following my own process."
        if control >= 0.70
        else "I am trying to stay disciplined, but outside events still affect my actions."
        if control >= 0.40
        else "I feel helpless, externally driven, and out of control because the market keeps forcing my hand."
    )

    return " ".join(
        [
            f"Market regime: {market_context['market_regime']}.",
            (
                f"Market detail: {market_context['market_description']} "
                f"(1m return {market_context['monthly_return_pct']:+.2f}%, "
                f"drawdown {market_context['drawdown_pct']:+.2f}%, "
                f"volatility score {market_context['volatility_score']:.2f})."
            ),
            f"Life context: {life_event}.",
            risk_text,
            patience_text,
            analytical_text,
            control_text,
            crowd_text,
            uncertainty_text,
            evidence_style_text,
            agency_text,
        ]
    )


def _market_ticker(price_history: pd.DataFrame, market_ticker: str = "NIFTYBEES.NS") -> str:
    if market_ticker in price_history.columns:
        return market_ticker
    return str(price_history.columns[0])


def _annualized_volatility(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * math.sqrt(252))


def _market_context_from_history(
    price_history: pd.DataFrame,
    *,
    month_dates: list[date],
    market_ticker: str = "NIFTYBEES.NS",
) -> list[dict[str, object]]:
    ticker = _market_ticker(price_history, market_ticker=market_ticker)
    series = price_history[ticker].dropna().sort_index()
    contexts: list[dict[str, object]] = []

    for current_date in month_dates:
        current_ts = pd.Timestamp(current_date.isoformat())
        history = series.loc[series.index <= current_ts]
        if len(history) < 5:
            contexts.append(
                {
                    "date": current_date.isoformat(),
                    "market_ticker": ticker,
                    "market_regime": "insufficient_history",
                    "market_description": "limited price history; using neutral market effect",
                    "monthly_return_pct": 0.0,
                    "drawdown_pct": 0.0,
                    "volatility_score": 0.5,
                    "market_drift": {
                        "risk_sensitivity": 0.0,
                        "patience_level": 0.0,
                        "analytical_thinking": 0.0,
                        "controlled_perception": 0.0,
                    },
                }
            )
            continue

        trailing_prices = history.tail(63)
        monthly_return = float((trailing_prices.iloc[-1] / trailing_prices.iloc[max(0, len(trailing_prices) - 22)] - 1.0)) if len(trailing_prices) >= 22 else 0.0
        peak_price = float(trailing_prices.max()) if not trailing_prices.empty else float(history.iloc[-1])
        current_price = float(history.iloc[-1])
        drawdown = ((current_price / peak_price) - 1.0) if peak_price > 0 else 0.0
        trailing_returns = history.pct_change().dropna().tail(21)
        annualized_vol = _annualized_volatility(trailing_returns)
        volatility_score = _clamp(annualized_vol / 0.40)

        if monthly_return <= -0.12 or drawdown <= -0.18:
            regime = "crash"
            description = "real market stress from Yahoo Finance history with sharp losses"
        elif monthly_return < 0 or drawdown <= -0.08:
            regime = "correction"
            description = "real market pullback from Yahoo Finance history"
        elif monthly_return >= 0.05 and drawdown > -0.03 and volatility_score < 0.45:
            regime = "calm_bull"
            description = "real market advance with low realized volatility"
        elif monthly_return >= 0:
            regime = "volatile_bull"
            description = "real market gains with unstable swings"
        else:
            regime = "recovery"
            description = "real market recovery after earlier stress"

        drift = {
            "risk_sensitivity": round(_clamp(0.14 * volatility_score + 0.18 * max(-monthly_return, 0.0) + 0.10 * max(-drawdown, 0.0), 0.0, 0.22), 4),
            "patience_level": round(-_clamp(0.10 * volatility_score + 0.14 * max(-monthly_return, 0.0), 0.0, 0.18), 4),
            "analytical_thinking": round(_clamp(0.05 * volatility_score - 0.05 * max(-monthly_return, 0.0), -0.06, 0.06), 4),
            "controlled_perception": round(-_clamp(0.12 * volatility_score + 0.12 * max(-drawdown, 0.0), 0.0, 0.20), 4),
        }
        if monthly_return > 0 and drawdown > -0.05:
            drift["risk_sensitivity"] = round(-min(0.08, 0.03 + monthly_return * 0.35), 4)
            drift["patience_level"] = round(min(0.07, 0.02 + monthly_return * 0.25), 4)
            drift["controlled_perception"] = round(min(0.08, 0.02 + monthly_return * 0.20), 4)

        contexts.append(
            {
                "date": current_date.isoformat(),
                "market_ticker": ticker,
                "market_regime": regime,
                "market_description": description,
                "monthly_return_pct": round(monthly_return * 100, 4),
                "drawdown_pct": round(drawdown * 100, 4),
                "volatility_score": round(volatility_score, 4),
                "market_drift": drift,
            }
        )

    return contexts


def _sample_life_event(archetype: Archetype, rng: random.Random) -> str:
    # Raise event incidence slightly so more users experience meaningful state changes
    # over the simulation horizon, which makes static onboarding profiles stale faster.
    threshold = 0.18 + 0.22 * archetype.life_event_sensitivity
    if rng.random() > threshold:
        return "none"

    weighted = [
        ("family_expense_shock", 0.24),
        ("job_uncertainty", 0.18),
        ("salary_growth", 0.16),
        ("new_dependents", 0.18),
        ("health_stress", 0.24),
    ]
    draw = rng.random()
    running = 0.0
    for name, weight in weighted:
        running += weight
        if draw <= running:
            return name
    return weighted[-1][0]


def generate_synthetic_users(
    *,
    num_users: int = 100,
    seed: int = 42,
    capital: float = 100_000.0,
    static_profile_noise_scale: float = 1.6,
    static_profile_midpoint_pull: float = 0.18,
    save_path: Path | str | None = DEFAULT_USERS_PATH,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    users: list[dict[str, object]] = []

    for index in range(num_users):
        archetype = ARCHETYPES[index % len(ARCHETYPES)]
        baseline_true_theta = _jitter_theta(archetype.base_theta, rng, scale=0.09)
        static_theta = _simulate_self_reported_theta(
            baseline_true_theta,
            rng=rng,
            noise=archetype.self_report_noise * static_profile_noise_scale,
            midpoint_pull=static_profile_midpoint_pull,
        )
        users.append(
            {
                "user_id": f"sim_user_{index + 1:03d}",
                "email": f"sim_user_{index + 1:03d}@example.com",
                "name": f"Simulated User {index + 1:03d}",
                "capital": capital,
                "archetype": archetype.name,
                "baseline_true_theta": baseline_true_theta,
                "static_theta": static_theta,
                "static_theta_scalar": round(theta_scalar(static_theta), 4),
                "market_sensitivity": round(min(1.0, archetype.market_sensitivity * 1.12), 4),
                "life_event_sensitivity": round(min(1.0, archetype.life_event_sensitivity * 1.15), 4),
                "recovery_bias": round(max(0.04, archetype.recovery_bias * 0.72), 4),
            }
        )

    if save_path is not None:
        save_target = Path(save_path)
        save_target.parent.mkdir(parents=True, exist_ok=True)
        with save_target.open("w", encoding="utf-8") as handle:
            json.dump(_safe_json_dump(users), handle, indent=2)

    return users


def _transition_theta(
    current_theta: dict[str, float],
    *,
    baseline_theta: dict[str, float],
    market_drift: dict[str, float],
    life_event: str,
    market_sensitivity: float,
    life_event_sensitivity: float,
    recovery_bias: float,
    rng: random.Random,
) -> dict[str, float]:
    life_drift = LIFE_EVENTS[life_event]["drift"]
    updated = {}

    for key in LATENT_KEYS:
        value = float(current_theta[key])
        pull_home = recovery_bias * (float(baseline_theta[key]) - value)
        market_component = 1.10 * market_sensitivity * float(market_drift[key])
        life_component = 1.15 * life_event_sensitivity * float(life_drift[key])
        noise = rng.uniform(-0.035, 0.035)
        updated[key] = round(_clamp(value + pull_home + market_component + life_component + noise), 4)

    return updated


def _dimension_error_series(
    events: list[dict[str, object]],
    baseline_theta: dict[str, float],
) -> dict[str, list[dict[str, float | str | None]]]:
    series: dict[str, list[dict[str, float | str | None]]] = {key: [] for key in LATENT_KEYS}
    if not events:
        return series

    start_date = date.fromisoformat(str(events[0]["date"]))
    for event in events:
        current_date = date.fromisoformat(str(event["date"]))
        month_offset = (current_date - start_date).days / 30.4375
        for key in LATENT_KEYS:
            error = abs(float(event["theta_true"][key]) - float(baseline_theta[key]))
            series[key].append(
                {
                    "date": str(event["date"]),
                    "month_offset": month_offset,
                    "absolute_error": error,
                }
            )
    return series


def _dimension_average_error_by_month(results: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[int, dict[str, list[float]]] = {}
    for result in results:
        for key, rows in result["dimension_error_series"].items():
            for row in rows:
                month_index = int(round(float(row["month_offset"])))
                month_bucket = buckets.setdefault(month_index, {latent_key: [] for latent_key in LATENT_KEYS})
                month_bucket[key].append(float(row["absolute_error"]))

    output = []
    for month, by_dimension in sorted(buckets.items()):
        output.append(
            {
                "month": float(month),
                "dimension_mae": {
                    key: (sum(values) / len(values)) if values else 0.0
                    for key, values in by_dimension.items()
                },
            }
        )
    return output


def _scalar_error_cdf(results: list[dict[str, object]]) -> list[dict[str, float]]:
    final_errors = sorted(
        float(result["scalar_error_series"][-1]["absolute_error"])
        for result in results
        if result["scalar_error_series"]
    )
    if not final_errors:
        return []

    total = len(final_errors)
    return [
        {
            "error_threshold": error,
            "cdf": (index + 1) / total,
        }
        for index, error in enumerate(final_errors)
    ]


def _series_growth_rate(series: list[dict[str, float | str | None]]) -> float | None:
    if len(series) < 2:
        return None
    start_month = series[0]["month_offset"]
    end_month = series[-1]["month_offset"]
    start_error = series[0]["absolute_error"]
    end_error = series[-1]["absolute_error"]
    if start_month is None or end_month is None or float(end_month) == float(start_month):
        return None
    return (float(end_error) - float(start_error)) / (float(end_month) - float(start_month))


def _user_level_metric_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for result in results:
        scalar_series = result["scalar_error_series"]
        final_dimension_error = {
            key: float(rows[-1]["absolute_error"]) if rows else None
            for key, rows in result["dimension_error_series"].items()
        }
        life_event_months = sum(1 for event in result["events"] if event["life_event"] != "none")
        output.append(
            {
                "user_id": result["user_id"],
                "archetype": result["archetype"],
                "final_scalar_error": float(scalar_series[-1]["absolute_error"]) if scalar_series else None,
                "average_scalar_error": (
                    sum(float(point["absolute_error"]) for point in scalar_series) / len(scalar_series)
                ) if scalar_series else None,
                "scalar_error_growth_rate": _series_growth_rate(scalar_series),
                "final_rmse": result["static_misalignment"]["final_rmse"],
                "rmse_growth_rate": (
                    result["static_misalignment"]["error_growth_rate"]["slope_per_month"]
                    if result["static_misalignment"]["error_growth_rate"]
                    else None
                ),
                "material_misalignment_rate": result["static_misalignment"]["material_misalignment_rate"],
                "life_event_months": life_event_months,
                "final_dimension_error": final_dimension_error,
            }
        )
    return output


def _dimension_level_metric_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for key in LATENT_KEYS:
        dimension_series = []
        for result in results:
            rows = result["dimension_error_series"][key]
            if not rows:
                continue
            dimension_series.append(rows)

        if not dimension_series:
            continue

        final_errors = sorted(float(rows[-1]["absolute_error"]) for rows in dimension_series)
        growth_rates = [rate for rate in (_series_growth_rate(rows) for rows in dimension_series) if rate is not None]
        output.append(
            {
                "dimension": key,
                "average_error_over_time": [
                    {
                        "month": float(month_row["month"]),
                        "average_error": float(month_row["dimension_mae"][key]),
                    }
                    for month_row in _dimension_average_error_by_month(results)
                ],
                "average_final_error": sum(final_errors) / len(final_errors),
                "median_final_error": statistics.median(final_errors),
                "average_growth_rate": (sum(growth_rates) / len(growth_rates)) if growth_rates else None,
                "final_error_cdf": [
                    {
                        "error_threshold": error,
                        "cdf": (index + 1) / len(final_errors),
                    }
                    for index, error in enumerate(final_errors)
                ],
            }
        )
    return output


def _plot_population_average_error_over_time(payload: dict[str, object]) -> None:
    series = payload.get("average_error_over_time", [])
    if not series:
        return

    months = [float(point["month"]) for point in series]
    errors = [float(point["average_error"]) for point in series]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(months, errors, marker="o", linewidth=2, color="#1f77b4")
    axis.set_title("Population H1 Average Misalignment Error Over Time")
    axis.set_xlabel("Months Since Onboarding")
    axis.set_ylabel("Average RMSE")
    axis.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(PLOTS_NEW_DIR / "population_h1_average_error_over_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_population_final_error_cdf(payload: dict[str, object]) -> None:
    cdf_points = payload.get("final_error_cdf", {}).get("cdf_points", [])
    if not cdf_points:
        return

    thresholds = [float(point["error_threshold"]) for point in cdf_points]
    cdfs = [float(point["cdf"]) for point in cdf_points]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.step(thresholds, cdfs, where="post", linewidth=2, color="#2ca02c")
    axis.set_title("Population H1 Final Error CDF")
    axis.set_xlabel("Final RMSE Threshold")
    axis.set_ylabel("CDF")
    axis.set_ylim(0, 1.05)
    axis.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(PLOTS_NEW_DIR / "population_h1_final_error_cdf.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_population_growth_rate_distribution(payload: dict[str, object]) -> None:
    user_metrics = payload.get("user_level_metrics", [])
    growth_rates = [
        float(item["rmse_growth_rate"])
        for item in user_metrics
        if item.get("rmse_growth_rate") is not None
    ]
    if not growth_rates:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(growth_rates, bins=min(15, max(5, len(growth_rates) // 5)), color="#ff7f0e", edgecolor="black", alpha=0.85)
    axes[0].set_title("Population H1 RMSE Growth Rates")
    axes[0].set_xlabel("Slope per Month")
    axes[0].set_ylabel("Count")

    axes[1].boxplot(growth_rates, vert=True, patch_artist=True, boxprops={"facecolor": "#4c78a8", "alpha": 0.65})
    axes[1].set_title("Population H1 Growth Rate Boxplot")
    axes[1].set_ylabel("Slope per Month")
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["Users"])

    fig.tight_layout()
    fig.savefig(PLOTS_NEW_DIR / "population_h1_growth_rate_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_population_dimension_errors(payload: dict[str, object]) -> None:
    rows = payload.get("dimension_level_metrics", [])
    if not rows:
        return

    labels = [str(row["dimension"]).replace("_", " ").title() for row in rows]
    final_errors = [float(row["average_final_error"]) for row in rows]
    growth_rates = [float(row["average_growth_rate"] or 0.0) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].bar(labels, final_errors, color="#d62728", alpha=0.85)
    axes[0].set_title("Population H1 Average Final Error by Latent Dimension")
    axes[0].set_ylabel("Average Final Absolute Error")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, growth_rates, color="#9467bd", alpha=0.85)
    axes[1].set_title("Population H1 Average Error Growth by Latent Dimension")
    axes[1].set_ylabel("Average Growth Rate")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(PLOTS_NEW_DIR / "population_h1_dimension_errors.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_population_sampled_trajectories(payload: dict[str, object], max_users: int = 12) -> None:
    results = payload.get("results", [])
    if not results:
        return

    sampled = sorted(
        results,
        key=lambda item: float(item["static_misalignment"]["final_rmse"] or 0.0),
        reverse=True,
    )[:max_users]

    fig, axis = plt.subplots(figsize=(11, 6))
    for result in sampled:
        series = result["static_misalignment"]["error_series"]
        months = [float(point["month_offset"]) for point in series if point["month_offset"] is not None]
        errors = [float(point["rmse"]) for point in series if point["month_offset"] is not None]
        if months and errors:
            axis.plot(months, errors, linewidth=1.7, label=result["user_id"])

    axis.set_title("Population H1 Sampled Misalignment Trajectories")
    axis.set_xlabel("Months Since Onboarding")
    axis.set_ylabel("Static Profile RMSE")
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS_NEW_DIR / "population_h1_sampled_trajectories.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_population_representative_theta(payload: dict[str, object]) -> None:
    results = payload.get("results", [])
    if not results:
        return

    representative = max(
        results,
        key=lambda item: float(item["static_misalignment"]["final_rmse"] or 0.0),
    )
    events = representative.get("events", [])
    if not events:
        return

    dates = [event["date"] for event in events]
    static_theta = representative["static_theta"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()

    for axis, key in zip(axes, LATENT_KEYS):
        true_values = [float(event["theta_true"][key]) for event in events]
        static_values = [float(static_theta[key])] * len(events)
        axis.plot(dates, true_values, linewidth=2.2, color="#1f77b4", label="True Theta")
        axis.plot(dates, static_values, linewidth=2.0, linestyle="--", color="#d62728", label="Static Theta")
        axis.set_title(key.replace("_", " ").title())
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.2, linestyle="--")

    axes[0].legend(loc="best")
    fig.suptitle(f"Population H1 Static vs True Theta: {representative['user_id']}", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_NEW_DIR / "population_h1_representative_theta.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_population_h1_plots(payload: dict[str, object]) -> None:
    _ensure_population_output_dirs()
    _plot_population_average_error_over_time(payload)
    _plot_population_final_error_cdf(payload)
    _plot_population_growth_rate_distribution(payload)
    _plot_population_dimension_errors(payload)
    _plot_population_sampled_trajectories(payload)
    _plot_population_representative_theta(payload)


def run_population_h1_simulation(
    *,
    num_users: int = 100,
    months: int = 24,
    seed: int = 42,
    start_date: str = "2022-01-01",
    material_threshold: float = 0.12,
    price_history: pd.DataFrame | None = None,
    market_ticker: str = "NIFTYBEES.NS",
    save_users_path: Path | str | None = DEFAULT_USERS_PATH,
    save_results_path: Path | str | None = DEFAULT_RESULTS_PATH,
    save_user_profiles_dir: Path | str | None = DEFAULT_USER_PROFILES_DIR,
    save_evaluation_results_path: Path | str | None = DEFAULT_EVALUATION_RESULTS_PATH,
    save_plots: bool = True,
    include_inference: bool = False,
    inference_strength: float | dict[str, float] = {
        "risk_sensitivity": 2.0,
        "patience_level": 2.0,
        "analytical_thinking": 3.2,
        "controlled_perception": 3.0,
    },
    inference_decay_factor: float | dict[str, float] = {
        "risk_sensitivity": 0.92,
        "patience_level": 0.92,
        "analytical_thinking": 0.82,
        "controlled_perception": 0.80,
    },
    posterior_concentration_scale: float | dict[str, float] = 1.0,
    static_profile_noise_scale: float = 1.6,
    static_profile_midpoint_pull: float = 0.18,
    signal_mode: str = "auto",
    require_real_nlp: bool = False,
    verbose: bool = True,
    log_prefix: str = "H1-POP",
) -> dict[str, object]:
    _ensure_population_output_dirs()
    users = generate_synthetic_users(
        num_users=num_users,
        seed=seed,
        static_profile_noise_scale=static_profile_noise_scale,
        static_profile_midpoint_pull=static_profile_midpoint_pull,
        save_path=save_users_path,
    )
    start_dt = date.fromisoformat(start_date)
    history = price_history if price_history is not None else get_price_history()
    month_dates = [_add_months(start_dt, month_index) for month_index in range(months)]
    market_timeline = _market_context_from_history(
        history,
        month_dates=month_dates,
        market_ticker=market_ticker,
    )
    results: list[dict[str, object]] = []
    archetype_lookup = {item.name: item for item in ARCHETYPES}

    for index, user in enumerate(users):
        if verbose:
            print(
                f"[{log_prefix}] User {index + 1}/{len(users)} "
                f"{user['user_id']} archetype={user['archetype']}",
                flush=True,
            )
        user_rng = random.Random(seed + 10_000 + index)
        current_true_theta = dict(user["baseline_true_theta"])
        engine = None
        if include_inference:
            from latent_state_engine import run_bayesian
            from latent_state_engine.bayesian_update import BayesianLatentEngine

            engine = BayesianLatentEngine()
        events: list[dict[str, object]] = []

        for month_index in range(months):
            current_date = month_dates[month_index]
            market_context = market_timeline[month_index]
            life_event = _sample_life_event(
                archetype_lookup[str(user["archetype"])],
                user_rng,
            )
            prior_theta = dict(current_true_theta)
            current_true_theta = _transition_theta(
                current_true_theta,
                baseline_theta=user["baseline_true_theta"],
                market_drift=market_context["market_drift"],
                life_event=life_event,
                market_sensitivity=float(user["market_sensitivity"]),
                life_event_sensitivity=float(user["life_event_sensitivity"]),
                recovery_bias=float(user["recovery_bias"]),
                rng=user_rng,
            )
            chat_text = _chat_text_for_state(
                current_true_theta,
                market_context=market_context,
                life_event=life_event,
            )
            market_drift = {
                key: round(float(user["market_sensitivity"]) * float(market_context["market_drift"][key]), 4)
                for key in LATENT_KEYS
            }
            life_drift = {
                key: round(float(user["life_event_sensitivity"]) * float(LIFE_EVENTS[life_event]["drift"][key]), 4)
                for key in LATENT_KEYS
            }
            event = {
                "date": current_date.isoformat(),
                "month_index": month_index,
                "market_regime": market_context["market_regime"],
                "market_description": market_context["market_description"],
                "market_ticker": market_context["market_ticker"],
                "market_monthly_return_pct": market_context["monthly_return_pct"],
                "market_drawdown_pct": market_context["drawdown_pct"],
                "market_volatility_score": market_context["volatility_score"],
                "life_event": life_event,
                "life_event_description": LIFE_EVENTS[life_event]["description"],
                "theta_prior": prior_theta,
                "theta_true": current_true_theta,
                "theta_static": user["static_theta"],
                "theta_true_scalar": round(theta_scalar(current_true_theta), 4),
                "theta_static_scalar": float(user["static_theta_scalar"]),
                "driver_effects": {
                    "market_drift": market_drift,
                    "life_event_drift": life_drift,
                },
                "chat_text": chat_text,
            }
            if include_inference:
                signals, signal_source, signal_metadata = _extract_signals_with_fallback(
                    chat_text,
                    current_true_theta,
                    signal_mode=signal_mode,
                    require_real_nlp=require_real_nlp,
                )
                if verbose:
                    print(
                        f"[{log_prefix}]   Month {month_index + 1}/{months} "
                        f"date={current_date.isoformat()} backend={signal_source}",
                        flush=True,
                    )
                theta_inferred = run_bayesian(
                    signals,
                    strength=inference_strength,
                    decay_factor=inference_decay_factor,
                    engine=engine,
                )
                posterior_params = _calibrate_posterior_params(
                    engine.get_params(),
                    concentration_scale=posterior_concentration_scale,
                )
                event["theta_inferred"] = theta_inferred
                event["posterior_params"] = posterior_params
                event["signals"] = signals
                event["signal_source"] = signal_source
                event["signal_metadata"] = signal_metadata
            event["scalar_absolute_error"] = abs(event["theta_true_scalar"] - event["theta_static_scalar"])
            events.append(event)

        static_summary = summarize_static_misalignment(
            events,
            user["static_theta"],
            baseline_name="self_reported_static_theta",
            material_threshold=material_threshold,
        )
        dimension_error_series = _dimension_error_series(events, user["static_theta"])
        scalar_error_series = [
            {
                "date": event["date"],
                "month_offset": point["month_offset"],
                "absolute_error": event["scalar_absolute_error"],
            }
            for event, point in zip(events, static_summary["error_series"])
        ]

        results.append(
            {
                "user_id": user["user_id"],
                "email": user["email"],
                "name": user["name"],
                "archetype": user["archetype"],
                "baseline_true_theta": user["baseline_true_theta"],
                "static_theta": user["static_theta"],
                "static_profile_origin": "self_reported_onboarding_preference",
                "events": events,
                "dimension_error_series": dimension_error_series,
                "scalar_error_series": scalar_error_series,
                "static_misalignment": static_summary,
                "hypothesis_supported": bool(
                    static_summary["final_rmse"] is not None
                    and static_summary["rmse_growth"] is not None
                    and static_summary["final_rmse"] > material_threshold
                    and static_summary["rmse_growth"] > 0
                ),
            }
        )
        if include_inference:
            from evaluation.metrics import compare_static_profile_to_dynamic

            results[-1]["static_vs_dynamic"] = compare_static_profile_to_dynamic(events, user["static_theta"])

        if save_user_profiles_dir is not None:
            _save_json(
                Path(save_user_profiles_dir) / f"{user['user_id']}.json",
                {
                    "user_profile": user,
                    "simulation_result": results[-1],
                },
            )
        if verbose:
            completion_source = (
                events[-1].get("signal_source")
                if include_inference and events
                else "static_only"
            )
            print(
                f"[{log_prefix}] Completed {user['user_id']} "
                f"months={len(events)} last_backend={completion_source}",
                flush=True,
            )

    investor_error_series = [
        {
            "user_id": result["user_id"],
            "error_series": result["static_misalignment"]["error_series"],
        }
        for result in results
    ]
    final_rmse_values = [
        float(result["static_misalignment"]["final_rmse"])
        for result in results
        if result["static_misalignment"]["final_rmse"] is not None
    ]
    growth_rates = [
        float(result["static_misalignment"]["error_growth_rate"]["slope_per_month"])
        for result in results
        if result["static_misalignment"]["error_growth_rate"]
        and result["static_misalignment"]["error_growth_rate"]["slope_per_month"] is not None
    ]
    supported_count = sum(1 for result in results if result["hypothesis_supported"])
    user_level_metrics = _user_level_metric_rows(results)
    dimension_level_metrics = _dimension_level_metric_rows(results)

    payload: dict[str, object] = {
        "generated_at": date.today().isoformat(),
        "num_users": num_users,
        "months": months,
        "start_date": start_date,
        "material_threshold_rmse": material_threshold,
        "market_data_source": "yfinance",
        "market_ticker": _market_ticker(history, market_ticker=market_ticker),
        "market_timeline": market_timeline,
        "include_inference": include_inference,
        "simulation_artifacts_dir": str(SIMULATION_DIR),
        "hypothesis_support_rate": (supported_count / len(results)) if results else None,
        "average_error_over_time": average_error_by_month(investor_error_series),
        "final_error_cdf": build_cross_investor_error_cdf(investor_error_series, month_index=-1),
        "scalar_error_cdf": _scalar_error_cdf(results),
        "dimension_average_error_over_time": _dimension_average_error_by_month(results),
        "dimension_level_metrics": dimension_level_metrics,
        "user_level_metrics": user_level_metrics,
        "average_final_rmse": (sum(final_rmse_values) / len(final_rmse_values)) if final_rmse_values else None,
        "median_final_rmse": statistics.median(final_rmse_values) if final_rmse_values else None,
        "average_error_growth_rate": (sum(growth_rates) / len(growth_rates)) if growth_rates else None,
        "results": results,
    }
    if include_inference:
        payload["inference_strength"] = inference_strength
        payload["inference_decay_factor"] = inference_decay_factor
        payload["posterior_concentration_scale"] = posterior_concentration_scale
        payload["static_profile_noise_scale"] = static_profile_noise_scale
        payload["static_profile_midpoint_pull"] = static_profile_midpoint_pull
        payload["signal_mode"] = signal_mode
        payload["require_real_nlp"] = require_real_nlp
        payload["signal_source_summary"] = {
            source: sum(
                1
                for result in results
                for event in result["events"]
                if event.get("signal_source") == source
            )
            for source in {
                str(event.get("signal_source"))
                for result in results
                for event in result["events"]
            }
        }

    if save_results_path is not None:
        _save_json(save_results_path, payload)
    if save_evaluation_results_path is not None:
        _save_json(save_evaluation_results_path, payload)
    if save_plots:
        _save_population_h1_plots(payload)
    if verbose:
        print(
            f"[{log_prefix}] Finished population run users={num_users} months={months}",
            flush=True,
        )

    return payload


if __name__ == "__main__":
    run_population_h1_simulation()

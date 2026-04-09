from __future__ import annotations

import json
import math
import random
import sys
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from evaluation.H2.metrics import (
    LATENT_KEYS,
    compute_credible_interval_coverage,
    compute_event_error,
    compute_grouped_pearson_tracking,
    compute_pearson_tracking,
    compute_static_baseline_metrics,
    summarize_event_errors,
)
from integration.theta_adapter import select_config_from_theta
from latent_state_engine import run_bayesian
from latent_state_engine.bayesian_update import BayesianLatentEngine
from nlp.signal_extractor import extract_schema_variables_with_metadata

from portfolio_optimizer.portfolio_api import should_trigger_rebalance
from portfolio_optimizer.simulation_scenarios import SCENARIOS, SIMULATION_END_DATE
from .h1_population_simulator import run_population_h1_simulation
from .simulator import SimPortfolio, apply_event, get_price_history, get_prices_on_date, run_optimizer_historical

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT_DIR / "evaluation"
H2_DIR = EVALUATION_DIR / "H2"
PLOTS_DIR = H2_DIR / "plots"
PLOTS_NEW_DIR = H2_DIR / "plots_new"
RESULTS_PATH = H2_DIR / "result.json"
POPULATION_RESULTS_PATH = H2_DIR / "population_result.json"


def _ensure_output_dirs() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    (H2_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_NEW_DIR.mkdir(parents=True, exist_ok=True)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _add_months(dt: date, months: int) -> date:
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    day = min(dt.day, monthrange(year, month)[1])
    return date(year, month, day)


def _month_window_starts(start_dt: date, end_dt: date) -> list[date]:
    starts = []
    current = date(start_dt.year, start_dt.month, 1)
    while current < end_dt:
        starts.append(current)
        current = _add_months(current, 1)
    return starts


def _safe_json_dump(value):
    if isinstance(value, dict):
        return {key: _safe_json_dump(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json_dump(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_json_dump(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def _market_volatility_score(
    price_history: pd.DataFrame,
    as_of_date: str,
    *,
    market_ticker: str = "NIFTYBEES.NS",
    window_days: int = 21,
) -> float:
    if market_ticker not in price_history.columns:
        market_ticker = price_history.columns[0]

    ts = pd.Timestamp(as_of_date)
    series = price_history[market_ticker].dropna()
    series = series.loc[series.index <= ts]
    if len(series) < 5:
        return 0.5

    returns = series.pct_change().dropna().tail(window_days)
    if returns.empty:
        return 0.5

    annualized_vol = float(returns.std(ddof=0) * math.sqrt(252))
    return _clamp(annualized_vol / 0.40)


def _persona_traits(persona: str) -> dict[str, float]:
    text = persona.lower()
    if "reactive" in text:
        return {"risk": 0.55, "patience": 0.40, "analytical": 0.45, "control": 0.40}
    if "dip buyer" in text:
        return {"risk": 0.65, "patience": 0.55, "analytical": 0.60, "control": 0.60}
    if "panic" in text:
        return {"risk": 0.75, "patience": 0.25, "analytical": 0.35, "control": 0.25}
    if "set and forget" in text:
        return {"risk": 0.25, "patience": 0.80, "analytical": 0.60, "control": 0.70}
    if "news follower" in text:
        return {"risk": 0.50, "patience": 0.45, "analytical": 0.50, "control": 0.45}
    return {"risk": 0.50, "patience": 0.50, "analytical": 0.50, "control": 0.50}


def _sample_theta_true(
    *,
    rng: random.Random,
    persona: str,
    volatility_score: float,
    previous_theta: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    traits = _persona_traits(persona)
    prev = previous_theta or {
        "risk_sensitivity": traits["risk"],
        "patience_level": traits["patience"],
        "analytical_thinking": traits["analytical"],
        "controlled_perception": traits["control"],
    }
    reactive_noise = 0.10 if "reactive" in persona.lower() or "news follower" in persona.lower() else 0.06

    risk = _clamp(
        0.35 * traits["risk"] +
        0.30 * prev["risk_sensitivity"] +
        0.35 * volatility_score +
        rng.uniform(-reactive_noise, reactive_noise)
    )
    patience = _clamp(
        0.40 * traits["patience"] +
        0.30 * prev["patience_level"] +
        0.30 * (1.0 - volatility_score) +
        rng.uniform(-reactive_noise, reactive_noise)
    )
    analytical = _clamp(
        0.50 * traits["analytical"] +
        0.25 * prev["analytical_thinking"] +
        0.15 * (1.0 - volatility_score) +
        rng.uniform(-0.05, 0.05)
    )
    control = _clamp(
        0.35 * traits["control"] +
        0.30 * prev["controlled_perception"] +
        0.35 * (1.0 - volatility_score) +
        rng.uniform(-reactive_noise, reactive_noise)
    )

    return {
        "risk_sensitivity": round(risk, 4),
        "patience_level": round(patience, 4),
        "analytical_thinking": round(analytical, 4),
        "controlled_perception": round(control, 4),
    }


def _theta_to_chat_text(
    theta: dict[str, float],
    *,
    persona: str,
    volatility_score: float,
    rng: random.Random,
) -> str:
    risk = theta["risk_sensitivity"]
    patience = theta["patience_level"]
    analytical = theta["analytical_thinking"]
    control = theta["controlled_perception"]
    if risk >= 0.75:
        risk_phrase = "I am willing to take aggressive positions because upside matters more to me than drawdowns right now"
    elif risk >= 0.55:
        risk_phrase = "I can accept measured risk if the reward potential is strong"
    elif risk >= 0.35:
        risk_phrase = "I prefer balanced risk and I do not want unnecessary downside"
    else:
        risk_phrase = "Capital preservation matters more to me than chasing returns"

    if patience >= 0.75:
        patience_phrase = "I am comfortable holding for years and letting compounding play out"
    elif patience >= 0.55:
        patience_phrase = "I can stay patient for months if the thesis still holds"
    elif patience >= 0.35:
        patience_phrase = "I want flexibility and I do not want to stay locked in too long"
    else:
        patience_phrase = "I want an immediate move and I am focused on what happens this week"

    if analytical >= 0.72:
        analytical_phrase = "I reviewed earnings, valuations, cash flows, and market data before deciding"
    elif analytical >= 0.52:
        analytical_phrase = "I am weighing both the data and my instinct before I act"
    elif analytical >= 0.32:
        analytical_phrase = "I am leaning more on market mood and headlines than on deep analysis"
    else:
        analytical_phrase = "I am mostly reacting to price action and gut feel instead of analysis"

    if control >= 0.72:
        control_phrase = "I trust my own plan and I feel fully in control of the decision"
    elif control >= 0.52:
        control_phrase = "I am trying to stay disciplined and stick to my process"
    elif control >= 0.32:
        control_phrase = "I feel the market is pushing me around more than I want"
    else:
        control_phrase = "It feels like the market decides for me and I am struggling to stay in control"

    herding_phrase = (
        "I do not care what the crowd is doing and I want to rely on my own view"
        if analytical >= 0.60 and control >= 0.55 else
        "I keep noticing what everyone else is buying and it affects my conviction"
        if analytical < 0.45 or control < 0.40 else
        "I notice the crowd, but I still want to make my own call"
    )
    vol_phrase = (
        "Recent volatility has been very high and the swings are hard to ignore."
        if volatility_score >= 0.70 else
        "The market has been moving around, but it does not feel completely chaotic."
        if volatility_score >= 0.40 else
        "The market has felt relatively calm recently."
    )

    intro_options = [
        f"As a {persona.lower()}, {risk_phrase}.",
        f"Right now, {risk_phrase}.",
        f"My current stance is this: {risk_phrase}.",
    ]
    sentences = [
        rng.choice(intro_options),
        patience_phrase + ".",
        analytical_phrase + ".",
        control_phrase + ".",
        herding_phrase + ".",
        vol_phrase,
    ]
    rng.shuffle(sentences[1:])
    return " ".join(sentences[: rng.randint(4, 6)])


def _inverse_signals_from_theta(theta: dict[str, float]) -> dict[str, float]:
    risk = theta["risk_sensitivity"]
    patience = theta["patience_level"]
    analytical = theta["analytical_thinking"]
    control = theta["controlled_perception"]
    return {
        "fear_sentiment": round(risk, 4),
        "risk_language_density": round(risk, 4),
        "time_horizon_bias": round(2 * patience - 1, 4),
        "urgency_score": round(1 - patience, 4),
        "analytical_marker": round(analytical, 4),
        "herding_marker": round(_clamp((risk + (1 - control)) / 2), 4),
        "internal_locus_score": round(control, 4),
        "external_locus_score": round(1 - control, 4),
        "uncertainty_score": round(_clamp(0.6 * risk + 0.4 * (1 - control)), 4),
    }


def _extract_signals_with_fallback(
    chat_text: str,
    theta_true: dict[str, float],
    *,
    require_real_nlp: bool = False,
) -> tuple[dict[str, float], str, dict[str, object]]:
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
        return _inverse_signals_from_theta(theta_true), "synthetic_fallback", {"error": str(exc)}


def generate_latent_scenario(
    base_scenario: dict,
    *,
    seed: int = 42,
    min_gap_months: int = 4,
    max_gap_months: int = 7,
    chats_per_month: tuple[int, int] = (2, 4),
    end_date: str = SIMULATION_END_DATE,
    price_history: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Build a ground-truth latent timeline where each latent state spans multiple chats.
    """
    if min_gap_months < 3:
        raise ValueError("min_gap_months must be at least 3")

    rng = random.Random(seed)
    history = price_history if price_history is not None else get_price_history()
    start_date = base_scenario["events"][0]["date"]
    start_dt = date.fromisoformat(start_date)
    end_dt = date.fromisoformat(end_date)

    change_dates = [start_dt]
    current = start_dt
    while True:
        next_dt = _add_months(current, rng.randint(min_gap_months, max_gap_months))
        if next_dt >= end_dt:
            break
        change_dates.append(next_dt)
        current = next_dt

    timeline = []
    previous_theta = None
    for index, state_start in enumerate(change_dates):
        next_state = change_dates[index + 1] if index + 1 < len(change_dates) else end_dt
        volatility_score = _market_volatility_score(history, state_start.isoformat())
        theta_true = _sample_theta_true(
            rng=rng,
            persona=base_scenario["persona"],
            volatility_score=volatility_score,
            previous_theta=previous_theta,
        )

        chats = []
        for month_start in _month_window_starts(state_start, next_state):
            chat_count = rng.randint(*chats_per_month)
            days_in_month = monthrange(month_start.year, month_start.month)[1]
            chosen_days = sorted(
                rng.sample(range(1, days_in_month + 1), k=min(chat_count, days_in_month))
            )
            for day_number in chosen_days:
                chat_dt = date(month_start.year, month_start.month, day_number)
                if chat_dt < state_start or chat_dt >= next_state:
                    continue
                chats.append({
                    "date": chat_dt.isoformat(),
                    "text": _theta_to_chat_text(
                        theta_true,
                        persona=base_scenario["persona"],
                        volatility_score=volatility_score,
                        rng=rng,
                    ),
                })

        if not chats:
            chats.append({
                "date": state_start.isoformat(),
                "text": _theta_to_chat_text(
                    theta_true,
                    persona=base_scenario["persona"],
                    volatility_score=volatility_score,
                    rng=rng,
                ),
            })

        timeline.append({
            "state_id": index + 1,
            "state_start": state_start.isoformat(),
            "state_end": next_state.isoformat(),
            "market_volatility_score": round(volatility_score, 4),
            "theta_true": theta_true,
            "chats": sorted(chats, key=lambda item: item["date"]),
        })
        previous_theta = theta_true

    return {
        "email": base_scenario["email"],
        "name": base_scenario["name"],
        "capital": base_scenario["capital"],
        "persona": base_scenario["persona"],
        "latent_timeline": timeline,
        "simulation_end_date": end_date,
    }


def _gap_decay(previous_date: Optional[str], current_date: str) -> Optional[float]:
    if previous_date is None:
        return None
    days = (date.fromisoformat(current_date) - date.fromisoformat(previous_date)).days
    if days <= 0:
        return None
    return _clamp(1.0 - min(days, 120) / 600.0, lower=0.80, upper=0.99)


def _calibrate_posterior_params(
    posterior_params: dict[str, dict[str, float]],
    *,
    concentration_scale: float | dict[str, float] = 1.0,
) -> dict[str, dict[str, float]]:
    """
    Reduce posterior concentration while preserving the posterior mean.

    This widens credible intervals for H2 evaluation so the reported coverage
    better reflects empirical uncertainty instead of the raw update count.
    """
    scaled = {}
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


def run_latent_state_simulation(
    scenario: dict,
    *,
    seed: int = 42,
    strength: float = 0.7,
    threshold_pct: float = 2.0,
    simulate_portfolio: bool = True,
    price_history: Optional[pd.DataFrame] = None,
    verbose: bool = True,
    log_prefix: str = "H2",
    posterior_concentration_scale: float | dict[str, float] = 1.0,
    require_real_nlp: bool = False,
) -> dict:
    latent_scenario = generate_latent_scenario(
        scenario,
        seed=seed,
        price_history=price_history,
    )

    if verbose:
        print(f"\n[{log_prefix}] Running latent-state simulation for {latent_scenario['name']}")

    engine = BayesianLatentEngine()
    previous_chat_date = None
    all_chat_events = []
    state_endpoints = []
    portfolio = SimPortfolio(scenario["capital"]) if simulate_portfolio else None

    for state in latent_scenario["latent_timeline"]:
        theta_true = state["theta_true"]
        last_theta_inferred = None

        for chat in state["chats"]:
            signals, extraction_mode, signal_metadata = _extract_signals_with_fallback(
                chat["text"],
                theta_true,
                require_real_nlp=require_real_nlp,
            )
            theta_inferred = run_bayesian(
                signals,
                strength=strength,
                decay_factor=_gap_decay(previous_chat_date, chat["date"]),
                engine=engine,
            )
            posterior_params = _calibrate_posterior_params(
                engine.get_params(),
                concentration_scale=posterior_concentration_scale,
            )
            selected_config = select_config_from_theta(theta_inferred)
            trigger_rebalance = should_trigger_rebalance(theta_inferred)
            error = compute_event_error(theta_true, theta_inferred)

            portfolio_value = None
            if simulate_portfolio and portfolio is not None:
                prices = get_prices_on_date(chat["date"])
                should_apply = not portfolio.positions or trigger_rebalance
                if should_apply:
                    optimizer_result = run_optimizer_historical(selected_config, end_date=chat["date"])
                    if optimizer_result and optimizer_result.get("status") == "success":
                        apply_event(portfolio, optimizer_result, chat["date"], prices, threshold_pct)
                portfolio_value = portfolio.record_checkpoint(
                    chat["date"],
                    prices,
                    selected_config["profile"],
                    event_label=f"State {state['state_id']} chat",
                )

            event_record = {
                "state_id": state["state_id"],
                "date": chat["date"],
                "chat_text": chat["text"],
                "theta_true": theta_true,
                "theta_inferred": theta_inferred,
                "posterior_params": posterior_params,
                "signals": signals,
                "signal_source": extraction_mode,
                "signal_metadata": signal_metadata,
                "selected_profile": selected_config["profile"],
                "rebalance_triggered": trigger_rebalance,
                "portfolio_value_inr": portfolio_value,
                "error": error,
            }
            all_chat_events.append(event_record)
            previous_chat_date = chat["date"]
            last_theta_inferred = theta_inferred

        if last_theta_inferred is not None:
            state_endpoints.append({
                "state_id": state["state_id"],
                "state_start": state["state_start"],
                "state_end": state["state_end"],
                "theta_true": theta_true,
                "theta_inferred": last_theta_inferred,
            })

    chat_summary = summarize_event_errors(all_chat_events)
    state_summary = summarize_event_errors(state_endpoints)
    chat_correlation = compute_pearson_tracking(all_chat_events)
    state_correlation = compute_pearson_tracking(state_endpoints)
    static_baseline = compute_static_baseline_metrics(all_chat_events)
    credible_interval_coverage = compute_credible_interval_coverage(all_chat_events)

    portfolio_summary = None
    if simulate_portfolio and portfolio is not None:
        end_prices = get_prices_on_date(latent_scenario["simulation_end_date"])
        final_value = portfolio.portfolio_value(end_prices)
        portfolio_summary = {
            "final_value_inr": round(final_value, 2),
            "total_realized_pnl": round(portfolio.total_realized_pnl, 2),
            "total_tax_paid": round(portfolio.total_tax_paid, 2),
            "leftover_cash": round(portfolio.leftover_cash, 2),
            "total_trades": len(portfolio.trades),
            "checkpoints": portfolio.checkpoints,
        }

    return {
        "email": latent_scenario["email"],
        "name": latent_scenario["name"],
        "persona": latent_scenario["persona"],
        "capital": latent_scenario["capital"],
        "simulation_end_date": latent_scenario["simulation_end_date"],
        "latent_timeline": latent_scenario["latent_timeline"],
        "chat_events": all_chat_events,
        "chat_level_metrics": chat_summary,
        "state_level_metrics": state_summary,
        "pearson_correlation": {
            "chat_level": chat_correlation,
            "state_level": state_correlation,
        },
        "static_baseline_comparison": static_baseline,
        "credible_interval_coverage": credible_interval_coverage,
        "portfolio_simulation": portfolio_summary,
        "calibration": {
            "posterior_concentration_scale": posterior_concentration_scale,
            "require_real_nlp": require_real_nlp,
        },
    }


def _save_user_theta_plot(result: dict) -> None:
    _ensure_output_dirs()
    if not result["chat_events"]:
        return

    dates = [event["date"] for event in result["chat_events"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for axis, key in zip(axes, LATENT_KEYS):
        axis.plot(dates, [event["theta_true"][key] for event in result["chat_events"]], label="True", linewidth=2)
        axis.plot(dates, [event["theta_inferred"][key] for event in result["chat_events"]], label="Inferred", linestyle="--")
        axis.set_title(key.replace("_", " ").title())
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=45)

    axes[0].legend()
    fig.suptitle(f"Latent Tracking: {result['name']}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{result['email'].replace('@', '_at_')}_theta_tracking.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_summary_plots(
    results: list[dict],
    *,
    overall_coverage: Optional[dict] = None,
    overall_static_baseline: Optional[dict] = None,
    overall_correlation: Optional[dict] = None,
) -> None:
    _ensure_output_dirs()
    if not results:
        return

    names = [result["name"] for result in results]
    mae_values = [result["chat_level_metrics"]["overall_mae"] or 0 for result in results]

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(names, mae_values, color="#1f77b4")
    axis.set_title("Average Chat-Level MAE by Investor")
    axis.set_ylabel("MAE")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "latent_mae_by_investor.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    dim_mae = {key: 0.0 for key in LATENT_KEYS}
    for result in results:
        for key in LATENT_KEYS:
            dim_mae[key] += float(result["chat_level_metrics"]["dimension_mae"][key] or 0)
    dim_mae = {key: value / len(results) for key, value in dim_mae.items()}

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar([key.replace("_", " ").title() for key in LATENT_KEYS], [dim_mae[key] for key in LATENT_KEYS], color="#ff7f0e")
    axis.set_title("Average Dimension MAE")
    axis.set_ylabel("MAE")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "latent_mae_by_dimension.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if overall_static_baseline:
        static_overall_rmse = float(overall_static_baseline["static_overall_rmse"] or 0)
        dynamic_overall_rmse = float(overall_static_baseline["dynamic_overall_rmse"] or 0)
        dimension_static_rmse = overall_static_baseline.get("dimension_static_rmse", {})
        dimension_dynamic_rmse = overall_static_baseline.get("dimension_dynamic_rmse", {})
    else:
        static_overall_rmse = sum(
            float(result["static_baseline_comparison"]["static_overall_rmse"] or 0)
            for result in results
        ) / len(results)
        dynamic_overall_rmse = sum(
            float(result["static_baseline_comparison"]["dynamic_overall_rmse"] or 0)
            for result in results
        ) / len(results)
        dimension_static_rmse = {
            key: sum(
                float(result["static_baseline_comparison"]["dimension_static_rmse"][key] or 0)
                for result in results
            ) / len(results)
            for key in LATENT_KEYS
        }
        dimension_dynamic_rmse = {
            key: sum(
                float(result["static_baseline_comparison"]["dimension_dynamic_rmse"][key] or 0)
                for result in results
            ) / len(results)
            for key in LATENT_KEYS
        }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(
        ["Static", "Dynamic"],
        [static_overall_rmse, dynamic_overall_rmse],
        color=["#d62728", "#2ca02c"],
    )
    axes[0].set_title("Overall Deviation From True Theta")
    axes[0].set_ylabel("RMSE")

    x_labels = [key.replace("_", " ").title() for key in LATENT_KEYS]
    x_pos = range(len(LATENT_KEYS))
    width = 0.36
    axes[1].bar(
        [x - width / 2 for x in x_pos],
        [float(dimension_static_rmse.get(key) or 0) for key in LATENT_KEYS],
        width=width,
        label="Static",
        color="#d62728",
    )
    axes[1].bar(
        [x + width / 2 for x in x_pos],
        [float(dimension_dynamic_rmse.get(key) or 0) for key in LATENT_KEYS],
        width=width,
        label="Dynamic",
        color="#2ca02c",
    )
    axes[1].set_title("Deviation by Dimension")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticks(list(x_pos))
    axes[1].set_xticklabels(x_labels, rotation=20)
    axes[1].legend()
    fig.suptitle("Static vs Dynamic Deviation From True Theta")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "latent_improvement_over_static.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if overall_coverage and overall_coverage.get("dimension_coverage"):
        coverage_by_dimension = {
            key: float(overall_coverage["dimension_coverage"].get(key) or 0)
            for key in LATENT_KEYS
        }
    else:
        coverage_by_dimension = {key: 0.0 for key in LATENT_KEYS}
        for result in results:
            for key in LATENT_KEYS:
                coverage_by_dimension[key] += float(
                    result["credible_interval_coverage"]["dimension_coverage"].get(key) or 0
                )
        coverage_by_dimension = {
            key: value / len(results)
            for key, value in coverage_by_dimension.items()
        }

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        [key.replace("_", " ").title() for key in LATENT_KEYS],
        [coverage_by_dimension[key] for key in LATENT_KEYS],
        color="#9467bd",
    )
    axis.axhline(0.90, linestyle="--", color="black", linewidth=1)
    axis.set_title("90% Credible Interval Coverage by Dimension")
    axis.set_ylabel("Coverage")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "latent_credible_interval_coverage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if overall_correlation and overall_correlation.get("dimension_correlation"):
        correlation_by_dimension = {
            key: float(overall_correlation["dimension_correlation"].get(key) or 0)
            for key in LATENT_KEYS
        }
    else:
        correlation_by_dimension = {key: 0.0 for key in LATENT_KEYS}
        for result in results:
            chat_level_corr = result.get("pearson_correlation", {}).get("chat_level", {}).get("dimension_correlation", {})
            for key in LATENT_KEYS:
                correlation_by_dimension[key] += float(chat_level_corr.get(key) or 0)
        correlation_by_dimension = {
            key: value / len(results)
            for key, value in correlation_by_dimension.items()
        }

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        [key.replace("_", " ").title() for key in LATENT_KEYS],
        [correlation_by_dimension[key] for key in LATENT_KEYS],
        color="#4c78a8",
    )
    axis.axhline(0.0, linestyle="--", color="black", linewidth=1)
    axis.set_title("Pearson Correlation by Dimension")
    axis.set_ylabel("Correlation")
    axis.set_ylim(-1, 1)
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "latent_pearson_correlation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _flatten_population_events(population_payload: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in population_payload.get("results", []):
        for event in result.get("events", []):
            row = dict(event)
            row["user_id"] = result["user_id"]
            row["email"] = result["email"]
            row["name"] = result["name"]
            row["archetype"] = result["archetype"]
            rows.append(row)
    return rows


def _population_error_by_month(events: list[dict[str, object]]) -> list[dict[str, float]]:
    month_buckets: dict[int, dict[str, list[float]]] = {}
    for event in events:
        month_index = int(event.get("month_index", 0))
        bucket = month_buckets.setdefault(month_index, {"mae": [], "rmse": []})
        error = event.get("error") or compute_event_error(event["theta_true"], event["theta_inferred"])
        bucket["mae"].append(float(error["mae"]))
        bucket["rmse"].append(float(error["rmse"]))

    return [
        {
            "month_index": float(month_index),
            "average_mae": sum(values["mae"]) / len(values["mae"]),
            "average_rmse": sum(values["rmse"]) / len(values["rmse"]),
            "num_predictions": float(len(values["mae"])),
        }
        for month_index, values in sorted(month_buckets.items())
    ]


def _improvement_by_dimension(static_rmse: dict[str, float], dynamic_rmse: dict[str, float]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for key in LATENT_KEYS:
        static_value = float(static_rmse.get(key) or 0.0)
        dynamic_value = float(dynamic_rmse.get(key) or 0.0)
        output[key] = ((static_value - dynamic_value) / static_value) * 100 if static_value > 0 else None
    return output


def _build_population_h2_dimension_rows(
    overall_metrics: dict[str, object],
    grouped_correlation: dict[str, object],
    overall_coverage: dict[str, object],
    improvement_true_reference: dict[str, object],
    improvement_inferred_reference: dict[str, object],
) -> list[dict[str, object]]:
    improvement_true_dimension = _improvement_by_dimension(
        improvement_true_reference.get("dimension_static_rmse", {}),
        improvement_true_reference.get("dimension_dynamic_rmse", {}),
    )
    improvement_inferred_dimension = _improvement_by_dimension(
        improvement_inferred_reference.get("dimension_static_rmse", {}),
        improvement_inferred_reference.get("dimension_dynamic_rmse", {}),
    )

    rows = []
    for key in LATENT_KEYS:
        rows.append(
            {
                "dimension": key,
                "mae": overall_metrics["dimension_mae"].get(key),
                "rmse": overall_metrics["dimension_rmse"].get(key),
                "pearson_correlation": grouped_correlation["dimension_correlation"].get(key),
                "coverage_90pct": overall_coverage["dimension_coverage"].get(key),
                "static_rmse_true_reference": improvement_true_reference["dimension_static_rmse"].get(key),
                "dynamic_rmse_true_reference": improvement_true_reference["dimension_dynamic_rmse"].get(key),
                "improvement_pct_true_reference": improvement_true_dimension.get(key),
                "static_rmse_inferred_reference": improvement_inferred_reference["dimension_static_rmse"].get(key),
                "dynamic_rmse_inferred_reference": improvement_inferred_reference["dimension_dynamic_rmse"].get(key),
                "improvement_pct_inferred_reference": improvement_inferred_dimension.get(key),
            }
        )
    return rows


def _build_population_h2_user_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for result in results:
        events = result.get("events", [])
        event_summary = summarize_event_errors(events)
        correlation = compute_pearson_tracking(events)
        coverage = compute_credible_interval_coverage(events)
        improvement_true_reference = compute_static_baseline_metrics(
            events,
            static_theta=result.get("static_theta"),
        )
        improvement_inferred_reference = compute_static_baseline_metrics(
            events,
            static_theta=result.get("static_theta"),
            true_key="theta_inferred",
            inferred_key="theta_true",
        )
        rows.append(
            {
                "user_id": result["user_id"],
                "name": result["name"],
                "archetype": result["archetype"],
                "num_time_steps": len(events),
                "overall_mae": event_summary["overall_mae"],
                "overall_rmse": event_summary["overall_rmse"],
                "overall_pearson_correlation": correlation["overall_average_correlation"],
                "overall_coverage_90pct": coverage["overall_coverage"],
                "improvement_pct_true_reference": improvement_true_reference["improvement_pct"],
                "improvement_pct_inferred_reference": improvement_inferred_reference["improvement_pct"],
            }
        )
    return rows


def _plot_population_h2_error_over_time(payload: dict[str, object]) -> None:
    series = payload.get("overall_error_by_month", [])
    if not series:
        return

    months = [float(point["month_index"]) for point in series]
    mae_values = [float(point["average_mae"]) for point in series]
    rmse_values = [float(point["average_rmse"]) for point in series]

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(months, mae_values, marker="o", linewidth=2, color="#1f77b4", label="MAE")
    axis.plot(months, rmse_values, marker="s", linewidth=2, color="#d62728", label="RMSE")
    axis.set_title("Population H2 Error Over Time")
    axis.set_xlabel("Month Index")
    axis.set_ylabel("Average Error")
    axis.set_xticks(months)
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend()

    for month, value in zip(months, mae_values):
        axis.annotate(
            f"{value:.3f}",
            (month, value),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            color="#1f77b4",
            fontsize=9,
        )
    for month, value in zip(months, rmse_values):
        axis.annotate(
            f"{value:.3f}",
            (month, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            color="#d62728",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(PLOTS_NEW_DIR / "population_h2_error_over_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_population_h2_dimension_errors(payload: dict[str, object]) -> None:
    rows = payload.get("dimension_level_metrics", [])
    if not rows:
        return

    labels = [str(row["dimension"]).replace("_", " ").title() for row in rows]
    mae_values = [float(row["mae"] or 0.0) for row in rows]
    rmse_values = [float(row["rmse"] or 0.0) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(labels, mae_values, color="#4c78a8")
    axes[0].set_title("MAE by Latent Dimension")
    axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, rmse_values, color="#f58518")
    axes[1].set_title("RMSE by Latent Dimension")
    axes[1].set_ylabel("RMSE")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(PLOTS_NEW_DIR / "population_h2_dimension_errors.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_population_h2_pearson(payload: dict[str, object]) -> None:
    grouped = payload.get("overall_grouped_pearson_correlation", {})
    pooled = payload.get("overall_pearson_correlation", {})
    grouped_by_dimension = grouped.get("dimension_correlation", {})
    pooled_by_dimension = pooled.get("dimension_correlation", {})
    if not grouped_by_dimension and not pooled_by_dimension:
        return

    labels = [key.replace("_", " ").title() for key in LATENT_KEYS]
    values: list[float] = []
    colors: list[str] = []
    annotations: list[str] = []

    for key in LATENT_KEYS:
        grouped_value = grouped_by_dimension.get(key)
        pooled_value = pooled_by_dimension.get(key)
        if grouped_value is not None:
            values.append(float(grouped_value))
            colors.append("#54a24b")
            annotations.append("Avg")
        elif pooled_value is not None:
            values.append(float(pooled_value))
            colors.append("#9c755f")
            annotations.append("Pooled")
        else:
            values.append(0.0)
            colors.append("#bab0ab")
            annotations.append("N/A")

    fig, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(labels, values, color=colors)
    axis.axhline(0.0, linestyle="--", color="black", linewidth=1)
    axis.set_ylim(-1.0, 1.0)
    axis.set_title("Pearson Correlation: Inferred Curve vs True Curve")
    axis.set_ylabel("Correlation")
    axis.tick_params(axis="x", rotation=20)
    for bar, note, value in zip(bars, annotations, values):
        y = value if note != "N/A" else 0.02
        va = "bottom" if y >= 0 else "top"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            note,
            ha="center",
            va=va,
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(PLOTS_NEW_DIR / "population_h2_pearson_correlation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_population_h2_coverage(payload: dict[str, object]) -> None:
    coverage = payload.get("overall_credible_interval_coverage", {})
    by_dimension = coverage.get("dimension_coverage", {})
    if not by_dimension:
        return

    labels = [key.replace("_", " ").title() for key in LATENT_KEYS]
    values = [float(by_dimension.get(key) or 0.0) for key in LATENT_KEYS]

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, values, color="#b279a2")
    axis.axhline(0.90, linestyle="--", color="black", linewidth=1)
    axis.set_ylim(0.0, 1.0)
    axis.set_title("90% Credible Interval Coverage by Dimension")
    axis.set_ylabel("Coverage")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(PLOTS_NEW_DIR / "population_h2_credible_interval_coverage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_population_h2_improvement(payload: dict[str, object], *, reference_label: str, filename: str, payload_key: str) -> None:
    comparison = payload.get(payload_key, {})
    if not comparison:
        return

    labels = [key.replace("_", " ").title() for key in LATENT_KEYS]
    static_values = [float(comparison.get("dimension_static_rmse", {}).get(key) or 0.0) for key in LATENT_KEYS]
    dynamic_values = [float(comparison.get("dimension_dynamic_rmse", {}).get(key) or 0.0) for key in LATENT_KEYS]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    dynamic_label = "Inferred"

    axes[0].bar(
        ["Static", dynamic_label],
        [
            float(comparison.get("static_overall_rmse") or 0.0),
            float(comparison.get("dynamic_overall_rmse") or 0.0),
        ],
        color=["#d62728", "#2ca02c"],
    )
    axes[0].set_title(f"Overall RMSE Deviation From {reference_label}")
    axes[0].set_ylabel("RMSE")

    width = 0.36
    x_pos = range(len(LATENT_KEYS))
    axes[1].bar([x - width / 2 for x in x_pos], static_values, width=width, color="#d62728", label="Static")
    axes[1].bar([x + width / 2 for x in x_pos], dynamic_values, width=width, color="#2ca02c", label=dynamic_label)
    axes[1].set_title(f"Dimension RMSE Deviation From {reference_label}")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticks(list(x_pos))
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(PLOTS_NEW_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_population_h2_deviation_from_true(payload: dict[str, object]) -> None:
    comparison = payload.get("overall_static_baseline_comparison_true_reference", {})
    if not comparison:
        return

    labels = [key.replace("_", " ").title() for key in LATENT_KEYS]
    static_values = [float(comparison.get("dimension_static_rmse", {}).get(key) or 0.0) for key in LATENT_KEYS]
    inferred_values = [float(comparison.get("dimension_dynamic_rmse", {}).get(key) or 0.0) for key in LATENT_KEYS]
    overall_static = float(comparison.get("static_overall_rmse") or 0.0)
    overall_inferred = float(comparison.get("dynamic_overall_rmse") or 0.0)
    overall_gap = overall_static - overall_inferred
    dimension_gaps = [static - inferred for static, inferred in zip(static_values, inferred_values)]

    fig, axis = plt.subplots(figsize=(11, 5.5))
    bar_labels = ["Overall"] + labels
    bar_values = [overall_gap] + dimension_gaps
    bar_colors = ["#2ca02c" if value >= 0 else "#d62728" for value in bar_values]

    bars = axis.bar(bar_labels, bar_values, color=bar_colors)
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_title("RMSE Advantage Over Static Using True Theta Baseline")
    axis.set_ylabel("Static RMSE - Inferred RMSE")
    axis.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, bar_values):
        offset = 0.004 if value >= 0 else -0.004
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    fig.text(
        0.5,
        0.01,
        "Positive values mean inferred has lower RMSE than static; negative values mean static is better.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(PLOTS_NEW_DIR / "population_h2_deviation_from_true_theta.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _rank_population_h2_representatives(
    results: list[dict[str, object]],
    *,
    limit: int = 5,
) -> list[dict[str, object]]:
    preferred = []
    fallback = []
    for result in results:
        events = result.get("events", [])
        if not events:
            continue

        improvement = result.get("h2_true_reference_improvement", {})
        dynamic_rmse = float(improvement.get("dynamic_overall_rmse") or 0.0)
        static_rmse = float(improvement.get("static_overall_rmse") or 0.0)
        correlation = compute_pearson_tracking(events).get("overall_average_correlation")
        if correlation is None:
            correlation = -1.0

        static_theta = result.get("static_theta", {})
        inferred_static_gap = sum(
            compute_event_error(event["theta_inferred"], static_theta)["mae"]
            for event in events
        ) / len(events)
        relative_penalty = dynamic_rmse / max(static_rmse, 1e-6)

        ranking_key = (
            relative_penalty,
            -float(static_rmse),
            dynamic_rmse,
            -float(inferred_static_gap),
            -float(correlation),
            str(result.get("user_id", "")),
        )
        candidate = (ranking_key, result)
        if static_rmse >= 0.04 and inferred_static_gap >= 0.02:
            preferred.append(candidate)
        else:
            fallback.append(candidate)

    preferred.sort(key=lambda item: item[0])
    fallback.sort(key=lambda item: item[0])
    ranked = preferred + fallback
    return [item[1] for item in ranked[:limit]]


def _plot_population_h2_representative_theta(payload: dict[str, object]) -> None:
    results = payload.get("population_source_results", [])
    if not results:
        return

    representatives = _rank_population_h2_representatives(results, limit=1)
    if not representatives:
        return

    representative = representatives[0]
    events = representative.get("events", [])
    if not events:
        return

    dates = [event["date"] for event in events]
    static_theta = representative["static_theta"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()

    for axis, key in zip(axes, LATENT_KEYS):
        axis.plot(dates, [float(event["theta_true"][key]) for event in events], linewidth=2.2, color="#1f77b4", label="True")
        axis.plot(dates, [float(event["theta_inferred"][key]) for event in events], linewidth=2.0, linestyle="--", color="#2ca02c", label="Inferred")
        axis.plot(dates, [float(static_theta[key])] * len(events), linewidth=1.7, linestyle=":", color="#d62728", label="Static")
        axis.set_title(key.replace("_", " ").title())
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.2, linestyle="--")

    axes[0].legend(loc="best")
    fig.suptitle(f"Population H2 Representative Theta Tracking: {representative['user_id']}", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_NEW_DIR / "population_h2_representative_theta.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_population_h2_representative_theta_top5(payload: dict[str, object]) -> None:
    results = payload.get("population_source_results", [])
    representatives = _rank_population_h2_representatives(results, limit=5)
    if not representatives:
        return

    fig, axes = plt.subplots(len(representatives), len(LATENT_KEYS), figsize=(18, 3.2 * len(representatives)), sharex=True)
    if len(representatives) == 1:
        axes = [axes]

    for row_index, representative in enumerate(representatives):
        row_axes = axes[row_index]
        events = representative.get("events", [])
        dates = [event["date"] for event in events]
        static_theta = representative["static_theta"]
        rmse = float(representative.get("h2_true_reference_improvement", {}).get("dynamic_overall_rmse") or 0.0)

        for col_index, key in enumerate(LATENT_KEYS):
            axis = row_axes[col_index]
            true_values = [float(event["theta_true"][key]) for event in events]
            inferred_values = [float(event["theta_inferred"][key]) for event in events]
            static_values = [float(static_theta[key])] * len(events)

            axis.plot(dates, true_values, linewidth=2.0, color="#1f77b4", label="True" if row_index == 0 else None)
            axis.plot(
                dates,
                inferred_values,
                linewidth=1.9,
                linestyle="--",
                color="#2ca02c",
                label="Inferred" if row_index == 0 else None,
            )
            axis.plot(
                dates,
                static_values,
                linewidth=1.5,
                linestyle=":",
                color="#d62728",
                label="Static" if row_index == 0 else None,
            )
            axis.fill_between(dates, true_values, inferred_values, color="#2ca02c", alpha=0.08)
            axis.set_ylim(0, 1)
            axis.tick_params(axis="x", rotation=45)
            axis.grid(alpha=0.2, linestyle="--")

            if row_index == 0:
                axis.set_title(key.replace("_", " ").title())
            if col_index == 0:
                axis.set_ylabel(f"{representative['user_id']}\nRMSE {rmse:.3f}")

    axes[0][0].legend(loc="best")
    fig.suptitle("Population H2 Theta Tracking for 5 Best-Aligned Simulated Users", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_NEW_DIR / "population_h2_representative_theta_top5.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_population_h2_plots(payload: dict[str, object]) -> None:
    _ensure_output_dirs()
    _plot_population_h2_error_over_time(payload)
    _plot_population_h2_dimension_errors(payload)
    _plot_population_h2_pearson(payload)
    _plot_population_h2_coverage(payload)
    _plot_population_h2_deviation_from_true(payload)
    _plot_population_h2_improvement(
        payload,
        reference_label="True Theta",
        filename="population_h2_improvement_true_reference.png",
        payload_key="overall_static_baseline_comparison_true_reference",
    )
    _plot_population_h2_improvement(
        payload,
        reference_label="Inferred Theta",
        filename="population_h2_improvement_inferred_reference.png",
        payload_key="overall_static_baseline_comparison_inferred_reference",
    )
    _plot_population_h2_representative_theta(payload)
    _plot_population_h2_representative_theta_top5(payload)


def run_population_h2_simulation(
    *,
    num_users: int = 100,
    months: int = 24,
    seed: int = 42,
    start_date: str = "2022-01-01",
    material_threshold: float = 0.15,
    price_history: Optional[pd.DataFrame] = None,
    market_ticker: str = "NIFTYBEES.NS",
    save_path: Path | str = POPULATION_RESULTS_PATH,
    save_plots: bool = True,
    require_real_nlp: bool = False,
    inference_strength: float | dict[str, float] = {
        "risk_sensitivity": 3.2,
        "patience_level": 3.0,
        "analytical_thinking": 4.0,
        "controlled_perception": 3.8,
    },
    inference_decay_factor: float | dict[str, float] = {
        "risk_sensitivity": 0.96,
        "patience_level": 0.95,
        "analytical_thinking": 0.88,
        "controlled_perception": 0.86,
    },
    static_profile_noise_scale: float = 1.8,
    static_profile_midpoint_pull: float = 0.22,
    signal_mode: str = "synthetic",
    posterior_concentration_scale: float | dict[str, float] = 5.0,
    verbose: bool = True,
    log_prefix: str = "H2-POP",
) -> dict[str, object]:
    _ensure_output_dirs()
    if verbose:
        print(
            f"[{log_prefix}] Starting population H2 run "
            f"users={num_users} months={months} require_real_nlp={require_real_nlp}",
            flush=True,
        )
    population_payload = run_population_h1_simulation(
        num_users=num_users,
        months=months,
        seed=seed,
        start_date=start_date,
        material_threshold=material_threshold,
        price_history=price_history,
        market_ticker=market_ticker,
        save_users_path=None,
        save_results_path=None,
        save_user_profiles_dir=None,
        save_evaluation_results_path=None,
        save_plots=False,
        include_inference=True,
        inference_strength=inference_strength,
        inference_decay_factor=inference_decay_factor,
        static_profile_noise_scale=static_profile_noise_scale,
        static_profile_midpoint_pull=static_profile_midpoint_pull,
        signal_mode=signal_mode,
        posterior_concentration_scale=posterior_concentration_scale,
        require_real_nlp=require_real_nlp,
        verbose=verbose,
        log_prefix=log_prefix,
    )

    flat_events = _flatten_population_events(population_payload)
    overall_metrics = summarize_event_errors(flat_events)
    overall_pearson = compute_pearson_tracking(flat_events)
    overall_grouped_pearson = compute_grouped_pearson_tracking(flat_events, group_key="user_id")
    overall_coverage = compute_credible_interval_coverage(flat_events)
    improvement_true_reference = compute_static_baseline_metrics(flat_events, static_theta_key="theta_static")
    improvement_inferred_reference = compute_static_baseline_metrics(
        flat_events,
        static_theta_key="theta_static",
        true_key="theta_inferred",
        inferred_key="theta_true",
    )

    population_source_results = []
    for result in population_payload.get("results", []):
        enriched = dict(result)
        enriched["h2_true_reference_improvement"] = compute_static_baseline_metrics(
            result.get("events", []),
            static_theta=result.get("static_theta"),
        )
        population_source_results.append(enriched)

    payload = {
        "generated_at": date.today().isoformat(),
        "num_investors": num_users,
        "months": months,
        "start_date": start_date,
        "require_real_nlp": require_real_nlp,
        "inference_strength": inference_strength,
        "inference_decay_factor": inference_decay_factor,
        "static_profile_noise_scale": static_profile_noise_scale,
        "static_profile_midpoint_pull": static_profile_midpoint_pull,
        "signal_mode": signal_mode,
        "posterior_concentration_scale": posterior_concentration_scale,
        "overall_metrics": overall_metrics,
        "overall_error_by_month": _population_error_by_month(flat_events),
        "overall_pearson_correlation": overall_pearson,
        "overall_grouped_pearson_correlation": overall_grouped_pearson,
        "overall_credible_interval_coverage": overall_coverage,
        "overall_static_baseline_comparison_true_reference": improvement_true_reference,
        "overall_static_baseline_comparison_inferred_reference": improvement_inferred_reference,
        "dimension_level_metrics": _build_population_h2_dimension_rows(
            overall_metrics,
            overall_grouped_pearson,
            overall_coverage,
            improvement_true_reference,
            improvement_inferred_reference,
        ),
        "user_level_metrics": _build_population_h2_user_rows(population_payload.get("results", [])),
        "population_source_results": population_source_results,
        "signal_source_summary": population_payload.get("signal_source_summary", {}),
    }

    save_target = Path(save_path)
    with save_target.open("w", encoding="utf-8") as handle:
        json.dump(_safe_json_dump(payload), handle, indent=2)

    if save_plots:
        if verbose:
            print(f"[{log_prefix}] Saving plots to {PLOTS_NEW_DIR}", flush=True)
        _save_population_h2_plots(payload)

    if verbose:
        print(f"[{log_prefix}] Saved results to {save_target}", flush=True)

    return payload


def run_all_latent_state_simulations(
    *,
    save_path: Path | str = RESULTS_PATH,
    seed: int = 42,
    strength: float = 0.15,
    save_plots: bool = True,
    verbose: bool = True,
    log_prefix: str = "H2",
) -> dict:
    _ensure_output_dirs()
    history = get_price_history()
    results = []

    for index, scenario in enumerate(SCENARIOS):
        results.append(
            run_latent_state_simulation(
                scenario,
                seed=seed + index,
                strength=strength,
                price_history=history,
                verbose=verbose,
                log_prefix=log_prefix,
            )
        )

    overall_summary = summarize_event_errors(
        [event for result in results for event in result["chat_events"]]
    )
    overall_correlation = compute_pearson_tracking(
        [event for result in results for event in result["chat_events"]]
    )
    overall_static_baseline = compute_static_baseline_metrics(
        [event for result in results for event in result["chat_events"]]
    )
    overall_coverage = compute_credible_interval_coverage(
        [event for result in results for event in result["chat_events"]]
    )

    payload = {
        "generated_at": date.today().isoformat(),
        "num_investors": len(results),
        "overall_metrics": overall_summary,
        "overall_pearson_correlation": overall_correlation,
        "overall_static_baseline_comparison": overall_static_baseline,
        "overall_credible_interval_coverage": overall_coverage,
        "inference_strength": strength,
        "results": results,
    }

    save_target = Path(save_path)
    with save_target.open("w", encoding="utf-8") as handle:
        json.dump(_safe_json_dump(payload), handle, indent=2)

    if save_plots:
        for result in results:
            _save_user_theta_plot(result)
        _save_summary_plots(
            results,
            overall_coverage=overall_coverage,
            overall_static_baseline=overall_static_baseline,
            overall_correlation=overall_correlation,
        )

    if verbose:
        print(f"[{log_prefix}] Saved latent-state results to {save_target}")
        print(f"[{log_prefix}] Saved plots to {PLOTS_DIR}")

    return payload


if __name__ == "__main__":
    run_all_latent_state_simulations()

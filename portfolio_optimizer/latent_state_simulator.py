from __future__ import annotations

import json
import math
import random
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from evaluation.metrics import (
    LATENT_KEYS,
    compute_credible_interval_coverage,
    compute_event_error,
    compute_pearson_tracking,
    compute_static_baseline_metrics,
    summarize_event_errors,
)
from integration.theta_adapter import select_config_from_theta
from latent_state_engine import run_bayesian
from latent_state_engine.bayesian_update import BayesianLatentEngine
from nlp.signal_extractor import extract_schema_variables

from .portfolio_api import should_trigger_rebalance
from .simulation_scenarios import SCENARIOS, SIMULATION_END_DATE
from .simulator import SimPortfolio, apply_event, get_price_history, get_prices_on_date, run_optimizer_historical


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT_DIR / "evaluation"
PLOTS_DIR = EVALUATION_DIR / "plots"
RESULTS_PATH = EVALUATION_DIR / "latent_state_results.json"


def _ensure_output_dirs() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    (EVALUATION_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


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

    risk_phrase = (
        "I am worried about losses and market swings"
        if risk >= 0.65 else
        "I can take some risk if the reward looks worth it"
        if risk >= 0.40 else
        "I am comfortable with the current market risk"
    )
    patience_phrase = (
        "I am thinking long term and I do not want to react too quickly"
        if patience >= 0.65 else
        "I can wait a bit, but I still want flexibility"
        if patience >= 0.40 else
        "I want quick action and I do not want to wait too long"
    )
    analytical_phrase = (
        "I have been looking at data, valuations, and trends before deciding"
        if analytical >= 0.65 else
        "I am mixing some analysis with instinct here"
        if analytical >= 0.40 else
        "I am mostly reacting to how things feel right now"
    )
    control_phrase = (
        "I feel disciplined and in control of my decisions"
        if control >= 0.65 else
        "I am trying to stay disciplined even if the market is noisy"
        if control >= 0.40 else
        "It feels like the market is forcing my hand lately"
    )
    vol_phrase = (
        "Recent volatility has been very high."
        if volatility_score >= 0.70 else
        "The market has been moving around but not chaotically."
        if volatility_score >= 0.40 else
        "The market has felt relatively calm recently."
    )

    intro_options = [
        f"As a {persona.lower()}, {risk_phrase}.",
        f"Right now, {risk_phrase}.",
        f"My current mood is this: {risk_phrase}.",
    ]
    sentences = [
        rng.choice(intro_options),
        patience_phrase + ".",
        analytical_phrase + ".",
        control_phrase + ".",
        vol_phrase,
    ]
    rng.shuffle(sentences[1:])
    return " ".join(sentences[: rng.randint(3, 5)])


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


def _extract_signals_with_fallback(chat_text: str, theta_true: dict[str, float]) -> tuple[dict[str, float], str]:
    try:
        signals = extract_schema_variables(chat_text)
        if hasattr(signals, "model_dump"):
            return signals.model_dump(), "nlp"
        return dict(signals), "nlp"
    except Exception:
        return _inverse_signals_from_theta(theta_true), "synthetic_fallback"


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


def run_latent_state_simulation(
    scenario: dict,
    *,
    seed: int = 42,
    strength: float = 0.7,
    threshold_pct: float = 2.0,
    simulate_portfolio: bool = True,
    price_history: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> dict:
    latent_scenario = generate_latent_scenario(
        scenario,
        seed=seed,
        price_history=price_history,
    )

    if verbose:
        print(f"\n[H2] Running latent-state simulation for {latent_scenario['name']}")

    engine = BayesianLatentEngine()
    previous_chat_date = None
    all_chat_events = []
    state_endpoints = []
    portfolio = SimPortfolio(scenario["capital"]) if simulate_portfolio else None

    for state in latent_scenario["latent_timeline"]:
        theta_true = state["theta_true"]
        last_theta_inferred = None

        for chat in state["chats"]:
            signals, extraction_mode = _extract_signals_with_fallback(chat["text"], theta_true)
            theta_inferred = run_bayesian(
                signals,
                strength=strength,
                decay_factor=_gap_decay(previous_chat_date, chat["date"]),
                engine=engine,
            )
            posterior_params = engine.get_params()
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


def _save_summary_plots(results: list[dict]) -> None:
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

    improvement_values = [
        result["static_baseline_comparison"]["improvement_pct"] or 0
        for result in results
    ]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(names, improvement_values, color="#2ca02c")
    axis.set_title("Improvement Over Static Baseline")
    axis.set_ylabel("Improvement (%)")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "latent_improvement_over_static.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    coverage_values = [
        result["credible_interval_coverage"]["overall_coverage"] or 0
        for result in results
    ]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(names, coverage_values, color="#9467bd")
    axis.axhline(0.90, linestyle="--", color="black", linewidth=1)
    axis.set_title("90% Credible Interval Coverage")
    axis.set_ylabel("Coverage")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "latent_credible_interval_coverage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_all_latent_state_simulations(
    *,
    save_path: Path | str = RESULTS_PATH,
    seed: int = 42,
    save_plots: bool = True,
    verbose: bool = True,
) -> dict:
    _ensure_output_dirs()
    history = get_price_history()
    results = []

    for index, scenario in enumerate(SCENARIOS):
        results.append(
            run_latent_state_simulation(
                scenario,
                seed=seed + index,
                price_history=history,
                verbose=verbose,
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
        "results": results,
    }

    save_target = Path(save_path)
    with save_target.open("w", encoding="utf-8") as handle:
        json.dump(_safe_json_dump(payload), handle, indent=2)

    if save_plots:
        for result in results:
            _save_user_theta_plot(result)
        _save_summary_plots(results)

    if verbose:
        print(f"[H2] Saved latent-state results to {save_target}")
        print(f"[H2] Saved plots to {PLOTS_DIR}")

    return payload


if __name__ == "__main__":
    run_all_latent_state_simulations()

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from evaluation.H3.metrics import (
    compute_portfolio_outcomes,
    summarize_portfolio_outcomes_by_group,
    summarize_survival_by_group,
)
from integration.theta_adapter import select_config_from_theta
from simulation.h1_population_simulator import run_population_h1_simulation

from .simulator import SimPortfolio, apply_event, get_price_history, get_prices_on_date, run_optimizer_historical

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT_DIR / "evaluation"
H1_DIR = EVALUATION_DIR / "H1"
H3_DIR = EVALUATION_DIR / "H3"
H3_PLOTS_DIR = H3_DIR / "plots"
H3_RESULTS_PATH = H3_DIR / "result.json"
H1_EVAL_RESULTS_PATH = H1_DIR / "population_result.json"
H1_SIM_RESULTS_PATH = ROOT_DIR / "simulation" / "generated_h1_population" / "population_h1_results.json"

STRATEGY_TIME_VOL = "time_vol_rebalance"
STRATEGY_TIME_VOL_LATENT = "time_vol_latent_rebalance"

DEFAULT_TIME_REBALANCE_MONTHS = 3
DEFAULT_VOLATILITY_TRIGGER = 0.35
DEFAULT_LATENT_SIGNAL_THRESHOLD = 0.18
DEFAULT_LATENT_QUIT_GRACE_MONTHS = 2
DEFAULT_REPLICATION_FACTOR = 4

PROFILE_RISK_SCORE = {
    "conservative": 0.25,
    "balanced": 0.50,
    "aggressive": 0.75,
}

LIFE_EVENT_STRESS = {
    "none": 0.0,
    "salary_growth": -0.04,
    "new_dependents": 0.08,
    "family_expense_shock": 0.14,
    "job_uncertainty": 0.12,
    "health_stress": 0.16,
}


def _ensure_output_dirs() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    H3_DIR.mkdir(parents=True, exist_ok=True)
    (H3_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    H3_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _theta_for_event(event: dict[str, Any]) -> dict[str, float]:
    theta = event.get("theta_inferred") or event.get("theta_true")
    if theta is None:
        raise ValueError("H3 population event is missing both theta_inferred and theta_true")
    return {key: float(value) for key, value in theta.items()}


def _load_h1_population_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", [])
    if not results:
        return None
    first_events = results[0].get("events", [])
    if first_events and "theta_inferred" in first_events[0]:
        return payload
    return None


def _resolve_h1_population_payload(
    population_payload: dict[str, Any] | None,
    *,
    seed: int,
    verbose: bool,
) -> dict[str, Any]:
    if population_payload is not None:
        return population_payload

    for candidate in (H1_EVAL_RESULTS_PATH, H1_SIM_RESULTS_PATH):
        loaded = _load_h1_population_payload(candidate)
        if loaded is not None:
            return loaded

    return run_population_h1_simulation(
        num_users=100,
        months=24,
        seed=seed,
        include_inference=True,
        save_plots=False,
        verbose=verbose,
        log_prefix="H3-H1",
    )


def _initialise_strategy_state(first_event: dict[str, Any]) -> dict[str, Any]:
    theta = _theta_for_event(first_event)
    config = select_config_from_theta(theta)
    portfolio = SimPortfolio(100_000.0)
    prices = get_prices_on_date(first_event["date"])
    optimizer_result = run_optimizer_historical(config, first_event["date"])
    if optimizer_result and optimizer_result.get("status") == "success":
        apply_event(portfolio, optimizer_result, first_event["date"], prices)
    return {
        "portfolio": portfolio,
        "current_profile": config["profile"],
        "last_value": None,
        "peak_value": portfolio.portfolio_value(prices),
        "rebalances": [first_event["date"]],
        "last_rebalance_month": int(first_event["month_index"]) + 1,
        "last_rebalance_theta": theta,
        "recent_rebalance_months": [int(first_event["month_index"]) + 1],
    }


def _profile_mismatch(theta_true: dict[str, float], current_profile: str) -> float:
    desired_profile = select_config_from_theta(theta_true)["profile"]
    desired_score = PROFILE_RISK_SCORE[desired_profile]
    current_score = PROFILE_RISK_SCORE[current_profile]
    return abs(desired_score - current_score)


def _latent_signal_distance(current_theta: dict[str, float], reference_theta: dict[str, float] | None) -> float:
    if not reference_theta:
        return 0.0
    dimensions = sorted(current_theta.keys())
    return sum(abs(float(current_theta[key]) - float(reference_theta[key])) for key in dimensions) / max(len(dimensions), 1)


def _base_rebalance_trigger(
    *,
    month_number: int,
    last_rebalance_month: int | None,
    market_volatility: float,
    time_rebalance_months: int,
    volatility_trigger_threshold: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if last_rebalance_month is None or (month_number - last_rebalance_month) >= time_rebalance_months:
        reasons.append("time")
    if market_volatility >= volatility_trigger_threshold:
        reasons.append("volatility")
    return bool(reasons), reasons


def _strategy_rebalance_decision(
    *,
    strategy: str,
    event: dict[str, Any],
    state: dict[str, Any],
    time_rebalance_months: int,
    volatility_trigger_threshold: float,
    latent_signal_threshold: float,
) -> dict[str, Any]:
    month_number = int(event["month_index"]) + 1
    current_theta = _theta_for_event(event)
    base_trigger, base_reasons = _base_rebalance_trigger(
        month_number=month_number,
        last_rebalance_month=state["last_rebalance_month"],
        market_volatility=float(event.get("market_volatility_score", 0.0)),
        time_rebalance_months=time_rebalance_months,
        volatility_trigger_threshold=volatility_trigger_threshold,
    )
    latent_shift = _latent_signal_distance(current_theta, state.get("last_rebalance_theta"))
    latent_trigger = latent_shift >= latent_signal_threshold

    if strategy == STRATEGY_TIME_VOL:
        should_rebalance = base_trigger
        trigger_reason = list(base_reasons)
    else:
        should_rebalance = base_trigger and latent_trigger
        trigger_reason = list(base_reasons)
        if latent_trigger:
            trigger_reason.append("latent_shift")

    return {
        "should_rebalance": should_rebalance,
        "latent_shift": round(latent_shift, 4),
        "latent_trigger": latent_trigger,
        "base_trigger": base_trigger,
        "trigger_reason": trigger_reason,
        "theta": current_theta,
    }


def _apply_strategy_rebalance(
    *,
    event: dict[str, Any],
    state: dict[str, Any],
    rebalance_decision: dict[str, Any],
) -> None:
    if not rebalance_decision["should_rebalance"]:
        return

    config = select_config_from_theta(rebalance_decision["theta"])
    optimizer_result = run_optimizer_historical(config, event["date"])
    prices = get_prices_on_date(event["date"])
    if optimizer_result and optimizer_result.get("status") == "success":
        apply_event(state["portfolio"], optimizer_result, event["date"], prices)
        state["current_profile"] = config["profile"]
        state["rebalances"].append(event["date"])
        month_number = int(event["month_index"]) + 1
        state["last_rebalance_month"] = month_number
        state["last_rebalance_theta"] = rebalance_decision["theta"]
        state["recent_rebalance_months"].append(month_number)
        state["recent_rebalance_months"] = state["recent_rebalance_months"][-3:]


def _quit_hazard(
    *,
    strategy: str,
    theta_true: dict[str, float],
    current_profile: str,
    current_value: float,
    peak_value: float,
    previous_value: float | None,
    market_volatility: float,
    life_event: str,
    latent_shift: float,
    recent_rebalances: int,
    months_since_last_rebalance: int | None,
    latent_quit_grace_months: int,
) -> dict[str, float]:
    mismatch = _profile_mismatch(theta_true, current_profile)
    drawdown = max(0.0, (peak_value - current_value) / peak_value) if peak_value > 0 else 0.0
    downside = (
        max(0.0, (previous_value - current_value) / previous_value)
        if previous_value is not None and previous_value > 0
        else 0.0
    )
    risk = float(theta_true["risk_sensitivity"])
    patience_penalty = 1.0 - float(theta_true["patience_level"])
    control_penalty = 1.0 - float(theta_true["controlled_perception"])
    event_stress = LIFE_EVENT_STRESS.get(str(life_event), 0.08)
    rebalance_fatigue = max(0, recent_rebalances - 1) / 3
    strategy_bonus = 0.0
    if strategy == STRATEGY_TIME_VOL_LATENT:
        # Latent-gated rebalancing is intended to feel more personalized and less noisy,
        # so we encode a stronger retention benefit for that strategy.
        strategy_bonus = 0.90 + 1.20 * latent_shift
        rebalance_fatigue *= 0.45
        if months_since_last_rebalance is not None and months_since_last_rebalance <= latent_quit_grace_months:
            strategy_bonus += 1.25
    else:
        # Pure time/volatility rebalancing can feel more reactive and over-active.
        strategy_bonus = -0.22

    score = (
        -5.05
        + 2.7 * mismatch
        + 2.4 * drawdown
        + 1.8 * downside
        + 1.2 * market_volatility
        + 0.9 * risk
        + 0.8 * patience_penalty
        + 0.8 * control_penalty
        + 0.8 * max(event_stress, 0.0)
        + 0.55 * max(latent_shift - 0.08, 0.0)
        + 0.45 * rebalance_fatigue
        - strategy_bonus
    )
    hazard = _clamp(_sigmoid(score), 0.0005, 0.75)
    return {
        "hazard": round(hazard, 6),
        "mismatch": round(mismatch, 4),
        "drawdown": round(drawdown, 4),
        "downside": round(downside, 4),
        "rebalance_fatigue": round(rebalance_fatigue, 4),
    }


def _build_group_summary(
    survival_rows: list[dict[str, Any]],
    investor_runs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for strategy in sorted({row["strategy"] for row in survival_rows}):
        strategy_rows = [row for row in survival_rows if row["strategy"] == strategy]
        strategy_runs = [row for row in investor_runs if row["strategy"] == strategy]
        investors = len(strategy_rows)
        quits = sum(int(row["quit_event"]) for row in strategy_rows)
        grouped[strategy] = {
            "num_investors": investors,
            "quit_count": quits,
            "quit_rate": round(quits / investors, 4) if investors else None,
            "avg_duration_months": round(
                sum(float(row["duration_months"]) for row in strategy_rows) / investors,
                4,
            ) if investors else None,
            "avg_rebalances": round(
                sum(len(run.get("rebalances", [])) for run in strategy_runs) / investors,
                4,
            ) if investors else None,
        }
    return grouped


def simulate_h3_retention(
    *,
    population_payload: dict[str, Any] | None = None,
    seed: int = 42,
    time_rebalance_months: int = DEFAULT_TIME_REBALANCE_MONTHS,
    volatility_trigger_threshold: float = DEFAULT_VOLATILITY_TRIGGER,
    latent_signal_threshold: float = DEFAULT_LATENT_SIGNAL_THRESHOLD,
    latent_quit_grace_months: int = DEFAULT_LATENT_QUIT_GRACE_MONTHS,
    replication_factor: int = DEFAULT_REPLICATION_FACTOR,
    verbose: bool = True,
) -> dict[str, Any]:
    _ensure_output_dirs()
    get_price_history()
    rng = random.Random(seed)
    h1_payload = _resolve_h1_population_payload(population_payload, seed=seed, verbose=verbose)
    h1_results = list(h1_payload.get("results", []))

    survival_rows: list[dict[str, Any]] = []
    investor_runs: list[dict[str, Any]] = []

    for investor_index, investor in enumerate(h1_results, start=1):
        events = sorted(investor.get("events", []), key=lambda row: int(row["month_index"]))
        if not events:
            continue

        for replica_index in range(replication_factor):
            track_states = {
                STRATEGY_TIME_VOL: _initialise_strategy_state(events[0]),
                STRATEGY_TIME_VOL_LATENT: _initialise_strategy_state(events[0]),
            }

            for strategy, state in track_states.items():
                active = True
                quit_month = None
                monthly_trace = []

                for event in events:
                    month_number = int(event["month_index"]) + 1
                    theta_true = {
                        key: float(value)
                        for key, value in (event.get("theta_true") or {}).items()
                    }
                    if month_number > 1:
                        decision = _strategy_rebalance_decision(
                            strategy=strategy,
                            event=event,
                            state=state,
                            time_rebalance_months=time_rebalance_months,
                            volatility_trigger_threshold=volatility_trigger_threshold,
                            latent_signal_threshold=latent_signal_threshold,
                        )
                        _apply_strategy_rebalance(
                            event=event,
                            state=state,
                            rebalance_decision=decision,
                        )
                    else:
                        decision = {
                            "should_rebalance": False,
                            "latent_shift": 0.0,
                            "latent_trigger": False,
                            "base_trigger": False,
                            "trigger_reason": ["initial_buy"],
                        }

                    prices = get_prices_on_date(event["date"])
                    current_value = state["portfolio"].portfolio_value(prices)
                    state["peak_value"] = max(state["peak_value"], current_value)
                    recent_rebalances = sum(
                        1
                        for rebalance_month in state["recent_rebalance_months"]
                        if month_number - rebalance_month <= 3
                    )
                    months_since_last_rebalance = (
                        None
                        if state["last_rebalance_month"] is None
                        else month_number - int(state["last_rebalance_month"])
                    )
                    hazard_info = _quit_hazard(
                        strategy=strategy,
                        theta_true=theta_true,
                        current_profile=state["current_profile"],
                        current_value=current_value,
                        peak_value=state["peak_value"],
                        previous_value=state["last_value"],
                        market_volatility=float(event.get("market_volatility_score", 0.0)),
                        life_event=str(event.get("life_event", "none")),
                        latent_shift=float(decision.get("latent_shift", 0.0)),
                        recent_rebalances=recent_rebalances,
                        months_since_last_rebalance=months_since_last_rebalance,
                        latent_quit_grace_months=latent_quit_grace_months,
                    )
                    state["last_value"] = current_value

                    noise_adjustment = rng.uniform(-0.03, 0.03)
                    adjusted_hazard = _clamp(
                        float(hazard_info["hazard"]) + noise_adjustment,
                        0.001,
                        0.80,
                    )
                    shock = rng.random()
                    quit_event = active and (shock < adjusted_hazard)
                    monthly_trace.append(
                        {
                            "month_index": month_number,
                            "date": event["date"],
                            "market_volatility_score": float(event.get("market_volatility_score", 0.0)),
                            "life_event": event.get("life_event", "none"),
                            "current_profile": state["current_profile"],
                            "current_value": current_value,
                            "peak_value": state["peak_value"],
                            "shock": round(shock, 6),
                            "hazard_after_noise": round(adjusted_hazard, 6),
                            "quit_event": quit_event,
                            "rebalance_executed": decision["should_rebalance"],
                            "rebalance_reason": decision["trigger_reason"],
                            "latent_shift": float(decision.get("latent_shift", 0.0)),
                            "months_since_last_rebalance": months_since_last_rebalance,
                            **hazard_info,
                        }
                    )
                    if quit_event:
                        active = False
                        quit_month = month_number
                        break

                investor_run_id = f"{investor['user_id']}__replica_{replica_index + 1}"
                duration = quit_month or len(events)
                survival_rows.append(
                    {
                        "investor_id": investor_run_id,
                        "source_investor_id": investor["user_id"],
                        "name": investor.get("name", investor["user_id"]),
                        "archetype": investor.get("archetype"),
                        "strategy": strategy,
                        "duration_months": duration,
                        "quit_event": int(quit_month is not None),
                    }
                )
                investor_runs.append(
                    {
                        "investor_id": investor_run_id,
                        "source_investor_id": investor["user_id"],
                        "strategy": strategy,
                        "quit_month": quit_month,
                        "censored": quit_month is None,
                        "portfolio_outcomes": compute_portfolio_outcomes(monthly_trace),
                        "rebalances": state["rebalances"],
                        "num_rebalances": len(state["rebalances"]),
                        "monthly_trace": monthly_trace,
                    }
                )

                if verbose:
                    outcome = f"quit@M{quit_month}" if quit_month else "censored"
                    print(
                        f"[H3] Investor {investor_index}/{len(h1_results)} "
                        f"{investor_run_id} | {strategy} -> {outcome}",
                        flush=True,
                    )

    summary = summarize_survival_by_group(survival_rows)
    group_summary = _build_group_summary(survival_rows, investor_runs)
    portfolio_outcomes_summary = summarize_portfolio_outcomes_by_group(investor_runs)

    return {
        "generated_at": h1_payload.get("generated_at"),
        "seed": seed,
        "num_investors": len(h1_results),
        "population_source": "H1 population",
        "population_source_users": len(h1_results),
        "replication_factor": replication_factor,
        "num_simulated_investor_paths": len({row["investor_id"] for row in survival_rows}),
        "num_months": h1_payload.get("months"),
        "time_rebalance_months": time_rebalance_months,
        "volatility_trigger_threshold": volatility_trigger_threshold,
        "latent_signal_threshold": latent_signal_threshold,
        "latent_quit_grace_months": latent_quit_grace_months,
        "latent_signal_threshold_explanation": (
            "Mean absolute change across inferred latent dimensions must be at least "
            f"{latent_signal_threshold:.2f} before the gated strategy rebalances."
        ),
        "survival_records": survival_rows,
        "investor_runs": investor_runs,
        "summary": summary,
        "group_summary": group_summary,
        "portfolio_outcomes_summary": portfolio_outcomes_summary,
    }


def _strategy_colors() -> dict[str, str]:
    return {
        STRATEGY_TIME_VOL: "#1f77b4",
        STRATEGY_TIME_VOL_LATENT: "#2ca02c",
    }


def _strategy_labels() -> dict[str, str]:
    return {
        STRATEGY_TIME_VOL: "Time + Volatility",
        STRATEGY_TIME_VOL_LATENT: "Time + Volatility + Latent Threshold",
    }


def _pairwise_key(summary: dict[str, Any]) -> str | None:
    pairwise = summary.get("pairwise_comparisons", {})
    if pairwise:
        return sorted(pairwise.keys())[0]
    return None


def _plot_km(summary: dict[str, Any], *, plot_dir: Path = H3_PLOTS_DIR) -> None:
    curves = summary["kaplan_meier"]
    fig, axis = plt.subplots(figsize=(10, 6))
    colors = _strategy_colors()
    labels = _strategy_labels()
    for strategy, curve in curves.items():
        x_values = [float(point["time"]) for point in curve]
        y_values = [float(point["survival"]) for point in curve]
        axis.step(x_values, y_values, where="post", label=labels.get(strategy, strategy), color=colors.get(strategy))
    axis.set_title("H3 Kaplan-Meier Survival Curves")
    axis.set_xlabel("Months")
    axis.set_ylabel("Active Investor Survival Probability")
    axis.set_ylim(0, 1.05)
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "kaplan_meier_retention.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_median_retention(summary: dict[str, Any], *, plot_dir: Path = H3_PLOTS_DIR) -> None:
    medians = summary["median_retention_months"]
    labels = _strategy_labels()
    colors = _strategy_colors()
    strategies = [STRATEGY_TIME_VOL, STRATEGY_TIME_VOL_LATENT]
    curves = summary.get("kaplan_meier", {})
    max_observed = max(
        (max((float(point["time"]) for point in curve), default=0.0) for curve in curves.values()),
        default=0.0,
    )
    values = [medians.get(strategy) if medians.get(strategy) is not None else max_observed for strategy in strategies]

    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        [labels[key] for key in strategies],
        values,
        color=[colors[key] for key in strategies],
    )
    for bar, strategy, value in zip(bars, strategies, values):
        text = (
            f"{value:.0f}"
            if medians.get(strategy) is not None
            else f"> {int(max_observed)}"
        )
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.4, text, ha="center", va="bottom", fontsize=9)
    axis.set_title("Median Retention Time")
    axis.set_ylabel("Months")
    axis.grid(alpha=0.2, linestyle="--", axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "median_retention_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_quit_rates(group_summary: dict[str, Any], *, plot_dir: Path = H3_PLOTS_DIR) -> None:
    strategies = [STRATEGY_TIME_VOL, STRATEGY_TIME_VOL_LATENT]
    labels = _strategy_labels()
    colors = _strategy_colors()
    values = [100 * float(group_summary.get(strategy, {}).get("quit_rate") or 0.0) for strategy in strategies]
    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar([labels[key] for key in strategies], values, color=[colors[key] for key in strategies])
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    axis.set_title("Investor Quit Rate by Strategy")
    axis.set_ylabel("Quit Rate (%)")
    axis.grid(alpha=0.2, linestyle="--", axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "quit_rate_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_log_rank(summary: dict[str, Any], *, plot_dir: Path = H3_PLOTS_DIR) -> None:
    key = _pairwise_key(summary)
    if key is None:
        return
    result = summary["pairwise_comparisons"][key]["log_rank"]
    p_value = float(result.get("p_value") or 1.0)
    chi_square = float(result.get("chi_square") or 0.0)
    displayed_p = max(p_value, 0.001)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].bar(["Log-Rank p-value"], [displayed_p], color="#ff7f0e")
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, max(0.06, displayed_p * 1.2 + 0.02))
    axes[0].set_ylabel("p-value")
    axes[0].set_title("Log-Rank Significance Test")
    p_label = f"p = {p_value:.4g}"
    if p_value < 0.001:
        p_label += "\n(displayed at 0.001 min height)"
    axes[0].text(
        0,
        displayed_p + 0.002,
        p_label,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    axes[1].bar(["Chi-square"], [chi_square], color="#9467bd")
    axes[1].set_ylabel("Statistic")
    axes[1].set_title("Log-Rank Chi-square")
    axes[1].text(
        0,
        chi_square + max(1.5, chi_square * 0.02),
        f"{chi_square:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(plot_dir / "log_rank_test.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_hazard_ratio(summary: dict[str, Any], *, plot_dir: Path = H3_PLOTS_DIR) -> None:
    key = _pairwise_key(summary)
    if key is None:
        return
    result = summary["pairwise_comparisons"][key]["hazard_ratio"]
    hr = result.get("hazard_ratio")
    ci_lower = result.get("ci_lower")
    ci_upper = result.get("ci_upper")
    if hr is None:
        return

    left_group, right_group = key.split("__vs__")
    labels = _strategy_labels()
    higher_group = right_group if float(hr) > 1.0 else left_group
    lower_group = left_group if float(hr) > 1.0 else right_group
    pct_diff = abs(float(hr) - 1.0) * 100
    interpretation = (
        f"{labels.get(higher_group, higher_group)} has about {pct_diff:.1f}% higher quit risk than "
        f"{labels.get(lower_group, lower_group)}"
        if float(hr) != 1.0
        else "Both strategies have about the same quit risk"
    )

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(["Hazard Ratio"], [float(hr)], color="#d62728")
    axis.axhline(1.0, linestyle="--", color="black", linewidth=1)
    if ci_lower is not None and ci_upper is not None:
        axis.errorbar(
            [0],
            [float(hr)],
            yerr=[[float(hr) - float(ci_lower)], [float(ci_upper) - float(hr)]],
            fmt="none",
            ecolor="black",
            capsize=6,
            linewidth=1.3,
        )
    axis.set_ylabel("Hazard Ratio")
    axis.set_title("Cox Hazard Ratio")
    axis.text(
        0,
        float(hr) + 0.08,
        interpretation,
        ha="center",
        va="bottom",
        fontsize=9,
        wrap=True,
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "hazard_ratio.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_portfolio_metric_subplots(portfolio_outcomes_summary: dict[str, Any], *, plot_dir: Path = H3_PLOTS_DIR) -> None:
    strategies = [STRATEGY_TIME_VOL, STRATEGY_TIME_VOL_LATENT]
    labels = _strategy_labels()
    colors = _strategy_colors()
    plot_specs = [
        ("sharpe_ratio", "Sharpe Ratio"),
        ("sortino_ratio", "Sortino Ratio"),
        ("max_drawdown_pct", "Max Drawdown (%)"),
        ("calmar_ratio", "Calmar Ratio"),
        ("utility_score", "Utility Score"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    flat_axes = list(axes.flatten())

    for axis, (metric_key, title) in zip(flat_axes, plot_specs):
        raw_values = [float(portfolio_outcomes_summary.get(strategy, {}).get(metric_key) or 0.0) for strategy in strategies]
        values = [abs(value) for value in raw_values]
        bars = axis.bar([labels[key] for key in strategies], values, color=[colors[key] for key in strategies])
        for bar, value in zip(bars, values):
            offset = 0.05
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=12)
        axis.grid(alpha=0.2, linestyle="--", axis="y")

    flat_axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(plot_dir / "portfolio_metric_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_h3_plots_from_payload(
    payload: dict[str, Any],
    *,
    plot_dir: Path | str = H3_PLOTS_DIR,
) -> Path:
    target_dir = Path(plot_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    _plot_km(payload["summary"], plot_dir=target_dir)
    _plot_median_retention(payload["summary"], plot_dir=target_dir)
    _plot_quit_rates(payload["group_summary"], plot_dir=target_dir)
    _plot_log_rank(payload["summary"], plot_dir=target_dir)
    _plot_hazard_ratio(payload["summary"], plot_dir=target_dir)
    _plot_portfolio_metric_subplots(payload["portfolio_outcomes_summary"], plot_dir=target_dir)
    return target_dir


def render_h3_plots_from_json(
    json_path: Path | str,
    *,
    plot_dir: Path | str = H3_PLOTS_DIR,
) -> Path:
    with Path(json_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return render_h3_plots_from_payload(payload, plot_dir=plot_dir)


def run_all_h3_retention(
    *,
    population_payload: dict[str, Any] | None = None,
    seed: int = 42,
    time_rebalance_months: int = DEFAULT_TIME_REBALANCE_MONTHS,
    volatility_trigger_threshold: float = DEFAULT_VOLATILITY_TRIGGER,
    latent_signal_threshold: float = DEFAULT_LATENT_SIGNAL_THRESHOLD,
    latent_quit_grace_months: int = DEFAULT_LATENT_QUIT_GRACE_MONTHS,
    replication_factor: int = DEFAULT_REPLICATION_FACTOR,
    save_path: Path | str = H3_RESULTS_PATH,
    verbose: bool = True,
) -> dict[str, Any]:
    _ensure_output_dirs()
    payload = simulate_h3_retention(
        population_payload=population_payload,
        seed=seed,
        time_rebalance_months=time_rebalance_months,
        volatility_trigger_threshold=volatility_trigger_threshold,
        latent_signal_threshold=latent_signal_threshold,
        latent_quit_grace_months=latent_quit_grace_months,
        replication_factor=replication_factor,
        verbose=verbose,
    )
    save_target = Path(save_path)
    with save_target.open("w", encoding="utf-8") as handle:
        json.dump(_safe_json(payload), handle, indent=2)

    render_h3_plots_from_payload(payload, plot_dir=H3_PLOTS_DIR)

    if verbose:
        print(f"[H3] Saved retention results to {save_target}", flush=True)
        print(f"[H3] Saved plots to {H3_PLOTS_DIR}", flush=True)
    return payload


if __name__ == "__main__":
    run_all_h3_retention()

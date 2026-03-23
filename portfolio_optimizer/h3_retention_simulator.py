from __future__ import annotations

import json
import math
import random
import sys
from calendar import monthrange
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt

from evaluation.metrics import compute_portfolio_outcomes, summarize_portfolio_outcomes_by_group, summarize_survival_by_group
from integration.theta_adapter import select_config_from_theta
from latent_state_engine import run_bayesian
from latent_state_engine.bayesian_update import BayesianLatentEngine
from nlp.signal_extractor import extract_schema_variables

from .latent_state_simulator import generate_latent_scenario
from .portfolio_api import should_trigger_rebalance
from .simulation_scenarios import SCENARIOS, SIMULATION_END_DATE
from .simulator import SimPortfolio, apply_event, get_price_history, get_prices_on_date, run_optimizer_historical

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT_DIR / "evaluation"
H3_PLOTS_DIR = EVALUATION_DIR / "plots" / "H3"
H3_RESULTS_PATH = EVALUATION_DIR / "H3_retention_results.json"

STRATEGY_NO_REBALANCE = "no_rebalance"
STRATEGY_NORMAL = "normal_rebalance"
STRATEGY_BEHAVIORAL = "behavioral_rebalance"

PROFILE_RISK_SCORE = {
    "conservative": 0.25,
    "balanced": 0.50,
    "aggressive": 0.75,
}


def _ensure_output_dirs() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    H3_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _add_months(dt: date, months: int) -> date:
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    day = min(dt.day, monthrange(year, month)[1])
    return date(year, month, day)


def _month_ends(start_dt: date, end_dt: date) -> list[date]:
    dates = []
    current = start_dt
    month_counter = 0
    while current < end_dt:
        month_counter += 1
        next_month = _add_months(start_dt, month_counter)
        candidate = min(next_month, end_dt)
        dates.append(candidate)
        current = candidate
    return dates


def _theta_for_date(latent_timeline: list[dict], as_of_date: str) -> dict[str, float]:
    target = date.fromisoformat(as_of_date)
    for state in latent_timeline:
        start = date.fromisoformat(state["state_start"])
        end = date.fromisoformat(state["state_end"])
        if start <= target < end:
            return state["theta_true"]
    return latent_timeline[-1]["theta_true"]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _drawdown(current_value: float, peak_value: float) -> float:
    if peak_value <= 0:
        return 0.0
    return max(0.0, (peak_value - current_value) / peak_value)


def _negative_return(current_value: float, previous_value: float | None) -> float:
    if previous_value is None or previous_value <= 0:
        return 0.0
    return max(0.0, (previous_value - current_value) / previous_value)


def _preference_mismatch(theta_true: dict[str, float], current_profile: str) -> float:
    desired_profile = select_config_from_theta(theta_true)["profile"]
    desired_score = PROFILE_RISK_SCORE[desired_profile]
    actual_score = PROFILE_RISK_SCORE[current_profile]
    return abs(desired_score - actual_score)


def _quit_hazard(
    *,
    theta_true: dict[str, float],
    current_profile: str,
    current_value: float,
    peak_value: float,
    previous_value: float | None,
    strategy: str,
) -> dict[str, float]:
    mismatch = _preference_mismatch(theta_true, current_profile)
    drawdown = _drawdown(current_value, peak_value)
    downside = _negative_return(current_value, previous_value)
    risk = float(theta_true["risk_sensitivity"])
    patience_penalty = 1.0 - float(theta_true["patience_level"])
    control_penalty = 1.0 - float(theta_true["controlled_perception"])

    alignment_bonus = 0.0
    if strategy == STRATEGY_NORMAL:
        alignment_bonus = 0.15
    elif strategy == STRATEGY_BEHAVIORAL:
        alignment_bonus = 0.35

    score = (
        -5.6
        + 3.0 * mismatch
        + 3.5 * drawdown
        + 2.5 * downside
        + 0.9 * risk
        + 0.8 * patience_penalty
        + 0.9 * control_penalty
        - 1.4 * alignment_bonus
    )
    hazard = _clamp(_sigmoid(score), 0.002, 0.65)
    return {
        "hazard": hazard,
        "mismatch": mismatch,
        "drawdown": drawdown,
        "downside": downside,
    }


def _safe_json(value):
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


def _run_no_rebalance(
    scenario: dict,
    month_dates: list[date],
) -> tuple[list[dict], dict]:
    first_event = scenario["events"][0]
    portfolio = SimPortfolio(scenario["capital"])
    prices = get_prices_on_date(first_event["date"])
    optimizer_result = run_optimizer_historical(first_event["config"], first_event["date"])
    if optimizer_result and optimizer_result.get("status") == "success":
        apply_event(portfolio, optimizer_result, first_event["date"], prices)
    return [], {
        "portfolio": portfolio,
        "current_profile": first_event["config"]["profile"],
        "last_value": None,
        "peak_value": scenario["capital"],
        "rebalances": [first_event["date"]],
    }


def _run_normal_rebalance(
    scenario: dict,
) -> dict:
    first_event = scenario["events"][0]
    portfolio = SimPortfolio(scenario["capital"])
    prices = get_prices_on_date(first_event["date"])
    optimizer_result = run_optimizer_historical(first_event["config"], first_event["date"])
    if optimizer_result and optimizer_result.get("status") == "success":
        apply_event(portfolio, optimizer_result, first_event["date"], prices)
    return {
        "portfolio": portfolio,
        "current_profile": first_event["config"]["profile"],
        "pending_events": list(scenario["events"][1:]),
        "last_value": None,
        "peak_value": scenario["capital"],
        "rebalances": [first_event["date"]],
    }


def _run_behavioral_rebalance(
    scenario: dict,
    latent_scenario: dict,
) -> dict:
    portfolio = SimPortfolio(scenario["capital"])
    engine = BayesianLatentEngine()
    pending_chats = sorted(
        [
            chat | {"theta_true": state["theta_true"]}
            for state in latent_scenario["latent_timeline"]
            for chat in state["chats"]
        ],
        key=lambda item: item["date"],
    )
    return {
        "portfolio": portfolio,
        "engine": engine,
        "pending_chats": pending_chats,
        "latest_theta_inferred": None,
        "current_profile": None,
        "last_value": None,
        "peak_value": scenario["capital"],
        "rebalances": [],
        "chat_trace": [],
        "last_rebalance_month": None,
    }


def _apply_normal_events(state: dict, month_end_iso: str) -> None:
    while state["pending_events"] and state["pending_events"][0]["date"] <= month_end_iso:
        event = state["pending_events"].pop(0)
        prices = get_prices_on_date(event["date"])
        optimizer_result = run_optimizer_historical(event["config"], event["date"])
        if optimizer_result and optimizer_result.get("status") == "success":
            apply_event(state["portfolio"], optimizer_result, event["date"], prices)
            state["current_profile"] = event["config"]["profile"]
            state["rebalances"].append(event["date"])


def _apply_behavioral_month(
    state: dict,
    month_end_iso: str,
    month_index: int,
) -> None:
    updated = False
    while state["pending_chats"] and state["pending_chats"][0]["date"] <= month_end_iso:
        chat = state["pending_chats"].pop(0)
        signals = extract_schema_variables(chat["text"])
        if hasattr(signals, "model_dump"):
            signals = signals.model_dump()
        theta_inferred = run_bayesian(signals, engine=state["engine"], strength=0.7)
        state["latest_theta_inferred"] = theta_inferred
        state["chat_trace"].append(
            {
                "date": chat["date"],
                "chat_text": chat["text"],
                "theta_true": chat["theta_true"],
                "theta_inferred": theta_inferred,
            }
        )
        updated = True

    if state["latest_theta_inferred"] is None:
        return

    selected_config = select_config_from_theta(state["latest_theta_inferred"])
    current_profile = selected_config["profile"]
    trigger = should_trigger_rebalance(state["latest_theta_inferred"])
    cooldown_ok = state["last_rebalance_month"] is None or (month_index - state["last_rebalance_month"] >= 3)
    should_apply = not state["portfolio"].positions or (updated and trigger and cooldown_ok)
    if should_apply:
        optimizer_result = run_optimizer_historical(selected_config, month_end_iso)
        prices = get_prices_on_date(month_end_iso)
        if optimizer_result and optimizer_result.get("status") == "success":
            apply_event(state["portfolio"], optimizer_result, month_end_iso, prices)
            state["current_profile"] = current_profile
            state["rebalances"].append(month_end_iso)
            state["last_rebalance_month"] = month_index
    elif state["current_profile"] is None:
        state["current_profile"] = current_profile


def _strategy_monthly_status(
    *,
    strategy: str,
    state: dict,
    theta_true: dict[str, float],
    month_end_iso: str,
) -> dict:
    prices = get_prices_on_date(month_end_iso)
    current_value = state["portfolio"].portfolio_value(prices)
    state["peak_value"] = max(state["peak_value"], current_value)
    hazard_info = _quit_hazard(
        theta_true=theta_true,
        current_profile=state["current_profile"] or "balanced",
        current_value=current_value,
        peak_value=state["peak_value"],
        previous_value=state["last_value"],
        strategy=strategy,
    )
    state["last_value"] = current_value
    return {
        "current_value": current_value,
        "current_profile": state["current_profile"] or "balanced",
        **hazard_info,
    }


def simulate_h3_retention(
    *,
    cohort_multiplier: int = 3,
    seed: int = 42,
    end_date: str = SIMULATION_END_DATE,
    verbose: bool = True,
) -> dict:
    _ensure_output_dirs()
    get_price_history()
    rng = random.Random(seed)
    start_dt = min(date.fromisoformat(scenario["events"][0]["date"]) for scenario in SCENARIOS)
    end_dt = date.fromisoformat(end_date)
    month_dates = _month_ends(start_dt, end_dt)

    survival_rows = []
    investor_runs = []

    for replica in range(cohort_multiplier):
        for base_index, scenario in enumerate(SCENARIOS):
            investor_seed = seed + replica * 100 + base_index
            latent_scenario = generate_latent_scenario(
                scenario,
                seed=investor_seed,
                end_date=end_date,
            )
            investor_id = f"{scenario['email']}__replica_{replica+1}"
            monthly_shocks = {
                month_index: rng.random()
                for month_index in range(1, len(month_dates) + 1)
            }

            track_states = {
                STRATEGY_NO_REBALANCE: _run_no_rebalance(scenario, month_dates)[1],
                STRATEGY_NORMAL: _run_normal_rebalance(scenario),
                STRATEGY_BEHAVIORAL: _run_behavioral_rebalance(scenario, latent_scenario),
            }

            for strategy in (STRATEGY_NO_REBALANCE, STRATEGY_NORMAL, STRATEGY_BEHAVIORAL):
                state = track_states[strategy]
                active = True
                quit_month = None
                monthly_trace = []

                for month_index, month_end in enumerate(month_dates, start=1):
                    month_end_iso = month_end.isoformat()
                    theta_true = _theta_for_date(latent_scenario["latent_timeline"], month_end_iso)

                    if strategy == STRATEGY_NORMAL:
                        _apply_normal_events(state, month_end_iso)
                    elif strategy == STRATEGY_BEHAVIORAL:
                        _apply_behavioral_month(state, month_end_iso, month_index)

                    status = _strategy_monthly_status(
                        strategy=strategy,
                        state=state,
                        theta_true=theta_true,
                        month_end_iso=month_end_iso,
                    )
                    shock = monthly_shocks[month_index]
                    quit_event = active and (shock < status["hazard"])
                    monthly_trace.append(
                        {
                            "month_index": month_index,
                            "date": month_end_iso,
                            "theta_true": theta_true,
                            "shock": shock,
                            "quit_event": quit_event,
                            **status,
                        }
                    )
                    if quit_event:
                        active = False
                        quit_month = month_index
                        break

                duration = quit_month or len(month_dates)
                survival_rows.append(
                    {
                        "investor_id": investor_id,
                        "name": scenario["name"],
                        "persona": scenario["persona"],
                        "strategy": strategy,
                        "duration_months": duration,
                        "quit_event": int(quit_month is not None),
                    }
                )
                investor_runs.append(
                    {
                        "investor_id": investor_id,
                        "strategy": strategy,
                        "quit_month": quit_month,
                        "censored": quit_month is None,
                        "portfolio_outcomes": compute_portfolio_outcomes(monthly_trace),
                        "monthly_trace": monthly_trace,
                        "rebalances": state["rebalances"],
                    }
                )

                if verbose:
                    outcome = f"quit@M{quit_month}" if quit_month else "censored"
                    print(f"[H3] {investor_id} | {strategy} -> {outcome}")

    summary = summarize_survival_by_group(survival_rows)
    portfolio_outcomes_summary = summarize_portfolio_outcomes_by_group(investor_runs)
    payload = {
        "generated_at": date.today().isoformat(),
        "cohort_multiplier": cohort_multiplier,
        "num_investors": len({row['investor_id'] for row in survival_rows}),
        "end_date": end_date,
        "survival_records": survival_rows,
        "investor_runs": investor_runs,
        "summary": summary,
        "portfolio_outcomes_summary": portfolio_outcomes_summary,
    }
    return payload


def _plot_km(summary: dict) -> None:
    curves = summary["kaplan_meier"]
    fig, axis = plt.subplots(figsize=(10, 6))
    colors = {
        STRATEGY_NO_REBALANCE: "#7f7f7f",
        STRATEGY_NORMAL: "#1f77b4",
        STRATEGY_BEHAVIORAL: "#2ca02c",
    }
    labels = {
        STRATEGY_NO_REBALANCE: "No Rebalance",
        STRATEGY_NORMAL: "Normal Rebalance",
        STRATEGY_BEHAVIORAL: "Behavioral Rebalance",
    }
    for strategy, curve in curves.items():
        x = [point["time"] for point in curve]
        y = [point["survival"] for point in curve]
        axis.step(x, y, where="post", label=labels.get(strategy, strategy), color=colors.get(strategy))
    axis.set_title("H3 Kaplan-Meier Retention Curves")
    axis.set_xlabel("Months")
    axis.set_ylabel("Active Investor Survival Probability")
    axis.set_ylim(0, 1.05)
    axis.legend()
    fig.tight_layout()
    fig.savefig(H3_PLOTS_DIR / "kaplan_meier_retention.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_median_retention(summary: dict) -> None:
    medians = summary["median_retention_months"]
    labels = ["No Rebalance", "Normal Rebalance", "Behavioral Rebalance"]
    keys = [STRATEGY_NO_REBALANCE, STRATEGY_NORMAL, STRATEGY_BEHAVIORAL]
    values = [medians.get(key) or 0 for key in keys]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, values, color=["#7f7f7f", "#1f77b4", "#2ca02c"])
    axis.set_title("Median Retention Time")
    axis.set_ylabel("Months")
    fig.tight_layout()
    fig.savefig(H3_PLOTS_DIR / "median_retention_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_hazard_ratios(summary: dict) -> None:
    pairwise = summary["pairwise_comparisons"]
    labels = []
    values = []
    for key, result in pairwise.items():
        labels.append(key.replace("__vs__", " vs "))
        values.append(result["hazard_ratio"]["hazard_ratio"] or 0)
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, values, color="#d62728")
    axis.axhline(1.0, linestyle="--", color="black", linewidth=1)
    axis.set_title("Pairwise Hazard Ratios")
    axis.set_ylabel("Hazard Ratio")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(H3_PLOTS_DIR / "hazard_ratios.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_portfolio_outcomes(portfolio_outcomes_summary: dict) -> None:
    strategy_order = [STRATEGY_NO_REBALANCE, STRATEGY_NORMAL, STRATEGY_BEHAVIORAL]
    labels = ["No Rebalance", "Normal Rebalance", "Behavioral Rebalance"]
    plot_specs = [
        ("sharpe_ratio", "Sharpe Ratio", "h3_sharpe_ratio.png"),
        ("sortino_ratio", "Sortino Ratio", "h3_sortino_ratio.png"),
        ("max_drawdown_pct", "Max Drawdown (%)", "h3_max_drawdown.png"),
        ("calmar_ratio", "Calmar Ratio", "h3_calmar_ratio.png"),
        ("utility_score", "Utility Score", "h3_utility_score.png"),
    ]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c"]

    for metric_key, title, filename in plot_specs:
        values = [
            portfolio_outcomes_summary.get(strategy, {}).get(metric_key) or 0
            for strategy in strategy_order
        ]
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylabel(title)
        axis.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(H3_PLOTS_DIR / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)


def run_all_h3_retention(
    *,
    cohort_multiplier: int = 3,
    seed: int = 42,
    save_path: Path | str = H3_RESULTS_PATH,
    verbose: bool = True,
) -> dict:
    _ensure_output_dirs()
    payload = simulate_h3_retention(
        cohort_multiplier=cohort_multiplier,
        seed=seed,
        verbose=verbose,
    )
    save_target = Path(save_path)
    with save_target.open("w", encoding="utf-8") as handle:
        json.dump(_safe_json(payload), handle, indent=2)
    _plot_km(payload["summary"])
    _plot_median_retention(payload["summary"])
    _plot_hazard_ratios(payload["summary"])
    _plot_portfolio_outcomes(payload["portfolio_outcomes_summary"])
    if verbose:
        print(f"[H3] Saved retention results to {save_target}")
        print(f"[H3] Saved plots to {H3_PLOTS_DIR}")
    return payload


if __name__ == "__main__":
    run_all_h3_retention()

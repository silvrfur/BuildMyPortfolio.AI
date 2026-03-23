from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from evaluation.metrics.H1 import (
    average_error_by_month,
    build_cross_investor_error_cdf,
    build_static_misalignment_series,
    compare_static_profile_to_dynamic,
    summarize_static_misalignment,
)

from .latent_state_simulator import _ensure_output_dirs, _safe_json_dump, get_price_history, run_latent_state_simulation
from .simulation_scenarios import SCENARIOS


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT_DIR / "evaluation"
RESULTS_PATH = EVALUATION_DIR / "H1_static_misalignment_results.json"
H1_PLOTS_DIR = EVALUATION_DIR / "plots" / "H1"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _ensure_h1_output_dirs() -> None:
    _ensure_output_dirs()
    H1_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _simulate_self_reported_theta(
    theta_initial_true: dict[str, float],
    *,
    rng: random.Random,
    noise: float = 0.06,
    midpoint_pull: float = 0.12,
) -> dict[str, float]:
    reported = {}
    for key, value in theta_initial_true.items():
        adjusted = ((1.0 - midpoint_pull) * float(value)) + (midpoint_pull * 0.5) + rng.uniform(-noise, noise)
        reported[key] = round(_clamp(adjusted), 4)
    return reported


def _attach_static_errors(
    chat_events: list[dict[str, object]],
    *,
    theta_initial_true: dict[str, float],
    theta_self_reported: dict[str, float],
) -> list[dict[str, object]]:
    by_initial = build_static_misalignment_series(chat_events, theta_initial_true)
    by_self_report = build_static_misalignment_series(chat_events, theta_self_reported)
    enriched = []
    for event, initial_error, self_report_error in zip(chat_events, by_initial, by_self_report):
        record = dict(event)
        record["static_initial_error"] = initial_error
        record["static_self_reported_error"] = self_report_error
        enriched.append(record)
    return enriched


def _plot_average_error_over_time(payload: dict) -> None:
    series = payload.get("average_error_over_time", [])
    if not series:
        return

    months = [point["month"] for point in series]
    errors = [point["average_error"] for point in series]

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(months, errors, marker="o", linewidth=2, color="#1f77b4")
    axis.set_title("H1 Average Misalignment Error Over Time")
    axis.set_xlabel("Months")
    axis.set_ylabel("Average RMSE")
    axis.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(H1_PLOTS_DIR / "h1_average_error_over_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_investor_error_trajectories(results: list[dict]) -> None:
    if not results:
        return

    fig, axis = plt.subplots(figsize=(11, 6))
    for result in results:
        series = result["self_reported_static_misalignment"]["error_series"]
        months = [point["month_offset"] for point in series if point["month_offset"] is not None]
        errors = [point["rmse"] for point in series if point["month_offset"] is not None]
        if months and errors:
            axis.plot(months, errors, linewidth=1.8, label=result["name"])

    axis.set_title("H1 Misalignment Trajectories by Investor")
    axis.set_xlabel("Months Since Onboarding")
    axis.set_ylabel("Static Profile RMSE")
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(H1_PLOTS_DIR / "h1_investor_error_trajectories.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_final_error_cdf(payload: dict) -> None:
    cdf = payload.get("final_error_cdf", {}).get("cdf_points", [])
    if not cdf:
        return

    thresholds = [point["error_threshold"] for point in cdf]
    probs = [point["cdf"] for point in cdf]

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.step(thresholds, probs, where="post", linewidth=2, color="#2ca02c")
    axis.set_title("H1 Final-Month Error CDF")
    axis.set_xlabel("RMSE Threshold")
    axis.set_ylabel("CDF")
    axis.set_ylim(0, 1.05)
    axis.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(H1_PLOTS_DIR / "h1_final_error_cdf.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_final_error_distribution(results: list[dict]) -> None:
    final_errors = [
        float(result["self_reported_static_misalignment"]["final_rmse"])
        for result in results
        if result["self_reported_static_misalignment"]["final_rmse"] is not None
    ]
    if not final_errors:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].hist(final_errors, bins=min(8, max(3, len(final_errors))), color="#ff7f0e", edgecolor="black", alpha=0.85)
    axes[0].set_title("Final Misalignment Error Histogram")
    axes[0].set_xlabel("Final RMSE")
    axes[0].set_ylabel("Count")

    axes[1].boxplot(final_errors, vert=True, patch_artist=True, boxprops={"facecolor": "#9467bd", "alpha": 0.6})
    axes[1].set_title("Final Misalignment Error Boxplot")
    axes[1].set_ylabel("Final RMSE")
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["Investors"])

    fig.tight_layout()
    fig.savefig(H1_PLOTS_DIR / "h1_final_error_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_growth_rate_comparison(results: list[dict]) -> None:
    if not results:
        return

    labels = [result["name"] for result in results]
    early = []
    late = []
    ratios = []
    for result in results:
        comparison = result["self_reported_static_misalignment"]["window_growth_comparison"]
        early_growth = comparison["early_window_growth"]["slope_per_month"] if comparison and comparison["early_window_growth"] else 0.0
        late_growth = comparison["late_window_growth"]["slope_per_month"] if comparison and comparison["late_window_growth"] else 0.0
        ratio = comparison["growth_ratio_late_vs_early"] if comparison else None
        early.append(early_growth or 0.0)
        late.append(late_growth or 0.0)
        ratios.append(ratio or 0.0)

    positions = list(range(len(labels)))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    axes[0].bar([pos - width / 2 for pos in positions], early, width=width, label="Early Window", color="#1f77b4")
    axes[0].bar([pos + width / 2 for pos in positions], late, width=width, label="Late Window", color="#d62728")
    axes[0].set_title("H1 Error Growth Rate by Investor")
    axes[0].set_ylabel("Slope per Month")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].legend()

    axes[1].bar(labels, ratios, color="#2ca02c")
    axes[1].axhline(1.0, linestyle="--", color="black", linewidth=1)
    axes[1].set_title("Late vs Early Growth-Rate Ratio")
    axes[1].set_ylabel("Ratio")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(H1_PLOTS_DIR / "h1_growth_rate_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_h1_plots(payload: dict) -> None:
    _ensure_h1_output_dirs()
    results = payload.get("results", [])
    _plot_average_error_over_time(payload)
    _plot_investor_error_trajectories(results)
    _plot_final_error_cdf(payload)
    _plot_final_error_distribution(results)
    _plot_growth_rate_comparison(results)


def run_h1_static_profile_simulation(
    scenario: dict,
    *,
    seed: int = 42,
    material_threshold: float = 0.15,
    price_history: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> dict:
    if verbose:
        print(f"\n[H1] Running static-profile misalignment simulation for {scenario['name']}")

    latent_result = run_latent_state_simulation(
        scenario,
        seed=seed,
        simulate_portfolio=False,
        price_history=price_history,
        verbose=verbose,
        log_prefix="H1",
    )
    if not latent_result["latent_timeline"]:
        raise ValueError("latent timeline is empty; cannot evaluate H1")

    rng = random.Random(seed + 10_000)
    theta_initial_true = dict(latent_result["latent_timeline"][0]["theta_true"])
    theta_self_reported = _simulate_self_reported_theta(theta_initial_true, rng=rng)
    chat_events = _attach_static_errors(
        latent_result["chat_events"],
        theta_initial_true=theta_initial_true,
        theta_self_reported=theta_self_reported,
    )

    initial_truth_summary = summarize_static_misalignment(
        chat_events,
        theta_initial_true,
        baseline_name="initial_true_theta",
        material_threshold=material_threshold,
    )
    self_report_summary = summarize_static_misalignment(
        chat_events,
        theta_self_reported,
        baseline_name="self_reported_theta",
        material_threshold=material_threshold,
    )
    static_vs_dynamic = compare_static_profile_to_dynamic(
        chat_events,
        theta_self_reported,
    )

    hypothesis_supported = bool(
        self_report_summary["final_rmse"] is not None
        and self_report_summary["rmse_growth"] is not None
        and self_report_summary["final_rmse"] > material_threshold
        and self_report_summary["rmse_growth"] > 0
    )

    return {
        "email": latent_result["email"],
        "name": latent_result["name"],
        "persona": latent_result["persona"],
        "generated_at": date.today().isoformat(),
        "theta_initial_true": theta_initial_true,
        "theta_initial_self_reported": theta_self_reported,
        "latent_timeline": latent_result["latent_timeline"],
        "chat_events": chat_events,
        "initial_truth_static_misalignment": initial_truth_summary,
        "self_reported_static_misalignment": self_report_summary,
        "self_reported_static_vs_dynamic": static_vs_dynamic,
        "hypothesis_supported": hypothesis_supported,
    }


def run_all_h1_static_profile_simulations(
    *,
    save_path: Path | str = RESULTS_PATH,
    seed: int = 42,
    material_threshold: float = 0.15,
    save_plots: bool = True,
    verbose: bool = True,
) -> dict:
    _ensure_h1_output_dirs()
    history = get_price_history()
    results = []

    for index, scenario in enumerate(SCENARIOS):
        results.append(
            run_h1_static_profile_simulation(
                scenario,
                seed=seed + index,
                material_threshold=material_threshold,
                price_history=history,
                verbose=verbose,
            )
        )

    supported = [result for result in results if result["hypothesis_supported"]]
    avg_final_rmse = sum(
        float(result["self_reported_static_misalignment"]["final_rmse"] or 0.0)
        for result in results
    ) / len(results)
    avg_improvement = sum(
        float(result["self_reported_static_vs_dynamic"]["improvement_pct"] or 0.0)
        for result in results
    ) / len(results)
    investor_error_series = [
        {
            "email": result["email"],
            "error_series": result["self_reported_static_misalignment"]["error_series"],
        }
        for result in results
    ]

    payload = {
        "generated_at": date.today().isoformat(),
        "num_investors": len(results),
        "material_threshold_rmse": material_threshold,
        "hypothesis_support_rate": len(supported) / len(results) if results else None,
        "average_final_static_rmse": avg_final_rmse if results else None,
        "average_dynamic_improvement_pct": avg_improvement if results else None,
        "average_error_over_time": average_error_by_month(investor_error_series),
        "final_error_cdf": build_cross_investor_error_cdf(investor_error_series, month_index=-1),
        "results": results,
    }

    save_target = Path(save_path)
    with save_target.open("w", encoding="utf-8") as handle:
        json.dump(_safe_json_dump(payload), handle, indent=2)

    if save_plots:
        _save_h1_plots(payload)

    if verbose:
        print(f"[H1] Saved static-profile results to {save_target}")
        if save_plots:
            print(f"[H1] Saved plots to {H1_PLOTS_DIR}")

    return payload


if __name__ == "__main__":
    run_all_h1_static_profile_simulations()

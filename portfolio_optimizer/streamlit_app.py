"""
streamlit_app.py - BuildMyPortfolio.AI dashboard

Run with:
    streamlit run portfolio_optimizer/streamlit_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from simulation_scenarios import SCENARIOS, SIMULATION_END_DATE

st.set_page_config(
    page_title="BuildMyPortfolio.AI Dashboard",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  .metric-card {
    background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
    border-radius: 16px;
    padding: 18px 20px;
    border: 1px solid #e6ebf2;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
    margin-bottom: 10px;
    min-height: 112px;
  }
  .metric-label {
    font-size: 12px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
    font-weight: 600;
  }
  .metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
  }
  .metric-help {
    margin-top: 8px;
    font-size: 13px;
    color: #475569;
    line-height: 1.4;
  }
  .section-card {
    background: #ffffff;
    border: 1px solid #e6ebf2;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.04);
  }
  .hypothesis-card {
    background: linear-gradient(135deg, #f8fbff 0%, #ffffff 100%);
    border: 1px solid #dbe5f1;
    border-radius: 18px;
    padding: 18px 18px 16px 18px;
    min-height: 190px;
  }
  .hypothesis-kicker {
    font-size: 12px;
    color: #2563eb;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    margin-bottom: 10px;
  }
  .hypothesis-title {
    font-size: 21px;
    color: #0f172a;
    font-weight: 700;
    margin-bottom: 10px;
    line-height: 1.2;
  }
  .hypothesis-copy {
    font-size: 14px;
    color: #475569;
    line-height: 1.5;
  }
</style>
""",
    unsafe_allow_html=True,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT_DIR / "evaluation"
RESULTS_FILE = ROOT_DIR / "portfolio_optimizer" / "simulation_results.json"

H1_RESULT_PATH = EVALUATION_DIR / "H1" / "population_result.json"
H2_RESULT_PATH = EVALUATION_DIR / "H2" / "population_result.json"
H3_RESULT_PATH = EVALUATION_DIR / "H3" / "result.json"

H1_PLOTS_DIR = EVALUATION_DIR / "H1" / "plots_new"
H2_PLOTS_DIR = EVALUATION_DIR / "H2" / "plots_new"
H3_PLOTS_DIR = EVALUATION_DIR / "H3" / "plots"

PROFILE_COLORS = {
    "conservative": "#2563eb",
    "balanced": "#059669",
    "aggressive": "#dc2626",
}


@st.cache_data(show_spinner=False)
def load_results():
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open(encoding="utf-8") as handle:
            return json.load(handle)
    return None


@st.cache_data(show_spinner=False)
def load_json_file(path: str):
    target = Path(path)
    if not target.exists():
        return None
    with target.open(encoding="utf-8") as handle:
        return json.load(handle)


def get_scenario_by_email(email: str):
    return next((s for s in SCENARIOS if s["email"] == email), None)


def format_value(value, *, kind: str = "number", decimals: int = 4):
    if value is None:
        return "N/A"
    if kind == "pct":
        return f"{float(value):.{decimals}f}%"
    if kind == "ratio":
        return f"{float(value):.{decimals}f}"
    if kind == "currency":
        return f"₹{float(value):,.2f}"
    if kind == "integer":
        return f"{int(value)}"
    return str(value)


def format_stat_value(value, *, decimals: int = 4):
    if value is None:
        return "N/A"
    numeric = float(value)
    if numeric == 0:
        return f"{numeric:.{decimals}f}"
    if abs(numeric) < (10 ** -decimals):
        return f"{numeric:.4e}"
    return f"{numeric:.{decimals}f}"


def get_h3_display_median(summary: dict, strategy: str):
    medians = summary.get("median_retention_months", {})
    median = medians.get(strategy)
    if median is not None:
        return float(median), False

    curve = summary.get("kaplan_meier", {}).get(strategy, [])
    if not curve:
        return None, False

    max_time = max((float(point.get("time", 0.0)) for point in curve), default=None)
    if max_time is None:
        return None, False
    return max_time, True


def get_h3_pairwise_summary(summary: dict):
    pairwise = summary.get("pairwise_comparisons", {})
    preferred = pairwise.get("time_vol_latent_rebalance__vs__time_vol_rebalance")
    if preferred:
        return preferred

    reverse = pairwise.get("time_vol_rebalance__vs__time_vol_latent_rebalance")
    if reverse:
        return reverse

    for key, value in pairwise.items():
        if "time_vol_rebalance" in key and "time_vol_latent_rebalance" in key:
            return value
    return {}


def metric_card(label: str, value: str, help_text: str | None = None):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          {f'<div class="metric-help">{help_text}</div>' if help_text else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def interpretation_block(title: str, body: str):
    st.info(f"**Interpretation - {title}:** {body}")


def image_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def render_plot_grid(paths: list[Path], *, columns: int = 2):
    valid_paths = [path for path in paths if image_exists(path)]
    if not valid_paths:
        st.info("Plots are not available yet for this hypothesis.")
        return
    for start in range(0, len(valid_paths), columns):
        cols = st.columns(columns)
        for col, path in zip(cols, valid_paths[start:start + columns]):
            with col:
                st.image(str(path), use_container_width=True, caption=path.stem.replace("_", " ").title())


def render_plot_with_interpretation(path: Path, interpretation: str):
    if image_exists(path):
        st.image(str(path), use_container_width=True, caption=path.stem.replace("_", " ").title())
        interpretation_block(path.stem.replace("_", " ").title(), interpretation)


def render_hypothesis_summary_cards():
    st.title("Result Analysis")
    st.caption("Professional hypothesis-wise review using saved outputs under the evaluation directory.")
    st.divider()

    cards = [
        (
            "H1",
            "Static Profile Misalignment Growth",
            "Tests whether a fixed onboarding profile becomes increasingly misaligned with the investor's evolving latent state over time.",
        ),
        (
            "H2",
            "Latent State Tracking Accuracy",
            "Tests whether the inferred latent state follows the underlying true latent trajectory with low error and strong correlation.",
        ),
        (
            "H3",
            "Retention Benefit of Latent-Aware Rebalancing",
            "Tests whether adding a latent-state threshold to time and market-volatility rebalancing improves investor retention and portfolio quality.",
        ),
    ]
    cols = st.columns(3)
    for col, (hid, title, description) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="hypothesis-card">
                  <div class="hypothesis-kicker">{hid}</div>
                  <div class="hypothesis-title">{title}</div>
                  <div class="hypothesis-copy">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_h1_analysis():
    payload = load_json_file(str(H1_RESULT_PATH))
    st.subheader("H1: Static Profile Misalignment Growth")
    st.caption("Higher support means static onboarding preferences drift away from the user's true evolving behavior.")
    if payload is None:
        st.warning(f"Missing evaluation file: {H1_RESULT_PATH}")
        return

    cols = st.columns(4)
    with cols[0]:
        metric_card("Users", format_value(payload.get("num_users"), kind="integer"))
        interpretation_block("Users", "This value reports the cohort size used for the H1 evaluation and therefore defines the population over which static-profile misalignment is assessed.")
    with cols[1]:
        metric_card("Avg Final RMSE", format_value(payload.get("average_final_rmse"), kind="ratio"))
        interpretation_block("Avg Final RMSE", "The average final RMSE captures the mean end-of-period distance between the fixed onboarding profile and the investor's true latent state. Higher values indicate stronger accumulated misalignment.")
    with cols[2]:
        metric_card("Median Final RMSE", format_value(payload.get("median_final_rmse"), kind="ratio"))
        interpretation_block("Median Final RMSE", "The median final RMSE reflects the typical end-state misalignment across investors and is less influenced by extreme user trajectories than the mean.")
    with cols[3]:
        metric_card("Hypothesis Support Rate", format_value((payload.get("hypothesis_support_rate") or 0) * 100, kind="pct", decimals=2))
        interpretation_block("Hypothesis Support Rate", "This is the proportion of investors for whom misalignment both exceeded the material threshold and increased over time. Higher values provide stronger support for H1.")

    cols = st.columns(2)
    with cols[0]:
        metric_card("Material Threshold", format_value(payload.get("material_threshold_rmse"), kind="ratio"))
        interpretation_block("Material Threshold", "This threshold defines the minimum RMSE level at which static-profile error is treated as materially meaningful rather than negligible.")
    with cols[1]:
        metric_card("Avg RMSE Growth Rate", format_value(payload.get("average_error_growth_rate"), kind="ratio"))
        interpretation_block("Avg RMSE Growth Rate", "A positive average growth rate indicates that static-profile error tends to accumulate over successive months, which is the central directional claim of H1.")

    cols = st.columns(2)
    with cols[0]:
        render_plot_with_interpretation(
            H1_PLOTS_DIR / "population_h1_average_error_over_time.png",
            "An upward trajectory indicates that the static onboarding profile becomes progressively less representative of the investor's evolving latent state over time.",
        )
        render_plot_with_interpretation(
            H1_PLOTS_DIR / "population_h1_growth_rate_distribution.png",
            "This distribution shows whether misalignment growth is broadly present across the population or concentrated in a small number of extreme cases.",
        )
        render_plot_with_interpretation(
            H1_PLOTS_DIR / "population_h1_final_error_cdf.png",
            "The CDF shows the proportion of investors falling below alternative final-error thresholds. A rightward profile implies larger end-of-period static-profile error.",
        )
    with cols[1]:
        render_plot_with_interpretation(
            H1_PLOTS_DIR / "population_h1_sampled_trajectories.png",
            "These representative user trajectories illustrate whether misalignment typically rises, stabilizes, or declines at the individual level.",
        )
        render_plot_with_interpretation(
            H1_PLOTS_DIR / "population_h1_dimension_errors.png",
            "This figure identifies the latent dimensions that contribute most to static-profile mismatch and whether some dimensions drift more rapidly than others.",
        )
        render_plot_with_interpretation(
            H1_PLOTS_DIR / "population_h1_representative_theta.png",
            "This representative theta plot provides a visual comparison between the investor's evolving latent state and the fixed static profile, thereby illustrating the mechanism underlying H1.",
        )


def render_h2_analysis():
    payload = load_json_file(str(H2_RESULT_PATH))
    st.subheader("H2: Latent State Tracking Accuracy")
    st.caption("Better H2 results mean inferred latent states closely match the hidden ground-truth trajectory.")
    if payload is None:
        st.warning(f"Missing evaluation file: {H2_RESULT_PATH}")
        return

    overall = payload.get("overall_metrics", {})
    pearson = payload.get("overall_pearson_correlation", {})
    baseline = payload.get("overall_static_baseline_comparison_true_reference", {})
    coverage = payload.get("overall_credible_interval_coverage", {})

    cols = st.columns(4)
    with cols[0]:
        metric_card("Investors", format_value(payload.get("num_investors"), kind="integer"))
        interpretation_block("Investors", "This value indicates the population size used to evaluate latent-state inference accuracy under H2.")
    with cols[1]:
        metric_card("Overall MAE", format_value(overall.get("overall_mae"), kind="ratio"))
        interpretation_block("Overall MAE", "Overall MAE measures the average absolute deviation between inferred and true latent states across events and dimensions. Lower values indicate more accurate tracking.")
    with cols[2]:
        metric_card("Overall RMSE", format_value(overall.get("overall_rmse"), kind="ratio"))
        interpretation_block("Overall RMSE", "Overall RMSE gives greater weight to larger inference errors and therefore summarizes the severity of latent-state tracking misses. Lower values are favorable.")
    with cols[3]:
        metric_card("Avg Pearson Correlation", format_value(pearson.get("overall_average_correlation"), kind="ratio"))
        interpretation_block("Avg Pearson Correlation", "Higher correlation indicates that inferred latent trajectories move in the same direction as the underlying true trajectories over time.")

    cols = st.columns(3)
    with cols[0]:
        metric_card("Static RMSE", format_value(baseline.get("static_overall_rmse"), kind="ratio"))
        interpretation_block("Static RMSE", "This is the error associated with a non-adaptive static baseline and serves as the benchmark against which inferred latent tracking is compared.")
    with cols[1]:
        metric_card("Inferred RMSE", format_value(baseline.get("dynamic_overall_rmse"), kind="ratio"))
        interpretation_block("Inferred RMSE", "This is the tracking error of the adaptive inferred-latent approach. Values below the static baseline are supportive of H2.")
    with cols[2]:
        metric_card("CI Coverage", format_value((coverage.get("overall_coverage") or 0) * 100, kind="pct", decimals=2))
        interpretation_block("CI Coverage", "Coverage near the target level suggests that the posterior uncertainty intervals are reasonably calibrated rather than systematically too narrow or too wide.")

    cols = st.columns(2)
    with cols[0]:
        render_plot_with_interpretation(
            H2_PLOTS_DIR / "population_h2_error_over_time.png",
            "If error remains low or stabilizes over time, the inferred latent state is consistently tracking the underlying truth rather than drifting away from it.",
        )
        render_plot_with_interpretation(
            H2_PLOTS_DIR / "population_h2_pearson_correlation.png",
            "Higher correlations indicate that the inferred series captures the same directional movement as the true latent dimensions, which supports temporal fidelity of inference.",
        )
        render_plot_with_interpretation(
            H2_PLOTS_DIR / "population_h2_improvement_true_reference.png",
            "This compares the adaptive inferred approach against the static baseline. Larger improvement in favor of the inferred model provides stronger support for H2.",
        )
        render_plot_with_interpretation(
            H2_PLOTS_DIR / "population_h2_representative_theta.png",
            "This representative-user view shows whether the inferred latent state visually follows the true latent curve over time.",
        )
    with cols[1]:
        render_plot_with_interpretation(
            H2_PLOTS_DIR / "population_h2_dimension_errors.png",
            "This identifies which latent dimensions are easier or harder for the inference engine to recover with low error.",
        )
        render_plot_with_interpretation(
            H2_PLOTS_DIR / "population_h2_credible_interval_coverage.png",
            "Coverage close to the target level indicates statistically well-behaved posterior intervals and therefore more credible uncertainty estimates.",
        )
        render_plot_with_interpretation(
            H2_PLOTS_DIR / "population_h2_deviation_from_true_theta.png",
            "Positive improvement relative to the static baseline indicates that the dynamic inferred model better approximates the true latent state.",
        )
        render_plot_with_interpretation(
            H2_PLOTS_DIR / "population_h2_representative_theta_top5.png",
            "These top-performing user examples illustrate strong latent-state tracking performance and make the H2 pattern easier to inspect qualitatively.",
        )


def render_h3_analysis():
    payload = load_json_file(str(H3_RESULT_PATH))
    st.subheader("H3: Retention Benefit of Latent-Aware Rebalancing")
    st.caption("Case 1 is time plus volatility. Case 2 adds a latent-state threshold before rebalancing.")
    if payload is None:
        st.warning(f"Missing evaluation file: {H3_RESULT_PATH}")
        return

    summary = payload.get("summary", {})
    group_summary = payload.get("group_summary", {})
    portfolio = payload.get("portfolio_outcomes_summary", {})

    base = group_summary.get("time_vol_rebalance", {})
    latent = group_summary.get("time_vol_latent_rebalance", {})
    case1_median, case1_median_censored = get_h3_display_median(summary, "time_vol_rebalance")
    case2_median, case2_median_censored = get_h3_display_median(summary, "time_vol_latent_rebalance")
    pairwise = get_h3_pairwise_summary(summary)
    log_rank = pairwise.get("log_rank", {})
    hazard = pairwise.get("hazard_ratio", {})

    cols = st.columns(4)
    with cols[0]:
        metric_card("Case 1 Quit Rate", format_value((base.get("quit_rate") or 0) * 100, kind="pct", decimals=2))
        interpretation_block("Case 1 Quit Rate", "This is the proportion of investors who quit under time-plus-volatility rebalancing. Lower values indicate better retention.")
    with cols[1]:
        metric_card("Case 2 Quit Rate", format_value((latent.get("quit_rate") or 0) * 100, kind="pct", decimals=2))
        interpretation_block("Case 2 Quit Rate", "This is the proportion of investors who quit under time-plus-volatility-plus-latent-threshold rebalancing. Lower values indicate stronger retention.")
    with cols[2]:
        case1_median_text = "N/A" if case1_median is None else f"{case1_median:.1f}{'+' if case1_median_censored else ''}"
        metric_card("Case 1 Median Retention", case1_median_text, "Months")
        interpretation_block("Case 1 Median Retention", "This is the month by which 50% of Case 1 investors have quit. Higher values indicate longer investor persistence.")
    with cols[3]:
        case2_median_text = "N/A" if case2_median is None else f"{case2_median:.1f}{'+' if case2_median_censored else ''}"
        metric_card("Case 2 Median Retention", case2_median_text, "Months")
        interpretation_block("Case 2 Median Retention", "This is the month by which 50% of Case 2 investors have quit. If the curve never falls to 50% during the observed window, the dashboard shows the latest observed month with a plus sign.")

    cols = st.columns(4)
    with cols[0]:
        metric_card("Log-Rank p-value", format_stat_value(log_rank.get("p_value")))
        interpretation_block("Log-Rank p-value", "A p-value below 0.05 indicates that the survival difference between Case 1 and Case 2 is statistically significant and unlikely to be explained by chance alone.")
    with cols[1]:
        metric_card("Log-Rank Chi-square", format_stat_value(log_rank.get("chi_square")))
        interpretation_block("Log-Rank Chi-square", "A larger chi-square statistic indicates stronger separation between the two survival curves.")
    with cols[2]:
        metric_card("Hazard Ratio", format_value(hazard.get("hazard_ratio"), kind="ratio"))
        interpretation_block("Hazard Ratio", "In this comparison, a hazard ratio above 1 indicates higher quitting risk in Case 1 relative to Case 2, whereas a value below 1 would indicate the reverse.")
    with cols[3]:
        metric_card("Simulated Paths", format_value(payload.get("num_simulated_investor_paths"), kind="integer"))
        interpretation_block("Simulated Paths", "This is the effective sample size used in the H3 survival comparison and therefore affects statistical stability and power.")

    case1 = portfolio.get("time_vol_rebalance", {})
    case2 = portfolio.get("time_vol_latent_rebalance", {})
    cols = st.columns(5)
    metrics = [
        ("Case 2 Sharpe", case2.get("sharpe_ratio"), case1.get("sharpe_ratio"), "ratio"),
        ("Case 2 Sortino", case2.get("sortino_ratio"), case1.get("sortino_ratio"), "ratio"),
        ("Case 2 Max Drawdown", case2.get("max_drawdown_pct"), case1.get("max_drawdown_pct"), "pct"),
        ("Case 2 Calmar", case2.get("calmar_ratio"), case1.get("calmar_ratio"), "ratio"),
        ("Case 2 Utility", case2.get("utility_score"), case1.get("utility_score"), "ratio"),
    ]
    for col, (label, value, baseline, kind) in zip(cols, metrics):
        with col:
            metric_card(label, format_value(value, kind=kind, decimals=2), f"Case 1: {format_value(baseline, kind=kind, decimals=2)}")
            if "Drawdown" in label:
                interpretation_block(label, "For drawdown, a less negative value corresponds to lower downside risk and therefore stronger capital preservation.")
            else:
                interpretation_block(label, "For these portfolio-quality indicators, higher values generally correspond to better risk-adjusted performance.")

    cols = st.columns(2)
    with cols[0]:
        render_plot_with_interpretation(
            H3_PLOTS_DIR / "kaplan_meier_retention.png",
            "The higher survival curve identifies the strategy that retains a larger share of active investors over time. If Case 2 remains above Case 1, it demonstrates superior retention.",
        )
        render_plot_with_interpretation(
            H3_PLOTS_DIR / "quit_rate_comparison.png",
            "This directly compares the final proportion of investors who quit in each case. Lower quit rates indicate stronger user retention.",
        )
        render_plot_with_interpretation(
            H3_PLOTS_DIR / "hazard_ratio.png",
            "This figure reports the relative quitting risk between the two cases. In this setup, a value above 1 indicates that Case 1 is riskier and Case 2 performs better on retention.",
        )
    with cols[1]:
        render_plot_with_interpretation(
            H3_PLOTS_DIR / "median_retention_time.png",
            "This compares the month at which half of the investors have quit. A higher bar indicates longer investor persistence within that strategy.",
        )
        render_plot_with_interpretation(
            H3_PLOTS_DIR / "log_rank_test.png",
            "This tests whether the survival difference is statistically meaningful. A p-value below 0.05 supports the conclusion that the observed difference is unlikely to be random.",
        )
        render_plot_with_interpretation(
            H3_PLOTS_DIR / "portfolio_metric_comparison.png",
            "This summarizes whether the retention-improving strategy also preserves or improves portfolio quality across Sharpe, Sortino, drawdown, Calmar, and utility metrics.",
        )


def render_result_analysis_page():
    render_hypothesis_summary_cards()
    st.divider()
    tab_h1, tab_h2, tab_h3 = st.tabs(["H1", "H2", "H3"])
    with tab_h1:
        render_h1_analysis()
    with tab_h2:
        render_h2_analysis()
    with tab_h3:
        render_h3_analysis()


def render_individual_user_page(results, selected_email):
    user_result = next(r for r in results if r["email"] == selected_email)
    rb = user_result["rebalanced"]
    ho = user_result["hold"]
    cmp = user_result["comparison"]

    st.title(user_result["name"])
    st.caption(
        f"{user_result['persona']} | Capital: ₹{user_result['capital']:,.0f} | "
        f"Period: 2022-01-03 to {SIMULATION_END_DATE}"
    )
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Growth", "Events", "Metrics"])

    with tab1:
        cols = st.columns(3)
        with cols[0]:
            st.metric("Rebalanced Final Value", f"₹{rb['final_value_inr']:,.2f}", f"{rb['total_return_pct']:+.2f}%")
        with cols[1]:
            st.metric("Hold Final Value", f"₹{ho['final_value_inr']:,.2f}", f"{ho['total_return_pct']:+.2f}%")
        with cols[2]:
            better = "Rebalanced" if cmp["rebalanced_better"] else "Hold"
            st.metric("Winner", better, f"₹{cmp['difference_inr']:,.2f} better")

        compare_df = pd.DataFrame(
            {
                "Metric": ["Final Value", "Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown", "Tax Paid", "Trades"],
                "Rebalanced": [
                    f"₹{rb['final_value_inr']:,.2f}",
                    f"{rb['total_return_pct']:+.2f}%",
                    f"{rb['cagr_pct']:+.2f}%",
                    str(rb.get("sharpe_ratio") or "N/A"),
                    f"{rb['max_drawdown_pct']:.2f}%",
                    f"₹{rb['total_tax_paid']:,.2f}",
                    str(rb["total_trades"]),
                ],
                "Hold": [
                    f"₹{ho['final_value_inr']:,.2f}",
                    f"{ho['total_return_pct']:+.2f}%",
                    f"{ho['cagr_pct']:+.2f}%",
                    str(ho.get("sharpe_ratio") or "N/A"),
                    f"{ho['max_drawdown_pct']:.2f}%",
                    f"₹{ho['total_tax_paid']:,.2f}",
                    str(ho["total_trades"]),
                ],
            }
        )
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

    with tab2:
        rb_checkpoints = pd.DataFrame(rb.get("checkpoints", []))
        ho_checkpoints = pd.DataFrame(ho.get("checkpoints", []))
        if not rb_checkpoints.empty and not ho_checkpoints.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rb_checkpoints["date"], y=rb_checkpoints["value"], name="Rebalanced", line=dict(color="#2563eb", width=3)))
            fig.add_trace(go.Scatter(x=ho_checkpoints["date"], y=ho_checkpoints["value"], name="Hold", line=dict(color="#94a3b8", width=2, dash="dash")))
            fig.add_hline(y=user_result["capital"], line_dash="dot", line_color="gray")
            fig.update_layout(height=420, xaxis_title="Date", yaxis_title="Portfolio Value (₹)", legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Checkpoint data is not available.")

    with tab3:
        st.subheader("Event Timeline")
        for index, event in enumerate(user_result["events"]):
            label = "Initial Buy" if index == 0 else f"Rebalance {index}"
            st.markdown(f"**{event['date']}** - {label} - `{event['profile']}`")
            st.caption(event["nlp_input"])

    with tab4:
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Rebalanced Track**")
            for label, value in [
                ("Final Value", f"₹{rb['final_value_inr']:,.2f}"),
                ("Sharpe Ratio", str(rb.get("sharpe_ratio") or "N/A")),
                ("Max Drawdown", f"{rb['max_drawdown_pct']:.2f}%"),
                ("Tax Paid", f"₹{rb['total_tax_paid']:,.2f}"),
                ("Rebalance Events", str(rb["rebalance_count"])),
            ]:
                st.write(f"{label}: **{value}**")
        with cols[1]:
            st.markdown("**Hold Track**")
            for label, value in [
                ("Final Value", f"₹{ho['final_value_inr']:,.2f}"),
                ("Sharpe Ratio", str(ho.get("sharpe_ratio") or "N/A")),
                ("Max Drawdown", f"{ho['max_drawdown_pct']:.2f}%"),
                ("Tax Paid", f"₹{ho['total_tax_paid']:,.2f}"),
                ("Trades", str(ho["total_trades"])),
            ]:
                st.write(f"{label}: **{value}**")


def render_leaderboard_page(results):
    st.title("Cross-User Leaderboard")
    st.caption(f"Simulation period: 2022-01-03 to {SIMULATION_END_DATE}")
    st.divider()

    rows = []
    for result in results:
        rb = result["rebalanced"]
        ho = result["hold"]
        rows.append(
            {
                "User": result["name"],
                "Persona": result["persona"],
                "Rebalanced Return": f"{rb['total_return_pct']:+.2f}%",
                "Hold Return": f"{ho['total_return_pct']:+.2f}%",
                "Rebalanced Final": f"₹{rb['final_value_inr']:,.0f}",
                "Hold Final": f"₹{ho['final_value_inr']:,.0f}",
                "Winner": result["comparison"]["winner"].upper(),
                "Tax Paid (Rebalanced)": f"₹{rb['total_tax_paid']:,.0f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    names = [r["name"] for r in results]
    rb_vals = [r["rebalanced"]["final_value_inr"] for r in results]
    ho_vals = [r["hold"]["final_value_inr"] for r in results]

    fig = go.Figure()
    fig.add_bar(name="Rebalanced", x=names, y=rb_vals, marker_color="#2563eb")
    fig.add_bar(name="Hold", x=names, y=ho_vals, marker_color="#94a3b8")
    fig.update_layout(barmode="group", height=420, yaxis_title="Portfolio Value (₹)", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    scatter_rows = []
    for result in results:
        for key, label in [("rebalanced", "Rebalanced"), ("hold", "Hold")]:
            track = result[key]
            scatter_rows.append(
                {
                    "User": f"{result['name']} ({label})",
                    "Return (%)": track["total_return_pct"],
                    "Drawdown (%)": abs(track["max_drawdown_pct"]),
                    "Sharpe": track.get("sharpe_ratio") or 0,
                    "Track": label,
                }
            )
    scatter_df = pd.DataFrame(scatter_rows)
    fig2 = px.scatter(
        scatter_df,
        x="Drawdown (%)",
        y="Return (%)",
        color="Track",
        text="User",
        color_discrete_map={"Rebalanced": "#2563eb", "Hold": "#94a3b8"},
        height=430,
    )
    fig2.update_traces(textposition="top center", textfont_size=10)
    st.plotly_chart(fig2, use_container_width=True)


with st.sidebar:
    st.title("BuildMyPortfolio.AI")
    st.caption("Simulation and Evaluation Dashboard")
    st.divider()

    results = load_results()
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Individual User"
    if "main_view" not in st.session_state:
        st.session_state["main_view"] = "Individual User"

    main_view = st.radio(
        "Main Views",
        ["Individual User", "Leaderboard"],
        key="main_view",
        label_visibility="collapsed",
    )
    if st.session_state["active_page"] != "Result Analysis":
        st.session_state["active_page"] = main_view

    st.divider()
    if st.button("Result Analysis", use_container_width=True):
        st.session_state["active_page"] = "Result Analysis"

    page = st.session_state["active_page"]

    selected_email = None
    if page in {"Individual User", "Leaderboard"}:
        if results is None:
            st.warning("No simulation results found.")
            st.caption(f"Expected file: {RESULTS_FILE}")
        else:
            if st.button("Re-run Simulations", use_container_width=True):
                with st.spinner("Running simulations..."):
                    from simulation.simulator import run_all_simulations

                    run_all_simulations(verbose=False)
                    st.cache_data.clear()
                    st.rerun()

    if page == "Individual User" and results:
        st.divider()
        user_options = {result["name"]: result["email"] for result in results}
        selected_name = st.selectbox("Select User", list(user_options.keys()))
        selected_email = user_options[selected_name]


if page == "Result Analysis":
    render_result_analysis_page()
elif page == "Leaderboard":
    if results is None:
        st.warning("No simulation results found for the leaderboard view.")
    else:
        render_leaderboard_page(results)
else:
    if results is None or selected_email is None:
        st.warning("No simulation results found for the individual user view.")
    else:
        render_individual_user_page(results, selected_email)

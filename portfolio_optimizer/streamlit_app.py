"""
streamlit_app.py — BuildMyPortfolio.AI Simulation Dashboard

Run with:
    streamlit run streamlit_app.py

Shows simulation results for each user:
- Timeline of decisions
- Rebalanced vs Hold portfolio value over time
- Performance metrics comparison
- Trade history
- Position breakdown
- Cross-user leaderboard
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from simulation_scenarios import SCENARIOS, SIMULATION_END_DATE

st.set_page_config(
    page_title="BuildMyPortfolio.AI — Simulation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f8f9fa; border-radius: 10px; padding: 16px 20px;
    border: 1px solid #e9ecef; margin-bottom: 8px;
  }
  .metric-label { font-size: 12px; color: #6c757d; text-transform: uppercase;
                  letter-spacing: 0.05em; margin-bottom: 4px; }
  .metric-value { font-size: 24px; font-weight: 600; color: #212529; }
  .metric-delta { font-size: 13px; margin-top: 2px; }
  .green  { color: #198754; }
  .red    { color: #dc3545; }
  .badge  { display: inline-block; padding: 2px 10px; border-radius: 20px;
            font-size: 12px; font-weight: 500; }
  .badge-cons { background: #cfe2ff; color: #084298; }
  .badge-bal  { background: #d1e7dd; color: #0a3622; }
  .badge-agg  { background: #f8d7da; color: #842029; }
  .timeline-item { padding: 8px 0; border-left: 2px solid #dee2e6;
                   padding-left: 16px; margin-left: 8px; position: relative; }
  .timeline-dot  { width: 10px; height: 10px; border-radius: 50%;
                   position: absolute; left: -6px; top: 12px; }
</style>
""", unsafe_allow_html=True)

PROFILE_COLORS = {
    "conservative": "#0d6efd",
    "balanced":     "#198754",
    "aggressive":   "#dc3545",
}
PROFILE_BADGE = {
    "conservative": "badge-cons",
    "balanced":     "badge-bal",
    "aggressive":   "badge-agg",
}


# ── LOAD OR RUN SIMULATION ────────────────────────────────────────────────────

RESULTS_FILE = "simulation_results.json"

@st.cache_data(show_spinner=False)
def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return None


def get_scenario_by_email(email: str):
    return next((s for s in SCENARIOS if s["email"] == email), None)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Portfolio Simulator")
    st.caption("BuildMyPortfolio.AI — Backtesting Dashboard")
    st.divider()

    results = load_results()

    if results is None:
        st.warning("No simulation results found.")
        st.caption(f"Expected file: `{RESULTS_FILE}`")
        if st.button("▶ Run All Simulations", type="primary", use_container_width=True):
            with st.spinner("Running simulations (this takes 5–10 min)..."):
                from simulator import run_all_simulations
                results = run_all_simulations(verbose=False)
                st.cache_data.clear()
                st.rerun()
        st.stop()
    else:
        if st.button("🔄 Re-run Simulations", use_container_width=True):
            with st.spinner("Running simulations..."):
                from simulator import run_all_simulations
                results = run_all_simulations(verbose=False)
                st.cache_data.clear()
                st.rerun()

    st.divider()
    page = st.radio(
        "View",
        ["👤 Individual User", "🏆 Leaderboard"],
        label_visibility="collapsed",
    )

    if page == "👤 Individual User":
        user_options = {r["name"]: r["email"] for r in results}
        selected_name  = st.selectbox("Select User", list(user_options.keys()))
        selected_email = user_options[selected_name]
        user_result    = next(r for r in results if r["email"] == selected_email)
        scenario       = get_scenario_by_email(selected_email)

        st.divider()
        st.caption("User Journey")
        for ev in user_result["events"]:
            profile = ev["profile"]
            badge_cls = PROFILE_BADGE.get(profile, "badge-cons")
            st.markdown(
                f'<div style="font-size:12px;margin:3px 0;">'
                f'<span style="color:#6c757d;">{ev["date"]}</span> '
                f'<span class="badge {badge_cls}">{profile}</span></div>',
                unsafe_allow_html=True,
            )


# ── LEADERBOARD PAGE ──────────────────────────────────────────────────────────

if page == "🏆 Leaderboard":
    st.title("🏆 Cross-User Leaderboard")
    st.caption(f"Simulation period: 2022-01-03 → {SIMULATION_END_DATE} | Capital: ₹1,00,000")
    st.divider()

    rows = []
    for r in results:
        rb = r["rebalanced"]
        ho = r["hold"]
        rows.append({
            "User":                r["name"],
            "Persona":             r["persona"],
            "Rebalanced Return":   f"{rb['total_return_pct']:+.2f}%",
            "Hold Return":         f"{ho['total_return_pct']:+.2f}%",
            "Rebalanced Final":    f"₹{rb['final_value_inr']:,.0f}",
            "Hold Final":          f"₹{ho['final_value_inr']:,.0f}",
            "Winner":              r["comparison"]["winner"].upper(),
            "Diff (₹)":            f"₹{r['comparison']['difference_inr']:,.0f}",
            "Tax Paid (Rebal)":    f"₹{rb['total_tax_paid']:,.0f}",
            "Trades (Rebal)":      rb["total_trades"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Bar chart — final value comparison
    st.subheader("Final Portfolio Value — Rebalanced vs Hold")
    names = [r["name"] for r in results]
    rb_vals = [r["rebalanced"]["final_value_inr"] for r in results]
    ho_vals = [r["hold"]["final_value_inr"] for r in results]

    fig = go.Figure()
    fig.add_bar(name="Rebalanced", x=names, y=rb_vals, marker_color="#0d6efd")
    fig.add_bar(name="Hold",       x=names, y=ho_vals, marker_color="#adb5bd")
    fig.add_hline(y=100_000, line_dash="dot", line_color="gray",
                  annotation_text="Initial ₹1,00,000")
    fig.update_layout(barmode="group", height=400, yaxis_title="Portfolio Value (₹)",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # Return % comparison
    st.subheader("Total Return % — All Users")
    rb_rets = [r["rebalanced"]["total_return_pct"] for r in results]
    ho_rets = [r["hold"]["total_return_pct"] for r in results]

    fig2 = go.Figure()
    fig2.add_bar(name="Rebalanced", x=names, y=rb_rets, marker_color="#0d6efd")
    fig2.add_bar(name="Hold",       x=names, y=ho_rets, marker_color="#adb5bd")
    fig2.add_hline(y=0, line_color="gray")
    fig2.update_layout(barmode="group", height=350, yaxis_title="Return (%)",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, use_container_width=True)

    # Risk-return scatter
    st.subheader("Risk vs Return — Sharpe Ratio")
    scatter_data = []
    for r in results:
        for track, label in [("rebalanced", "Rebalanced"), ("hold", "Hold")]:
            t = r[track]
            scatter_data.append({
                "User": f"{r['name']} ({label})",
                "Return (%)": t["total_return_pct"],
                "Max Drawdown (%)": abs(t["max_drawdown_pct"]),
                "Sharpe": t.get("sharpe_ratio") or 0,
                "Track": label,
            })
    scatter_df = pd.DataFrame(scatter_data)
    fig3 = px.scatter(
        scatter_df, x="Max Drawdown (%)", y="Return (%)",
        text="User", color="Track",
        color_discrete_map={"Rebalanced": "#0d6efd", "Hold": "#adb5bd"},
        size_max=12, height=400,
    )
    fig3.update_traces(textposition="top center", textfont_size=10)
    st.plotly_chart(fig3, use_container_width=True)

    st.stop()


# ── INDIVIDUAL USER PAGE ──────────────────────────────────────────────────────

rb = user_result["rebalanced"]
ho = user_result["hold"]
cmp = user_result["comparison"]

st.title(f"👤 {user_result['name']}")
st.caption(f"{user_result['persona']} | Capital: ₹{user_result['capital']:,.0f} | Period: 2022-01-03 → {SIMULATION_END_DATE}")
st.divider()

# ── TAB LAYOUT ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "📈 Growth Chart", "🔄 Events & Trades",
    "📋 Metrics", "🥇 vs Others"
])


# ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────────
with tab1:
    winner_label = "Rebalanced" if cmp["rebalanced_better"] else "Hold"
    winner_color = "green" if cmp["rebalanced_better"] else "red"

    st.subheader("Final Result")
    col1, col2, col3 = st.columns(3)
    with col1:
        delta = rb["final_value_inr"] - user_result["capital"]
        st.metric("Rebalanced Final Value",
                  f"₹{rb['final_value_inr']:,.2f}",
                  f"{rb['total_return_pct']:+.2f}%")
    with col2:
        delta2 = ho["final_value_inr"] - user_result["capital"]
        st.metric("Hold Final Value",
                  f"₹{ho['final_value_inr']:,.2f}",
                  f"{ho['total_return_pct']:+.2f}%")
    with col3:
        diff = cmp["difference_inr"]
        better = "Rebalancing" if cmp["rebalanced_better"] else "Holding"
        st.metric("Winner", better, f"₹{diff:,.2f} better")

    st.divider()
    st.subheader("Side-by-side Comparison")

    metrics_data = {
        "Metric":            ["Final Value", "Total Return", "CAGR",
                              "Sharpe Ratio", "Max Drawdown",
                              "Realized P&L", "Tax Paid", "Total Trades"],
        "Rebalanced": [
            f"₹{rb['final_value_inr']:,.2f}",
            f"{rb['total_return_pct']:+.2f}%",
            f"{rb['cagr_pct']:+.2f}%",
            f"{rb.get('sharpe_ratio') or 'N/A'}",
            f"{rb['max_drawdown_pct']:.2f}%",
            f"₹{rb['total_realized_pnl']:,.2f}",
            f"₹{rb['total_tax_paid']:,.2f}",
            str(rb["total_trades"]),
        ],
        "Hold": [
            f"₹{ho['final_value_inr']:,.2f}",
            f"{ho['total_return_pct']:+.2f}%",
            f"{ho['cagr_pct']:+.2f}%",
            f"{ho.get('sharpe_ratio') or 'N/A'}",
            f"{ho['max_drawdown_pct']:.2f}%",
            f"₹{ho['total_realized_pnl']:,.2f}",
            f"₹{ho['total_tax_paid']:,.2f}",
            str(ho["total_trades"]),
        ],
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    # Cost of rebalancing
    st.divider()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Tax cost of rebalancing",
                 f"₹{cmp['tax_cost_of_rebalancing']:,.2f}",
                 help="Extra tax paid due to rebalancing vs just holding")
    col_b.metric("Extra trades from rebalancing",
                 str(cmp["extra_trades"]))
    col_c.metric("Return difference",
                 f"{cmp['return_diff_pct']:+.2f}%",
                 help="Rebalanced return minus Hold return")


# ── TAB 2: GROWTH CHART ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Portfolio Value Over Time")

    rb_checkpoints = rb.get("checkpoints", [])
    ho_checkpoints = ho.get("checkpoints", [])

    if rb_checkpoints and ho_checkpoints:
        rb_df = pd.DataFrame(rb_checkpoints)
        ho_df = pd.DataFrame(ho_checkpoints)

        fig = go.Figure()

        # Rebalanced line
        fig.add_trace(go.Scatter(
            x=rb_df["date"], y=rb_df["value"],
            mode="lines+markers", name="Rebalanced",
            line=dict(color="#0d6efd", width=2.5),
            marker=dict(size=8),
            hovertemplate="<b>%{x}</b><br>Value: ₹%{y:,.0f}<extra>Rebalanced</extra>",
        ))

        # Hold line
        fig.add_trace(go.Scatter(
            x=ho_df["date"], y=ho_df["value"],
            mode="lines+markers", name="Hold",
            line=dict(color="#adb5bd", width=2, dash="dash"),
            marker=dict(size=6),
            hovertemplate="<b>%{x}</b><br>Value: ₹%{y:,.0f}<extra>Hold</extra>",
        ))

        # Initial capital line
        fig.add_hline(y=user_result["capital"], line_dash="dot",
                      line_color="gray", opacity=0.5,
                      annotation_text=f"Initial ₹{user_result['capital']:,.0f}",
                      annotation_position="bottom right")

        # Event markers on rebalanced track
        for ev in user_result["events"][1:]:   # skip initial buy
            profile = ev["profile"]
            color   = PROFILE_COLORS.get(profile, "#666")
            fig.add_vline(
                x=ev["date"], line_dash="dot",
                line_color=color, opacity=0.6,
                annotation_text=profile[:4].upper(),
                annotation_font_size=10,
                annotation_font_color=color,
            )

        fig.update_layout(
            height=450,
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹)",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Vertical dotted lines = rebalancing events | Dashed line = Hold track")
    else:
        st.info("No checkpoint data available.")

    # Returns chart
    st.subheader("Return % Over Time")
    if rb_checkpoints:
        initial = user_result["capital"]
        rb_returns = [(cp["value"] - initial) / initial * 100 for cp in rb_checkpoints]
        ho_returns = [(cp["value"] - initial) / initial * 100 for cp in ho_checkpoints]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=[cp["date"] for cp in rb_checkpoints], y=rb_returns,
            name="Rebalanced", line=dict(color="#0d6efd", width=2),
            fill="tozeroy", fillcolor="rgba(13,110,253,0.08)",
        ))
        fig2.add_trace(go.Scatter(
            x=[cp["date"] for cp in ho_checkpoints], y=ho_returns,
            name="Hold", line=dict(color="#adb5bd", width=2, dash="dash"),
        ))
        fig2.add_hline(y=0, line_color="gray", opacity=0.5)
        fig2.update_layout(height=300, yaxis_title="Return (%)",
                           hovermode="x unified",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)


# ── TAB 3: EVENTS & TRADES ────────────────────────────────────────────────────
with tab3:
    st.subheader("User Journey — Events Timeline")

    for i, ev in enumerate(user_result["events"]):
        profile   = ev["profile"]
        badge_cls = PROFILE_BADGE.get(profile, "badge-cons")
        color     = PROFILE_COLORS.get(profile, "#666")
        label     = "Initial Buy" if i == 0 else f"Rebalance {i}"
        st.markdown(
            f"""<div style="border-left: 3px solid {color}; padding: 8px 16px; margin: 6px 0; border-radius: 0 8px 8px 0; background: #f8f9fa;">
            <div style="font-size:12px;color:#6c757d;">{ev['date']} — {label}</div>
            <div><span class="badge {badge_cls}" style="margin-right:8px;">{profile.upper()}</span>
            <span style="font-size:13px;color:#212529;">"{ev['nlp_input']}"</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Trade History — Rebalanced Track")

    # Reconstruct from rebalance comparison data
    st.info("Trade-level details are stored in the simulator's portfolio object. "
            "Run inspect_db.py on a simulation DB or check simulation_results.json "
            "for the full checkpoint-level data available here.")

    # Show checkpoints as trades proxy
    if rb.get("checkpoints"):
        cp_df = pd.DataFrame(rb["checkpoints"])
        cp_df.columns = [c.replace("_", " ").title() for c in cp_df.columns]
        if "Value" in cp_df.columns:
            cp_df["Value"] = cp_df["Value"].apply(lambda x: f"₹{float(x):,.2f}")
        st.dataframe(cp_df, use_container_width=True, hide_index=True)


# ── TAB 4: DETAILED METRICS ───────────────────────────────────────────────────
with tab4:
    st.subheader("Detailed Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Rebalanced Track**")
        metrics_list = [
            ("Final Value",       f"₹{rb['final_value_inr']:,.2f}"),
            ("Total Return",      f"{rb['total_return_pct']:+.2f}%"),
            ("CAGR",              f"{rb['cagr_pct']:+.2f}%"),
            ("Sharpe Ratio",      str(rb.get("sharpe_ratio") or "N/A")),
            ("Max Drawdown",      f"{rb['max_drawdown_pct']:.2f}%"),
            ("Realized P&L",      f"₹{rb['total_realized_pnl']:,.2f}"),
            ("Tax Paid",          f"₹{rb['total_tax_paid']:,.2f}"),
            ("Leftover Cash",     f"₹{rb['leftover_cash']:,.2f}"),
            ("Total Trades",      str(rb["total_trades"])),
            ("Sells",             str(rb["sell_count"])),
            ("Buys",              str(rb["buy_count"])),
            ("Rebalance Events",  str(rb["rebalance_count"])),
        ]
        for label, val in metrics_list:
            c1, c2 = st.columns([2, 1])
            c1.caption(label)
            c2.markdown(f"**{val}**")

    with col2:
        st.markdown("**Hold Track**")
        hold_list = [
            ("Final Value",      f"₹{ho['final_value_inr']:,.2f}"),
            ("Total Return",     f"{ho['total_return_pct']:+.2f}%"),
            ("CAGR",             f"{ho['cagr_pct']:+.2f}%"),
            ("Sharpe Ratio",     str(ho.get("sharpe_ratio") or "N/A")),
            ("Max Drawdown",     f"{ho['max_drawdown_pct']:.2f}%"),
            ("Realized P&L",     f"₹{ho['total_realized_pnl']:,.2f}"),
            ("Tax Paid",         f"₹{ho['total_tax_paid']:,.2f}"),
            ("Leftover Cash",    f"₹{ho['leftover_cash']:,.2f}"),
            ("Total Trades",     str(ho["total_trades"])),
            ("Sells",            str(ho["sell_count"])),
            ("Buys",             str(ho["buy_count"])),
            ("Rebalance Events", "0"),
        ]
        for label, val in hold_list:
            c1, c2 = st.columns([2, 1])
            c1.caption(label)
            c2.markdown(f"**{val}**")

    st.divider()
    st.subheader("Optimizer Performance at Each Event")
    st.caption("Each rebalance event ran the optimizer on historical data up to that date — no look-ahead bias.")

    event_rows = []
    for ev in user_result["events"]:
        event_rows.append({
            "Date":     ev["date"],
            "Profile":  ev["profile"].title(),
            "NLP Input": ev["nlp_input"][:60] + ("..." if len(ev["nlp_input"]) > 60 else ""),
        })
    st.dataframe(pd.DataFrame(event_rows), use_container_width=True, hide_index=True)


# ── TAB 5: vs OTHERS ──────────────────────────────────────────────────────────
with tab5:
    st.subheader(f"{user_result['name']} vs All Other Users")

    # Rebalanced returns comparison
    fig = go.Figure()
    for r in results:
        color = "#0d6efd" if r["email"] == selected_email else "#adb5bd"
        width = 3 if r["email"] == selected_email else 1.5
        cps   = r["rebalanced"].get("checkpoints", [])
        if cps:
            initial = r["capital"]
            returns = [(cp["value"] - initial) / initial * 100 for cp in cps]
            fig.add_trace(go.Scatter(
                x=[cp["date"] for cp in cps], y=returns,
                name=r["name"], mode="lines",
                line=dict(color=color, width=width),
                opacity=1.0 if r["email"] == selected_email else 0.4,
            ))

    fig.add_hline(y=0, line_color="gray", opacity=0.3)
    fig.update_layout(
        title="Rebalanced Return % — All Users",
        height=400, yaxis_title="Return (%)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metrics comparison table
    st.subheader("All Users — Key Metrics")
    rows = []
    for r in results:
        highlight = "⭐ " if r["email"] == selected_email else ""
        rb_ = r["rebalanced"]
        ho_ = r["hold"]
        rows.append({
            "User":               highlight + r["name"],
            "Rebal Return":       f"{rb_['total_return_pct']:+.2f}%",
            "Hold Return":        f"{ho_['total_return_pct']:+.2f}%",
            "Rebal Sharpe":       str(rb_.get("sharpe_ratio") or "—"),
            "Rebal Max DD":       f"{rb_['max_drawdown_pct']:.2f}%",
            "Tax Paid":           f"₹{rb_['total_tax_paid']:,.0f}",
            "Winner":             r["comparison"]["winner"].upper(),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

import yfinance as yf
import pandas as pd
import numpy as np
import riskfolio as rp
import matplotlib.pyplot as plt
import json
from datetime import datetime
from assets import assets, asset_classes, constraints_data
from config import CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG


def run_portfolio(
    config,
    asset_list=assets,
    asset_class_map=asset_classes,
    constraints=constraints_data,
    start="2022-01-01",
    end="2026-01-01",
    plot=True,
    save_json=True,
    verbose=True,
):
    """
    Run the full portfolio optimisation pipeline.

    Parameters
    ----------
    config          : dict   — CONFIG dict from config.py
    asset_list      : list   — list of ticker strings
    asset_class_map : dict   — {ticker: class} mapping
    constraints     : list   — constraints_data rows
    start / end     : str    — data window (YYYY-MM-DD)
    plot            : bool   — show efficient frontier plot
    save_json       : bool   — write result to .json file
    verbose         : bool   — print console summary

    Returns
    -------
    dict  — full result payload (JSON-serialisable),
            or error dict with "status": "failed" on failure
    """

    rf_daily  = config["rf_daily"]
    annual_rf = rf_daily * 252

    # ── STEP 1: DOWNLOAD ─────────────────────────────────────────────────────
    data = yf.download(
        asset_list, start=start, end=end,
        threads=True, progress=False
    )['Close']

    # ── STEP 2: DEDUPLICATE COLUMNS ──────────────────────────────────────────
    if data.columns.duplicated().any():
        dupes = data.columns[data.columns.duplicated(keep=False)].unique().tolist()
        print(f"[WARNING] Duplicate columns from yfinance — keeping first: {dupes}")
        data = data.loc[:, ~data.columns.duplicated(keep='first')]

    # ── STEP 3: CLEAN DATA ───────────────────────────────────────────────────
    data = data.dropna(axis=1, how='all')
    threshold = int(0.9 * len(data))
    data = data.dropna(axis=1, thresh=threshold)
    data = data.ffill()
    # ── STEP 4: RETURNS ──────────────────────────────────────────────────────
    returns = data.pct_change().dropna()
    returns = returns[sorted(returns.columns)]
    returns = returns.loc[:, ~returns.columns.duplicated(keep='first')]

    # ── STEP 5: SYNC asset_class_map TO AVAILABLE TICKERS ───────────────────
    available            = set(returns.columns)
    dropped              = sorted([t for t in asset_list if t not in available])
    active_asset_classes = {t: c for t, c in asset_class_map.items() if t in available}

    if verbose:
        if dropped:
            print(f"[WARNING] {len(dropped)} tickers dropped: {dropped}")
        print(f"\n[INFO] Tickers available : {len(available)}")
        print(f"[INFO] Classes in use    : {sorted(set(active_asset_classes.values()))}")
        print("Data ready.\n")

    # ── STEP 6: CONSTRAINTS ──────────────────────────────────────────────────
    asset_info = pd.DataFrame({'Asset': returns.columns})
    asset_info['Class'] = asset_info['Asset'].map(active_asset_classes)

    unmapped = asset_info[asset_info['Class'].isna()]['Asset'].tolist()
    if unmapped and verbose:
        print(f"[WARNING] Unmapped tickers: {unmapped}")

    col_names = ['Disabled', 'Type', 'Set', 'Position', 'Sign', 'Weight',
                 'Type Relative', 'Relative Set', 'Relative', 'Factor']
    constraints_df = pd.DataFrame(constraints, columns=col_names)
    A, B = rp.assets_constraints(constraints_df, asset_info)

    # ── STEP 7: OPTIMISATION ─────────────────────────────────────────────────
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu=config["mu_method"], method_cov=config["cov_method"])
    port.ainequality = A
    port.binequality = B
    port.budget      = 1
    port.lowerbound  = config["lower_bound"]
    port.upperbound  = config["upper_bound"]

    try:
        weights = port.optimization(
            model=config["model"],
            rm=config["risk_measure"],
            obj=config["objective"],
            rf=rf_daily,
            l=config["lambda_val"],
            hist=True,
        )
        frontier = port.efficient_frontier(
            model=config["model"],
            rm=config["risk_measure"],
            points=50,
            rf=rf_daily,
            hist=True,
        )
    except Exception as e:
        error_result = {
            "meta": {
                "generated_at": datetime.now().isoformat(timespec='seconds'),
                "profile":      config.get("profile", "unknown"),
            },
            "status": "failed",
            "error":  str(e),
        }
        print(f"[ERROR] Optimization failed: {e}")
        return error_result

    if weights is None or weights.empty:
        error_result = {
            "meta": {
                "generated_at": datetime.now().isoformat(timespec='seconds'),
                "profile":      config.get("profile", "unknown"),
            },
            "status": "failed",
            "error":  "Optimizer returned empty weights. Check constraints or data.",
        }
        print("[ERROR] Optimizer returned empty weights.")
        return error_result

    # ── STEP 8: EXTRACT FRONTIER DATA POINTS ────────────────────────────────
    # frontier is a DataFrame — each column is one point, rows are asset weights.
    # We compute (risk%, return%, sharpe) for every point so the frontend can
    # render the curve with any charting library (Recharts, Chart.js, D3, etc.)
    frontier_points = []
    try:
        _mu  = port.mu.values.flatten()
        _cov = port.cov.values
        for col in frontier.columns:
            w_f   = frontier[col].values
            ret_f = float(np.dot(w_f, _mu) * 252)
            vol_f = float(np.sqrt(np.dot(w_f.T, np.dot(_cov, w_f))) * np.sqrt(252))
            sr_f  = round((ret_f - annual_rf) / vol_f, 4) if vol_f > 0 else None
            frontier_points.append({
                "return_pct":   round(ret_f * 100, 4),
                "risk_pct":     round(vol_f * 100, 4),
                "sharpe_ratio": sr_f,
            })
        # Sort left → right so the curve draws correctly
        frontier_points.sort(key=lambda x: x["risk_pct"])
    except Exception as e:
        print(f"[WARNING] Could not extract frontier points: {e}")

    # ── STEP 8b: OPTIONAL PLOT (riskfolio built-in) ───────────────────────────
    if plot:
        ax = rp.plot_frontier(
            w_frontier=frontier,
            returns=returns,
            mu=port.mu,
            cov=port.cov,
            rm=config["risk_measure"],
            rf=rf_daily,
            alpha=0.05,
            cmap='viridis',
            w=weights,
            label='Optimized Portfolio',
            marker='*',
            s=15,
            c='r',
            height=6,
            width=10,
            ax=None,
        )
        plt.title(f"Efficient Frontier — {config.get('profile', config['risk_measure'])}")
        plt.show()

    # ── STEP 9: METRICS ──────────────────────────────────────────────────────
    w_vec      = weights.values.flatten()
    mu_daily   = port.mu.values.flatten()
    cov_matrix = port.cov.values

    expected_daily_return  = np.dot(w_vec, mu_daily)
    daily_volatility       = np.sqrt(np.dot(w_vec.T, np.dot(cov_matrix, w_vec)))
    expected_annual_return = expected_daily_return * 252
    annual_volatility      = daily_volatility * np.sqrt(252)
    sharpe_ratio           = (expected_annual_return - annual_rf) / annual_volatility

    # Daily portfolio return series
    port_daily_returns = returns.dot(w_vec)

    # Sortino
    downside_rets = port_daily_returns[port_daily_returns < rf_daily]
    downside_std  = np.sqrt((downside_rets ** 2).mean()) * np.sqrt(252) if len(downside_rets) > 0 else None
    sortino_ratio = (expected_annual_return - annual_rf) / downside_std if downside_std else None

    # Max drawdown + Calmar
    cumulative   = (1 + port_daily_returns).cumprod()
    rolling_max  = cumulative.cummax()
    drawdowns    = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())
    calmar_ratio = expected_annual_return / abs(max_drawdown) if max_drawdown != 0 else None

    # CVaR @ 95%
    sorted_rets   = np.sort(port_daily_returns.values)
    var_95_daily  = float(np.percentile(sorted_rets, 5))
    cvar_95_daily = float(sorted_rets[sorted_rets <= var_95_daily].mean())
    cvar_95_ann   = cvar_95_daily * np.sqrt(252)

    # Diversification
    active_mask    = w_vec > 0.001
    active_weights = w_vec[active_mask]
    n_active       = int(active_mask.sum())
    hhi            = float(np.sum(active_weights ** 2))
    effective_n    = round(1 / hhi, 2) if hhi > 0 else None
    concentration  = (
        "well diversified"        if hhi < 0.10 else
        "moderately concentrated" if hhi < 0.20 else
        "highly concentrated"
    )

    # ── STEP 10: ALLOCATION TABLES ───────────────────────────────────────────
    weights_pct = (weights * 100).round(4)
    weights_pct.columns = ['weight_pct']
    weights_pct['asset_class'] = weights_pct.index.map(active_asset_classes)
    weights_pct = weights_pct.sort_values('weight_pct', ascending=False)

    asset_allocations = [
        {
            "ticker":      ticker,
            "asset_class": row['asset_class'],
            "weight_pct":  round(float(row['weight_pct']), 4),
        }
        for ticker, row in weights_pct.iterrows()
        if row['weight_pct'] > 0.01
    ]

    class_summary = (
        weights_pct.groupby('asset_class')['weight_pct']
        .sum()
        .sort_values(ascending=False)
        .round(4)
    )
    class_allocations = {cls: round(float(pct), 4) for cls, pct in class_summary.items()}

    # ── STEP 11: CONSTRAINT CHECK ────────────────────────────────────────────
    constraint_checks = []
    all_passed = True
    for _, row in constraints_df.iterrows():
        cls    = row['Position']
        sign   = row['Sign']
        limit  = row['Weight'] * 100
        actual = float(class_summary.get(cls, 0.0))
        passed = (sign == '>=' and actual >= limit - 0.01) or \
                 (sign == '<=' and actual <= limit + 0.01)
        if not passed:
            all_passed = False
        constraint_checks.append({
            "class":      cls,
            "sign":       sign,
            "limit_pct":  round(limit, 2),
            "actual_pct": round(actual, 4),
            "passed":     passed,
        })

    # ── STEP 12: ASSEMBLE RESULT DICT ────────────────────────────────────────
    result = {
        "status": "success",

        "meta": {
            "generated_at":  datetime.now().isoformat(timespec='seconds'),
            "profile":       config.get("profile", "unknown"),
            "model":         config["model"],
            "risk_measure":  config["risk_measure"],
            "objective":     config["objective"],
            "lambda":        config["lambda_val"],
            "cov_method":    config["cov_method"],
            "mu_method":     config["mu_method"],
            "data_start":    start,
            "data_end":      end,
            "rf_annual_pct": round(annual_rf * 100, 4),
        },

        "universe": {
            "total_requested":  len(asset_list),
            "total_available":  len(available),
            "total_dropped":    len(dropped),
            "dropped_tickers":  dropped,
            "active_positions": n_active,
        },

        "performance": {
            "expected_annual_return_pct": round(expected_annual_return * 100, 4),
            "annual_volatility_pct":      round(annual_volatility * 100, 4),
            "sharpe_ratio":               round(sharpe_ratio, 4),
            "sortino_ratio":              round(sortino_ratio, 4) if sortino_ratio else None,
            "calmar_ratio":               round(calmar_ratio, 4) if calmar_ratio else None,
            "max_drawdown_pct":           round(max_drawdown * 100, 4),
            "var_95_daily_pct":           round(var_95_daily * 100, 4),
            "cvar_95_daily_pct":          round(cvar_95_daily * 100, 4),
            "cvar_95_annualised_pct":     round(cvar_95_ann * 100, 4),
        },

        "diversification": {
            "hhi":                round(hhi, 4),
            "effective_n_assets": effective_n,
            "active_positions":   n_active,
            "concentration":      concentration,
        },

        "class_allocation": class_allocations,
        "asset_allocation": asset_allocations,

        # --- Efficient frontier data (use this to render custom charts) ---
        # Each point: { return_pct, risk_pct, sharpe_ratio }
        # optimized_point: where YOUR portfolio sits on the frontier
        "efficient_frontier": {
            "x_axis":           "risk_pct (annual volatility %)",
            "y_axis":           "return_pct (annual return %)",
            "points":           frontier_points,
            "optimized_point": {
                "return_pct":   round(expected_annual_return * 100, 4),
                "risk_pct":     round(annual_volatility * 100, 4),
                "sharpe_ratio": round(sharpe_ratio, 4),
            },
        },

        "constraints": {
            "all_passed": all_passed,
            "checks":     constraint_checks,
        },
    }

    # ── STEP 13: SAVE JSON ───────────────────────────────────────────────────
    if save_json:
        filename = f"portfolio_result_{config.get('profile', config['risk_measure'])}.json"
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"[INFO] Results saved → {filename}")

    # ── STEP 14: CONSOLE SUMMARY ─────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 42)
        print(f"  {config.get('profile', config['risk_measure']).upper()} PORTFOLIO")
        print("=" * 42)
        print(f"  Return        : {expected_annual_return:.2%}")
        print(f"  Volatility    : {annual_volatility:.2%}")
        print(f"  Sharpe        : {sharpe_ratio:.4f}")
        print(f"  Sortino       : {sortino_ratio:.4f}" if sortino_ratio else "  Sortino       : N/A")
        print(f"  Calmar        : {calmar_ratio:.4f}"  if calmar_ratio  else "  Calmar        : N/A")
        print(f"  Max Drawdown  : {max_drawdown:.2%}")
        print(f"  CVaR 95%/day  : {cvar_95_daily:.2%}")
        print(f"  Active pos.   : {n_active}  (HHI {hhi:.3f} — {concentration})")
        print()
        print("  Class Allocation:")
        for cls, pct in class_allocations.items():
            bar = "█" * int(pct / 2)
            print(f"    {cls:<15s} {pct:>6.2f}%  {bar}")
        print()
        print("  Top 10 Holdings:")
        for a in asset_allocations[:10]:
            print(f"    {a['ticker']:<20s} {a['weight_pct']:>6.2f}%  [{a['asset_class']}]")
        violations = sum(1 for c in constraint_checks if not c['passed'])
        print()
        print(f"  Constraints   : {'ALL PASSED ✓' if all_passed else f'{violations} VIOLATED ✗'}")
        print("=" * 42)

    return result


# ── ENTRY POINT ──────────────────────────────────────────────────────────────
# Runs when executed directly: python main.py
# Silently skipped when imported: from main import run_portfolio
if __name__ == "__main__":
    result = run_portfolio()
    print(json.dumps(result, indent=2))
import yfinance as yf
import pandas as pd
import numpy as np
import riskfolio as rp
import matplotlib.pyplot as plt
from assets import assets, asset_classes, constraints_data
from config import CONFIG


rf_daily = CONFIG["rf_daily"]


data = yf.download(assets, start="2021-01-01", end="2026-01-01")['Close']
data = data.dropna(axis=1, how='all')  

threshold = int(0.9 * len(data))  
data = data.dropna(axis=1, thresh=threshold)  
data = data.ffill()  
returns = data.pct_change().dropna()
returns = returns[sorted(returns.columns)] 

print("Data Downloaded and Sorted Successfully!")


asset_info = pd.DataFrame({'Asset': returns.columns})
asset_info['Class'] = asset_info['Asset'].map(asset_classes)

columns = ['Disabled', 'Type', 'Set', 'Position', 'Sign', 'Weight',
           'Type Relative', 'Relative Set', 'Relative', 'Factor']

constraints_df = pd.DataFrame(constraints_data, columns=columns)
A, B = rp.assets_constraints(constraints_df, asset_info)


def run_optimization(returns_df, config):
    port = rp.Portfolio(returns=returns_df)
    

    port.assets_stats(method_mu=config["mu_method"], method_cov=config["cov_method"])


    port.ainequality = A
    port.binequality = B
    port.budget = 1
    port.lowerbound = config["lower_bound"]
    port.upperbound = config["upper_bound"]

    try:
 
        w = port.optimization(
            model=config["model"],
            rm=config["risk_measure"],
            obj=config["objective"],
            rf=config["rf_daily"],
            l=config["lambda_val"],
            hist=True
        )
        
        # Generate Efficient Frontier
        frontier = port.efficient_frontier(
            model=config["model"],
            rm=config["risk_measure"],
            points=50,
            rf=config["rf_daily"],
            hist=True
        )
        return w, port, frontier

    except Exception as e:
        print(f"Optimization failed: {e}")
        return None, None, None

# Run the optimization
weights, portfolio_obj, frontier = run_optimization(returns, CONFIG)


if weights is not None and not weights.empty:
    # Using the frontier calculated inside the function
    ax = rp.plot_frontier(
        w_frontier=frontier,
        returns=returns,
        mu=portfolio_obj.mu,
        cov=portfolio_obj.cov,
        rm=CONFIG["risk_measure"],
        rf=CONFIG["rf_daily"],
        alpha=0.05,
        cmap='viridis',
        w=weights,   # highlight your optimized portfolio
        label='Optimized Portfolio',
        marker='*',
        s=15,
        c='r',
        height=6,
        width=10,
        ax=None
    )
    plt.title("Efficient Frontier")
    plt.show()

    # --- Performance Metrics ---
  
    w_vec = weights.values.flatten()

 
    mu_daily = portfolio_obj.mu.values.flatten()
    expected_daily_return = np.dot(w_vec, mu_daily)

    cov_matrix = portfolio_obj.cov.values
    daily_volatility = np.sqrt(np.dot(w_vec.T, np.dot(cov_matrix, w_vec)))

    expected_annual_return = expected_daily_return * 252
    annual_volatility = daily_volatility * np.sqrt(252)

    sharpe_ratio_annual = (expected_annual_return - (rf_daily * 252)) / annual_volatility

    print("\n" + "="*30)
    print("PORTFOLIO PERFORMANCE")
    print("="*30)
    print(f"Expected Annual Return : {expected_annual_return:.2%}")
    print(f"Annual Volatility      : {annual_volatility:.2%}")
    print(f"Sharpe Ratio (Annual)  : {sharpe_ratio_annual:.4f}")

    weights_percent = (weights * 100).round(2)
    # The optimization returns a column named 'weights'
    weights_percent = weights_percent.sort_values(by="weights", ascending=False)
    
    print("\nOptimized Allocation (%):")
    print(weights_percent)

else:
    print("Optimization could not be completed. Please check constraints or data.")
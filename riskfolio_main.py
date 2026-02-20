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

threshold = int(0.9*len(data))  
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




# Optimization Function

def run_optimization(returns_df, config):

    port = rp.Portfolio(returns=returns_df)
    model = config["model"]
    rf_daily = config["rf_daily"]
    lambda_val = config["lambda_val"]
    risk_measure = config["risk_measure"]
    cov_method = config["cov_method"]
    mu_method = config["mu_method"]
    objective = config["objective"]
    lower_bound = config["lower_bound"]
    upper_bound = config["upper_bound"]


    # Estimate statistics
    port.assets_stats(method_mu=mu_method, method_cov=cov_method)

    # Attach constraints
    port.ainequality = A
    port.binequality = B

    # Explicit constraints
    port.budget = 1
    port.lowerbound = lower_bound
    port.upperbound = upper_bound

    try:
        w = port.optimization(
            model=model,
            rm=risk_measure,
            obj=objective,
            rf=rf_daily,
            l=lambda_val,
            hist=True
        )

        return w, port

    except Exception as e:
        print(f"Optimization failed: {e}")
        return None, None


weights, portfolio_obj = run_optimization(returns, CONFIG)

#  Performance Metrics (Manual Calculation) 

# Convert weights to numpy vector
w_vec = weights.values.flatten()

# Expected daily return
mu_daily = portfolio_obj.mu.values.flatten()
expected_daily_return = np.dot(w_vec, mu_daily)

# Daily volatility
cov_matrix = portfolio_obj.cov.values
daily_volatility = np.sqrt(np.dot(w_vec.T, np.dot(cov_matrix, w_vec)))

# Annualize
expected_annual_return = expected_daily_return * 252
annual_volatility = daily_volatility * np.sqrt(252)

# Sharpe Ratio
sharpe_ratio = (expected_daily_return - rf_daily) / daily_volatility
sharpe_ratio_annual = sharpe_ratio * np.sqrt(252)

print("\nPortfolio Performance:")
print(f"Expected Annual Return : {expected_annual_return:.4f}")
print(f"Annual Volatility      : {annual_volatility:.4f}")
print(f"Sharpe Ratio (Annual)  : {sharpe_ratio_annual:.4f}")

weights_percent = (weights * 100).round(2)
weights_percent = weights_percent.sort_values(by="weights", ascending=False)

print("\nOptimized Allocation (%):")
print(weights_percent)
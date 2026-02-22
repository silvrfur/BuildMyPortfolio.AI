
#Conservative portfolio

# CONFIG = {
#     "model": "Classic",
#     "risk_measure": "CVaR",
#     "lambda_val": 6,
#     "objective": "Utility",
#     "cov_method": "ledoit",
#     "mu_method": "hist",
#     "lower_bound": 0.05,
#     "upper_bound": 0.25,
#     "rf_daily": (1 + 0.06)**(1/252) - 1
# }


#Aggressive portfolio

CONFIG = {
    "model": "Classic",
    "risk_measure": "MV",
    "lambda_val": 0.8,
    "objective": "Utility",
    "cov_method": "oas",
    "mu_method": "hist",
    "lower_bound": 0,
    "upper_bound": 0.60,
    "rf_daily": (1 + 0.07)**(1/252) - 1
}




# MODEL TYPE

# "Classic"     → Mean–Risk optimization (most common)
# "BL"          → Black-Litterman model
# "FM"          → Factor model
# "Robust"      → Robust optimization
# "RiskBudget"  → Risk budgeting framework
# "MAD"         → Mean Absolute Deviation model




# RISK MEASURE (rm)

# "MV"     → Variance (Markowitz)
# "MAD"    → Mean Absolute Deviation
# "MSV"    → Semi-Variance
# "VaR"    → Value at Risk
# "CVaR"   → Conditional Value at Risk
# "WR"     → Worst Realization
# "FLPM"   → First Lower Partial Moment
# "SLPM"   → Second Lower Partial Moment
# "EVaR"   → Entropic Value at Risk
# "CDaR"   → Conditional Drawdown at Risk
# "UCI"    → Ulcer Index
# "MDD"    → Maximum Drawdown
# "ADD"    → Average Drawdown
# "DaR"    → Drawdown at Risk
# "EDaR"   → Entropic Drawdown at Risk





# OBJECTIVE FUNCTION (obj)

# "MinRisk" → Minimize risk
# "MaxRet"  → Maximize expected return
# "Sharpe"  → Maximize Sharpe ratio
# "Utility" → Maximize mean - λ * risk
# "MaxDiv"  → Maximum diversification





# RISK AVERSION (λ)

# Only used when objective = "Utility"
# 0.1  → Very aggressive
# 1    → Aggressive
# 3    → Moderate (default)
# 5    → Conservative
# 10   → Very defensive
# No strict upper bound, but typical range: 0.1 – 10





# COVARIANCE ESTIMATION

# "hist"    → Historical sample covariance
# "ledoit"  → Ledoit-Wolf shrinkage (recommended)
# "oas"     → Oracle Approximating Shrinkage
# "shrunk"  → Basic shrinkage
# "gl"      → Graphical Lasso





# EXPECTED RETURN ESTIMATION

# "hist" → Historical mean return
# "ewma" → Exponentially weighted mean
# "capm" → CAPM expected return
# "bl"   → Black-Litterman mean (used with BL model)




# WEIGHT BOUNDS

# 0 ≤ lower_bound ≤ 1
# 0 ≤ upper_bound ≤ 1
#
# Examples:
# 0, 1      → Fully flexible long-only
# 0.02, 0.40 → Diversified institutional style
# 0.05, 0.25 → Strict diversification
# 0, 0.80   → Concentrated portfolio allowed




# RISK-FREE RATE (DAILY)

# Formula: (1 + annual_rate)**(1/252) - 1
#
# 5%  → (1.05)**(1/252) - 1
# 6%  → (1.06)**(1/252) - 1
# 7%  → (1.07)**(1/252) - 1

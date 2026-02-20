
CONFIG = {

    "model": "Classic",
    "risk_measure": "MV",       
    "lambda_val": 3,
    "objective": "Utility",
    "cov_method": "ledoit",
    "mu_method": "hist",
    "lower_bound": 0.02,
    "upper_bound": 0.40,
    "rf_daily": (1 + 0.07)**(1/252) - 1
}
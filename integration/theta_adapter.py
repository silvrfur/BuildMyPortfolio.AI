import copy
from portfolio_optimizer.config import (
    CONSERVATIVE_CONFIG,
    BALANCED_CONFIG,
    AGGRESSIVE_CONFIG,
)

def select_config_from_theta(theta):

    # Convert Pydantic → dict
    if hasattr(theta, "model_dump"):
        theta = theta.model_dump()
    elif hasattr(theta, "dict"):
        theta = theta.dict()

    risk = theta["risk_sensitivity"]
    patience = theta["patience_level"]
    analytical = theta["analytical_thinking"]
    control = theta["controlled_perception"]

    # Profile score
    score = (
        0.45 * risk +
        0.2 * analytical +
        0.25 * (1 - patience) +
        0.1 * control
    )

    if score < 0.4:
        base = CONSERVATIVE_CONFIG
    elif score < 0.7:
        base = BALANCED_CONFIG
    else:
        base = AGGRESSIVE_CONFIG

    config = copy.deepcopy(base)

    # Dynamic tuning
    config["lambda_val"] = max(1, min(10, 8 - 6*risk))
    config["upper_bound"] = min(0.25, 0.05 + 0.2*risk)

    # Use control properly
    if control < 0.3:
        config["risk_measure"] = "CVaR"
        config["lambda_val"] += 1

    elif control > 0.7:
        config["risk_measure"] = "MV"

    # Analytical tuning
    if analytical > 0.7:
        config["mu_method"] = "ewma1"

    return config
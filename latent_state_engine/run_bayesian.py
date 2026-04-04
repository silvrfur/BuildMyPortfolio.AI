from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from latent_state_engine.bayesian_update import BayesianLatentEngine
from latent_state_engine.mapper import map_signals_to_latent


def run_bayesian(
    nlp_signals: Mapping[str, float],
    *,
    strength: float | Mapping[str, float] = 0.5,
    include_uncertainty: bool = False,
    decay_factor: float | Mapping[str, float] | None = None,
    engine: BayesianLatentEngine | None = None,
) -> dict[str, float] | dict[str, dict[str, float]]:
    """
    Convert NLP signal dict -> latent evidence -> Bayesian posterior values.

    Args:
        nlp_signals: NLP engine output dictionary.
        strength: Update strength for the Bayesian posterior update.
        include_uncertainty: If True, include posterior variance.
        decay_factor: Optional temporal decay factor in [0, 1].
        engine: Optional existing engine instance for stateful updates.

    Returns:
        Posterior latent means, or means + variances.
    """
    active_engine = engine or BayesianLatentEngine()

    latent_evidence = map_signals_to_latent(nlp_signals)
    active_engine.update_batch(latent_evidence, strength=strength)

    if decay_factor is not None:
        active_engine.apply_decay(decay_factor=decay_factor)

    means = {key: float(value) for key, value in active_engine.get_means().items()}
    if not include_uncertainty:
        return means

    variances = {key: float(value) for key, value in active_engine.get_variances().items()}
    return {"means": means, "variances": variances}


def run_bayesian_from_any(
    nlp_signals: Mapping[str, Any],
    *,
    strength: float | Mapping[str, float] = 0.5,
    include_uncertainty: bool = False,
    decay_factor: float | Mapping[str, float] | None = None,
    engine: BayesianLatentEngine | None = None,
) -> dict[str, float] | dict[str, dict[str, float]]:
    """
    Same as run_bayesian, but accepts any scalar-like mapping values.
    Values are cast to float before schema validation.
    """
    casted = {key: float(value) for key, value in nlp_signals.items()}
    return run_bayesian(
        casted,
        strength=strength,
        include_uncertainty=include_uncertainty,
        decay_factor=decay_factor,
        engine=engine,
    )

# This module contains the core mapping logic that transforms observable NLP signals into latent-state evidence values.

from __future__ import annotations

from collections.abc import Mapping

from latent_state_engine.schemas import LatentStateSchema
from nlp.schemas import NLPSignals


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(max_val, value))


def normalize_time_bias(time_horizon_bias: float) -> float:
    """
    Converts [-1, 1] -> [0, 1].
    In NLPSignals: -1 = short-term, +1 = long-term.
    """
    return (time_horizon_bias + 1.0) / 2.0


def compute_risk_sensitivity(signal: NLPSignals) -> float:
    risk = (
        0.40 * signal.fear_sentiment
        + 0.25 * signal.uncertainty_score
        + 0.20 * signal.risk_language_density
        + 0.15 * signal.herding_marker
    )
    return clamp(risk)


def compute_patience_level(signal: NLPSignals) -> float:
    long_term_orientation = normalize_time_bias(signal.time_horizon_bias)

    patience = (
        0.50 * long_term_orientation
        + 0.35 * (1.0 - signal.urgency_score)
        + 0.15 * (1.0 - signal.fear_sentiment)
    )
    return clamp(patience)


def compute_analytical_thinking(signal: NLPSignals) -> float:
    analytical = (
        0.50 * signal.analytical_marker
        + 0.25 * (1.0 - signal.fear_sentiment)
        + 0.25 * (1.0 - signal.uncertainty_score)
    )
    return clamp(analytical)


def compute_controlled_perception(signal: NLPSignals) -> float:
    control = (
        0.50 * signal.internal_locus_score
        + 0.30 * (1.0 - signal.external_locus_score)
        + 0.20 * (1.0 - signal.herding_marker)
    )
    return clamp(control)


def map_signals_to_latent(signal: NLPSignals | Mapping[str, float]) -> LatentStateSchema:
    """
    Main mapper function.
    Converts observable NLP signals into latent-state evidence values.
    Returns LatentStateSchema with values in [0, 1].
    """
    nlp_signal = signal if isinstance(signal, NLPSignals) else NLPSignals.model_validate(signal)

    return LatentStateSchema(
        risk_sensitivity=compute_risk_sensitivity(nlp_signal),
        patience_level=compute_patience_level(nlp_signal),
        analytical_thinking=compute_analytical_thinking(nlp_signal),
        controlled_perception=compute_controlled_perception(nlp_signal),
    )

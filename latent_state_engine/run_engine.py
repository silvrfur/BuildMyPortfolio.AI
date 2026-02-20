from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from latent_state_engine.bayesian_update import BayesianLatentEngine, clamp
from latent_state_engine.mapper import map_signals_to_latent
from latent_state_engine.schemas import LatentStateSchema
from nlp.signal_extractor import extract_schema_variables
from nlp.schemas import NLPSignals


@dataclass(frozen=True)
class EngineResult:
    text: str
    observable_signals: NLPSignals
    latent_evidence: LatentStateSchema
    latent_posterior: LatentStateSchema
    portfolio_inputs: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "observable_signals": self.observable_signals.model_dump(),
            "latent_evidence": self.latent_evidence.model_dump(),
            "latent_posterior": self.latent_posterior.model_dump(),
            "portfolio_inputs": self.portfolio_inputs,
        }


def build_portfolio_inputs(latent: LatentStateSchema) -> dict[str, float]:
    """
    Translate latent states into normalized inputs for a portfolio engine.
    All outputs are in [0, 1] and can be consumed by an allocation module.
    """
    risk = latent.risk_sensitivity
    patience = latent.patience_level
    analytical = latent.analytical_thinking
    control = latent.controlled_perception

    return {
        "risk_budget": clamp(risk),
        "equity_bias": clamp(0.50 * risk + 0.30 * patience + 0.20 * control),
        "cash_buffer_bias": clamp(1.0 - (0.60 * risk + 0.40 * control)),
        "rebalance_frequency": clamp(1.0 - patience),
        "research_intensity": clamp(analytical),
        "active_management_tilt": clamp(0.55 * control + 0.45 * analytical),
    }


def run_engine(
    text: str,
    bayes_engine: BayesianLatentEngine | None = None,
    strength: float = 0.5,
) -> EngineResult:
    """
    End-to-end pipeline:
    user text -> NLP observable signals -> latent mapper -> Bayesian update -> portfolio inputs.
    """
    engine = bayes_engine or BayesianLatentEngine()

    observable_signals = extract_schema_variables(text)
    latent_evidence = map_signals_to_latent(observable_signals)

    engine.update_batch(latent_evidence, strength=strength)
    posterior = LatentStateSchema.model_validate(engine.get_means())

    portfolio_inputs = build_portfolio_inputs(posterior)

    return EngineResult(
        text=text,
        observable_signals=observable_signals,
        latent_evidence=latent_evidence,
        latent_posterior=posterior,
        portfolio_inputs=portfolio_inputs,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run text -> NLP -> latent-state -> portfolio-input pipeline."
    )
    parser.add_argument(
        "--text",
        required=True,
        help="User text to analyze.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.5,
        help="Bayesian update strength per observation (default: 0.5).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_engine(text=args.text, strength=args.strength)
    print(result.to_dict())


if __name__ == "__main__":
    main()

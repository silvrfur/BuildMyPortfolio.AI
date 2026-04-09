# from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from typing import Literal

from latent_state_engine.schemas import LatentStateSchema


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(max_val, value))


LatentName = Literal[
    "risk_sensitivity",
    "patience_level",
    "analytical_thinking",
    "controlled_perception",
]


@dataclass
class BetaState:
    alpha: float = 1.0
    beta: float = 1.0

    def mean(self) -> float:
        """Posterior mean of the Beta distribution."""
        total = self.alpha + self.beta
        return self.alpha / total if total > 0 else 0.5

    def variance(self) -> float:
        """Posterior variance (uncertainty) of the Beta distribution."""
        total = self.alpha + self.beta
        if total <= 0:
            return 0.0
        return (self.alpha * self.beta) / ((total**2) * (total + 1.0))


class BayesianLatentEngine:
    def __init__(self) -> None:
        self.states: dict[LatentName, BetaState] = {
            "risk_sensitivity": BetaState(),
            "patience_level": BetaState(),
            "analytical_thinking": BetaState(),
            "controlled_perception": BetaState(),
        }

    def update(
        self,
        latent_name: LatentName,
        evidence: float,
        strength: float | Mapping[str, float] = 0.5,
    ) -> None:
        """
        Update a single latent state using fractional evidence.

        `evidence` is expected in [0, 1].
        `strength` controls how strongly one observation affects belief.
        """
        evidence = clamp(evidence)
        if isinstance(strength, Mapping):
            strength_value = max(0.0, float(strength.get(latent_name, 0.5)))
        else:
            strength_value = max(0.0, float(strength))

        state = self.states[latent_name]
        state.alpha += strength_value * evidence
        state.beta += strength_value * (1.0 - evidence)

    def update_batch(
        self,
        evidence: LatentStateSchema | dict[str, float],
        strength: float | Mapping[str, float] = 0.5,
    ) -> None:
        """Update all latent variables at once."""
        latent = evidence if isinstance(evidence, LatentStateSchema) else LatentStateSchema.model_validate(evidence)
        payload = latent.model_dump()

        for latent_name, value in payload.items():
            self.update(latent_name, value, strength=strength)

    def get_means(self) -> dict[LatentName, float]:
        """Return posterior means for all latent states."""
        return {name: state.mean() for name, state in self.states.items()}

    def get_variances(self) -> dict[LatentName, float]:
        """Return posterior variances for all latent states."""
        return {name: state.variance() for name, state in self.states.items()}

    def get_params(self) -> dict[LatentName, dict[str, float]]:
        """Return posterior Beta parameters for all latent states."""
        return {
            name: {"alpha": float(state.alpha), "beta": float(state.beta)}
            for name, state in self.states.items()
        }

    def apply_decay(self, decay_factor: float | Mapping[str, float] = 0.99) -> None:
        """
        Apply temporal decay so older evidence loses influence.
        Values are clamped to keep alpha/beta numerically stable and non-negative.
        """
        if isinstance(decay_factor, Mapping):
            for latent_name, state in self.states.items():
                current_decay = clamp(float(decay_factor.get(latent_name, 0.99)))
                state.alpha = max(1e-9, state.alpha * current_decay)
                state.beta = max(1e-9, state.beta * current_decay)
            return

        current_decay = clamp(float(decay_factor))
        for state in self.states.values():
            state.alpha = max(1e-9, state.alpha * current_decay)
            state.beta = max(1e-9, state.beta * current_decay)

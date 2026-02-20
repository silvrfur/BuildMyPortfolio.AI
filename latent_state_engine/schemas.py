# This module defines the schema for the latent state probabilities, ensuring that all values are constrained to the [0.0, 1.0] range.

from typing import Annotated

from pydantic import BaseModel, Field


Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class LatentStateSchema(BaseModel):
    risk_sensitivity: Probability
    patience_level: Probability
    analytical_thinking: Probability
    controlled_perception: Probability

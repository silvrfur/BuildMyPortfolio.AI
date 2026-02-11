from pydantic import BaseModel, Field, confloat


class NLPSignals(BaseModel):
    """
    Observable NLP-derived behavioral signals.

    These variables represent measurable linguistic evidence
    extracted from user text. They are NOT latent states.
    """

   
    # Risk Psychology 
    
    #--confloat means constrained float with ge (greater equal) and le (less equal)--
    #-- ... in Field means required field with description--

    fear_sentiment: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Probability of fear/negative sentiment in user language"
    )

    risk_language_density: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Proportion of risk- and loss-related terms in text"
    )

    # Time Psychology 

    #Note: Time_horizon_bias is bipolar(directional) while rest are unipolar (intensity)
    
    time_horizon_bias: confloat(ge=-1.0, le=1.0) = Field(
        ...,
        description="Temporal orientation: -1 (short-term) to +1 (long-term)"
    )

    urgency_score: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Degree of urgency or immediacy expressed in language"
    )

    # Information Processing 
    
    analytical_marker: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Evidence of analytical, data-driven reasoning"
    )

    herding_marker: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Evidence of social imitation or trend-following language"
    )

    # Control Perception 
    
    internal_locus_score: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Degree of internal locus of control (personal agency)"
    )

    external_locus_score: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Degree of external locus of control (market/fate driven)"
    )
   
    # Meta / Confidence
   
    uncertainty_score: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Degree of hedging, doubt, or epistemic uncertainty"
    )

    class Config:
        #--Enforces strict schema usage and prevents silent errors--
        extra = "forbid"          # --No undeclared fields allowed--
        validate_assignment = True # --Re-validate on assignment--
        frozen = True             # --Immutable once created--

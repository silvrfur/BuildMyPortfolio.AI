"""
Extract all NLP schema variables through the transformer signal encoder.
"""

from typing import Dict

from nlp.signal_encoder import TransformerSignals
from nlp.schemas import NLPSignals

_transformers = None

#--create the tranformer when needed, not at the start of the test--
def _get_transformers() -> TransformerSignals:
    global _transformers
    if _transformers is None:
        _transformers = TransformerSignals()  # created only on first use
    return _transformers

# Main extractor

def extract_schema_variables(text: str) -> NLPSignals:
    signal_values: Dict[str, float] = {}
    transformers = _get_transformers()

    signal_values["fear_sentiment"] = transformers.extract_fear_sentiment(text)
    analytical, herding = transformers.extract_analytical_and_herding_markers(text)
    signal_values["analytical_marker"] = analytical
    signal_values["herding_marker"] = herding
    signal_values["uncertainty_score"] = transformers.extract_uncertainty_score(text)
    signal_values["risk_language_density"] = (
        transformers.extract_risk_language_density(text)
    )
    signal_values["urgency_score"] = transformers.extract_urgency_score(text)
    signal_values["time_horizon_bias"] = transformers.extract_time_horizon_bias(text)
    internal_locus, external_locus = transformers.extract_locus_of_control(text)
    signal_values["internal_locus_score"] = internal_locus
    signal_values["external_locus_score"] = external_locus


    # Schema validation
    return NLPSignals(**signal_values)


def extract_schema_variables_with_metadata(text: str) -> tuple[NLPSignals, dict[str, object]]:
    signals = extract_schema_variables(text)
    transformers = _get_transformers()
    metadata = transformers.diagnostics()
    return signals, metadata

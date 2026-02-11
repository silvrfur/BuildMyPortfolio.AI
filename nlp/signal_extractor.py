"""
Extracts the valuese for each of the signal variables using hybrid of keyword matchingand transformer models.
""" 

from typing import Dict
from nlp.keywords import KEYWORD_SIGNALS
from nlp.signal_encoder import TransformerSignals
from nlp.schemas import NLPSignals

_transformers = None

#--create the tranformer when needed, not at the start of the test--
def _get_transformers() -> TransformerSignals:
    global _transformers
    if _transformers is None:
        _transformers = TransformerSignals()  # created only on first use
    return _transformers


# Helper functions (Keyword Utility)

def _keyword_count(text: str, keywords: list) -> int:
    text = text.lower()
    return sum(1 for kw in keywords if kw in text)


def _normalize(count: int, total_words: int, mode: str) -> float:
    if mode == "binary": #checks for presence or absence of keywords
        return float(count > 0)
    if mode == "density": #reflects "how much" keyword language appears 
        return min(count / max(total_words, 1), 1.0) #--Note: total_words is the length of they keyword list--
    return 0.0



# Main extractor

def extract_schema_variables(text: str) -> NLPSignals:
    text_lower = text.lower()
    total_words = len(text_lower.split())

    signal_values: Dict[str, float] = {}

    # Keyword-based signals (dynamic)
    for field_name,cfg in KEYWORD_SIGNALS.items():
        if field_name=="time_horizon_bias":
            continue #skip this one for now, since it's a directional signal and needs special handling

        count = _keyword_count(text_lower, cfg["keywords"])
        #!!! normalization doen't exist yet, but we can add it to the config if needed, passing "density" as default for now
        value = _normalize(count, total_words, "density")
        signal_values[field_name] = value

    # Transformer-based signals (fear_sentiment, analytical_marker,herding_marker, uncertainty_score)
    transformers = _get_transformers()
    signal_values["fear_sentiment"] = transformers.extract_fear_sentiment(text)
    analytical, herding = transformers.extract_analytical_and_herding_markers(text)
    signal_values["analytical_marker"] = analytical
    signal_values["herding_marker"] = herding
    signal_values["uncertainty_score"] = transformers.extract_uncertainty_score(text)

    #!!! Currently values are discrete {-1,,0,1}, later to be made continuous [-1.-,1.0]
    # Time horizon (bi-directional signal)
    time_cfg = KEYWORD_SIGNALS.get("time_horizon_bias", {})
    long_terms = time_cfg.get("long_term", [])
    short_terms = time_cfg.get("short_term", [])

    long_hit = any(term in text_lower for term in long_terms) #--returns True if at least one item is True--
    short_hit = any(term in text_lower for term in short_terms)

    if long_hit and not short_hit:
        signal_values["time_horizon_bias"] = 1.0
    elif short_hit and not long_hit:
        signal_values["time_horizon_bias"] = -1.0
    else:
        signal_values["time_horizon_bias"] = 0.0


    # Schema validation
    return NLPSignals(**signal_values)

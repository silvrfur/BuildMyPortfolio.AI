import pytest
import nlp.signal_extractor as signal_extractor_module
from nlp.signal_extractor import extract_schema_variables
from nlp.signal_extractor import extract_schema_variables_with_metadata
from nlp.schemas import NLPSignals
import logging

logger=logging.getLogger(__name__)

# Helper function to print schema per test
def _show(name, signals):
    logger.info("[%s] %s", name, signals.model_dump())

# Basic integration test

def test_extractor_returns_schema():
    text = "I am scared the market will crash soon"
    signals = extract_schema_variables(text)
    _show(text, signals)

    assert isinstance(signals, NLPSignals)


# Fear sentiment sanity check

def test_fear_sentiment_high_for_fearful_text():
    text = "I am very scared and worried about losing money"
    signals = extract_schema_variables(text)
    _show(text, signals)

    assert 0.0 <= signals.fear_sentiment <= 1.0
    assert signals.fear_sentiment > 0.5


# Herding vs analytical behavior

def test_herding_vs_analytical_difference():
    herding_text = "Everyone is buying this stock so I will too"
    analytical_text = "Based on financial statements and data analysis"

    herding_signals = extract_schema_variables(herding_text)
    analytical_signals = extract_schema_variables(analytical_text)
    _show(herding_text, herding_signals)
    _show(analytical_text, analytical_signals)

    assert herding_signals.herding_marker >= analytical_signals.herding_marker
    assert analytical_signals.analytical_marker >= herding_signals.analytical_marker


# Time horizon direction

def test_long_term_time_horizon():
    text = "I want to hold this investment for the long term"
    signals = extract_schema_variables(text)
    _show(text, signals)

    assert signals.time_horizon_bias >= 0.0


def test_short_term_time_horizon():
    text = "I want quick returns today"
    signals = extract_schema_variables(text)
    _show(text, signals)

    assert signals.time_horizon_bias <= 0.0


# Uncertainty detection

def test_uncertainty_signal():
    text = "I am not sure what to do, maybe I should wait"
    signals = extract_schema_variables(text)
    _show(text, signals)

    assert 0.0 <= signals.uncertainty_score <= 1.0
    assert signals.uncertainty_score > 0.4


# Schema strictness

def test_schema_is_fully_populated():
    text = "Neutral statement with no strong signals"
    signals = extract_schema_variables(text)
    _show(text, signals)

    for field in signals.__fields__:
        value = getattr(signals, field)
        assert value is not None


def test_transformer_backed_fields_override_keyword_only_path(monkeypatch):
    class FakeTransformers:
        def has_remote_support(self):
            return True

        def diagnostics(self):
            return {"backend_mode": "local_transformers"}

        def extract_fear_sentiment(self, _text):
            return 0.7

        def extract_analytical_and_herding_markers(self, _text):
            return 0.8, 0.2

        def extract_uncertainty_score(self, _text):
            return 0.1

        def extract_risk_language_density(self, _text):
            return 0.9

        def extract_urgency_score(self, _text):
            return 0.85

        def extract_time_horizon_bias(self, _text):
            return -0.6

        def extract_locus_of_control(self, _text):
            return 0.75, 0.15

    monkeypatch.setattr(signal_extractor_module, "_get_transformers", lambda: FakeTransformers())
    signals = extract_schema_variables("neutral text without obvious keywords")

    assert signals.risk_language_density == pytest.approx(0.9)
    assert signals.urgency_score == pytest.approx(0.85)
    assert signals.time_horizon_bias == pytest.approx(-0.6)
    assert signals.internal_locus_score == pytest.approx(0.75)
    assert signals.external_locus_score == pytest.approx(0.15)


def test_extract_schema_variables_with_metadata_returns_backend(monkeypatch):
    class FakeTransformers:
        def diagnostics(self):
            return {"backend_mode": "local_transformers", "backend_preference": "local"}

        def extract_fear_sentiment(self, _text):
            return 0.6

        def extract_analytical_and_herding_markers(self, _text):
            return 0.7, 0.3

        def extract_uncertainty_score(self, _text):
            return 0.2

        def extract_risk_language_density(self, _text):
            return 0.65

        def extract_urgency_score(self, _text):
            return 0.5

        def extract_time_horizon_bias(self, _text):
            return 0.1

        def extract_locus_of_control(self, _text):
            return 0.8, 0.2

    monkeypatch.setattr(signal_extractor_module, "_get_transformers", lambda: FakeTransformers())
    signals, metadata = extract_schema_variables_with_metadata("test")

    assert isinstance(signals, NLPSignals)
    assert metadata["backend_mode"] == "local_transformers"

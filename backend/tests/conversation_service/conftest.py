# tests/conversation/conftest.py
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_nlp_output():
    return {
        "sentiment": {"label": "NEGATIVE", "score": 0.91},
        "emotions": {
            "fear": 0.8,
            "anger": 0.1,
            "joy": 0.05
        },
        "urgency": 0.7
    }

@pytest.fixture
def sample_message():
    return "I am scared the market will crash and I want to sell now"

@pytest.fixture
def user_id():
    return "user_123"

@pytest.fixture
def mock_llm_response():
    return "I understand your concern. Let's slow down and assess calmly."

@pytest.fixture
def mock_signal():
    return {
        "loss_reactivity": 0.82,
        "temporal_compression": 0.76,
        "gain_attention": 0.12
    }

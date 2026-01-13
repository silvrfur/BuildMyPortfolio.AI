# tests/conversation/test_signals.py

def test_signal_schema(mock_signal):
    required_keys = {
        "loss_reactivity",
        "temporal_compression",
        "gain_attention"
    }

    assert required_keys.issubset(mock_signal.keys())

def test_signal_value_ranges(mock_signal):
    for value in mock_signal.values():
        assert 0 <= value <= 1

# tests/conversation/test_nlp.py

def test_nlp_features_structure(mock_nlp_output):
    assert "sentiment" in mock_nlp_output
    assert "emotions" in mock_nlp_output
    assert "urgency" in mock_nlp_output

def test_emotion_values_range(mock_nlp_output):
    for value in mock_nlp_output["emotions"].values():
        assert 0 <= value <= 1

def test_urgency_score_range(mock_nlp_output):
    assert 0 <= mock_nlp_output["urgency"] <= 1

# tests/conversation/test_service.py
from unittest.mock import Mock

def test_process_message_returns_response(
    conversation_service,
    sample_message,
    user_id,
    mock_llm_response
):
    conversation_service.generate_response = Mock(
        return_value=mock_llm_response
    )

    result = conversation_service.process_message(
        user_id=user_id,
        message=sample_message
    )

    assert "response" in result
    assert result["response"] == mock_llm_response

def test_process_message_emits_signal(
    conversation_service,
    sample_message,
    user_id
):
    conversation_service.emit_signal = Mock()

    conversation_service.process_message(
        user_id=user_id,
        message=sample_message
    )

    conversation_service.emit_signal.assert_called_once()

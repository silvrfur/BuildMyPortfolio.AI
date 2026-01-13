# tests/conversation/test_schemas.py
from pydantic import ValidationError
import pytest

def test_valid_chat_request_schema(ChatRequest):
    data = {
        "user_id": "u1",
        "message": "Should I sell?",
    }
    req = ChatRequest(**data)
    assert req.user_id == "u1"

def test_invalid_chat_request_schema(ChatRequest):
    with pytest.raises(ValidationError):
        ChatRequest(user_id="u1")  # missing message

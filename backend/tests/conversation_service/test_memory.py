# tests/conversation/test_memory.py

def test_add_message_to_memory(memory_store, user_id, sample_message):
    memory_store.add(user_id, sample_message)
    history = memory_store.get(user_id)

    assert len(history) == 1
    assert history[0]["content"] == sample_message

def test_message_order_preserved(memory_store, user_id):
    memory_store.add(user_id, "msg1")
    memory_store.add(user_id, "msg2")

    history = memory_store.get(user_id)
    assert history[0]["content"] == "msg1"
    assert history[1]["content"] == "msg2"

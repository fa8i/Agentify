import pytest

from agentify.memory.interfaces import MemoryAddress, Message
from agentify.memory.stores.sqlite_store import SQLiteStore


def test_memory_address_encoding_roundtrip(tmp_path):
    db_path = tmp_path / "memory.db"
    store = SQLiteStore(str(db_path))

    addr = MemoryAddress(
        user_id="user:1",
        conversation_id="conv=1",
        agent_id="agent/name",
        extras=(("chan:1", "a:b"),),
    )
    store.append_message(addr, Message(role="user", content="hi"))

    conversations = store.list_conversations()
    assert len(conversations) == 1

    recovered = conversations[0]
    assert recovered.user_id == addr.user_id
    assert recovered.conversation_id == addr.conversation_id
    assert recovered.agent_id == addr.agent_id
    assert recovered.extras == addr.extras

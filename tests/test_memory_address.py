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


def test_memory_address_from_key_str_basic():
    """Test basic from_key_str functionality."""
    addr = MemoryAddress(
        user_id="user1",
        conversation_id="conv1",
        agent_id="agent1",
    )
    key = addr.key_str()
    recovered = MemoryAddress.from_key_str(key)
    
    assert recovered.user_id == addr.user_id
    assert recovered.conversation_id == addr.conversation_id
    assert recovered.agent_id == addr.agent_id


def test_memory_address_from_key_str_all_fields():
    """Test from_key_str with all fields populated."""
    addr = MemoryAddress(
        api_version="v1",
        tenant_id="tenant123",
        user_id="user456",
        conversation_id="conv789",
        agent_id="agent_abc",
    )
    key = addr.key_str()
    recovered = MemoryAddress.from_key_str(key)
    
    assert recovered == addr


def test_memory_address_from_key_str_with_special_chars():
    """Test roundtrip with URL-unsafe characters."""
    addr = MemoryAddress(
        user_id="user:with/special=chars",
        conversation_id="conv:id",
        extras=(("key:1", "val=ue"),),
    )
    key = addr.key_str()
    recovered = MemoryAddress.from_key_str(key)
    
    assert recovered.user_id == addr.user_id
    assert recovered.conversation_id == addr.conversation_id
    assert recovered.extras == addr.extras


def test_memory_address_from_key_str_empty():
    """Test parsing empty/minimal key."""
    addr = MemoryAddress()
    key = addr.key_str()  # Should be just "mem"
    recovered = MemoryAddress.from_key_str(key)
    
    assert recovered == addr


def test_memory_address_from_key_str_custom_prefix():
    """Test with custom prefix."""
    addr = MemoryAddress(user_id="user1")
    key = addr.key_str(prefix="custom")
    recovered = MemoryAddress.from_key_str(key, prefix="custom")
    
    assert recovered.user_id == addr.user_id

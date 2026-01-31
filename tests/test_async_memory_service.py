"""Tests for AsyncMemoryService."""
import pytest
import asyncio

from agentify.memory.interfaces import MemoryAddress, Message
from agentify.memory.stores.in_memory_store import InMemoryStore
from agentify.memory.service import MemoryService
from agentify.memory.async_service import AsyncMemoryService


@pytest.fixture
def memory_setup():
    """Create a store, sync service, and async service for testing."""
    store = InMemoryStore()
    sync_service = MemoryService(store, log_enabled=False)
    async_service = AsyncMemoryService(store, log_enabled=False)
    addr = MemoryAddress(user_id="test_user", conversation_id="test_conv")
    return store, sync_service, async_service, addr


@pytest.mark.asyncio
async def test_async_append_and_get_history(memory_setup):
    """Test that async append and get work correctly."""
    store, sync_service, async_service, addr = memory_setup
    
    # Append via async
    await async_service.append_history(addr, {"role": "user", "content": "Hello async"})
    
    # Get via async
    history = await async_service.get_history(addr)
    
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello async"


@pytest.mark.asyncio
async def test_async_reset_history(memory_setup):
    """Test async reset history."""
    store, sync_service, async_service, addr = memory_setup
    
    # Add some messages
    await async_service.append_history(addr, {"role": "user", "content": "Message 1"})
    await async_service.append_history(addr, {"role": "assistant", "content": "Response 1"})
    
    # Reset to system message
    await async_service.reset_history(addr, {"role": "system", "content": "You are helpful."})
    
    history = await async_service.get_history(addr)
    assert len(history) == 1
    assert history[0]["role"] == "system"


@pytest.mark.asyncio
async def test_async_delete_history(memory_setup):
    """Test async delete history."""
    store, sync_service, async_service, addr = memory_setup
    
    await async_service.append_history(addr, {"role": "user", "content": "To delete"})
    await async_service.delete_history(addr)
    
    history = await async_service.get_history(addr)
    assert len(history) == 0


@pytest.mark.asyncio
async def test_async_service_from_sync(memory_setup):
    """Test creating AsyncMemoryService from existing MemoryService."""
    store, sync_service, async_service, addr = memory_setup
    
    # Create async service from sync
    async_from_sync = AsyncMemoryService.from_sync(sync_service)
    
    # Use it
    await async_from_sync.append_history(addr, {"role": "user", "content": "From sync wrapper"})
    
    # Verify via sync service (should see the same data)
    history = sync_service.get_history(addr)
    assert len(history) == 1
    assert history[0]["content"] == "From sync wrapper"


@pytest.mark.asyncio
async def test_async_concurrent_operations(memory_setup):
    """Test that multiple async operations can run concurrently."""
    store, sync_service, async_service, addr = memory_setup
    
    # Create multiple addresses
    addrs = [
        MemoryAddress(user_id="user1", conversation_id=f"conv_{i}")
        for i in range(5)
    ]
    
    # Concurrent appends
    await asyncio.gather(*[
        async_service.append_history(a, {"role": "user", "content": f"Message to {a.conversation_id}"})
        for a in addrs
    ])
    
    # Concurrent reads
    histories = await asyncio.gather(*[
        async_service.get_history(a)
        for a in addrs
    ])
    
    # Verify all operations completed correctly
    for i, history in enumerate(histories):
        assert len(history) == 1
        assert f"conv_{i}" in history[0]["content"]

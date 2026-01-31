from __future__ import annotations
import asyncio
from concurrent.futures import Executor
from functools import partial
from typing import Any, Dict, List, Optional, Union

from .interfaces import ConversationStore, MemoryAddress, Message
from .policies import MemoryPolicy
from .service import MemoryService


class AsyncMemoryService:
    """Async facade for memory operations.
    
    Wraps sync ConversationStore operations using run_in_executor to avoid
    blocking the event loop. This provides async compatibility for all existing
    stores without requiring them to implement async methods.
    
    For stores that natively support async (future), this class can detect
    and use native async methods when available.
    
    Attributes:
        store: The underlying conversation store (sync).
        policy: Memory policy for pruning and TTL.
        executor: Optional executor for running sync operations.
            If None, uses the default ThreadPoolExecutor.
    
    Example:
        >>> store = InMemoryStore()
        >>> async_memory = AsyncMemoryService(store)
        >>> history = await async_memory.get_history(addr)
    """

    def __init__(
        self,
        store: ConversationStore,
        policy: Optional[MemoryPolicy] = None,
        executor: Optional[Executor] = None,
        log_enabled: bool = True,
        max_log_length: Optional[int] = 5000,
    ) -> None:
        self.store = store
        self.policy = policy or MemoryPolicy(store)
        self._executor = executor
        # Create a sync service for operations that need logging/normalization
        self._sync_service = MemoryService(
            store=store,
            policy=policy,
            log_enabled=log_enabled,
            max_log_length=max_log_length,
        )

    async def _run_sync(self, func, *args, **kwargs):
        """Run a sync function in the executor without blocking."""
        loop = asyncio.get_running_loop()
        if kwargs:
            func = partial(func, **kwargs)
        return await loop.run_in_executor(self._executor, func, *args)

    async def append_history(self, addr: MemoryAddress, message: Dict[str, Any]) -> None:
        """Append a message to history (async, non-blocking)."""
        await self._run_sync(self._sync_service.append_history, addr, message)

    async def get_history(self, addr: MemoryAddress) -> List[Dict[str, Any]]:
        """Get conversation history as OpenAI-formatted dicts (async)."""
        return await self._run_sync(self._sync_service.get_history, addr)

    async def reset_history(self, addr: MemoryAddress, system_message: Dict[str, Any]) -> None:
        """Reset history to a single system message (async)."""
        await self._run_sync(self._sync_service.reset_history, addr, system_message)

    async def delete_history(self, addr: MemoryAddress) -> None:
        """Delete all messages for the given address (async)."""
        await self._run_sync(self._sync_service.delete_history, addr)

    async def list_conversations(self, limit: int = 100, offset: int = 0) -> List[MemoryAddress]:
        """List active conversations (async)."""
        return await self._run_sync(
            self._sync_service.list_conversations,
            limit,
            offset,
        )

    @classmethod
    def from_sync(cls, sync_service: MemoryService, executor: Optional[Executor] = None) -> "AsyncMemoryService":
        """Create an AsyncMemoryService from an existing MemoryService.
        
        This is useful when you already have a configured MemoryService
        and want to use it in async contexts.
        
        Args:
            sync_service: The existing sync MemoryService.
            executor: Optional executor for sync operations.
            
        Returns:
            An AsyncMemoryService wrapping the sync service's store.
        """
        instance = cls(
            store=sync_service.store,
            policy=sync_service.policy,
            executor=executor,
            log_enabled=sync_service.log_enabled,
            max_log_length=sync_service.max_log_length,
        )
        # Reuse the sync service directly to preserve all configuration
        instance._sync_service = sync_service
        return instance

from __future__ import annotations
import asyncio
from concurrent.futures import Executor
from functools import partial
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .interfaces import ConversationStore, MemoryAddress
from .policies import MemoryPolicy
from .service import MemoryService


T = TypeVar("T")


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
        default_timeout: Optional default timeout for sync operations (seconds).
            If None, no timeout is applied. Can be overridden per-call.
    
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
        default_timeout: Optional[float] = None,
    ) -> None:
        self.store = store
        self._executor = executor
        self._default_timeout = default_timeout
        
        # Create policy once, share with sync service
        resolved_policy = policy or MemoryPolicy(store)
        self.policy = resolved_policy
        
        # Create sync service reusing the same policy instance
        self._sync_service = MemoryService(
            store=store,
            policy=resolved_policy,
            log_enabled=log_enabled,
            max_log_length=max_log_length,
        )

    @property
    def log_enabled(self) -> bool:
        """Whether logging is enabled for memory operations."""
        return self._sync_service.log_enabled

    @property
    def max_log_length(self) -> Optional[int]:
        """Maximum length of logged content (None = unlimited)."""
        return self._sync_service.max_log_length

    @property
    def default_timeout(self) -> Optional[float]:
        """Default timeout for sync operations in seconds (None = no timeout)."""
        return self._default_timeout

    async def _run_sync(
        self,
        func: Callable[..., T],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """Run a sync function in the executor without blocking.
        
        Args:
            func: The sync function to run.
            *args: Positional arguments for the function.
            timeout: Optional timeout in seconds. If None, uses default_timeout.
                If both are None, no timeout is applied.
            **kwargs: Keyword arguments for the function.
            
        Returns:
            The result of the function.
            
        Raises:
            asyncio.TimeoutError: If the operation exceeds the timeout.
        """
        loop = asyncio.get_running_loop()
        if kwargs:
            func = partial(func, **kwargs)
        
        future = loop.run_in_executor(self._executor, func, *args)
        
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout is not None:
            return await asyncio.wait_for(future, timeout=effective_timeout)
        return await future

    async def append_history(self, addr: MemoryAddress, message: Dict[str, Any]) -> None:
        """Append a message to history (async, non-blocking)."""
        await self._run_sync(self._sync_service.append_history, addr, message)

    async def get_history(self, addr: MemoryAddress) -> List[Dict[str, Any]]:
        """Get conversation history as OpenAI-formatted dicts (async)."""
        return await self._run_sync(self._sync_service.get_history, addr)

    async def reset_history(self, addr: MemoryAddress, system_message: Dict[str, Any]) -> None:
        """Reset history to a single system message (async)."""
        await self._run_sync(self._sync_service.reset_history, addr, system_message)

    async def replace_history(self, addr: MemoryAddress, history: List[Dict[str, Any]]) -> None:
        """Replace all messages for the given address with the provided history (async)."""
        await self._run_sync(self._sync_service.replace_history, addr, history)

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
    def from_sync(
        cls,
        sync_service: MemoryService,
        executor: Optional[Executor] = None,
        default_timeout: Optional[float] = None,
    ) -> "AsyncMemoryService":
        """Create an AsyncMemoryService from an existing MemoryService.
        
        This is useful when you already have a configured MemoryService
        and want to use it in async contexts.
        
        Args:
            sync_service: The existing sync MemoryService.
            executor: Optional executor for sync operations.
            default_timeout: Optional default timeout for operations (seconds).
            
        Returns:
            An AsyncMemoryService wrapping the sync service's store.
        """
        instance = cls(
            store=sync_service.store,
            policy=sync_service.policy,
            executor=executor,
            log_enabled=sync_service.log_enabled,
            max_log_length=sync_service.max_log_length,
            default_timeout=default_timeout,
        )
        # Reuse the sync service directly to preserve all configuration
        instance._sync_service = sync_service
        return instance


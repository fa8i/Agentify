from __future__ import annotations

import asyncio
import contextvars
import queue
import threading
from collections.abc import Awaitable, Generator
from typing import Any, AsyncGenerator, Callable, Optional, TypeVar

T = TypeVar("T")


def has_running_loop() -> bool:
    """Return True when an event loop is currently running in this thread."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return True
    except RuntimeError:
        pass

    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def ensure_sync_call_allowed(*, api_name: str, async_api_name: str) -> None:
    """Raise a clear error if a sync API is called in an async context."""
    if has_running_loop():
        raise RuntimeError(
            f"{api_name}() cannot be used while an event loop is running. "
            f"Use `await {async_api_name}()` in async contexts."
        )


def run_coro_blocking(
    coro: Awaitable[T],
    *,
    api_name: str,
    async_api_name: str,
) -> T:
    """Execute a coroutine from sync code with loop safety checks."""
    try:
        ensure_sync_call_allowed(api_name=api_name, async_api_name=async_api_name)
    except Exception:
        close = getattr(coro, "close", None)
        if callable(close):
            close()
        raise
    return asyncio.run(coro)


def stream_async_to_sync(
    async_entrypoint: Callable[[], Awaitable[object]],
    *,
    api_name: str,
    async_api_name: str,
    queue_maxsize: int = 64,
    put_timeout: float = 0.1,
    join_timeout: float = 1.0,
) -> Generator[str, None, None]:
    """Bridge an async streaming entrypoint into a sync generator.

    The bridge runs the async entrypoint inside a background thread with a dedicated
    event loop, forwarding chunks over a bounded queue. It supports cooperative
    cancellation to avoid zombie threads when sync iteration stops early.
    """

    ensure_sync_call_allowed(api_name=api_name, async_api_name=async_api_name)

    q: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=queue_maxsize)
    cancel_event = threading.Event()
    sentinel = object()
    copied_context = contextvars.copy_context()

    def safe_put(item: tuple[str, Any]) -> bool:
        while True:
            if cancel_event.is_set():
                return False
            try:
                q.put(item, timeout=put_timeout)
                return True
            except queue.Full:
                continue

    async def pump() -> None:
        async_stream: Optional[AsyncGenerator[str, None]] = None
        try:
            result = await async_entrypoint()
            if isinstance(result, str):
                safe_put(("chunk", result))
                return

            if not hasattr(result, "__aiter__"):
                safe_put(("chunk", str(result)))
                return

            async_stream = result  # type: ignore[assignment]
            async for chunk in async_stream:
                if cancel_event.is_set():
                    break
                if not safe_put(("chunk", chunk)):
                    break
        except Exception as exc:  # pragma: no cover - exercised in integration paths
            safe_put(("error", exc))
        finally:
            if async_stream is not None:
                try:
                    await async_stream.aclose()
                except Exception:
                    pass
            safe_put(("end", sentinel))

    def worker() -> None:
        try:
            asyncio.run(pump())
        except Exception as exc:  # pragma: no cover - defensive fallback
            safe_put(("error", exc))
            safe_put(("end", sentinel))

    thread = threading.Thread(
        target=lambda: copied_context.run(worker),
        name=f"agentify-{api_name}-stream-bridge",
        daemon=True,
    )
    thread.start()

    try:
        while True:
            kind, payload = q.get()
            if kind == "chunk":
                yield payload
            elif kind == "error":
                raise payload
            else:
                break
    finally:
        cancel_event.set()
        thread.join(timeout=join_timeout)

import asyncio
import contextvars
import threading
import sys
from unittest.mock import MagicMock

import pytest

# Mock dependencies before importing agentify package
mock_openai = MagicMock()
mock_openai.RateLimitError = Exception
sys.modules["openai"] = mock_openai
sys.modules["openai.types.chat"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()

from agentify.core.sync_bridge import ensure_sync_call_allowed, stream_async_to_sync


def test_stream_bridge_propagates_context_and_stops_on_close() -> None:
    trace_id = contextvars.ContextVar("trace_id", default="missing")
    trace_id.set("trace-123")
    stopped = threading.Event()

    async def async_entrypoint():
        async def producer():
            counter = 0
            try:
                while True:
                    yield f"{trace_id.get()}:{counter}"
                    counter += 1
                    await asyncio.sleep(0)
            finally:
                stopped.set()

        return producer()

    stream = stream_async_to_sync(
        async_entrypoint,
        api_name="run",
        async_api_name="arun",
        queue_maxsize=1,
        put_timeout=0.05,
    )

    first_chunk = next(stream)
    assert first_chunk.startswith("trace-123:")

    stream.close()
    assert stopped.wait(1.0)

    active_bridge_threads = [
        t for t in threading.enumerate() if t.name.startswith("agentify-run-stream-bridge")
    ]
    assert not active_bridge_threads


@pytest.mark.asyncio
async def test_sync_call_blocked_when_loop_is_running() -> None:
    with pytest.raises(RuntimeError, match="cannot be used while an event loop is running"):
        ensure_sync_call_allowed(api_name="run", async_api_name="arun")

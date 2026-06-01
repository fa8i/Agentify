import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentify.llm.codex_backend import CodexThreadBackend, DummyResponse
import agentify.llm.codex_backend as codex_module
from agentify.core.agent import BaseAgent
from agentify.core.config import AgentConfig
from agentify.core.tool import Tool
from agentify.memory.interfaces import MemoryAddress
from agentify.memory.service import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore


class MockThread:
    """Simulates an AsyncThread returned by thread_start / thread_resume."""

    def __init__(self, thread_id: str):
        self.id = thread_id
        self.run_called_with = []

    async def run(self, prompt: str):
        self.run_called_with.append(prompt)
        mock_result = MagicMock()
        mock_result.final_response = f"Response to: {prompt}"
        return mock_result

    async def turn(self, prompt: str):
        class Stream:
            def __init__(self, text: str):
                self._events = iter(
                    [
                        MagicMock(
                            method="item/agentMessage/delta",
                            payload=MagicMock(delta=text),
                        ),
                        MagicMock(
                            method="turn/completed",
                            payload=MagicMock(turn=MagicMock(error=None)),
                        ),
                    ]
                )

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._events)
                except StopIteration:
                    raise StopAsyncIteration

            async def aclose(self):
                pass

        class Turn:
            def stream(self):
                return Stream(f"Response to: {prompt}")

        return Turn()


class MockAsyncCodex:
    """Simulates the AsyncCodex client."""

    def __init__(self, **kwargs):
        self._thread_counter = 0
        self.resumed_ids = []

    async def thread_start(self, model: str):
        self._thread_counter += 1
        return MockThread(thread_id=f"codex_thread_{self._thread_counter}")

    async def thread_resume(self, thread_id: str, **kwargs):
        self.resumed_ids.append(thread_id)
        return MockThread(thread_id=thread_id)


@pytest.fixture
def mock_codex():
    with patch.object(codex_module, "AsyncCodex", MockAsyncCodex):
        yield


def test_codex_backend_init_raises_without_dependency():
    """Importing without openai-codex installed raises ImportError."""
    with patch.object(codex_module, "AsyncCodex", None):
        with pytest.raises(ImportError, match="openai-codex is not installed"):
            CodexThreadBackend(config={}, timeout=30)


def test_codex_backend_stores_thread_id_not_object(mock_codex):
    """thread_ids dict stores string IDs, not live AsyncThread objects."""
    backend = CodexThreadBackend(config={}, timeout=30)

    asyncio.run(
        backend.run_native(session_id="session1", model="gpt-5.4", prompt="Hello")
    )

    assert "session1" in backend.thread_ids
    assert isinstance(backend.thread_ids["session1"], str)
    assert backend.thread_ids["session1"] == "codex_thread_1"


def test_codex_backend_resumes_thread_on_second_call(mock_codex):
    """Second call for the same session uses thread_resume with the stored ID."""
    backend = CodexThreadBackend(config={}, timeout=30)

    # First call → thread_start
    asyncio.run(
        backend.run_native(session_id="session1", model="gpt-5.4", prompt="Hello")
    )
    assert backend.codex._thread_counter == 1

    # Second call → thread_resume (not thread_start again)
    response = asyncio.run(
        backend.run_native(session_id="session1", model="gpt-5.4", prompt="How are you?")
    )
    # thread_start should NOT have been called again
    assert backend.codex._thread_counter == 1
    # thread_resume should have been called with the stored ID
    assert backend.codex.resumed_ids == ["codex_thread_1"]
    assert response.choices[0].message.content == "Response to: How are you?"


def test_codex_backend_separate_sessions_separate_threads(mock_codex):
    """Different session IDs produce different Codex threads."""
    backend = CodexThreadBackend(config={}, timeout=30)

    asyncio.run(
        backend.run_native(session_id="session_a", model="gpt-5.4", prompt="Task A")
    )
    asyncio.run(
        backend.run_native(session_id="session_b", model="gpt-5.4", prompt="Task B")
    )

    assert backend.thread_ids["session_a"] == "codex_thread_1"
    assert backend.thread_ids["session_b"] == "codex_thread_2"
    assert backend.codex._thread_counter == 2


def test_codex_backend_adapter_compatibility(mock_codex):
    """Fallback adapter (chat.completions.create) works correctly."""
    backend = CodexThreadBackend(config={}, timeout=30)

    response = asyncio.run(
        backend.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "user", "content": "Tell me a joke"},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
                {"role": "user", "content": "To get to the other side!"},
            ],
            session_id="session2",
        )
    )

    assert response.choices[0].message.content == "Response to: To get to the other side!"
    assert "session2" in backend.thread_ids


def test_codex_backend_rejects_streaming(mock_codex):
    backend = CodexThreadBackend(config={}, timeout=30)

    with pytest.raises(NotImplementedError, match="Streaming is not supported"):
        asyncio.run(
            backend.run_native(
                session_id="session1", model="gpt-5.4", prompt="Hello", stream=True
            )
        )


def test_codex_dummy_response_does_not_emit_empty_reasoning():
    message = DummyResponse("Hello").choices[0].message

    assert not hasattr(message, "reasoning_content")


def test_codex_backend_extracts_mcp_result_when_final_response_is_empty(mock_codex):
    backend = CodexThreadBackend(config={}, timeout=30)

    content_item = MagicMock()
    content_item.text = "ECHO_FROM_AGENTIFY: hola-agentify"
    mcp_result = MagicMock()
    mcp_result.content = [content_item]
    root = MagicMock()
    root.text = None
    root.result = mcp_result
    item = MagicMock()
    item.root = root
    turn_result = MagicMock()
    turn_result.final_response = None
    turn_result.items = [item]

    assert backend._extract_response_content(turn_result) == "ECHO_FROM_AGENTIFY: hola-agentify"


def test_codex_backend_run_native_uses_event_stream_when_available(mock_codex):
    class Stream:
        def __init__(self):
            self._events = iter(
                [
                    MagicMock(
                        method="item/agentMessage/delta",
                        payload=MagicMock(delta="ECHO_FROM_AGENTIFY: hola-agentify"),
                    ),
                    MagicMock(
                        method="turn/completed",
                        payload=MagicMock(turn=MagicMock(error=None)),
                    ),
                ]
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

        async def aclose(self):
            pass

    class Turn:
        def stream(self):
            return Stream()

    class Thread:
        async def turn(self, prompt):
            return Turn()

    backend = CodexThreadBackend(config={}, timeout=30)
    backend._get_or_create_thread = AsyncMock(return_value=Thread())

    response = asyncio.run(backend.run_native(session_id="s", model="m", prompt="p"))

    assert response.choices[0].message.content == "ECHO_FROM_AGENTIFY: hola-agentify"


def test_codex_backend_flags_are_explicit(mock_codex):
    backend = CodexThreadBackend(config={}, timeout=30)

    assert backend.is_native_thread_backend is True
    assert backend.supports_tools is False
    assert backend.supports_openai_tool_calls is False
    assert backend.supports_mcp_tools is True
    assert backend.supports_streaming is False


def test_codex_backend_requires_turn_api_when_mcp_tools_enabled(mock_codex):
    class ThreadWithoutTurn:
        async def run(self, prompt):
            raise AssertionError("run should not be called when MCP tools are enabled")

    backend = CodexThreadBackend(config={}, timeout=30)
    backend._get_or_create_thread = AsyncMock(return_value=ThreadWithoutTurn())

    with pytest.raises(RuntimeError, match="Codex MCP tools require event streaming"):
        asyncio.run(backend.run_native(session_id="s", model="m", prompt="p"))


def test_codex_backend_allows_run_fallback_when_mcp_tools_disabled(mock_codex):
    class ThreadWithoutTurn:
        async def run(self, prompt):
            result = MagicMock()
            result.final_response = "simple response"
            return result

    backend = CodexThreadBackend(config={"mcp_tools_enabled": False}, timeout=30)
    backend._get_or_create_thread = AsyncMock(return_value=ThreadWithoutTurn())

    response = asyncio.run(backend.run_native(session_id="s", model="m", prompt="p"))

    assert response.choices[0].message.content == "simple response"


def test_codex_backend_errors_when_turn_completes_without_text(mock_codex):
    class Stream:
        def __init__(self):
            self._events = iter(
                [MagicMock(method="turn/completed", payload=MagicMock(turn=MagicMock(error=None)))]
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

        async def aclose(self):
            pass

    class Turn:
        def stream(self):
            return Stream()

    class Thread:
        async def turn(self, prompt):
            return Turn()

    backend = CodexThreadBackend(config={}, timeout=30)
    backend._get_or_create_thread = AsyncMock(return_value=Thread())

    with pytest.raises(RuntimeError, match="without reconstructible text"):
        asyncio.run(backend.run_native(session_id="s", model="m", prompt="p"))


def test_codex_backend_timeout_waiting_for_events(mock_codex):
    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(1)
            return MagicMock(method="noop", payload=MagicMock())

        async def aclose(self):
            pass

    class Turn:
        def stream(self):
            return Stream()

    class Thread:
        async def turn(self, prompt):
            return Turn()

    backend = CodexThreadBackend(config={}, timeout=0.01)
    backend._get_or_create_thread = AsyncMock(return_value=Thread())

    with pytest.raises(TimeoutError, match="Timed out waiting for Codex turn events"):
        asyncio.run(backend.run_native(session_id="s", model="m", prompt="p"))


def test_stream_processor_ignores_empty_reasoning():
    class Delta:
        content = "Hello"
        reasoning_content = None
        tool_calls = None

    class Choice:
        delta = Delta()

    class Chunk:
        choices = [Choice()]

    async def response_stream():
        yield Chunk()

    class Factory:
        def create_client(self, **kwargs):
            return MagicMock()

        def create_async_client(self, **kwargs):
            return MagicMock()

    async def run_test():
        agent = BaseAgent(
            config=AgentConfig(
                name="StreamAgent",
                system_prompt="Test agent.",
                provider="mock",
                model_name="mock-model",
            ),
            memory=MemoryService(store=InMemoryStore(), log_enabled=False),
            memory_address=MemoryAddress(conversation_id="session1"),
            client_factory=Factory(),
        )
        chunks = []
        async for chunk in agent._aprocess_stream_response(response_stream()):
            chunks.append(chunk)
        return chunks, agent._last_stream_reasoning

    chunks, reasoning = asyncio.run(run_test())

    assert chunks == ["Hello"]
    assert reasoning is None


def test_agent_rejects_tools_for_native_codex_backend():
    class NativeCodexMock:
        is_native_thread_backend = True
        supports_tools = False
        supports_streaming = False

        async def run_native(self, **kwargs):
            raise AssertionError("run_native should not be called when tools are present")

    class Factory:
        def create_client(self, **kwargs):
            return MagicMock()

        def create_async_client(self, **kwargs):
            return NativeCodexMock()

    tool = Tool(
        schema={"name": "example", "parameters": {"type": "object"}},
        func=lambda: "ok",
    )
    agent = BaseAgent(
        config=AgentConfig(
            name="CodexAgent",
            system_prompt="Test agent.",
            provider="codex",
            model_name="gpt-5.4",
        ),
        memory=MemoryService(store=InMemoryStore(), log_enabled=False),
        memory_address=MemoryAddress(conversation_id="session1"),
        client_factory=Factory(),
        tools=[tool],
    )

    with pytest.raises(NotImplementedError, match="classic tool loop is not supported"):
        asyncio.run(agent.arun("Use the tool"))


def test_supports_tools_false_does_not_block_codex_without_classic_tools():
    class NativeCodexMock:
        is_native_thread_backend = True
        supports_tools = False
        supports_openai_tool_calls = False
        supports_mcp_tools = True
        supports_streaming = False

        async def run_native(self, **kwargs):
            return DummyResponse("ok")

    class Factory:
        def create_client(self, **kwargs):
            return MagicMock()

        def create_async_client(self, **kwargs):
            return NativeCodexMock()

    agent = BaseAgent(
        config=AgentConfig(
            name="CodexAgent",
            system_prompt="Test agent.",
            provider="codex",
            model_name="gpt-5.3-codex",
        ),
        memory=MemoryService(store=InMemoryStore(), log_enabled=False),
        memory_address=MemoryAddress(conversation_id="session1"),
        client_factory=Factory(),
    )

    assert asyncio.run(agent.arun("Hello")) == "ok"


def test_agent_rejects_streaming_for_native_codex_backend():
    class NativeCodexMock:
        is_native_thread_backend = True
        supports_tools = False
        supports_streaming = False

        async def run_native(self, **kwargs):
            raise AssertionError("run_native should not be called when streaming is enabled")

    class Factory:
        def create_client(self, **kwargs):
            return MagicMock()

        def create_async_client(self, **kwargs):
            return NativeCodexMock()

    agent = BaseAgent(
        config=AgentConfig(
            name="CodexAgent",
            system_prompt="Test agent.",
            provider="codex",
            model_name="gpt-5.4",
            stream=True,
        ),
        memory=MemoryService(store=InMemoryStore(), log_enabled=False),
        memory_address=MemoryAddress(conversation_id="session1"),
        client_factory=Factory(),
    )

    async def run_streaming_agent():
        response = await agent.arun("Stream this")
        async for _ in response:
            pass

    with pytest.raises(NotImplementedError, match="Streaming is not supported"):
        asyncio.run(run_streaming_agent())


def test_magicmock_does_not_activate_native_codex_branch():
    response = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello", tool_calls=None))]
    )
    async_client = MagicMock()
    async_client.chat.completions.create = AsyncMock(return_value=response)

    class Factory:
        def create_client(self, **kwargs):
            return MagicMock()

        def create_async_client(self, **kwargs):
            return async_client

    agent = BaseAgent(
        config=AgentConfig(
            name="MockAgent",
            system_prompt="Test agent.",
            provider="mock",
            model_name="mock-model",
        ),
        memory=MemoryService(store=InMemoryStore(), log_enabled=False),
        memory_address=MemoryAddress(conversation_id="session1"),
        client_factory=Factory(),
    )

    result = asyncio.run(agent.arun("Hello"))

    assert result == "Hello"
    async_client.chat.completions.create.assert_awaited_once()

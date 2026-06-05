import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentify.llm.codex_backend import (
    CodexThreadBackend,
    DummyResponse,
)
from agentify.llm.codex_errors import NonRetryableCodexError, raise_codex_errors
from agentify.llm.codex_inputs import build_codex_turn_input, extract_output_schema
from agentify.llm.tool_adapters import CodexMCPToolAdapter
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
        self.thread_start_kwargs = []
        self.closed = False

    async def thread_start(self, model: str, **kwargs):
        self.thread_start_kwargs.append(kwargs)
        self._thread_counter += 1
        return MockThread(thread_id=f"codex_thread_{self._thread_counter}")

    async def thread_resume(self, thread_id: str, **kwargs):
        self.resumed_ids.append(thread_id)
        return MockThread(thread_id=thread_id)

    async def close(self):
        self.closed = True


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
    backend = CodexThreadBackend(config={"memory_mode": "codex_thread"}, timeout=30)

    asyncio.run(
        backend.run_native(session_id="session1", model="gpt-5.4", prompt="Hello")
    )

    assert "session1" in backend.thread_ids
    assert isinstance(backend.thread_ids["session1"], str)
    assert backend.thread_ids["session1"] == "codex_thread_1"


def test_codex_backend_resumes_thread_on_second_call(mock_codex):
    """Second call for the same session uses thread_resume with the stored ID."""
    backend = CodexThreadBackend(config={"memory_mode": "codex_thread"}, timeout=30)

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
    backend = CodexThreadBackend(config={"memory_mode": "codex_thread"}, timeout=30)

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
    backend = CodexThreadBackend(config={"memory_mode": "codex_thread"}, timeout=30)

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


def test_codex_backend_defaults_to_agentify_memory_without_thread_reuse(mock_codex):
    backend = CodexThreadBackend(config={}, timeout=30)

    asyncio.run(
        backend.run_native(session_id="session1", model="gpt-5.4", prompt="Hello")
    )
    asyncio.run(
        backend.run_native(session_id="session1", model="gpt-5.4", prompt="Again")
    )

    assert backend.thread_ids == {}
    assert backend.codex._thread_counter == 2
    assert backend.codex.resumed_ids == []


def test_codex_backend_builds_prompt_from_agentify_messages(mock_codex):
    prompt = build_codex_turn_input(
        "latest only",
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Remember: my city is Madrid."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "What is my city?"},
        ],
        memory_mode="agentify",
    )

    assert "Agentify-managed conversation state" in prompt
    assert "SYSTEM: You are helpful." in prompt
    assert "USER: Remember: my city is Madrid." in prompt
    assert "ASSISTANT: Got it." in prompt
    assert "USER: What is my city?" in prompt


def test_codex_backend_builds_multimodal_prompt_from_agentify_messages(mock_codex):
    prompt = build_codex_turn_input(
        "latest only",
        [
            {"role": "system", "content": "You can inspect images."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc123"},
                    },
                    {"type": "text", "text": "What is in this image?"},
                ],
            },
        ],
        memory_mode="agentify",
    )

    assert isinstance(prompt, list)
    assert any(getattr(item, "text", "").startswith("Use the following") for item in prompt)
    assert any(getattr(item, "url", "") == "data:image/jpeg;base64,abc123" for item in prompt)
    assert any("What is in this image?" in getattr(item, "text", "") for item in prompt)


def test_codex_backend_codex_thread_mode_uses_latest_prompt(mock_codex):
    assert (
        build_codex_turn_input(
            "latest only",
            [{"role": "user", "content": "history"}],
            memory_mode="codex_thread",
        )
        == "latest only"
    )


def test_codex_backend_rejects_invalid_memory_mode(mock_codex):
    with pytest.raises(ValueError, match="memory_mode"):
        CodexThreadBackend(config={"memory_mode": "bad"}, timeout=30)


def test_codex_backend_auto_mcp_tools_adds_runtime_config(mock_codex):
    tool = Tool(
        schema={
            "name": "echo_tool",
            "description": "Echo text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
        func=lambda text: f"echo: {text}",
    )
    backend = CodexThreadBackend(config={}, timeout=30)

    try:
        asyncio.run(
            backend.run_native(
                session_id="session1",
                model="gpt-5.4",
                prompt="Use echo_tool",
                agentify_tools=[tool],
            )
        )
    finally:
        if backend._runtime_mcp_bridge is not None:
            asyncio.run(backend._runtime_mcp_bridge.close())

    config = backend.codex.thread_start_kwargs[0]["config"]
    server = config["mcp_servers"]["agentify-runtime-tools"]
    assert server["command"]
    assert "agentify.mcp.runtime_server" in server["args"]
    assert server["enabled_tools"] == ["echo_tool"]


def test_codex_backend_close_releases_runtime_bridge_and_client(mock_codex):
    tool = Tool(schema={"name": "example", "parameters": {"type": "object"}}, func=lambda: "ok")
    backend = CodexThreadBackend(config={}, timeout=30)

    asyncio.run(
        backend.run_native(
            session_id="session1",
            model="gpt-5.4",
            prompt="Use tool",
            agentify_tools=[tool],
        )
    )

    assert backend._runtime_mcp_bridge is not None
    asyncio.run(backend.close())

    assert backend._runtime_mcp_bridge is None
    assert backend.codex.closed is True


def test_codex_backend_auto_mcp_tools_can_be_disabled(mock_codex):
    tool = Tool(
        schema={"name": "example", "parameters": {"type": "object"}},
        func=lambda: "ok",
    )
    backend = CodexThreadBackend(config={"auto_mcp_tools": False}, timeout=30)

    with pytest.raises(NotImplementedError, match="auto_mcp_tools=False"):
        asyncio.run(
            backend.run_native(
                session_id="session1",
                model="gpt-5.4",
                prompt="Use tool",
                agentify_tools=[tool],
            )
        )


def test_codex_backend_streams_turn_event_deltas(mock_codex):
    backend = CodexThreadBackend(config={}, timeout=30)

    async def run_test():
        stream = await backend.run_native(
            session_id="session1",
            model="gpt-5.4",
            prompt="Hello",
            stream=True,
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk.choices[0].delta.content)
        return chunks

    assert asyncio.run(run_test()) == ["Response to: Hello"]


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


def test_codex_backend_passes_output_schema_to_turn(mock_codex):
    class Stream:
        def __init__(self):
            self._events = iter(
                [
                    MagicMock(
                        method="item/agentMessage/delta",
                        payload=MagicMock(delta='{"ok":true}'),
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
        def __init__(self):
            self.output_schema = None

        async def turn(self, prompt, **kwargs):
            self.output_schema = kwargs.get("output_schema")
            return Turn()

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    thread = Thread()
    backend = CodexThreadBackend(config={}, timeout=30)
    backend._get_or_create_thread = AsyncMock(return_value=thread)

    response = asyncio.run(
        backend.run_native(
            session_id="s",
            model="m",
            prompt="p",
            output_schema=schema,
        )
    )

    assert response.choices[0].message.content == '{"ok":true}'
    assert thread.output_schema == schema


def test_codex_backend_extracts_openai_style_response_format_schema(mock_codex):
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    assert extract_output_schema(
        {"response_format": {"type": "json_schema", "json_schema": {"schema": schema}}}
    ) == schema


def test_codex_backend_flags_are_explicit(mock_codex):
    backend = CodexThreadBackend(config={}, timeout=30)

    assert backend.is_native_thread_backend is True
    assert backend.supports_tools is False
    assert backend.supports_openai_tool_calls is False
    assert backend.supports_mcp_tools is True
    assert backend.supports_streaming is True


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


def test_codex_backend_marks_usage_limit_non_retryable(mock_codex):
    with pytest.raises(NonRetryableCodexError, match="usage limit exceeded"):
        raise_codex_errors(["Codex usage limit exceeded: try later"])


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


def test_agent_passes_tools_as_agentify_tools_for_native_codex_mcp_backend():
    class NativeCodexMock:
        is_native_thread_backend = True
        supports_tools = False
        supports_openai_tool_calls = False
        supports_mcp_tools = True
        supports_streaming = False

        def __init__(self):
            self.kwargs = None

        async def run_native(self, **kwargs):
            self.kwargs = kwargs
            return DummyResponse("ok")

    native_client = NativeCodexMock()

    class Factory:
        def create_client(self, **kwargs):
            return MagicMock()

        def create_async_client(self, **kwargs):
            return native_client

    tool = Tool(
        schema={"name": "example", "parameters": {"type": "object"}},
        func=lambda: "ok",
    )
    agent = BaseAgent(
        config=AgentConfig(
            name="CodexAgent",
            system_prompt="Test agent.",
            provider="codex",
            model_name="gpt-5.3-codex",
            tool_timeout=17,
        ),
        memory=MemoryService(store=InMemoryStore(), log_enabled=False),
        memory_address=MemoryAddress(conversation_id="session1"),
        client_factory=Factory(),
        tools=[tool],
    )

    assert asyncio.run(agent.arun("Use the tool")) == "ok"
    assert native_client.kwargs is not None
    assert native_client.kwargs["agentify_tools"] == [tool]
    assert native_client.kwargs["agentify_tool_timeout"] == 17
    assert callable(native_client.kwargs["agentify_tool_executor"])
    assert "tools" not in native_client.kwargs
    assert "tool_choice" not in native_client.kwargs


def test_codex_mcp_tool_call_is_recorded_in_memory():
    memory = MemoryService(store=InMemoryStore(), log_enabled=False)
    addr = MemoryAddress(conversation_id="session1")
    async def add_memory(**kwargs):
        message = {key: value for key, value in kwargs.items() if key != "addr"}
        memory.append_history(kwargs["addr"], message)

    async def execute_tool(tool_name, arguments):
        assert tool_name == "echo_tool"
        return f"echo: {arguments['text']}"

    adapter = CodexMCPToolAdapter(
        agent_name="CodexAgent",
        provider="codex",
        tool_timeout=30,
        max_tool_iter=10,
        callbacks=[],
        execute_tool=execute_tool,
        add_memory=add_memory,
    )

    result = asyncio.run(
        adapter.execute_tool_call(
            "echo_tool",
            {"text": "hola"},
            addr=addr,
        )
    )
    history = memory.get_history(addr)

    assert result == "echo: hola"
    assert history[0]["role"] == "assistant"
    assert history[0]["source"] == "codex_mcp"
    assert history[0]["tool_calls"][0]["function"]["name"] == "echo_tool"
    assert history[1]["role"] == "tool"
    assert history[1]["name"] == "echo_tool"
    assert history[1]["content"] == "echo: hola"
    assert history[1]["source"] == "codex_mcp"


def test_codex_mcp_tool_adapter_enforces_max_tool_iter():
    memory = MemoryService(store=InMemoryStore(), log_enabled=False)
    addr = MemoryAddress(conversation_id="session1")
    executed = []

    async def add_memory(**kwargs):
        message = {key: value for key, value in kwargs.items() if key != "addr"}
        memory.append_history(kwargs["addr"], message)

    async def execute_tool(tool_name, arguments):
        executed.append((tool_name, arguments))
        return "ok"

    adapter = CodexMCPToolAdapter(
        agent_name="CodexAgent",
        provider="codex",
        tool_timeout=30,
        max_tool_iter=1,
        callbacks=[],
        execute_tool=execute_tool,
        add_memory=add_memory,
    )

    first = asyncio.run(adapter.execute_tool_call("echo_tool", {}, addr=addr))
    second = asyncio.run(adapter.execute_tool_call("echo_tool", {}, addr=addr))
    history = memory.get_history(addr)

    assert first == "ok"
    assert "Tool iteration limit reached" in second
    assert executed == [("echo_tool", {})]
    assert history[-1]["role"] == "tool"
    assert "Tool iteration limit reached" in history[-1]["content"]


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


def test_native_codex_backend_receives_agentify_memory_messages():
    class NativeCodexMock:
        is_native_thread_backend = True
        supports_tools = False
        supports_openai_tool_calls = False
        supports_mcp_tools = True
        supports_streaming = False

        def __init__(self):
            self.kwargs = None

        async def run_native(self, **kwargs):
            self.kwargs = kwargs
            return DummyResponse("ok")

    native_client = NativeCodexMock()

    class Factory:
        def create_client(self, **kwargs):
            return MagicMock()

        def create_async_client(self, **kwargs):
            return native_client

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
    agent.add("user", "Remember: my city is Madrid")
    agent.add("assistant", "Got it")

    assert asyncio.run(agent.arun("What is my city?")) == "ok"
    assert native_client.kwargs is not None
    assert native_client.kwargs["prompt"] == "What is my city?"
    assert native_client.kwargs["messages"] == [
        {"role": "system", "content": "Test agent."},
        {"role": "user", "content": "Remember: my city is Madrid"},
        {"role": "assistant", "content": "Got it"},
        {"role": "user", "content": "What is my city?"},
    ]


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


def test_agent_does_not_retry_non_retryable_provider_errors():
    class ProviderError(RuntimeError):
        non_retryable_provider_error = True

    class NativeCodexMock:
        is_native_thread_backend = True
        supports_tools = False
        supports_streaming = True

        def __init__(self):
            self.calls = 0

        async def run_native(self, **kwargs):
            self.calls += 1
            raise ProviderError("usage limit exceeded")

    native_client = NativeCodexMock()

    class Factory:
        def create_client(self, **kwargs):
            return MagicMock()

        def create_async_client(self, **kwargs):
            return native_client

    agent = BaseAgent(
        config=AgentConfig(
            name="CodexAgent",
            system_prompt="Test agent.",
            provider="codex",
            model_name="gpt-5.3-codex",
            max_retries=3,
        ),
        memory=MemoryService(store=InMemoryStore(), log_enabled=False),
        memory_address=MemoryAddress(conversation_id="session1"),
        client_factory=Factory(),
    )

    with pytest.raises(ProviderError):
        asyncio.run(agent.arun("Hello"))
    assert native_client.calls == 1


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

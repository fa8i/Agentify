import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agentify.llm.codex_backend import CodexThreadBackend, DummyResponse
import agentify.llm.codex_backend as codex_module


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


@pytest.mark.asyncio
async def test_codex_backend_init_raises_without_dependency():
    """Importing without openai-codex installed raises ImportError."""
    with patch.object(codex_module, "AsyncCodex", None):
        with pytest.raises(ImportError, match="openai-codex is not installed"):
            CodexThreadBackend(config={}, timeout=30)


@pytest.mark.asyncio
async def test_codex_backend_stores_thread_id_not_object(mock_codex):
    """thread_ids dict stores string IDs, not live AsyncThread objects."""
    backend = CodexThreadBackend(config={}, timeout=30)

    await backend.run_native(
        session_id="session1", model="gpt-5.4", prompt="Hello"
    )

    assert "session1" in backend.thread_ids
    assert isinstance(backend.thread_ids["session1"], str)
    assert backend.thread_ids["session1"] == "codex_thread_1"


@pytest.mark.asyncio
async def test_codex_backend_resumes_thread_on_second_call(mock_codex):
    """Second call for the same session uses thread_resume with the stored ID."""
    backend = CodexThreadBackend(config={}, timeout=30)

    # First call → thread_start
    await backend.run_native(
        session_id="session1", model="gpt-5.4", prompt="Hello"
    )
    assert backend.codex._thread_counter == 1

    # Second call → thread_resume (not thread_start again)
    response = await backend.run_native(
        session_id="session1", model="gpt-5.4", prompt="How are you?"
    )
    # thread_start should NOT have been called again
    assert backend.codex._thread_counter == 1
    # thread_resume should have been called with the stored ID
    assert backend.codex.resumed_ids == ["codex_thread_1"]
    assert response.choices[0].message.content == "Response to: How are you?"


@pytest.mark.asyncio
async def test_codex_backend_separate_sessions_separate_threads(mock_codex):
    """Different session IDs produce different Codex threads."""
    backend = CodexThreadBackend(config={}, timeout=30)

    await backend.run_native(
        session_id="session_a", model="gpt-5.4", prompt="Task A"
    )
    await backend.run_native(
        session_id="session_b", model="gpt-5.4", prompt="Task B"
    )

    assert backend.thread_ids["session_a"] == "codex_thread_1"
    assert backend.thread_ids["session_b"] == "codex_thread_2"
    assert backend.codex._thread_counter == 2


@pytest.mark.asyncio
async def test_codex_backend_adapter_compatibility(mock_codex):
    """Fallback adapter (chat.completions.create) works correctly."""
    backend = CodexThreadBackend(config={}, timeout=30)

    response = await backend.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "user", "content": "Tell me a joke"},
            {"role": "assistant", "content": "Why did the chicken cross the road?"},
            {"role": "user", "content": "To get to the other side!"},
        ],
        session_id="session2",
    )

    assert response.choices[0].message.content == "Response to: To get to the other side!"
    assert "session2" in backend.thread_ids

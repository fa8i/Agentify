import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agentify.llm.codex_backend import CodexThreadBackend, DummyResponse
import agentify.llm.codex_backend as codex_module

class MockThread:
    def __init__(self):
        self.run_called_with = []

    async def run(self, prompt: str):
        self.run_called_with.append(prompt)
        mock_result = MagicMock()
        mock_result.final_response = f"Response to: {prompt}"
        return mock_result

class MockAsyncCodex:
    def __init__(self):
        self.threads_created = 0

    async def thread_start(self, model: str):
        self.threads_created += 1
        return MockThread()

@pytest.fixture
def mock_codex():
    with patch.object(codex_module, "AsyncCodex", MockAsyncCodex):
        yield

@pytest.mark.asyncio
async def test_codex_backend_init_raises_without_dependency():
    with patch.object(codex_module, "AsyncCodex", None):
        with pytest.raises(ImportError, match="openai-codex is not installed"):
            CodexThreadBackend(config={}, timeout=30)

@pytest.mark.asyncio
async def test_codex_backend_run_native(mock_codex):
    backend = CodexThreadBackend(config={}, timeout=30)
    
    # Run first time
    response = await backend.run_native(
        session_id="session1", 
        model="gpt-5.4", 
        prompt="Hello world"
    )
    
    assert isinstance(response, DummyResponse)
    assert response.choices[0].message.content == "Response to: Hello world"
    assert "session1" in backend.threads
    
    # Run second time for same session (should reuse thread)
    response2 = await backend.run_native(
        session_id="session1", 
        model="gpt-5.4", 
        prompt="How are you?"
    )
    
    assert response2.choices[0].message.content == "Response to: How are you?"
    assert backend.codex.threads_created == 1  # Should only be 1 thread for session1
    
    thread = backend.threads["session1"]
    assert thread.run_called_with == ["Hello world", "How are you?"]

@pytest.mark.asyncio
async def test_codex_backend_adapter_compatibility(mock_codex):
    backend = CodexThreadBackend(config={}, timeout=30)
    
    # Use the adapter interface
    response = await backend.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "user", "content": "Tell me a joke"},
            {"role": "assistant", "content": "Why did the chicken cross the road?"},
            {"role": "user", "content": "To get to the other side!"}
        ],
        session_id="session2"
    )
    
    assert response.choices[0].message.content == "Response to: To get to the other side!"
    assert "session2" in backend.threads
    thread = backend.threads["session2"]
    assert thread.run_called_with == ["To get to the other side!"]

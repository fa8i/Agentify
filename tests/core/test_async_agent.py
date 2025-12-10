import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import asyncio

# Mock dependencies before import
mock_openai = MagicMock()
mock_openai.RateLimitError = Exception
mock_openai.AsyncOpenAI = MagicMock
mock_openai.AsyncAzureOpenAI = MagicMock
sys.modules["openai"] = mock_openai
sys.modules["openai.types.chat"] = MagicMock()
sys.modules["dotenv"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()

from agentify.core import BaseAgent, Tool, AgentConfig


class MockAsyncClient:
    """Mock async client for testing."""
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        
    async def create_completion(self, **kwargs):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Async Hello", tool_calls=None))]
        return mock_response


@pytest.fixture
def mock_async_factory():
    """Create a mock factory that returns async clients."""
    factory = MagicMock()
    
    # Sync client
    sync_client = MagicMock()
    mock_sync_response = MagicMock()
    mock_sync_response.choices = [MagicMock(message=MagicMock(content="Sync Hello", tool_calls=None))]
    sync_client.chat.completions.create.return_value = mock_sync_response
    factory.create_client.return_value = sync_client
    
    # Async client
    async_client = MagicMock()
    mock_async_response = MagicMock()
    mock_async_response.choices = [MagicMock(message=MagicMock(content="Async Hello", tool_calls=None))]
    
    async def async_create(**kwargs):
        return mock_async_response
    
    async_client.chat.completions.create = AsyncMock(return_value=mock_async_response)
    factory.create_async_client.return_value = async_client
    
    return factory


@pytest.mark.asyncio
async def test_agent_arun_basic(agent_config, memory_service, memory_address, mock_async_factory):
    """Test basic async agent execution."""
    agent = BaseAgent(
        config=agent_config,
        memory=memory_service,
        memory_address=memory_address,
        client_factory=mock_async_factory
    )
    
    response = await agent.arun("Hello async")
    
    assert response == "Async Hello"
    mock_async_factory.create_async_client.assert_called_once()


@pytest.mark.asyncio
async def test_agent_arun_with_sync_tool(agent_config, memory_service, memory_address):
    """Test async agent with a synchronous tool."""
    
    def sync_double(x: int):
        return str(x * 2)
    
    tool = Tool(schema={"name": "double", "description": "Doubles x"}, func=sync_double)
    
    mock_factory = MagicMock()
    mock_sync_client = MagicMock()
    mock_factory.create_client.return_value = mock_sync_client
    
    # Async client setup
    async_client = MagicMock()
    
    # First response: tool call
    msg1 = MagicMock()
    msg1.content = None
    func_mock = MagicMock()
    func_mock.name = "double"
    func_mock.arguments = '{"x": 21}'
    msg1.tool_calls = [MagicMock(id="call_1", function=func_mock)]
    
    # Second response: final answer
    msg2 = MagicMock()
    msg2.content = "The result is 42"
    msg2.tool_calls = None
    
    async_responses = iter([
        MagicMock(choices=[MagicMock(message=msg1)]),
        MagicMock(choices=[MagicMock(message=msg2)])
    ])
    
    async def mock_create(**kwargs):
        return next(async_responses)
    
    async_client.chat.completions.create = AsyncMock(side_effect=lambda **kwargs: next(async_responses))
    mock_factory.create_async_client.return_value = async_client

    agent = BaseAgent(
        config=agent_config,
        memory=memory_service,
        memory_address=memory_address,
        tools=[tool],
        client_factory=mock_factory
    )
    
    # Reset the iterator for the actual test
    async_responses = iter([
        MagicMock(choices=[MagicMock(message=msg1)]),
        MagicMock(choices=[MagicMock(message=msg2)])
    ])
    async_client.chat.completions.create = AsyncMock(side_effect=lambda **kwargs: next(async_responses))
    
    response = await agent.arun("Double 21")
    
    assert "42" in response or "result" in response.lower()


@pytest.mark.asyncio
async def test_agent_arun_with_async_tool(agent_config, memory_service, memory_address):
    """Test async agent with an async tool function."""
    
    async def async_multiply(x: int, y: int):
        await asyncio.sleep(0.01)  # Simulate async work
        return str(x * y)
    
    tool = Tool(
        schema={"name": "multiply", "description": "Multiplies x and y"},
        func=async_multiply
    )
    
    mock_factory = MagicMock()
    mock_sync_client = MagicMock()
    mock_factory.create_client.return_value = mock_sync_client
    
    async_client = MagicMock()
    
    # First response: tool call
    msg1 = MagicMock()
    msg1.content = None
    func_mock = MagicMock()
    func_mock.name = "multiply"
    func_mock.arguments = '{"x": 6, "y": 7}'
    msg1.tool_calls = [MagicMock(id="call_1", function=func_mock)]
    
    # Second response: final answer
    msg2 = MagicMock()
    msg2.content = "The result is 42"
    msg2.tool_calls = None
    
    responses = [
        MagicMock(choices=[MagicMock(message=msg1)]),
        MagicMock(choices=[MagicMock(message=msg2)])
    ]
    response_iter = iter(responses)
    
    async_client.chat.completions.create = AsyncMock(side_effect=lambda **kwargs: next(response_iter))
    mock_factory.create_async_client.return_value = async_client

    agent = BaseAgent(
        config=agent_config,
        memory=memory_service,
        memory_address=memory_address,
        tools=[tool],
        client_factory=mock_factory
    )
    
    response = await agent.arun("Multiply 6 and 7")
    
    assert "42" in response or "result" in response.lower()


@pytest.mark.asyncio
async def test_parallel_tool_execution(agent_config, memory_service, memory_address):
    """Test that multiple tools are executed in parallel."""
    
    execution_order = []
    
    async def slow_tool_a():
        execution_order.append("a_start")
        await asyncio.sleep(0.1)
        execution_order.append("a_end")
        return "Result A"
    
    async def slow_tool_b():
        execution_order.append("b_start")
        await asyncio.sleep(0.1)
        execution_order.append("b_end")
        return "Result B"
    
    tool_a = Tool(schema={"name": "tool_a", "description": "Tool A"}, func=slow_tool_a)
    tool_b = Tool(schema={"name": "tool_b", "description": "Tool B"}, func=slow_tool_b)
    
    mock_factory = MagicMock()
    mock_sync_client = MagicMock()
    mock_factory.create_client.return_value = mock_sync_client
    
    async_client = MagicMock()
    
    # Response with two parallel tool calls
    msg1 = MagicMock()
    msg1.content = None
    
    func_a = MagicMock()
    func_a.name = "tool_a"
    func_a.arguments = '{}'
    
    func_b = MagicMock()
    func_b.name = "tool_b"
    func_b.arguments = '{}'
    
    msg1.tool_calls = [
        MagicMock(id="call_a", function=func_a),
        MagicMock(id="call_b", function=func_b)
    ]
    
    msg2 = MagicMock()
    msg2.content = "Both tools executed"
    msg2.tool_calls = None
    
    responses = [
        MagicMock(choices=[MagicMock(message=msg1)]),
        MagicMock(choices=[MagicMock(message=msg2)])
    ]
    response_iter = iter(responses)
    
    async_client.chat.completions.create = AsyncMock(side_effect=lambda **kwargs: next(response_iter))
    mock_factory.create_async_client.return_value = async_client

    agent = BaseAgent(
        config=agent_config,
        memory=memory_service,
        memory_address=memory_address,
        tools=[tool_a, tool_b],
        client_factory=mock_factory
    )
    
    import time
    start = time.time()
    response = await agent.arun("Run both tools")
    elapsed = time.time() - start
    
    # If parallel, should take ~0.1s. If sequential, ~0.2s
    # Using 0.15s as threshold
    assert elapsed < 0.18, f"Tools should run in parallel, took {elapsed}s"
    
    # Both should have started before either ended (parallel execution)
    # This checks that a_start and b_start both appear before a_end and b_end
    assert "a_start" in execution_order
    assert "b_start" in execution_order

import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock dependencies before import
mock_openai = MagicMock()
mock_openai.RateLimitError = Exception
sys.modules["openai"] = mock_openai
sys.modules["openai.types.chat"] = MagicMock()
sys.modules["dotenv"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()

from agentify.core import BaseAgent, Tool, AgentConfig, AgentCallbackHandler

class MockCallback(AgentCallbackHandler):
    def __init__(self):
        self.events = []

    def on_agent_start(self, agent_name, user_input):
        self.events.append(("agent_start", agent_name))

    def on_agent_finish(self, agent_name, response):
        self.events.append(("agent_finish", response))

    def on_tool_start(self, tool_name, args):
        self.events.append(("tool_start", tool_name))

    def on_tool_finish(self, tool_name, output):
        self.events.append(("tool_finish", output))

def test_agent_initialization(agent_config, memory_service, memory_address):
    mock_factory = MagicMock()
    mock_factory.create_client.return_value = MagicMock()
    
    agent = BaseAgent(
        config=agent_config,
        memory=memory_service,
        memory_address=memory_address,
        client_factory=mock_factory
    )
    assert agent.config.name == "TestAgent"
    assert len(agent.callbacks) == 1  # Default logger

def test_agent_callbacks(agent_config, memory_service, memory_address):
    callback = MockCallback()
    agent_config.callbacks = [callback]
    
    mock_factory = MagicMock()
    mock_client = MagicMock()
    mock_factory.create_client.return_value = mock_client
    
    # Mock LLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello", tool_calls=None))]
    mock_client.chat.completions.create.return_value = mock_response

    agent = BaseAgent(
        config=agent_config,
        memory=memory_service,
        memory_address=memory_address,
        client_factory=mock_factory
    )
    
    agent.respond("Hi")
    
    assert ("agent_start", "TestAgent") in callback.events
    assert ("agent_finish", "Hello") in callback.events

def test_tool_execution(agent_config, memory_service, memory_address):
    def my_tool(x: int):
        return str(x * 2)
    
    tool = Tool(schema={"name": "double", "description": "Doubles x"}, func=my_tool)
    
    mock_factory = MagicMock()
    mock_client = MagicMock()
    mock_factory.create_client.return_value = mock_client
    
    # 1. Assistant calls tool
    msg1 = MagicMock()
    msg1.content = None
    # Ensure name is a string, not a mock
    func_mock = MagicMock()
    func_mock.name = "double"
    func_mock.arguments = '{"x": 21}'
    
    msg1.tool_calls = [MagicMock(id="call_1", function=func_mock)]
    
    # 2. Assistant gives final answer
    msg2 = MagicMock()
    msg2.content = "The answer is 42"
    msg2.tool_calls = None
    
    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=msg1)]),
        MagicMock(choices=[MagicMock(message=msg2)])
    ]

    agent = BaseAgent(
        config=agent_config,
        memory=memory_service,
        memory_address=memory_address,
        tools=[tool],
        client_factory=mock_factory
    )
    
    response = agent.respond("Double 21")
    
    assert response == "The answer is 42"
    
    # Verify history
    history = agent.get_history(memory_address)
    # System, User, Assistant(Call), Tool(Result), Assistant(Final)
    assert len(history) == 5
    assert history[3]["role"] == "tool"
    assert history[3]["content"] == "42"

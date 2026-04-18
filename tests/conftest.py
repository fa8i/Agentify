import pytest
import sys
from unittest.mock import MagicMock
from typing import List, Dict, Any

# Mock optional runtime dependencies for test collection environments
mock_openai = MagicMock()
mock_openai.RateLimitError = Exception
sys.modules.setdefault("openai", mock_openai)
sys.modules.setdefault("openai.types.chat", MagicMock())
sys.modules.setdefault("dotenv", MagicMock())
sys.modules.setdefault("PIL", MagicMock())
sys.modules.setdefault("PIL.Image", MagicMock())

from agentify.memory.interfaces import MemoryAddress, Message, ConversationStore
from agentify.memory.service import MemoryService
from agentify.core.config import AgentConfig

class InMemoryStore:
    def __init__(self):
        self.data = {}

    def append_message(self, addr: MemoryAddress, msg: Message) -> None:
        key = addr.key_str()
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(msg)

    def read_messages(self, addr: MemoryAddress, start: int = 0, end: int = -1) -> List[Message]:
        key = addr.key_str()
        msgs = self.data.get(key, [])
        if end == -1:
            return msgs[start:]
        return msgs[start:end]

    def replace_messages(self, addr: MemoryAddress, messages: List[Message]) -> None:
        key = addr.key_str()
        self.data[key] = list(messages)

    def delete_conversation(self, addr: MemoryAddress) -> None:
        key = addr.key_str()
        if key in self.data:
            del self.data[key]

    def set_ttl(self, addr: MemoryAddress, seconds: int) -> None:
        pass

@pytest.fixture
def memory_store():
    return InMemoryStore()

@pytest.fixture
def memory_service(memory_store):
    return MemoryService(memory_store)

@pytest.fixture
def memory_address():
    return MemoryAddress(conversation_id="test_conv")

@pytest.fixture
def agent_config():
    return AgentConfig(
        name="TestAgent",
        system_prompt="You are a test agent.",
        provider="mock",
        model_name="mock-model"
    )

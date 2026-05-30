import sys
import os
import asyncio
from typing import cast
from unittest.mock import MagicMock, AsyncMock

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentify.core.agent import BaseAgent
from agentify.core.config import AgentConfig
from agentify.memory.service import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore
from agentify.memory.interfaces import MemoryAddress
from agentify.llm.client import LLMClientFactory, LLMClientType, AsyncLLMClientType

# Mock LLM Client
class MockLLMClient:
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Hello from Mock LLM", tool_calls=None))]
        )

class MockFactory(LLMClientFactory):
    def create_client(self, *args, **kwargs) -> LLMClientType:
        return cast(LLMClientType, MockLLMClient())

    def create_async_client(self, *args, **kwargs) -> AsyncLLMClientType:
        client = MockLLMClient()
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="Hello from Mock LLM", tool_calls=None))]
            )
        )
        return cast(AsyncLLMClientType, client)

# --- Hooks with different signatures ---

def hook_simple(user_input):
    print(f"HOOK_SIMPLE: Input='{user_input}'")

def hook_full(agent, user_input):
    print(f"HOOK_FULL: Agent='{agent.config.name}', Input='{user_input}'")

def hook_no_args():
    print("HOOK_NO_ARGS: Executed")

def post_hook_response(response):
    print(f"POST_HOOK_RESPONSE: Response='{response}'")

def post_hook_full(agent, response):
    print(f"POST_HOOK_FULL: Agent='{agent.config.name}', Response='{response}'")

async def main():
    config = AgentConfig(
        name="SmartAgent",
        system_prompt="You are a smart agent.",
        provider="deepseek",
        model_name="deepseek-v4-flash",
    )
    memory = MemoryService(store=InMemoryStore())
    addr = MemoryAddress(conversation_id="test-smart-hooks")

    agent = BaseAgent(
        config=config,
        memory=memory,
        memory_address=addr,
        client_factory=MockFactory(),
        pre_hooks=[hook_simple, hook_full, hook_no_args],
        post_hooks=[post_hook_response, post_hook_full],
    )

    print("--- Starting Smart Agent Interaction ---")
    response = await agent.arun("Hello smart agent!", addr=addr)
    print(f"--- Interaction Finished. Final Response: {response} ---")

if __name__ == "__main__":
    asyncio.run(main())

import sys
import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentify.core.agent import BaseAgent
from agentify.core.config import AgentConfig
from agentify.memory.service import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore
from agentify.memory.interfaces import MemoryAddress
from agentify.llm.client import LLMClientFactory

# Mock LLM Client
class MockLLMClient:
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Hello from Mock LLM", tool_calls=None))]
        )

class MockFactory(LLMClientFactory):
    def create_client(self, *args, **kwargs):
        return MockLLMClient()

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

def main():
    config = AgentConfig(
        name="SmartAgent",
        system_prompt="You are a smart agent.",
        provider="openai",
        model_name="gpt-4o",
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
    response = agent.run("Hello smart agent!", addr=addr)
    print(f"--- Interaction Finished. Final Response: {response} ---")

if __name__ == "__main__":
    main()

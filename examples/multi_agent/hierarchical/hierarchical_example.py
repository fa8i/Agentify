import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from agentify.core import BaseAgent, AgentConfig
from agentify.memory import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore
from agentify.multi_agent import HierarchicalTeam

from dotenv import load_dotenv

load_dotenv()

def main():
    store = InMemoryStore()
    memory_service = MemoryService(store=store, log_enabled=True, max_log_length=200)

    # 1. Define Agents

    # Level 2: Specialists
    coder_config = AgentConfig(
        name="Coder",
        system_prompt="You are a Python expert. Write code snippets based on instructions.",
        model_name="gpt-4.1-mini",
        provider="openai",
    )
    coder = BaseAgent(config=coder_config, memory=memory_service)

    tester_config = AgentConfig(
        name="Tester",
        system_prompt="You are a QA engineer. Review code and suggest tests.",
        model_name="gpt-4.1-mini",
        provider="openai",
    )
    tester = BaseAgent(config=tester_config, memory=memory_service)

    # Level 1: Manager (manages Coder and Tester)
    manager_config = AgentConfig(
        name="TechLead",
        system_prompt="You are a Tech Lead. You manage a Coder and a Tester. Delegate tasks to them to solve the user's request.",
        model_name="gpt-4.1-mini",
        provider="openai",
    )
    manager = BaseAgent(config=manager_config, memory=memory_service)

    # Level 0: Root (Director) - manages the TechLead
    director_config = AgentConfig(
        name="Director",
        system_prompt="You are the Director of Engineering. You receive high-level requests and delegate them to your Tech Lead.",
        model_name="gpt-4.1-mini",
        provider="openai",
    )
    director = BaseAgent(config=director_config, memory=memory_service)

    # 2. Define Hierarchy
    # Director -> [TechLead]
    # TechLead -> [Coder, Tester]
    hierarchy = {director: [manager], manager: [coder, tester]}

    # 3. Create Hierarchical Team
    team = HierarchicalTeam(root=director, hierarchy=hierarchy)

    # 4. Run
    request = "Create a Python function to calculate Fibonacci numbers and verify it."
    print(f"Request: {request}\n")

    result = team.run(user_input=request, session_id="hier_demo_1")

    print("-" * 20)
    print("Final Output:")
    print(result)


if __name__ == "__main__":
    main()

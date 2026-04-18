import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from agentify import BaseAgent, AgentConfig, tool
from agentify.memory import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore
from agentify.multi_agent import Team
import datetime

load_dotenv()


# Define custom tools with decorator
@tool
def get_current_time() -> dict:
    """Returns the current date and time."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"current_time": now}


@tool
def calculate(expression: str) -> dict:
    """Calculates a mathematical expression.
    
    Args:
        expression: The math expression to evaluate (e.g., '25 * 4').
    """
    try:
        result = eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def create_demo_team():
    store = InMemoryStore()
    memory_service = MemoryService(store=store, log_enabled=True, max_log_length=200)

    # 1. Create Worker: Researcher (has tools)
    researcher_config = AgentConfig(
        name="Researcher",
        system_prompt="You are a researcher. You can calculate things and check time. Be concise.",
        provider="openai",
        model_name="gpt-4.1",
        temperature=0.0,
    )
    researcher = BaseAgent(
        config=researcher_config,
        memory=memory_service,
        tools=[get_current_time, calculate],  # Using decorator tools
    )

    # 2. Create Supervisor: Manager (no tools initially, will get Researcher as tool)
    manager_config = AgentConfig(
        name="Manager",
        system_prompt="You are a manager. You delegate tasks to your researcher. Summarize their findings for the user.",
        provider="openai",
        model_name="gpt-4.1",
        temperature=0.7,
    )
    manager = BaseAgent(
        config=manager_config,
        memory=memory_service,
    )

    # 3. Create Team
    team = Team(agents=[manager, researcher], supervisor=manager)
    return team


def main():
    print("Initializing Team...")
    team = create_demo_team()

    print("\n--- Test 1: Simple Delegation ---")
    query = "What time is it and what is 25 * 4?"
    print(f"User: {query}")

    response = team.run(query, session_id="demo_session_1")
    print(f"\nManager Response:\n{response}")


if __name__ == "__main__":
    main()

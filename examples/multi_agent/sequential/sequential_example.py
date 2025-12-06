import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from agentify.core import BaseAgent, AgentConfig
from agentify.memory import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore
from agentify.multi_agent import Pipeline


def main():
    # 1. Setup Memory
    store = InMemoryStore()
    memory_service = MemoryService(store=store, log_enabled=True, max_log_length=200)

    # 2. Define Agents
    # Step 1: Researcher
    researcher_config = AgentConfig(
        name="Researcher",
        system_prompt="You are a researcher. Given a topic, provide 3 key facts about it. Be concise.",
        model_name="gpt-4o-mini",
        provider="openai",
    )
    researcher = BaseAgent(config=researcher_config, memory=memory_service)

    # Step 2: Writer
    writer_config = AgentConfig(
        name="Writer",
        system_prompt="You are a writer. Given a list of facts, write a short poem incorporating them.",
        model_name="gpt-4o-mini",
        provider="openai",
    )
    writer = BaseAgent(config=writer_config, memory=memory_service)

    # 3. Create Pipeline
    pipeline = Pipeline(steps=[researcher, writer])

    # 4. Run Pipeline
    topic = "The planet Mars"
    print(f"Input: {topic}\n")

    result = pipeline.run(user_input=topic, session_id="seq_demo_1")

    print("-" * 20)
    print("Final Output:")
    print(result)


if __name__ == "__main__":
    main()

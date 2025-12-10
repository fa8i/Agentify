import os
import sys
from dotenv import load_dotenv

# Ensure we can import agentify
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentify.core.agent import BaseAgent
from agentify.core.config import AgentConfig
from agentify.memory.service import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore
from agentify.memory.interfaces import MemoryAddress
from agentify.extensions.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from agentify.extensions.tools.planning import TodoTool
from agentify.multi_agent.tool_wrapper import SpawnAgentTool



def main():
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY in .env to run this demo.")
        return

    # 1. Setup Agent Configuration
    config = AgentConfig(
        name="DeepAgent-01",
        provider="openai",
        model_name="gpt-5",
        temperature=1.0, # always 1.0 on gpt-5 family models
        max_tool_iter=None,
        system_prompt=(
            "You are a sophisticated 'Deep Agent' capable of planning, "
            "managing files, and spawning sub-agents. "
            "Use your tools to solve complex tasks autonomously. "
            "First, always create a plan using 'manage_plan'. "
            "Then execute the plan step by step."
        ),
    )

    # 2. Setup Memory Service (Session Memory)
    memory_service = MemoryService(store=InMemoryStore())
    main_addr = MemoryAddress(user_id="user_1", conversation_id="session_deep_1", agent_id="main")

    # 3. Initialize Tools
    # Filesystem tools (Sandbox in current dir for safety)
    fs_tools = [
        ListDirTool(sandbox_dir="."),
        ReadFileTool(sandbox_dir="."),
        WriteFileTool(sandbox_dir="."),
    ]
    
    # Planning tool
    plan_tool = TodoTool()
    
    # Sub-agent spawner
    # Note: passing memory_service so sub-agents share the same store backend (but different addresses)
    spawn_tool = SpawnAgentTool(
        base_config=config, 
        memory_service=memory_service, 
        parent_addr=main_addr
    )

    tools = fs_tools + [plan_tool, spawn_tool]

    # 4. Initialize Agent
    agent = BaseAgent(
        config=config,
        memory=memory_service,
        memory_address=main_addr,
        tools=tools
    )

    print("\n--- Starting Deep Agent Demo ---\n")
    
    user_request = (
        "I need you to check the current directory for any markdown files. "
        "If you find my 'README.md', read it and summarize it."
        "Spawn a 'Summarizer' sub-agent to do it. "
        "Finally, save the summary to 'summary_report.txt'."
    )

    print(f"User Request: {user_request}\n")

    # Run the agent
    response_gen = agent.run(user_request)
    
    if hasattr(response_gen, "__iter__") and not isinstance(response_gen, str):
        full_response = ""
        for chunk in response_gen:
            print(chunk, end="", flush=True)
            full_response += chunk
        print("\n")
    else:
        print(f"Agent Response: {response_gen}")

    # Check long-term memory or file output
    print("\n--- Verification ---")
    if os.path.exists("summary_report.txt"):
        with open("summary_report.txt", "r") as f:
            print(f"Report content:\n{f.read()}")
    else:
        print("Report file was not created.")

if __name__ == "__main__":
    main()

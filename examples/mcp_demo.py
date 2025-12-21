"""MCP Demo: Connects to a public MCP server and uses its tools via an agent."""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentify.mcp import MCPConnection
from agentify.core.agent import BaseAgent
from agentify.core.config import AgentConfig
from agentify.memory.service import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore
from agentify.memory.interfaces import MemoryAddress


async def main():
    # Use the official "mcp-server-fetch" for fetching web content (no API key needed)
    print("Connecting to 'mcp-server-fetch'...")

    async with MCPConnection(command="uvx", args=["mcp-server-fetch"]) as mcp:
        tools = await mcp.get_tools()
        print(f"Loaded tools: {[t.name for t in tools]}")

        memory = MemoryService(store=InMemoryStore(), log_enabled=True)
        config = AgentConfig(
            name="WebFetchAgent",
            system_prompt="You are a web research assistant. Use the fetch tool to get web content when asked.",
            provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0,
        )

        agent = BaseAgent(
            config=config,
            memory=memory,
            tools=tools,
            memory_address=MemoryAddress(conversation_id="mcp_demo", agent_id="WebFetchAgent")
        )

        prompt = "Fetch the content from https://httpbin.org/get and summarize what you see."
        print(f"\nUser: {prompt}\n")

        try:
            response = await agent.arun(prompt)
            print(f"\nAgent Response:\n{response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

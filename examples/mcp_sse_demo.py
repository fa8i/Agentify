"""MCP SSE Demo: Connects to a remote MCP server via SSE transport."""
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
    # Replace with your SSE-enabled MCP server URL
    sse_url = os.getenv("MCP_SSE_URL", "http://localhost:8080/sse")

    print(f"Connecting to MCP server via SSE: {sse_url}...")

    try:
        async with MCPConnection.sse(url=sse_url) as mcp:
            tools = await mcp.get_tools()
            print(f"Loaded tools: {[t.name for t in tools]}")

            memory = MemoryService(store=InMemoryStore(), log_enabled=True)
            config = AgentConfig(
                name="SSEAgent",
                system_prompt="You are an assistant with access to remote tools.",
                provider="openai",
                model_name="gpt-4.1-mini",
                temperature=0,
            )

            agent = BaseAgent(
                config=config,
                memory=memory,
                tools=tools,
                memory_address=MemoryAddress(conversation_id="sse_demo", agent_id="SSEAgent")
            )

            prompt = "List the available tools and describe what they do."
            print(f"\nUser: {prompt}\n")

            response = await agent.arun(prompt)
            print(f"\nAgent Response:\n{response}")

    except Exception as e:
        print(f"Error connecting to SSE server: {e}")
        print("Make sure you have an MCP server running with SSE transport.")
        print("Set MCP_SSE_URL environment variable to your server's SSE endpoint.")


if __name__ == "__main__":
    asyncio.run(main())

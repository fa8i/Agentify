"""
Async Multi-Agent Demo

This example demonstrates the async capabilities of Agentify:
1. Basic async agent execution with run()
2. Async Team orchestration
3. Performance comparison between sync and async execution

Key benefits of async:
- LLM calls don't block the event loop
- Multiple tools execute in parallel via asyncio.gather()
- Better resource utilization in async applications (FastAPI, etc.)
"""

import asyncio
import time
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agentify.core import BaseAgent, AgentConfig, Tool
from agentify.memory import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore
from agentify.multi_agent import Team


# ----- Simulated async tools -----

async def async_search(query: str) -> str:
    """Simulates an async API call (e.g., web search)."""
    print(f"  [async_search] Starting search for: {query}")
    await asyncio.sleep(1)  # Simulate network delay
    print(f"  [async_search] Completed search for: {query}")
    return f"Search results for '{query}': Found 3 relevant articles about {query}."


async def async_summarize(text: str) -> str:
    """Simulates an async summarization task."""
    print("  [async_summarize] Starting summarization...")
    await asyncio.sleep(1)  # Simulate processing
    print("  [async_summarize] Completed summarization.")
    return f"Summary: The text discusses key points about {text[:50]}..."


def sync_calculate(expression: str) -> str:
    """A synchronous tool for comparison."""
    print(f"  [sync_calculate] Calculating: {expression}")
    # Simulates CPU work
    result = eval(expression)  # Note: unsafe in production!
    return f"Result: {result}"


# ----- Demo functions -----

async def demo_basic_async_agent():
    """Demonstrates basic async agent execution."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Async Agent")
    print("=" * 60)
    
    store = InMemoryStore()
    memory = MemoryService(store=store, log_enabled=True, max_log_length=100)
    
    agent = BaseAgent(
        config=AgentConfig(
            name="AsyncAssistant",
            system_prompt="You are a helpful assistant. Answer questions concisely.",
            provider="gemini",
            model_name="gemini-2.5-flash",
            temperature=0.3,
        ),
        memory=memory,
    )
    
    from agentify.memory.interfaces import MemoryAddress
    addr = MemoryAddress(conversation_id="async_demo_1", user_id="demo_user")
    
    print("\nCalling agent.arun() (async)...")
    start = time.time()
    response = await agent.arun("What is the capital of France? Answer in one word.", addr=addr)
    elapsed = time.time() - start
    
    print(f"\nResponse: {response}")
    print(f"Time: {elapsed:.2f}s")


async def demo_async_tools():
    """Demonstrates async tool execution with parallel calls."""
    print("\n" + "=" * 60)
    print("DEMO 2: Async Tools (Parallel Execution)")
    print("=" * 60)
    
    store = InMemoryStore()
    memory = MemoryService(store=store, log_enabled=True, max_log_length=100)
    
    # Create tools
    search_tool = Tool(
        schema={
            "name": "search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        },
        func=async_search
    )
    
    summarize_tool = Tool(
        schema={
            "name": "summarize",
            "description": "Summarize text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to summarize"}
                },
                "required": ["text"]
            }
        },
        func=async_summarize
    )
    
    agent = BaseAgent(
        config=AgentConfig(
            name="ResearchAgent",
            system_prompt=(
                "You are a research assistant. When asked to research a topic, "
                "use the search tool to find information, then summarize. "
                "Try to call multiple tools if beneficial."
            ),
            provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.3,
            max_tool_iter=3,
        ),
        memory=memory,
        tools=[search_tool, summarize_tool],
    )
    
    from agentify.memory.interfaces import MemoryAddress
    addr = MemoryAddress(conversation_id="async_tools_demo", user_id="demo_user")
    
    print("\nCalling agent.arun() with async tools...")
    print("If the model calls multiple tools, they will execute IN PARALLEL!")
    
    start = time.time()
    response = await agent.arun(
        "Search for 'Apple stock price' AND search for 'Microsoft stock price' at the same time.",
        addr=addr
    )
    elapsed = time.time() - start
    
    print(f"\nResponse: {response[:300]}...")
    print(f"\nTotal time: {elapsed:.2f}s")
    print("(If tools ran in parallel, this should be ~1-2s instead of 2-3s)")


async def demo_async_team():
    """Demonstrates async Team orchestration."""
    print("\n" + "=" * 60)
    print("DEMO 3: Async Team Orchestration")
    print("=" * 60)
    
    store = InMemoryStore()
    memory = MemoryService(store=store, log_enabled=True, max_log_length=100)
    
    # Supervisor
    supervisor = BaseAgent(
        config=AgentConfig(
            name="Supervisor",
            system_prompt=(
                "You are a team supervisor. Delegate tasks to your workers: "
                "Researcher and Writer. First get research, then get writing."
            ),
            provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.3,
            max_tool_iter=4,
        ),
        memory=memory,
    )
    
    # Workers
    researcher = BaseAgent(
        config=AgentConfig(
            name="Researcher",
            system_prompt="You are a researcher. Provide factual information briefly.",
            provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.3,
        ),
        memory=memory,
    )
    
    writer = BaseAgent(
        config=AgentConfig(
            name="Writer",
            system_prompt="You are a writer. Create engaging content based on input.",
            provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.5,
        ),
        memory=memory,
    )
    
    # Create team
    team = Team(
        agents=[supervisor, researcher, writer],
        supervisor=supervisor
    )
    
    print("\nCalling team.arun() (async orchestration)...")
    
    start = time.time()
    response = await team.arun(
        user_input="Create a short paragraph about renewable energy.",
        session_id="team_async_demo",
        user_id="demo_user"
    )
    elapsed = time.time() - start
    
    print(f"\nFinal Response:\n{response[:500]}...")
    print(f"\nTotal time: {elapsed:.2f}s")


async def main():
    """Run all demos."""
    print("=" * 60)
    print("AGENTIFY ASYNC CAPABILITIES DEMO")
    print("=" * 60)
    print("\nThis demo showcases the async features of the Agentify library.")
    print("All executions use async run() for non-blocking I/O.\n")
    
    # Demo 1: Basic async
    await demo_basic_async_agent()
    
    # Demo 2: Async tools
    await demo_async_tools()
    
    # Demo 3: Async team
    await demo_async_team()
    
    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

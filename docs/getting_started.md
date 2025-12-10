# Getting Started with Agentify

## Installation

```bash
pip install agentify-core
```

For development:
```bash
git clone https://github.com/fa8i/Agentify.git
cd Agentify
pip install -e .
```

## Prerequisites

- Python 3.10+
- API Key from your chosen provider (OpenAI, DeepSeek, Gemini, etc.)

## Environment Setup

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your-key-here
# Or for other providers:
# DEEPSEEK_API_KEY=your-key-here
# GEMINI_API_KEY=your-key-here
# ANTHROPIC_API_KEY=your-key-here
```

## Your First Agent

```python
import os
from dotenv import load_dotenv
from agentify import BaseAgent, AgentConfig, MemoryService, MemoryAddress
from agentify.memory.stores import InMemoryStore

load_dotenv()

# 1. Setup Memory
memory = MemoryService(store=InMemoryStore())
addr = MemoryAddress(conversation_id="my_first_chat")

# 2. Create Agent
agent = BaseAgent(
    config=AgentConfig(
        name="MyFirstAgent",
        system_prompt="You are a helpful assistant.",
        provider="provider",
        model_name="model_name",
        temperature=0.7,
    ),
    memory=memory,
    memory_address=addr
)

# 3. Chat
response = agent.run("Hello! Who are you?")
print(response)
```

## Streaming Responses

Enable streaming for real-time output:

```python
agent = BaseAgent(
    config=AgentConfig(
        name="StreamAgent",
        system_prompt="You are a helpful assistant.",
        provider="provider",
        model_name="model_name",
        stream=True,  # Enable streaming
    ),
    memory=memory,
    memory_address=addr
)

# Get a generator
response = agent.run("Tell me a story")

# Stream the response
for chunk in response:
    print(chunk, end="", flush=True)
```

## Adding Tools

Tools give your agent capabilities:

```python
from agentify.extensions.tools import TimeTool, CalculatorTool

agent = BaseAgent(
    config=AgentConfig(...),
    memory=memory,
    memory_address=addr,
    tools=[TimeTool(), CalculatorTool()]  # Add tools here
)

response = agent.run("What time is it? Also calculate 15 * 23")
print(response)
```

## Async Execution (Parallelism)

For high-performance applications, use `arun()` instead of `run()`. This allows:
1.  **Non-blocking execution:** Your server stays responsive while waiting for the LLM.
2.  **Parallel Tool Calls:** If the agent needs multiple tools (e.g., getting weather for 3 cities), it executes them **simultaneously**, saving time.

```python
import asyncio

async def main():
    # ... setup agent as above ...
    
    # Use 'await agent.arun()'
    response = await agent.arun("Get weather for Tokyo, London, and NY")
    print(response)

# Run the async loop
asyncio.run(main())
```

## Next Steps

- [Core Concepts](core_concepts.md) - Understand agents, memory, and tools
- [Multi-Agent Systems](multi_agent.md) - Build teams and pipelines
- [API Reference](api_reference.md) - Detailed API documentation
- [Examples](../examples/) - Complete working examples

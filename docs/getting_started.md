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
# LOCAL_API_BASE=http://localhost:1234/v1
# LOCAL_API_KEY=api_key (dummy for local servers)
```

## Your First Agent

The quickest path is the `Agent` helper. Only `model` is required; the store and
conversation address are created for you, and `provider` defaults to `"openai"`:

```python
from dotenv import load_dotenv
from agentify import Agent

load_dotenv()

agent = Agent(
    "You are a helpful assistant.",
    model="gpt-5.5",
    temperature=0.7,
)

# Chat (sync)
print(agent.run("Hello! Who are you?"))

# Async alternative:
# print(await agent.arun("Hello! Who are you?"))
```

### Full control with `BaseAgent`

When you need a custom store, a shared `MemoryService`, or multi-tenant memory
addressing, build the components explicitly:

```python
from dotenv import load_dotenv
from agentify import BaseAgent, AgentConfig, MemoryService, MemoryAddress, InMemoryStore

load_dotenv()

# 1. Setup Memory
memory = MemoryService(store=InMemoryStore())
addr = MemoryAddress(conversation_id="my_first_chat")

# 2. Create Agent
agent = BaseAgent(
    config=AgentConfig(
        name="MyFirstAgent",
        system_prompt="You are a helpful assistant.",
        provider="openai",
        model_name="gpt-5.5",
        temperature=0.7,
    ),
    memory=memory,
    memory_address=addr,
)

# 3. Chat (sync)
print(agent.run("Hello! Who are you?"))
```

## Streaming Responses

Enable streaming for real-time output:

```python
agent = Agent(
    "You are a helpful assistant.",
    model="gpt-5.5",
    stream=True,  # Enable streaming
)

# Get a sync generator
response = agent.run("Tell me a story")

# Stream the response (sync)
for chunk in response:
    print(chunk, end="", flush=True)

# Async streaming alternative:
# response = await agent.arun("Tell me a story")
# async for chunk in response:
#     print(chunk, end="", flush=True)
```

## Adding Tools

Tools give your agent capabilities:

```python
from agentify.extensions.tools import TimeTool, CalculatorTool

agent = Agent(
    "You are a helpful assistant.",
    model="gpt-5.5",
    tools=[TimeTool(), CalculatorTool()],  # Add tools here
)

response = agent.run("What time is it? Also calculate 15 * 23")
print(response)
```

## Async Execution (Parallelism)

`arun()` enables non-blocking execution and parallel tool calls:
1.  **Non-blocking execution:** Your server stays responsive while waiting for the LLM.
2.  **Parallel Tool Calls:** If the agent needs multiple tools (e.g., getting weather for 3 cities), it executes them **simultaneously**, saving time.

```python
import asyncio

async def main():
    # ... setup agent as above ...
    
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

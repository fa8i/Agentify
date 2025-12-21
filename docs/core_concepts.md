# Core Concepts

## Architecture Overview

Agentify is built around three main concepts:

```
┌─────────────┐
│    Agent    │ ◄── Executes tasks using tools and memory
└─────────────┘
       │
       ├── Memory Service ◄── Manages conversation history
       │
       └── Tools ◄── Extends agent capabilities
```

## Agents

### BaseAgent

The fundamental unit of work in Agentify. An agent:
- Receives user input
- Processes it using an LLM
- Can call tools to perform actions
- Maintains conversation history via memory

```python
from agentify import BaseAgent, AgentConfig

agent = BaseAgent(
    config=AgentConfig(
        name="MyAgent",
        system_prompt="You are a helpful assistant",
        provider="provider",
        model_name="model_name"
    ),
    memory=memory_service,
    memory_address=addr
)
```

### AgentConfig

Configuration class for agents:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `name` | `str` | Agent identifier | Required |
| `system_prompt` | `str` | System instructions | Required |
| `provider` | `str` | LLM provider | Required |
| `model_name` | `str` | Model to use | Required |
| `temperature` | `float` | Creativity (0-1) | `0.7` |
| `max_tool_iter` | `int\|None` | Max tool calls | `10` |
| `stream` | `bool` | Enable streaming | `False` |
| `timeout` | `int` | Request timeout (s) | `60` |
| `reasoning_effort` | `str\|None` | For reasoning models | `None` |

## Memory System

### Memory Architecture

```
MemoryService
    ├── ConversationStore (Backend)
    │   ├── InMemoryStore
    │   └── RedisStore
    │
    └── MemoryPolicy (Rules)
        ├── Message limit
        ├── TTL
        └── Token budget
```

### MemoryAddress

Identifies where conversations are stored:

```python
from agentify.memory import MemoryAddress

addr = MemoryAddress(
    user_id="user_123",
    conversation_id="chat_456",
    agent_id="agent_007"
)
```

### Memory Stores

**InMemoryStore** - For development/testing:
```python
from agentify.memory.stores import InMemoryStore
store = InMemoryStore()
```

**RedisStore** - For production:
```python
from agentify.memory.stores import RedisStore
store = RedisStore(url="redis://localhost:6379/0")
```

### Memory Policy

Control memory behavior:

```python
from agentify.memory import MemoryPolicy

policy = MemoryPolicy(
    store=store,
    ttl_seconds=3600,        # Expire after 1 hour
    max_user_msgs=10,        # Keep last 10 user messages
    max_assistant_msgs=10,   # Keep last 10 assistant messages
)
```

## Tools

Tools extend agent capabilities.

### Built-in Tools

```python
from agentify.extensions.tools import (
    TimeTool,         # Get current date/time
    CalculatorTool,   # Math calculations
    WeatherTool,      # Weather info
    TodoTool,         # Task planning
    ListDirTool,      # List files
    ReadFileTool,     # Read files
    WriteFileTool,    # Write files
)

agent = BaseAgent(
    config=config,
    memory=memory,
    tools=[
        TimeTool(),
        CalculatorTool(),
        TodoTool()
    ]
)
```

### Creating Custom Tools

Agentify offers two ways to create tools: the `@tool` decorator (recommended) or subclassing `Tool`.

#### Using the `@tool` Decorator (Recommended)

The `@tool` decorator expects **Google Style** docstring and automatically generates the JSON Schema from your function signature:


```python
from agentify import tool

@tool
def get_current_time() -> dict:
    """Returns the current date and time in ISO 8601 format."""
    import datetime
    return {"current_time": datetime.datetime.now().isoformat()}
```

**With Parameters:**

```python
@tool
def calculate(expression: str) -> dict:
    """Evaluates a mathematical expression.
    
    Args:
        expression: The math expression to evaluate (e.g., '2 + 2').
    """
    import ast
    # ... calculation logic ...
    return {"result": result}
```

**Note:** The `Returns:` section is purely for documentation and does NOT affect the generated JSON Schema.

**Using Decorator Tools:**

```python
from agentify import BaseAgent, AgentConfig, tool

@tool
def my_tool(param: str) -> dict:
    """My custom tool."""
    return {"result": param}

agent = BaseAgent(
    config=config,
    memory=memory,
    tools=[my_tool]  # Use directly, no need to instantiate
)
```

#### Subclassing Tool (Advanced)

For complex tools with state or initialization logic:

```python
from agentify import Tool

class CustomTool(Tool):
    def __init__(self):
        schema = {
            "name": "my_custom_tool",
            "description": "What this tool does",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "First parameter"
                    }
                },
                "required": ["param1"]
            }
        }
        super().__init__(schema, self._execute)
    
    def _execute(self, param1: str) -> dict:
        # Your logic here
        return {"result": f"Processed: {param1}"}

# Use it
agent = BaseAgent(
    config=config,
    memory=memory,
    tools=[CustomTool()]  # Instantiate for class-based tools
)
```

### MCP Tools (Model Context Protocol)

Agentify supports **MCP** (Model Context Protocol), an open standard for connecting AI agents to external tools and data sources.

#### How it Works

MCP uses a client-server architecture. Agentify acts as the **client**, connecting to any **MCP server** (local or remote) via `stdio` transport. The server exposes tools that Agentify automatically converts to its native `Tool` format.

#### Usage

Use `MCPConnection` as an async context manager to ensure proper resource cleanup:

```python
import asyncio
from agentify import BaseAgent, AgentConfig, MemoryService, MemoryAddress
from agentify.memory.stores import InMemoryStore
from agentify.mcp import MCPConnection

async def main():
    # Connect to an MCP server (e.g., the official 'fetch' server)
    async with MCPConnection(command="uvx", args=["mcp-server-fetch"]) as mcp:
        mcp_tools = await mcp.get_tools()

        agent = BaseAgent(
            config=AgentConfig(
                name="MCPAgent",
                system_prompt="You are an assistant.",
                provider="provider",
                model_name="model_name"
            ),
            memory=MemoryService(store=InMemoryStore()),
            memory_address=MemoryAddress(agent_id="MCPAgent"),
            tools=mcp_tools
        )

        response = await agent.arun("Fetch content from https://example.com")
        print(response)

asyncio.run(main())
```

> **Note:** The `mcp` package must be installed: `pip install mcp`

## Callbacks

Monitor agent behavior:

```python
from agentify.core.callbacks import AgentCallbackHandler

class MyCallback(AgentCallbackHandler):
    def on_agent_start(self, agent_id: str, input_text: str):
        print(f"Agent {agent_id} starting with: {input_text}")
    
    def on_tool_start(self, tool_name: str, arguments: dict):
        print(f"Calling tool {tool_name}: {arguments}")
    
    def on_agent_finish(self, agent_id: str, output: str):
        print(f"Agent {agent_id} finished: {output}")

agent = BaseAgent(
    config=AgentConfig(
        name="CallbackAgent",
        callbacks=[MyCallback()]
    ),
    memory=memory
)
```

## Providers

Agentify supports multiple LLM providers:

### OpenAI
```python
config = AgentConfig(
    provider="openai",
    model_name="gpt-4.1-mini"
)
```

### DeepSeek
```python
config = AgentConfig(
    provider="deepseek",
    model_name="deepseek-chat"
)
```

### Gemini
```python
config = AgentConfig(
    provider="gemini",
    model_name="gemini-2.5-flash"
)
```

### Azure OpenAI
```python
config = AgentConfig(
    provider="azure",
    model_name="model",
    client_config_override={
        "api_version": "2024-02-15-preview"
    }
)
```

## Next Steps

- [Multi-Agent Systems](multi_agent.md)
- [Advanced Features](advanced.md)
- [API Reference](api_reference.md)

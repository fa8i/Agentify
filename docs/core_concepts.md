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
    │   └── SQLiteStore
    │   └── ElasticsearchStore
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

# Delete conversation
store.delete_conversation(addr)
```

**SQLiteStore** - For zero-dependency persistence:
```python
from agentify.memory.stores import SQLiteStore

# Persistent (single file)
store = SQLiteStore(db_path="agentify.db")

# In-memory (transient) by default
store = SQLiteStore(db_path=":memory:")
```

**ElasticsearchStore** - For advanced search and durability:
```python
from agentify.memory.stores import ElasticsearchStore
store = ElasticsearchStore(url="http://localhost:9200", index_name="agentify-memory")

# Delete conversation
store.delete_conversation(addr)
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

Tools extend agent capabilities. Tool arguments are validated against their JSON schema before execution.

### Built-in Tools

```python
from agentify.extensions.tools import (
    TimeTool,         # Get current date/time
    CalculatorTool,   # Math calculations
    WeatherTool,      # Weather info
    TodoTool,         # Task planning
    ListDirTool,      # List files
    ReadFileTool,     # Read files (supports max_bytes)
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

#### Transports

| Transport | Use Case | Factory Method | Arguments |
|-----------|----------|----------------|-----------|
| **StdIO** | Local servers (scripts, CLIs) | `MCPConnection.stdio(...)` | `command`, `args`, `env` |
| **SSE** | Remote HTTP servers | `MCPConnection.sse(...)` | `url`, `headers`, `timeout`, `sse_read_timeout` |

#### StdIO Example (Local Servers)

```python
from agentify.mcp import MCPConnection

async with MCPConnection.stdio(command="uvx", args=["mcp-server-fetch"]) as mcp:
    tools = await mcp.get_tools()
    agent = BaseAgent(config=config, memory=memory, tools=tools, ...)
    await agent.arun("Fetch https://example.com")
```

#### SSE Example (Remote Servers)

```python
from agentify.mcp import MCPConnection

async with MCPConnection.sse(url="http://localhost:8080/sse", headers={"Authorization": "Bearer token"}) as mcp:
    tools = await mcp.get_tools()
    agent = BaseAgent(config=config, memory=memory, tools=tools, ...)
    await agent.arun("Use the remote tools")
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

> **Security Note**: The default `LoggingCallbackHandler` (enabled when `verbose=True`) automatically redacts sensitive keys like `password`, `api_key`, or `token` from tool arguments in the logs.


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

### Local LLMs (LM Studio, Ollama, etc.)
```python
config = AgentConfig(
    provider="local",
    model_name="google/gemma-4-e4b",  # Example
    # Optional override if not using env vars
    client_config_override={
        "base_url": "http://localhost:1234/v1"
    }
)
```

### Codex Native Provider

The `codex` provider uses ChatGPT OAuth via `codex login` and native Codex
threads. It sends the latest Agentify prompt string to a native Codex thread and
lets Codex maintain thread context.

```python
config = AgentConfig(
    provider="codex",
    model_name="gpt-5.3-codex"
)
```

The model is always taken from `AgentConfig.model_name`. In the real validation
environment, `gpt-5.3-codex` worked. Other models can fail depending on Codex
CLI version, ChatGPT account type, and quota.

Supported:

- ChatGPT OAuth through the Codex CLI
- Native Codex threads
- Persistent context per Codex thread
- Plain string prompts sent to native Codex turns
- Agentify tools exposed via MCP stdio

Not supported:

- OpenAI-style `tool_calls`
- Agentify's classic OpenAI-style tool loop
- Real streaming

Use `provider="openai"` for agents that need OpenAI-style function calling.
For Codex, tools must be exposed through MCP. Codex decides and invokes those
tools internally; it does not return `assistant_message.tool_calls` to Agentify.
Agentify reconstructs the final answer from `thread.turn(...).stream()` events.

Agentify provides a transport-agnostic `AgentifyMCPServer` adapter that converts
registered Agentify tools into MCP tool definitions and handlers. Wire this
adapter into a local MCP transport, then configure Codex to load that MCP server
from its Codex MCP configuration.

First define an importable registry that returns the tools for the current
agent/scope:

```python
# my_project/tools.py
from agentify.core.tool import tool

@tool
def echo_tool(text: str) -> str:
    """Echo text back to the caller."""
    return f"echo: {text}"

def build_agentify_tools():
    return [echo_tool]
```

Generate the Codex MCP config block:

```bash
agentify codex mcp config \
  --name agentify-my-agent \
  --registry my_project.tools:build_agentify_tools \
  --allow echo_tool \
  --command /absolute/path/to/.venv/bin/python
```

Add the generated block to `~/.codex/config.toml` for the first version. Global
configuration is the recommended path initially because project-scoped
`.codex/config.toml` can behave differently across Codex CLI, Desktop, and IDE
surfaces.

Equivalent manual config:

```toml
[mcp_servers.agentify-my-agent]
command = "python"
args = [
  "-m", "agentify.mcp.server",
  "--registry", "my_project.tools:build_agentify_tools",
  "--allow", "echo_tool"
]
enabled_tools = ["echo_tool"]
```

Manual verification:

- Run `codex login` if needed.
- Add the config block to `~/.codex/config.toml`.
- Start Codex and ask it to call `echo_tool` with a short string.
- Confirm the MCP tool result returns `echo: <your string>`.

#### Codex MCP end-to-end validation

The repository includes a manual validation script for the full flow:

```text
Agentify provider="codex"
  -> CodexThreadBackend
  -> openai-codex SDK
  -> thread.turn(prompt).stream()
  -> Codex MCP config
  -> Agentify MCP stdio server
  -> echo_tool
```

Run it from the project root after `codex login`:

```bash
.venv/bin/python scripts/manual_codex_mcp_e2e.py
```

To validate the complete Agentify runtime with `BaseAgent(provider="codex")`, run:

```bash
.venv/bin/python scripts/manual_codex_agentify_e2e.py --model gpt-5.3-codex
```

This script verifies both the BaseAgent response and the Agentify MCP debug log.
The expected success marker is `ECHO_FROM_AGENTIFY: hola-agentify` and the log
must show `call_tool called: echo_tool`.

The script:

- Creates a temporary backup of `~/.codex/config.toml`.
- Appends a temporary `[mcp_servers.agentify-e2e]` block using the absolute
  Python executable running the script.
- Starts Codex through `CodexThreadBackend` and asks it to call `echo_tool`.
- Verifies the response contains `ECHO_FROM_AGENTIFY: hola-agentify`.
- Restores the original Codex config before exit unless `--keep-config` is used.

Exit codes:

- `0`: Codex invoked the MCP tool and returned the expected marker.
- `1`: Codex returned a final response, but it did not contain the expected marker.
- `3`: The Agentify MCP debug log shows `echo_tool` was invoked, but the
  `openai-codex` SDK turn result did not expose a final response containing the
  tool output.

Agentify uses the Codex turn event stream (`thread.turn(...).stream()`) for the
native Codex provider because the convenience `thread.run(prompt)` API can
collapse MCP turns into `TurnResult(final_response=None, items=[])`. The event
stream exposes `item/agentMessage/delta`, `item/completed` for MCP calls, and
`turn/completed`, which is enough to reconstruct a usable response.

If an installed `openai-codex` SDK does not expose `thread.turn`, Agentify only
falls back to `thread.run()` when MCP tools are explicitly disabled through the
backend config. MCP-backed tools require event streaming.

Current diagnostic status:

- MCP server loaded: yes
- MCP tools invoked: yes
- MCP output visible in events: yes
- `thread.run(prompt).final_response` reliable for MCP turns: no

Alternative dynamic tools route:

- Route A, MCP external: Codex loads `AgentifyMCPServer` from Codex config and
  Agentify reconstructs the final response from Codex turn events. This is the
  implemented route.
- Route B, dynamic tools: the installed `openai-codex` SDK advertises
  `experimentalApi=True` internally, but its public Python API currently exposes
  no client methods for registering dynamic tools or responding to dynamic tool
  calls directly. This route is not implemented.

Debug logs from the Agentify MCP subprocess are written to
`/tmp/agentify-codex-mcp-e2e.log` by default. The stdio server also supports
manual logging with:

```bash
python -m agentify.mcp.server \
  --registry tests.fixtures.codex_mcp_registry:build_agentify_tools \
  --allow echo_tool \
  --debug-log /tmp/agentify-mcp.log
```

This manual E2E is intentionally not part of CI because it requires local
`codex login` authentication and a ChatGPT account with Codex access.

## Next Steps

- [Multi-Agent Systems](multi_agent.md)
- [Advanced Features](advanced.md)
- [API Reference](api_reference.md)

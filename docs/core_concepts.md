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
    model_name="gpt-5.5-mini"
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
threads. By default, Agentify remains the source of truth for memory: the
provider reads the configured Agentify memory store, formats that conversation
state into the Codex turn prompt, and starts a fresh Codex thread for each turn.

Install and authenticate Codex first:

```bash
pip install agentify-core[codex]
codex login
codex login status
```

Use ChatGPT login in the Codex CLI flow when you want to run the Codex models
available to your ChatGPT account. `codex login status` should report an active
login before running Agentify with `provider="codex"`. Model availability and
quota depend on the Codex CLI version and the authenticated account.

```python
config = AgentConfig(
    provider="codex",
    model_name="gpt-5.5"
)
```

The model is always taken from `AgentConfig.model_name`. In the real validation
environment, `gpt-5.5` worked. Other models can fail depending on Codex
CLI version, ChatGPT account type, and quota.

Supported:

- ChatGPT OAuth through the Codex CLI
- Native Codex threads
- Agentify-managed memory from SQLite, in-memory, Elastic, or any configured store
- Optional persistent context per Codex thread with `memory_mode="codex_thread"`
- Prompt context built from Agentify memory and sent to native Codex turns
- Normal `BaseAgent(tools=[...])` tools exposed to Codex through runtime MCP
- Multimodal image inputs from Agentify `image_path` converted to Codex image inputs
- Structured output through Codex `output_schema`
- Streaming text deltas reconstructed from Codex turn events

Not supported:

- OpenAI-style `tool_calls`
- Agentify's classic OpenAI-style tool loop

Codex streaming is event-based. Agentify maps `item/agentMessage/delta` events
to normal streaming chunks, but tool calls still happen inside Codex MCP turns
rather than through OpenAI-style `assistant.tool_calls` responses.

Use `provider="openai"` for agents that need OpenAI-style function calling.
For Codex, Agentify adapts normal `tools=[...]` to MCP internally. Codex decides
and invokes those tools through MCP; it does not return
`assistant_message.tool_calls` to Agentify. Agentify reconstructs the final
answer from `thread.turn(...).stream()` events.

Example with Agentify tools:

```python
from agentify.core.agent import BaseAgent
from agentify.core.config import AgentConfig
from agentify.extensions.tools.filesystem import ListDirTool, ReadFileTool
from agentify.memory.interfaces import MemoryAddress
from agentify.memory.service import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore

addr = MemoryAddress(conversation_id="codex-tools", agent_id="codex-agent")

agent = BaseAgent(
    config=AgentConfig(
        name="CodexToolsAgent",
        system_prompt="Use tools when they help answer the user.",
        provider="codex",
        model_name="gpt-5.5",
    ),
    memory=MemoryService(store=InMemoryStore()),
    memory_address=addr,
    tools=[
        ListDirTool(sandbox_dir="."),
        ReadFileTool(sandbox_dir="."),
    ],
)
```

Internally, Agentify starts a local runtime MCP bridge for the live `Tool`
objects and passes that MCP configuration to the Codex thread. This does not
require editing `~/.codex/config.toml` and it supports stateful Python tool
objects, closures, and tools that share the agent's memory service. Tool calls
use `AgentConfig.tool_timeout` and respect `AgentConfig.max_tool_iter` within a
Codex turn. MCP tool calls are persisted to Agentify memory as an assistant tool
intent followed by a `tool` result message with `metadata.source="codex_mcp"`.

Structured output can be requested with either a direct Codex schema:

```python
config = AgentConfig(
    provider="codex",
    model_name="gpt-5.5",
    model_kwargs={
        "output_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        }
    },
)
```

or with the OpenAI-style `response_format={"type": "json_schema", ...}` shape;
Agentify maps it to Codex `output_schema` internally.

For multimodal use, keep the normal Agentify API:

```python
agent.run("What is in this image?", image_path="diagram.png")
```

Agentify stores the multimodal user message in memory and sends the image part to
Codex as an SDK image input. If the installed Codex SDK does not expose image
input classes, Agentify degrades to an explicit text marker instead of silently
pretending the image was inspected.

By default, Codex uses Agentify-managed memory. Agentify formats the current
conversation history from its configured memory store into the prompt sent to a
fresh Codex thread, so SQLite/in-memory/Elastic remain the source of truth. To
opt into native Codex thread memory instead, configure:

```python
config = AgentConfig(
    provider="codex",
    model_name="gpt-5.5",
    client_config_override={"memory_mode": "codex_thread"}
)
```

The default is equivalent to:

```python
client_config_override={"memory_mode": "agentify"}
```

In `memory_mode="agentify"`, Codex threads are not reused as memory. In
`memory_mode="codex_thread"`, Codex thread IDs are reused per Agentify session.

Choosing a memory mode:

- `memory_mode="agentify"` (default) keeps Agentify memory authoritative and
  portable across providers, but every turn starts a fresh ephemeral Codex
  thread and resends the full conversation history. Per-turn latency grows with
  conversation length, and when Agentify tools are attached the MCP server is
  restarted on every turn. Use it for one-shot/batch calls or when memory must
  be readable and editable through the Agentify store.
- `memory_mode="codex_thread"` is the recommended mode for interactive,
  multi-turn assistants. One persistent Codex thread per Agentify session means
  only the latest message is sent per turn and the MCP server stays warm, so
  latency stays flat. Agentify memory still records the conversation and tool
  calls, but Codex thread state is what the model sees, so out-of-band edits to
  Agentify memory will not reach the model.

Measured difference (gpt-5.5, 4 short turns, real ChatGPT OAuth): with an
Agentify tool attached, `codex_thread` averaged ~8s/turn vs ~12s/turn for
`agentify` (~1.5x); without tools, ~3.5s vs ~5.8s (~1.7x). The gap grows with
conversation length because `agentify` resends the full history every turn.
Reproduce with `scripts/benchmark_codex_memory_modes.py`.

### How Codex thread memory works

In `memory_mode="codex_thread"`, conversation memory is managed by the Codex
CLI (`codex app-server`), not by Agentify:

- **Storage**: each thread is persisted on disk under `~/.codex/` as a rollout
  JSONL file in `sessions/<year>/...`, indexed by `session_index.jsonl` and
  SQLite state databases. Archived threads move to `archived_sessions/`.
  `~/.codex/auth.json` holds the OAuth tokens — Agentify never reads it, and
  neither should your application or logs.
- **Lifetime**: threads survive process restarts and are resumable by ID. The
  ephemeral threads used by `memory_mode="agentify"` are *not* persisted.
- **Context management**: Codex tracks token usage per thread and
  automatically compacts (summarizes) the thread when it approaches the model
  context window, so long-running sessions do not need manual pruning.
- **System prompt**: the agent system prompt is passed as thread-level
  instructions on every `thread_start`/`thread_resume` (not as conversation
  text). Codex keeps it in the compaction-preserved prefix and Agentify
  re-applies it each turn, so the agent persona does not degrade as the thread
  is compacted. Use `instructions_mode="base"` to fully replace Codex's native
  coding-agent harness, or the default `"developer"` to layer the prompt on top
  of it (see the API reference).

Agentify keeps a session → Codex thread ID mapping in the backend. By default
it lives in memory, so a process restart starts fresh threads. To make
sessions durable across restarts, persist the mapping (a plain JSON file of
IDs, no secrets):

```python
config = AgentConfig(
    provider="codex",
    model_name="gpt-5.5",
    client_config_override={
        "memory_mode": "codex_thread",
        "thread_map_path": "~/.agentify/codex_threads.json",
    },
)
```

If a mapped thread no longer exists on disk (deleted, archived elsewhere),
Agentify logs a warning and transparently starts a new thread for that session
instead of failing; transient resume errors are still raised so a network blip
never silently discards context.

To inspect the native thread state, the Codex backend exposes:

```python
backend = agent._get_async_client()          # CodexThreadBackend
thread_id = backend.get_thread_id(session_id)
response = await backend.read_session_history(session_id)  # ThreadReadResponse
for item in response.thread.items:
    ...
```

Call `agent.close()` or `await agent.aclose()` in long-running applications to
release provider resources such as the runtime MCP bridge.

Example with SQLite memory:

```python
from agentify.core.agent import BaseAgent
from agentify.core.config import AgentConfig
from agentify.memory.interfaces import MemoryAddress
from agentify.memory.service import MemoryService
from agentify.memory.stores.sqlite_store import SQLiteStore

memory = MemoryService(store=SQLiteStore("agentify-memory.db"))
addr = MemoryAddress(
    tenant_id="tenant-1",
    user_id="user-1",
    conversation_id="conversation-1",
    agent_id="codex-agent",
)

agent = BaseAgent(
    config=AgentConfig(
        name="CodexAgent",
        system_prompt="You are a helpful assistant.",
        provider="codex",
        model_name="gpt-5.5",
    ),
    memory=memory,
    memory_address=addr,
)

agent.run("Remember that I live in Madrid.")
agent.run("Where do I live?")
```

Both user turns and assistant responses are persisted in SQLite under the same
`MemoryAddress`. On the second call, Agentify reads that SQLite history and sends
the current conversation state to Codex. If a `MemoryPolicy` trims or summarizes
history, Codex receives the policy-filtered history rather than unbounded raw
history.

For advanced deployments, Agentify also provides a transport-agnostic
`AgentifyMCPServer` adapter that converts registered Agentify tools into MCP tool
definitions and handlers. Use this when you want a long-running external MCP
server or an explicit Codex config instead of the automatic runtime bridge.

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
.venv/bin/python scripts/manual_codex_agentify_e2e.py --model gpt-5.5
```

To run the broader manual diagnostics for structured output, image input,
streaming, and runtime MCP tool limits, run:

```bash
.venv/bin/python scripts/manual_codex_feature_diagnostics.py --case all --image IMAGEN.png
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

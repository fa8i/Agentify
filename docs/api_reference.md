# API Reference

Complete reference for Agentify classes and methods.

## Core

### BaseAgent

Main agent class.

```python
class BaseAgent:
    def __init__(
        self,
        config: AgentConfig,
        memory: MemoryService,
        *,
        memory_address: Optional[MemoryAddress] = None,
        client_factory: Optional[LLMClientFactory] = None,
        tools: Optional[List[Tool]] = None,
        image_config: Optional[ImageConfig] = None,
        pre_hooks: Optional[List[Callable]] = None,
        post_hooks: Optional[List[Callable]] = None,
    )
```

**Parameters:**
- `config`: Agent configuration
- `memory`: Memory service instance
- `memory_address`: Optional memory address
- `client_factory`: Optional custom LLM client factory
- `tools`: List of tools
- `image_config`: Image processing configuration
- `pre_hooks`: Functions to run before execution
- `post_hooks`: Functions to run after execution

**Methods:**

#### `run(user_input, *, addr=None, image_path=None, image_detail_override=None)`
Execute the agent with user input (sync API).

**Returns:** `str` or `Generator[str, None, None]` (if streaming)

#### `arun(user_input, *, addr=None, image_path=None, image_detail_override=None)`
Execute the agent with user input (async API).

**Returns:** `str` or `AsyncGenerator[str, None]` (if streaming)

#### `add(role, content=None, *, addr=None, **kwargs)`
Add a message to memory.

#### `clear_memory(*, addr=None)`
Reset conversation history.

#### `get_history(addr)`
Get conversation history.

**Returns:** `List[Dict[str, Any]]`

#### `save_history(path, *, addr=None)`
Save history to JSON file.

#### `load_history(path, *, addr=None)`
Load history from JSON file.

#### `register_tool(tool)`
Register a new tool.

#### `unregister_tool(name)`
Remove a tool.

**Returns:** `bool` (True if removed)

#### `tool_exists(name)`
Check if tool is registered.

**Returns:** `bool`

**Properties:**
- `tool_defs`: List of tool definitions
- `list_tools`: List of tool names

### AgentConfig

Configuration for agents.

```python
@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    provider: str
    model_name: str
    temperature: float = 1.0
    timeout: int = 300
    tool_timeout: int = 300
    stream: bool = False
    max_retries: int = 3
    verbose: bool = True
    max_tool_iter: Optional[int] = 10
    delegation_recovery_enabled: bool = True
    delegation_recovery_mode: str = "retry_isolated"
    delegation_max_retries: int = 1
    delegation_retry_backoff_ms: int = 200
    reasoning_effort: Optional[str] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    client_config_override: Optional[Dict[str, Any]] = None
    callbacks: Optional[List[AgentCallbackHandler]] = None
```

### Tool

Base class for tools.

```python
class Tool:
    def __init__(
        self,
        schema: Dict[str, Any],
        func: Callable
    )
    
    @property
    def name(self) -> str
    
    @property
    def schema(self) -> Dict[str, Any]
    
    def __call__(self, **kwargs) -> Any
```
Tool arguments are validated against the JSON schema before execution.

## Memory

### MemoryService

Main memory management interface.

```python
class MemoryService:
    def __init__(
        self,
        store: ConversationStore,
        policy: Optional[MemoryPolicy] = None,
        log_enabled: bool = True,
        max_log_length: Optional[int] = 5000,
    )
```

**Methods:**

#### `append_history(addr, message)`
Add a message to history. Logging truncates content to `max_log_length` and redacts common secrets.

#### `reset_history(addr, system_message)`
Replace history with system message.

#### `get_history(addr)`
Get all messages.

**Returns:** `List[Dict[str, Any]]`

#### `delete_history(addr)`
Remove all messages for address.

### MemoryAddress

Identifier for conversations.

```python
@dataclass(frozen=True)
class MemoryAddress:
    api_version: Optional[str] = None
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    agent_id: Optional[str] = None
    extras: Tuple[Tuple[str, str], ...] = ()
```

**Methods:**

#### `key_str(prefix="mem")`
Generate storage key. Keys and values are URL-encoded for safe storage.

**Returns:** `str`

#### `as_tuple()`
Get hashable tuple representation.

**Returns:** `Tuple`

### MemoryPolicy

Control memory behavior.

```python
class MemoryPolicy:
    def __init__(
        self,
        store: ConversationStore,
        *,
        ttl_seconds: Optional[int] = None,
        max_user_msgs: int = 6,
        max_assistant_msgs: int = 6,
        tokenizer: Optional[TokenCounter] = None,
        max_tokens: Optional[int] = None,
        summarizer: Optional[Callable] = None,
    )
```

### InMemoryStore

In-memory conversation storage.

```python
class InMemoryStore:
    def __init__(self)
    
    def append_message(self, addr, msg)
    def read_messages(self, addr, start=0, end=-1)
    def replace_messages(self, addr, messages)
    def delete_conversation(self, addr)
    def set_ttl(self, addr, seconds)
```

### RedisStore

Redis-backed conversation storage.

```python
class RedisStore:
    def __init__(self, url: str)
    
    # Same methods as InMemoryStore
```

## LLM Clients

### LLMClientFactory

```python
class LLMClientFactory:
    def create_client(
        self,
        provider: str,
        config_override: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
    ) -> LLMClientType
```

Supported providers:
- `"openai"`
- `"azure"`
- `"deepseek"`
- `"gemini"`
- `"anthropic"`
- `"llama"`
- `"local"` (e.g., LM Studio, Ollama)
- `"codex"` (experimental native Codex threads via ChatGPT OAuth; Agentify
  memory is the default source of truth; normal `BaseAgent(tools=[...])` tools
  are adapted to runtime MCP instead of OpenAI-style `tool_calls`; responses are
  reconstructed or streamed from `thread.turn(...).stream()` events)

Codex-specific `client_config_override` keys:

- `memory_mode`: `"agentify"` by default. Uses the configured Agentify memory
  store and sends the current history returned by `MemoryService` to a fresh
  Codex thread each turn. Stores such as SQLite, in-memory, and Elastic remain
  the memory source of truth. Set `"codex_thread"` to reuse native Codex thread
  memory per Agentify session — recommended for interactive multi-turn
  assistants, since it avoids resending the full history and restarting the
  runtime MCP server on every turn (~1.5–1.7x faster per turn in benchmarks;
  see `scripts/benchmark_codex_memory_modes.py`).
- `thread_map_path`: optional path to a JSON file persisting the
  session → Codex thread ID mapping across process restarts (only meaningful
  with `memory_mode="codex_thread"`). Codex threads themselves are persisted
  by the Codex CLI under `~/.codex/`. If a mapped thread no longer exists,
  Agentify starts a new one for that session and logs a warning.
- `mcp_tools_enabled`: `True` by default. Requires the Codex SDK
  `thread.turn(...).stream()` API when MCP tools are configured.
- `auto_mcp_tools`: `True` by default. When `tools=[...]` are passed to a Codex
  `BaseAgent`, Agentify starts a local runtime MCP bridge and passes that MCP
  config to the Codex thread automatically. Set to `False` only if tools are
  configured through an external MCP server manually.
- `output_schema`: optional JSON schema passed to Codex `thread.turn(...)` for
  structured output. OpenAI-style `response_format={"type": "json_schema", ...}`
  is also mapped to this internally.

Codex-specific notes:

- `stream=True` emits text deltas from Codex turn events.
- `image_path` multimodal input is converted to Codex SDK image input when
  supported by the installed SDK.
- Runtime MCP tool calls respect `AgentConfig.tool_timeout` and
  `AgentConfig.max_tool_iter`. Tools are registered on the runtime MCP bridge
  per session, so concurrent turns for different sessions are isolated.
- `CodexThreadBackend.get_thread_id(session_id)` returns the mapped Codex
  thread ID; `await CodexThreadBackend.read_session_history(session_id)`
  returns the native thread history (`ThreadReadResponse`) in
  `memory_mode="codex_thread"`.
- Use `agent.close()` or `await agent.aclose()` to release provider resources in
  long-running applications.


## Multi-Agent

### Team

Supervisor-workers pattern.

```python
class Team:
    def __init__(
        self,
        supervisor: BaseAgent,
        workers: List[Union[BaseAgent, "Team", "SequentialPipeline"]],
        session_id: str = "default_session",
        user_id: str = "default_user",
    )
    
    def run(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Union[str, Generator[str, None, None]]

    async def arun(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Union[str, AsyncGenerator[str, None]]
```

### SequentialPipeline

Sequential execution pipeline.

```python
class SequentialPipeline:
    def __init__(
        self,
        steps: List[PipelineStep],
        session_id: str = "default_session",
        user_id: str = "default_user",
    )
    
    def run(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Union[str, Generator[str, None, None]]

    async def arun(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Union[str, AsyncGenerator[str, None]]
```

### HierarchicalTeam

Multi-level hierarchy.

```python
class HierarchicalTeam:
    def __init__(
        self,
        root: BaseAgent,
        hierarchy: Dict[BaseAgent, List[Union[BaseAgent, Team, SequentialPipeline]]],
        session_id: str = "default_session",
        user_id: str = "default_user",
    )
    
    def run(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Union[str, Generator[str, None, None]]

    async def arun(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Union[str, AsyncGenerator[str, None]]
```

## Tools

### AgentTool

Wrap agent as a tool.

```python
class AgentTool(Tool):
    def __init__(
        self,
        agent: BaseAgent,
        parent_addr: MemoryAddress,
        description_override: Optional[str] = None,
    )
```

### FlowTool

Wrap multi-agent flow as a tool.

```python
class FlowTool(Tool):
    def __init__(
        self,
        flow: Any,  # Team, Pipeline, or Hierarchy
        name: str,
        description: str,
        parent_addr: MemoryAddress,
    )
```

### SpawnAgentTool

Dynamically spawn sub-agents.

```python
class SpawnAgentTool(Tool):
    def __init__(
        self,
        base_config: AgentConfig,
        memory_service: MemoryService,
        parent_addr: MemoryAddress,
        client_factory: Optional[Any] = None,
    )
```

## Extensions

### Built-in Tools

#### TimeTool
```python
class TimeTool(Tool):
    def __init__(self)
```
Returns current date/time in ISO 8601 format.

#### CalculatorTool
```python
class CalculatorTool(Tool):
    def __init__(self)
```
Evaluates safe mathematical expressions.

#### WeatherTool
```python
class WeatherTool(Tool):
    def __init__(self)
```
Gets weather information (requires `OPENWEATHER_API_KEY`).

#### TodoTool
```python
class TodoTool(Tool):
    def __init__(self)
```
Manages task lists with actions: `add`, `complete`, `list`, `remove`.

#### ListDirTool
```python
class ListDirTool(Tool):
    def __init__(self, sandbox_dir: Optional[str] = None)
```
Lists files and directories.

#### ReadFileTool
```python
class ReadFileTool(Tool):
    def __init__(self, sandbox_dir: Optional[str] = None)
```
Reads file contents. Accepts `max_bytes` per call (default 1MB, hard cap 5MB).

#### WriteFileTool
```python
class WriteFileTool(Tool):  
    def __init__(self, sandbox_dir: Optional[str] = None)
```
Writes content to files.

## MCP

### AgentifyMCPServer

Transport-agnostic adapter for exposing a scoped set of Agentify tools as MCP
tool definitions and handlers. This is the foundation for using tools with the
native Codex provider, where Codex consumes tools through MCP instead of
OpenAI-style `tool_calls`.

```python
class AgentifyMCPServer:
    def __init__(self, tools: Sequence[Tool], *, allowlist: Iterable[str] | None = None)
    def list_tools(self) -> list[mcp.types.Tool]
    async def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> mcp.types.CallToolResult
```

Use `allowlist` to limit which Agentify tools are exposed to Codex through MCP.

Stdio entrypoint:

```bash
python -m agentify.mcp.server \
  --registry my_project.tools:build_agentify_tools \
  --allow search_docs \
  --debug-log /tmp/agentify-mcp.log
```

Config helper:

```bash
agentify codex mcp config --name agentify-my-agent --registry my_project.tools:build_agentify_tools --allow search_docs
```

## Callbacks

### AgentCallbackHandler

Base class for callbacks.

```python
class AgentCallbackHandler(Protocol):
    def on_agent_start(self, agent_id: str, input_text: str)
    def on_agent_finish(self, agent_id: str, output: str)
    def on_llm_start(self, model_name: str, messages: List[Dict])
    def on_llm_end(self, response: Any)
    def on_llm_new_token(self, token: str)
    def on_reasoning_step(self, content: str)
    def on_tool_start(self, tool_name: str, arguments: Dict)
    def on_tool_finish(self, tool_name: str, result: str)
    def on_error(self, error: Exception, context: str)
```

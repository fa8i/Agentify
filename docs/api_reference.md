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
Execute the agent with user input.

**Returns:** `str` or `Generator[str, None, None]` (if streaming)

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
    temperature: float = 0.7
    timeout: int = 60
    stream: bool = False
    max_retries: int = 3
    max_tool_iter: Optional[int] = 10
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
        max_log_length: Optional[int] = None,
    )
```

**Methods:**

#### `append_history(addr, message)`
Add a message to history.

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
Generate storage key.

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
    ) -> str
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
    ) -> str
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
    ) -> str
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
Reads file contents.

#### WriteFileTool
```python
class WriteFileTool(Tool):  
    def __init__(self, sandbox_dir: Optional[str] = None)
```
Writes content to files.

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


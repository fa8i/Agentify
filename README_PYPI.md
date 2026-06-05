# Agentify

**Production-ready AI agent library built on the OpenAI SDK**

Build and orchestrate AI agents—from simple assistants to complex multi-agent systems. Agentify targets the OpenAI-compatible Chat Completions interface, enabling seamless switching between providers (OpenAI, Azure, DeepSeek, Gemini, Anthropic, Llama, **Local LLMs**) without code changes. It also includes experimental native Codex support through ChatGPT OAuth.

---

## Why Agentify?

| Feature | Benefit |
|---------|---------|
| **Production-first** | Clear abstractions, explicit config, robust error handling |
| **Multi-provider** | Switch providers with one line—no agent code changes |
| **Orchestration primitives** | Uniform `run()`/`arun()` across agents, teams, pipelines, hierarchies |
| **Async-native** | Non-blocking I/O, parallel tool execution, event-loop friendly |
| **Pluggable memory** | In-memory, SQLite, Redis, Elasticsearch—same API |
| **Local LLMs** | Support for LM Studio, Ollama and custom local servers |
| **Codex provider** | Native Codex threads with Agentify memory, tools via runtime MCP, image input, structured output and streaming events |

---

## Key Features

- **Single agents & multi-agent patterns**  
  Agents with tools and memory • Supervisor–worker Teams • Sequential Pipelines • Hierarchical delegation • Dynamic routing

- **Memory system**  
  Pluggable backends with policies (TTL, message limits, pruning) • Memory isolation per conversation • Async-safe operations

- **Reasoning models**  
  Configurable thinking depth (`reasoning_effort`) • Chain-of-thought storage • Real-time reasoning logs

- **Tools**  
  `@tool` decorator with automatic JSON Schema • Type-annotated interface • Argument validation

- **Async & parallel execution**  
  Native `arun()` for async apps • `run()` bridge for sync apps • Parallel tool calls

- **Observability**  
  Callback hooks for logging, monitoring, debugging

- **Multimodal**  
  Vision/image support • Streaming responses

---

## Installation

```bash
pip install agentify-core
```

Optional backends:
```bash
pip install agentify-core[redis]      # Redis memory store
pip install agentify-core[elastic]    # Elasticsearch store
pip install agentify-core[codex]      # Native Codex provider
pip install agentify-core[all]        # All optional dependencies
```

---

## Quick Start

```python
from agentify import BaseAgent, AgentConfig, MemoryService, MemoryAddress, tool
from agentify.memory.stores import InMemoryStore

@tool
def get_time() -> dict:
    """Returns the current time."""
    from datetime import datetime
    return {"time": datetime.now().strftime("%H:%M:%S")}

# Setup
memory = MemoryService(store=InMemoryStore())
addr = MemoryAddress(conversation_id="session_1")

agent = BaseAgent(
    config=AgentConfig(
        name="Assistant",
        system_prompt="You are a helpful assistant.",
        provider="provider",
        model_name="model",
        reasoning_effort="high",  # optional param:"low", "medium", "high"
        model_kwargs={"max_completion_tokens": 5000}, # Pass model-specific params
        verbose=True, # Controls logging
    ),
    memory=memory,
    memory_address=addr,
    tools=[get_time],
)

response = agent.run("What time is it?")
print(response)

# Async usage is also available:
# response = await agent.arun("What time is it?")
```

---

## Native Codex Provider

Codex support is experimental and uses ChatGPT OAuth through the Codex CLI:

```bash
pip install agentify-core[codex]
codex login
codex login status
```

`codex login` opens the Codex CLI authentication flow. Choose ChatGPT login when
you want to use the Codex models available to your ChatGPT account. If the login
is successful, `codex login status` should report that you are logged in. Model
availability and quota depend on your Codex CLI version and ChatGPT account.

Then use the same Agentify API:

```python
agent = BaseAgent(
    config=AgentConfig(
        name="CodexAgent",
        system_prompt="You are a helpful assistant.",
        provider="codex",
        model_name="gpt-5.3-codex",
    ),
    memory=memory,
    memory_address=addr,
    tools=[get_time],
)

response = agent.run("Use the tool and answer concisely.")
```

Agentify keeps memory as the source of truth and sends the current conversation
state to Codex. Normal `tools=[...]` are exposed to Codex through an internal
runtime MCP bridge, so users do not need to manually wrap tools for common use.

Supported Codex features include:

- Agentify-managed memory with SQLite, in-memory, Redis or Elasticsearch stores.
- Runtime MCP tools with logs and persisted tool-call history.
- `stream=True` using Codex turn events.
- `image_path=...` multimodal input when supported by the installed Codex SDK.
- Structured output with `model_kwargs={"output_schema": ...}` or OpenAI-style `response_format`.

For long-running apps, call `agent.close()` or `await agent.aclose()` to release
provider resources.

---

## Memory Backends

```python
from agentify.memory.stores import InMemoryStore
from agentify.memory.stores.sqlite_store import SQLiteStore
from agentify.memory.stores.redis_store import RedisStore

# In-memory (default, for development)
store = InMemoryStore()

# SQLite (persistent, zero-config)
store = SQLiteStore(db_path="./agent.db")

# Redis (production, distributed)
store = RedisStore(url="redis://localhost:6379/0")
```

---

## Composable Flows

All primitives share the same `run()`/`arun()` interface:

- **BaseAgent** — Single agent with tools
- **Team** — Supervisor routes to worker agents
- **SequentialPipeline** — Output flows step-to-step
- **HierarchicalTeam** — Tree structures for delegation

Nest freely: Teams of Pipelines, Pipelines of Teams, dynamic routing at runtime.

---

## Links

- **Documentation & Examples**: [GitHub Repository](https://github.com/fa8i/Agentify)
- **Issues & PRs**: [Contribute](https://github.com/fa8i/Agentify/issues)

---

## License

MIT License

## Author

**Fabian Melchor** — [fabianmp_98@hotmail.com](mailto:fabianmp_98@hotmail.com)

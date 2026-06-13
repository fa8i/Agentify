<div align="center">

# Agentify

[![PyPI version](https://img.shields.io/pypi/v/agentify-core?color=orange&style=for-the-badge)](https://pypi.org/project/agentify-core/)
[![Downloads](https://img.shields.io/pepy/dt/agentify-core?style=for-the-badge)](https://pepy.tech/project/agentify-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/agentify-core?style=for-the-badge)](https://pypi.org/project/agentify-core/)

<h3>Independent AI Agent Library based on the OpenAI SDK</h3>

<p>
Agentify is a lightweight, clean, and powerful Python library for building AI agents and multi-agent systems. 
It provides simple abstractions for memory, tools, and orchestration without the heavy framework lock-in.
</p>

[Getting Started](docs/getting_started.md) • [Documentation](docs/api_reference.md) • [Examples](examples/) • [Changelog](CHANGELOG.md)

</div>

---

## Key Features

| Feature | Description |
| :--- | :--- |
| **Multi-Agent Orchestration** | Teams, sequential pipelines, hierarchical structures, and dynamic sub-agent spawning. |
| **Memory Service** | Pluggable backends (In-Memory, SQLite, Redis, Elasticsearch) with configurable TTL, limits and token budgets. |
| **Tools & MCP** | Easy `@tool` decorator, custom classes, and full **Model Context Protocol (MCP)** integration. |
| **Local Models** | First-class support for **LM Studio**, **Ollama**, and other local servers via OpenAI-compatible endpoints. |
| **Codex Provider** | Experimental native Codex support with Agentify memory, runtime MCP tools, image input, structured output, and event streaming. |
| **Async & Parallel** | Dual API: simple `run()` for sync usage and `arun()` for native async execution. |
| **Observability** | Comprehensive callback system for monitoring, debugging, and tracing agent thoughts. |
| **Reasoning & Planning** | Configure thinking depth, chain-of-thought storage, and real-time reasoning logs. |

## Installation

Install the core package:

```bash
pip install agentify-core
```

For all optional features (Redis, vector stores, etc.):

```bash
pip install agentify-core[all]
```

For native Codex support:

```bash
pip install agentify-core[codex]
codex login
codex login status
```

`codex login` starts the Codex CLI authentication flow. Use ChatGPT login to run
Codex models available to your ChatGPT account. `codex login status` confirms the
active session; available models and quota depend on your account and Codex CLI
version.

## Quick Start

Here is how to create a simple agent with memory and tools:

```python
# Note: Agentify does not auto-load .env. Load it manually if needed.
# from dotenv import load_dotenv; load_dotenv()

from agentify import BaseAgent, AgentConfig, MemoryService, MemoryAddress, tool
from agentify.memory.stores import InMemoryStore

# 1. Define a tool
@tool
def get_time() -> dict:
    """Returns the current local time."""
    from datetime import datetime
    return {"time": datetime.now().strftime("%H:%M:%S")}

# 2. Initialize Memory
memory = MemoryService(store=InMemoryStore())
addr = MemoryAddress(conversation_id="session_1")

# 3. Create the Agent
agent = BaseAgent(
    config=AgentConfig(
        name="ReasoningAgent",
        system_prompt="You are a helpful assistant.",
        provider="provider",
        model_name="model",
        reasoning_effort="low",  # optional param:"low", "medium", "high"
        model_kwargs={"max_completion_tokens": 5000}, # Pass model-specific params
        verbose=True, # Controls logging (True by default)
    ),
    memory=memory,
    memory_address=addr,
    tools=[get_time]
)

# 4. Run it (sync)
response = agent.run(user_input="What time is it?")
print(response)

# Async usage is also available:
# response = await agent.arun(user_input="What time is it?")
```

## Native Codex Provider

Codex support uses ChatGPT OAuth via the Codex CLI and keeps the normal Agentify
API:

```python
agent = BaseAgent(
    config=AgentConfig(
        name="CodexAgent",
        system_prompt="You are a helpful assistant.",
        provider="codex",
        model_name="gpt-5.4",
        stream=True,
    ),
    memory=memory,
    memory_address=addr,
    tools=[get_time],
)
```

Agentify keeps its memory stores as the source of truth and adapts normal
`tools=[...]` to Codex through runtime MCP internally. The provider also supports
structured output, image input via `image_path`, streaming text deltas, and
persisted MCP tool-call history.

For interactive multi-turn assistants, prefer native Codex thread memory — it
avoids resending the full history each turn (~1.5–1.7x faster per turn in our
benchmarks) and can persist sessions across restarts:

```python
client_config_override={
    "memory_mode": "codex_thread",
    "thread_map_path": "~/.agentify/codex_threads.json",  # optional
}
```

See [Core Concepts](docs/core_concepts.md) for how Codex thread memory works.

## Documentation

Detailed guides and API references are available in the `docs/` directory:

- **[Getting Started](docs/getting_started.md)**: Installation and first steps.
- **[Core Concepts](docs/core_concepts.md)**: Deep dive into Agents, Memory, and Tools.
- **[Multi-Agent Systems](docs/multi_agent.md)**: Building Teams, Pipelines, and Hierarchies.
- **[Advanced Features](docs/advanced.md)**: Vision, Screening, Hooks, and more.

## Examples

Explore the `examples/` directory for production-ready implementations:

- **[Chatbot](examples/chatbot/)**: A simple conversational agent.
- **[Multi-Agent Team](examples/multi_agent/team/)**: Agents working together.
- **[Pipelines](examples/multi_agent/pipeline/)**: Sequential task processing.
- **[Hierarchies](examples/multi_agent/hierarchical/)**: Complex delegated decision making.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">
Created by <b>Fabian Melchor</b><br>
<a href="mailto:fabianmp_98@hotmail.com">fabianmp_98@hotmail.com</a>
</div>

# Agentify

[![PyPI version](https://img.shields.io/pypi/v/agentify-core?color=orange)](https://pypi.org/project/agentify-core/)
[![Downloads](https://img.shields.io/pepy/dt/agentify-core)](https://pepy.tech/project/agentify-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/agentify-core)](https://pypi.org/project/agentify-core/)

**Independent AI agent library based on the OpenAI SDK**

Agentify is a Python library for building AI agents and multi-agent systems. Built on the OpenAI-compatible Chat Completions interface, it supports multiple providers (OpenAI, Azure, DeepSeek, Gemini, Claude) with clear abstractions for memory, tools, and orchestration—no heavy framework lock-in.


## Key Features

- **Multi-agent orchestration**: Teams, pipelines, hierarchies, and dynamic sub-agent spawning
- **Memory service**: Pluggable backends (in-memory, SQLite, Redis, Elasticsearch) with policies (TTL, limits, token budgets)
- **Tools**: `@tool` decorator for auto-schema generation, or custom tool classes. Built-in file I/O, planning, weather, and more
- **MCP Integration**: Easy connection to MCP servers via StdIO (local) or SSE/HTTP (remote) to use external tools
- **Reasoning models**: Configure thinking depth, store chain-of-thought, real-time reasoning logs
- **Async & parallel**: `arun()` support with automatic parallel tool and agent execution
- **Observability**: Callback system for monitoring and debugging
- **Advanced capabilities**: Dynamic workflows, file/directory operations, complex state management


## Installation

```bash
pip install agentify-core
```

For optional features:
```bash
pip install agentify-core[all]  # Installs all optional dependencies
```

## Quick Start

```python
from agentify import BaseAgent, AgentConfig, MemoryService, MemoryAddress, tool
from agentify.memory.stores import InMemoryStore

# 1. Create a simple tool with @tool decorator
@tool
def get_time() -> dict:
    """Returns the current time."""
    from datetime import datetime
    return {"time": datetime.now().strftime("%H:%M:%S")}

# 2. Create memory service
memory = MemoryService(store=InMemoryStore(), log_enabled=True, max_log_length=100)
addr = MemoryAddress(conversation_id="session_1")

# 3. Create an Agent with the tool
agent = BaseAgent(
    config=AgentConfig(
        name="ReasoningAgent",
        system_prompt="You are a helpful assistant.",
        provider="provider",
        model_name="model",
        reasoning_effort="high",  # optional param:"low", "medium", "high"
        model_kwargs={"max_completion_tokens": 5000} # Pass model-specific params
    ),
    memory=memory,
    memory_address=addr,
    tools=[get_time]  # Add your tools here
)

# 4. Run a conversation
response = agent.run(user_input="What time is it?")
```

## Composable Flows

Agentify provides powerful primitives that can be combined to build arbitrarily complex systems:

* **BaseAgent**: The fundamental unit of work.
* **Teams**: A group of agents managed by a supervisor.
* **Pipelines**: A sequence of steps where output passes from one to the next.
* **Hierarchies**: Tree structures for massive delegation.

Because all flows share the same `run()` interface, you can build Teams made of Pipelines, Pipelines made of Teams, and deeply nested Hierarchies.

Agentify supports both **strict workflows** (fixed, pre-defined Pipelines and Hierarchies) and **dynamic agentic flows**, where a supervisor/router agent decides at runtime which agent, Team or Pipeline to call next.


## Documentation

- [Getting Started](docs/getting_started.md) - Installation and first steps
- [Core Concepts](docs/core_concepts.md) - Agents, memory, and tools
- [Multi-Agent Systems](docs/multi_agent.md) - Teams, pipelines, and hierarchies
- [Advanced Features](docs/advanced.md) - Vision, streaming, hooks, and more
- [API Reference](docs/api_reference.md) - Complete API documentation


### More Examples

Check out the [examples](examples/) directory for detailed implementations:

*   [Single Agent Chatbot](examples/chatbot/)
*   [Multi-Agent Teams](examples/multi_agent/team/)
*   [Sequential Pipelines](examples/multi_agent/pipeline/)
*   [Hierarchical Structures](examples/multi_agent/hierarchical/)


## Author

- **Fabian Melchor** [fabianmp_98@hotmail.com](mailto:fabianmp_98@hotmail.com)


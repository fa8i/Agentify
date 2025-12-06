# Agentify

**Framework-agnostic AI agent library for building single and multi-agent systems**

Agentify is a Python library for building and orchestrating AI agents, from simple assistants to complex multi-agent systems. It focuses on a small set of composable primitives for LLM integration, memory, tools and coordination, so you can focus on product logic instead of framework details.

## Why Agentify?

- **Built for production**: clear abstractions, explicit configuration, error handling and extension points that map well to real deployments.
- **Orchestration-first design**: a uniform `run()` interface for agents, teams, pipelines and hierarchies makes it straightforward to compose and refactor flows.
- **Providers**: switch between OpenAI, Gemini, Azure OpenAI, DeepSeek, Claude and others without changing your agent code.


## Key Features

- **Agents and multi-agent patterns**  
  Single Agents with tools and memory, supervisor–worker Multi-Agent Teams, Sequential Pipelines where output flows from step to step, Hierarchical Structures for complex delegation, and Dynamic Flows where a controller decides at runtime which sub-agents or teams to invoke.

- **Memory service and isolation**  
  Pluggable backends (in-memory, Redis, …) with per-use-case policies (TTL, maximum messages, etc.), plus optional memory isolation so each agent can maintain its own conversation history for scalability and privacy.

- **Tools and actions**  
  Type-annotated tool interface, straightforward registration of custom tools.

- **Observability hooks**  
  Callback system for logging, monitoring and debugging agent behaviour across complex flows.

- **I/O capabilities**  
  Streaming support for real-time responses and vision/image models for multimodal interactions.


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
from agentify import BaseAgent, AgentConfig, MemoryService, MemoryAddress
from agentify.memory.stores import InMemoryStore

# 1. Create memory service
memory = MemoryService(store=InMemoryStore(), log_enabled=True, max_log_length=100)
addr = MemoryAddress(conversation_id="session_1")

# 2. Create an Agent
agent = BaseAgent(
    config=AgentConfig(
        name="Assistant",
        system_prompt="You are a helpful AI assistant.",
        provider="openai",
        model_name="gpt-4.1-mini"
    ),
    memory=memory,
    memory_address=addr
)

# 3. Run a conversation
response = agent.respond(user_input="Hello! How can you help me?")
```

## Composable Flows

Agentify provides powerful primitives that can be combined to build arbitrarily complex systems:

* **BaseAgent**: The fundamental unit of work.
* **Teams**: A group of agents managed by a supervisor.
* **Pipelines**: A sequence of steps where output passes from one to the next.
* **Hierarchies**: Tree structures for massive delegation.

Because all flows share the same `run()` interface, you can build Teams made of Pipelines, Pipelines made of Teams, and deeply nested Hierarchies.

Agentify supports both **strict workflows** (fixed, pre-defined Pipelines and Hierarchies) and **dynamic agentic flows**, where a supervisor/router agent decides at runtime which agent, Team or Pipeline to call next.


### More Examples

Check out the [examples](examples/) directory for detailed implementations:

*   [Single Agent Chatbot](examples/chatbot/)
*   [Multi-Agent Teams](examples/multi_agent/team/)
*   [Sequential Pipelines](examples/multi_agent/pipeline/)
*   [Hierarchical Structures](examples/multi_agent/hierarchical/)


## Author

- **Fabian Melchor** [fabianmp_98@hotmail.com](mailto:fabianmp_98@hotmail.com)


## Links

- **Repository**: https://github.com/fa8i/Agentify
- **Issues**: https://github.com/fa8i/Agentify/issues

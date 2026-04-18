# Agentify Documentation

Complete documentation for the Agentify AI agent library.

## Quick Links

- [Getting Started](getting_started.md) - Install and create your first agent
- [Core Concepts](core_concepts.md) - Agents, memory, tools, and providers
- [Multi-Agent Systems](multi_agent.md) - Teams, pipelines, and hierarchies
- [Advanced Features](advanced.md) - Vision, streaming, hooks, and more
- [📖 API Reference](api_reference.md) - Complete API documentation
- [Personal Assistant MVP](personal_assistant_mvp/README.md) - Linux app execution plan

## Documentation Structure

### First Steps


1. [Getting Started](getting_started.md)
   - Installation
   - Your first agent
   - Adding tools
   - Streaming responses

2. [Core Concepts](core_concepts.md)
   - Architecture overview
   - Agents and configuration
   - Memory system
   - Tools
   - Callbacks
   - Providers

### For Building Systems

Learn to build complex multi-agent applications:

3. [Multi-Agent Systems](multi_agent.md)
   - Teams (Supervisor-Workers)
   - Sequential Pipelines
   - Hierarchical Teams
   - Composability
   - Dynamic sub-agent spawning
   - Best practices

### For Advanced Use

Deep dive into advanced features:

4. [Advanced Features](advanced.md)
   - Vision & multimodal inputs
   - Streaming
   - Hooks (pre/post execution)
   - Reasoning models
   - Error handling & retries
   - Tool iteration control
   - Model-specific parameters
   - Custom memory policies
   - Production best practices

### For Reference

Complete API documentation:

5. [API Reference](api_reference.md)
   - Core classes
   - Memory system
   - Multi-agent patterns
   - Tools
   - Callbacks
   - LLM clients

## Examples

Working code examples are available in the [`examples/`](../examples/) directory:

- **Single Agent**: [`examples/chatbot/`](../examples/chatbot/)
- **Multi-Agent Teams**: [`examples/multi_agent/team/`](../examples/multi_agent/team/)
- **Sequential Pipelines**: [`examples/multi_agent/pipeline/`](../examples/multi_agent/pipeline/)
- **Hierarchical Systems**: [`examples/multi_agent/hierarchical/`](../examples/multi_agent/hierarchical/)
- **Deep Agent Demo**: [`examples/deep_agent_demo.py`](../examples/deep_agent_demo.py)

## Quick Reference

### Create an Agent

```python
from agentify import BaseAgent, AgentConfig, MemoryService
from agentify.memory.stores import InMemoryStore
from agentify.memory import MemoryAddress

memory = MemoryService(store=InMemoryStore())
addr = MemoryAddress(conversation_id="chat_1")

agent = BaseAgent(
    config=AgentConfig(
        name="Assistant",
        system_prompt="You are helpful.",
        provider="provider",
        model_name="model_name"
    ),
    memory=memory,
    memory_address=addr
)

response = agent.run("Hello!")
```

### Create a Team

```python
from agentify.multi_agent import Team

team = Team(
    agents=[manager_agent, worker1, worker2],
    supervisor=manager_agent,
)

result = team.run("Complete this task")
```

### Add Tools

```python
from agentify.extensions.tools import TimeTool, CalculatorTool

agent = BaseAgent(
    config=config,
    memory=memory,
    tools=[TimeTool(), CalculatorTool()]
)
```

## Contributing

Found an issue or want to contribute? Visit:
- **Repository**: https://github.com/fa8i/Agentify
- **Issues**: https://github.com/fa8i/Agentify/issues

## License

MIT License - see LICENSE file for details.

## Author

**Fabian Melchor** - [fabianmp_98@hotmail.com](mailto:fabianmp_98@hotmail.com)

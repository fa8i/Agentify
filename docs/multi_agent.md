# Multi-Agent Systems

Agentify provides powerful patterns for coordinating multiple agents.

## Overview

All multi-agent patterns share the same dual interface: sync `run()` and async `arun()`.

```python
# Sync usage
agent_response = agent.run(user_input)
team_response = team.run(user_input)
pipeline_response = pipeline.run(user_input)
hierarchy_response = hierarchy.run(user_input)

# Async usage
agent_response = await agent.arun(user_input)
team_response = await team.arun(user_input)
pipeline_response = await pipeline.arun(user_input)
hierarchy_response = await hierarchy.arun(user_input)
```

## Team (Supervisor-Workers)

A supervisor agent manages and delegates to worker agents.

### Architecture

```
      ┌────────────┐
      │ Supervisor │
      └─────┬──────┘
            │
     ┌──────┴──────┐
     │             │
┌────▼────┐   ┌────▼────┐
│ Worker1 │   │ Worker2 │
└─────────┘   └─────────┘
```

### Example

```python
from agentify import BaseAgent, AgentConfig, Team
from agentify.memory import MemoryService
from agentify.memory.stores import InMemoryStore
from agentify.extensions.tools import CalculatorTool, TimeTool

memory = MemoryService(store=InMemoryStore())

# Create workers
researcher = BaseAgent(
    config=AgentConfig(
        name="Researcher",
        system_prompt="You research and gather information.",
        provider="provider",
        model_name="model_name"
    ),
    memory=memory,
    tools=[TimeTool()]
)

analyst = BaseAgent(
    config=AgentConfig(
        name="Analyst",
        system_prompt="You analyze data and calculate.",
        provider="provider",
        model_name="model_name"
    ),
    memory=memory,
    tools=[CalculatorTool()]
)

# Create supervisor
supervisor = BaseAgent(
    config=AgentConfig(
        name="Manager",
        system_prompt="You delegate tasks to specialists.",
        provider="provider",
        model_name="model_name"
    ),
    memory=memory
)

# Create team
team = Team(
    agents=[supervisor, researcher, analyst],
    supervisor=supervisor,
)

# Run the team
response = team.run("Research today's date and calculate 42 * 17")
print(response)
```

### How It Works

1. User input goes to supervisor
2. Supervisor receives workers as tools (e.g., `call_researcher`, `call_analyst`)
3. Supervisor decides which worker(s) to call
4. Workers execute and return results
5. Supervisor synthesizes final response

## Sequential Pipeline

Execute agents in sequence, passing output from one to the next.

### Architecture

```
Input → Agent1 → Agent2 → Agent3 → Output
```

### Example

```python
from agentify.multi_agent import SequentialPipeline

# Create pipeline steps
step1 = BaseAgent(
    config=AgentConfig(
        name="Researcher",
        system_prompt="Research the topic.",
        provider="provider",
        model_name="model_name"
    ),
    memory=memory
)

step2 = BaseAgent(
    config=AgentConfig(
        name="Writer",
        system_prompt="Write based on research.",
        provider="provider",
        model_name="model_name"
    ),
    memory=memory
)

step3 = BaseAgent(
    config=AgentConfig(
        name="Editor",
        system_prompt="Edit and polish the text.",
        provider="provider",
        model_name="model_name"
    ),
    memory=memory
)

# Create pipeline
pipeline = SequentialPipeline(
    steps=[step1, step2, step3]
)

# Run pipeline
result = pipeline.run("Write an article about AI")
print(result)
```

### How It Works

1. Step 1 receives user input
2. Step 1's output becomes Step 2's input
3. Step 2's output becomes Step 3's input
4. Final step's output is the pipeline result

## Hierarchical Team

Multi-level delegation with parent-child relationships.

### Architecture

```
         ┌────────┐
         │  Root  │
         └───┬────┘
             │
      ┌──────┴──────┐
      │             │
  ┌───▼────┐    ┌───▼────┐
  │ Child1 │    │ Child2 │
  └───┬────┘    └────────┘
      │
  ┌───▼───┐
  │ Grand │
  └───────┘
```

### Example

```python
from agentify.multi_agent import HierarchicalTeam

# Create hierarchy
ceo = BaseAgent(config=AgentConfig(name="CEO", ...))
cto = BaseAgent(config=AgentConfig(name="CTO", ...))
cfo = BaseAgent(config=AgentConfig(name="CFO", ...))
engineer = BaseAgent(config=AgentConfig(name="Engineer", ...))

hierarchy = HierarchicalTeam(
    root=ceo,
    hierarchy={
        ceo: [cto, cfo],     # CEO manages CTO and CFO
        cto: [engineer]      # CTO manages Engineer
    }
)

# Run hierarchy
result = hierarchy.run("Build a new feature")
```

### How It Works

1. Root agent receives user input
2. Each parent can call their direct children as tools
3. Children can delegate to their own children
4. Results bubble up the hierarchy

## Composability

Mix and match patterns:

### Team of Pipelines

```python
# Create pipelines
research_pipeline = SequentialPipeline(steps=[...])
writing_pipeline = SequentialPipeline(steps=[...])

# Use as team workers
team = Team(
    agents=[manager, research_pipeline, writing_pipeline],
    supervisor=manager,
)
```

### Pipeline of Teams

```python
# Create teams
data_team = Team(agents=[data_lead, ...], supervisor=data_lead)
analysis_team = Team(agents=[analysis_lead, ...], supervisor=analysis_lead)

# Chain teams in pipeline
pipeline = SequentialPipeline(
    steps=[data_team, analysis_team]
)
```

## Dynamic Sub-Agent Spawning

Create agents on-the-fly during execution:

```python
from agentify.multi_agent.tool_wrapper import SpawnAgentTool

spawn_tool = SpawnAgentTool(
    base_config=config,
    memory_service=memory,
    parent_addr=addr
)

agent = BaseAgent(
    config=config,
    memory=memory,
    tools=[spawn_tool]
)

# Agent can now spawn sub-agents dynamically
response = agent.run(
    "Spawn a 'Researcher' agent to find information about Python"
)
```

## Best Practices

### 1. **Use Teams for Specialization**
```python
# Good: specialized workers
team = Team(
    agents=[manager, data_expert, code_expert, doc_expert],
    supervisor=manager,
)
```

### 2. **Use Pipelines for Sequential Workflows**
```python
# Good: clear flow
pipeline = SequentialPipeline(
    steps=[gather, process, analyze, report]
)
```

### 3. **Use Hierarchies for Large Organizations**
```python
# Good: clear reporting structure
hierarchy = HierarchicalTeam(
    root=ceo,
    hierarchy={
        ceo: [vp_eng, vp_sales],
        vp_eng: [tech_lead1, tech_lead2],
        tech_lead1: [dev1, dev2]
    }
)
```

### 4. **Memory Isolation**
Each agent should have its own memory address:
```python
worker1 = BaseAgent(
    config=config,
    memory=memory,
    memory_address=MemoryAddress(agent_id="worker1")
)

worker2 = BaseAgent(
    config=config,
    memory=memory,
    memory_address=MemoryAddress(agent_id="worker2")
)
```

## Examples

See complete working examples in:
- [`examples/multi_agent/team/`](../examples/multi_agent/team/)
- [`examples/multi_agent/pipeline/`](../examples/multi_agent/pipeline/)
- [`examples/multi_agent/hierarchical/`](../examples/multi_agent/hierarchical/)

## Next Steps

- [Advanced Features](advanced.md)
- [API Reference](api_reference.md)

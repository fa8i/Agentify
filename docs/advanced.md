# Advanced Features

## Vision & Multimodal

Agentify supports image inputs with vision-capable models.

### Basic Image Input

```python
agent = BaseAgent(
    config=AgentConfig(
        name="VisionAgent",
        provider="provider",
        model_name="model_name",  # Vision-capable model
    ),
    memory=memory
)

response = agent.run(
    user_input="What's in this image?",
    image_path="/path/to/image.jpg"
)
```

### Image Configuration

Control image processing:

```python
from agentify.core.config import ImageConfig

agent = BaseAgent(
    config=config,
    memory=memory,
    image_config=ImageConfig(
        max_side_px=1024,    # Max dimension
        quality=85,          # JPEG quality (1-100)
        detail="high"        # "low", "high", or "auto"
    )
)
```

## Streaming

### Basic Streaming

```python
agent = BaseAgent(
    config=AgentConfig(
        stream=True  # Enable streaming
    ),
    memory=memory
)

response = agent.run("Tell me a long story")

for chunk in response:
    print(chunk, end="", flush=True)
```

### Tool Calls with Streaming

When streaming is enabled, text content streams in real-time, but tool calls are assembled and executed normally:

```python
from agentify.extensions.tools import CalculatorTool

agent = BaseAgent(
    config=AgentConfig(stream=True),
    memory=memory,
    tools=[CalculatorTool()]
)

# Both text and tool calls work seamlessly
for chunk in agent.run("Calculate 15 * 23 and explain the result"):
    print(chunk, end="", flush=True)
```

## Hooks (Pre/Post Execution)

Execute custom logic before and after agent runs:

```python
def pre_hook(agent, user_input):
    print(f"Starting: {user_input}")
    # Your logic here

def post_hook(agent, user_input, response):
    print(f"Finished. Response length: {len(response)}")
    # Your logic here

agent = BaseAgent(
    config=config,
    memory=memory,
    pre_hooks=[pre_hook],
    post_hooks=[post_hook]
)
```

### Signature Injection

Hooks automatically inject available arguments:

```python
# Minimal hook
def simple_hook():
    print("Hook called!")

# Hook with specific args
def detailed_hook(agent, user_input):
    print(f"Agent {agent.config.name}: {user_input}")

# Hook with all args (post-hook only)
def full_hook(agent, user_input, response):
    print(f"Full context available")

# Hook with **kwargs to accept any args
def flexible_hook(**kwargs):
    print(f"Available: {kwargs.keys()}")
```

## Reasoning Models

Models with built-in reasoning (like o1, DeepSeek-R1) are fully supported:

```python
agent = BaseAgent(
    config=AgentConfig(
        name="ReasoningAgent",
       provider="provider",
        model_name="model_name",
        reasoning_effort="high",  # "low", "medium", "high"
    ),
    memory=memory
)

response = agent.run("Solve this complex problem...")
```

### Reasoning Callbacks

Monitor reasoning steps in real-time:

```python
from agentify.core.callbacks import AgentCallbackHandler

class ReasoningCallback(AgentCallbackHandler):
    def on_reasoning_step(self, content: str):
        print(f"[Thinking] {content}")

agent = BaseAgent(
    config=AgentConfig(
        name="Reasoner",
        provider="provider"
        model_name="model_name",
        reasoning_effort="high", 
        callbacks=[ReasoningCallback()]
    ),
    memory=memory
)
```

### Reasoning in Memory

Reasoning content is automatically stored in message metadata:

```python
history = agent.get_history(addr)
for msg in history:
    if msg.get("metadata", {}).get("reasoning_content"):
        print(f"Reasoning: {msg['metadata']['reasoning_content']}")
```

## Error Handling & Retries

Built-in retry logic with exponential backoff:

```python
agent = BaseAgent(
    config=AgentConfig(
        max_retries=5,  # Retry up to 5 times
        timeout=120     # 2 minute timeout
    ),
    memory=memory
)
```

### Custom Error Handling

```python
from agentify.core.callbacks import AgentCallbackHandler

class ErrorCallback(AgentCallbackHandler):
    def on_error(self, error: Exception, context: str):
        print(f"Error in {context}: {error}")
        # Log to monitoring system, send alerts, etc.

agent = BaseAgent(
    config=AgentConfig(callbacks=[ErrorCallback()]),
    memory=memory
)
```

## Tool Iteration Control

Control how many times an agent can call tools:

```python
# Limited iterations
agent = BaseAgent(
    config=AgentConfig(
        max_tool_iter=5  # Max 5 tool calls
    ),
    memory=memory,
    tools=tools
)

# Unlimited iterations (use with caution!)
agent = BaseAgent(
    config=AgentConfig(
        max_tool_iter=None  # No limit
    ),
    memory=memory,
    tools=tools
)
```

## Model-Specific Parameters

Pass parameters directly to the model:

```python
agent = BaseAgent(
    config=AgentConfig(
      provider="provider",
        model_name="model_name",
        model_kwargs={
            "max_completion_tokens": 5000,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
        }
    ),
    memory=memory
)
```

## Custom Memory Policies

Fine-tune memory management:

```python
from agentify.memory import MemoryPolicy

# Token-based policy
def count_tokens(messages):
    # Your tokenizer logic
    return sum(len(m.get("content", "")) for m in messages)

policy = MemoryPolicy(
    store=store,
    tokenizer=count_tokens,
    max_tokens=4000,  # Keep under 4K tokens
    max_user_msgs=20,
    max_assistant_msgs=20,
)

memory = MemoryService(store, policy)
# Logs are truncated to max_log_length and redact common secrets.
```

## Client Configuration Override

Customize LLM client settings:

```python
agent = BaseAgent(
    config=AgentConfig(
        provider="azure",
        model_name="model_name",
        client_config_override={
            "api_version": "2024-02-15-preview",
            "azure_endpoint": "https://your-resource.openai.azure.com",
        }
    ),
    memory=memory
)
```

## Production Best Practices

### 1. Use Redis for Memory
```python
from agentify.memory.stores import RedisStore

store = RedisStore(url="redis://localhost:6379/0")
```

### 2. Enable Monitoring
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### 3. Set Reasonable Limits
```python
config = AgentConfig(
    max_tool_iter=10,
    max_retries=3,
    timeout=60
)
```

### 4. Use Memory Policies
```python
policy = MemoryPolicy(
    ttl_seconds=3600,  # 1 hour TTL
    max_user_msgs=50,
    max_assistant_msgs=50
)
```

### 5. Implement Error Callbacks
```python
class ProductionCallback(AgentCallbackHandler):
    def on_error(self, error, context):
        # Send to monitoring service
        sentry.capture_exception(error)
```

## Next Steps

- [API Reference](api_reference.md) - Complete API documentation
- [Examples](../examples/) - Working code examples

# Tool Decorator Guide

## What does the @tool decorator do?

Converts any Python function into a Tool that an agent can use, automatically generating the JSON Schema from type hints.

## Basic Usage

### 1. Without Parameters

```python
from agentify import tool

@tool
def get_time() -> dict:
    """Returns the current time."""
    import datetime
    return {"time": datetime.datetime.now().isoformat()}
```

**What it generates:**
- Name: `get_time`
- Description: "Returns the current time."
- Parameters: none

### 2. With Parameters (Type Hints Required)

```python
@tool
def calculate(expression: str) -> dict:
    """Calculates a mathematical expression.
    
    Args:
        expression: The expression to calculate (e.g., '2 + 2').
    """
    return {"result": eval(expression)}
```

**What it generates:**
- Name: `calculate`
- Description: "Calculates a mathematical expression."
- Parameter `expression`:
  - Type: `"string"`
  - Description: "The expression to calculate (e.g., '2 + 2')."
  - Required: `true` (no default)

### 3. With Optional Parameters

```python
@tool
def greet(name: str, greeting: str = "Hello") -> dict:
    """Greets a person.
    
    Args:
        name: The person's name.
        greeting: The greeting to use (default: "Hello").
    """
    return {"message": f"{greeting}, {name}!"}
```

**What it generates:**
- `name` is **required**
- `greeting` is **optional** (has default = "Hello")

### 4. Custom Name and Description

```python
@tool(name="weather_api", description="Fetches weather data from API")
def get_weather(city: str) -> dict:
    """Weather lookup."""
    return {"temperature": 20, "conditions": "sunny"}
```

## Docstring Format (Google Style Required)

The `@tool` decorator expects **Google Style** docstrings:

```python
@tool
def my_tool(param1: str, param2: int = 10) -> dict:
    """Brief one-line summary of what the tool does.
    
    Optional extended description can go here.
    This part is not used by the schema generator.
    
    Args:
        param1: Description of param1. This becomes the parameter
            description in the JSON Schema.
        param2: Description of param2. Can span multiple lines
            if properly indented.
    
    Returns:
        (Optional) Description of return value. NOT used by schema generator.
    
    Raises:
        ValueError: When something goes wrong. Also ignored by schema.
    """
    return {"result": f"{param1}-{param2}"}
```

### What the Decorator Extracts

| Docstring Section | Used for Schema? | Purpose |
|-------------------|------------------|---------|
| First line | Yes | Tool description |
| Extended description | No | Human documentation only |
| `Args:` section | Yes | Parameter descriptions |
| `Returns:` section | No | Human documentation only |
| `Raises:` section | No | Human documentation only |

**Important:** The `Returns:` section does NOT affect the generated schema. The decorator only uses:
- Parameter type hints (e.g., `param: str`)
- First line of docstring
- `Args:` section for parameter descriptions

## Supported Types

| Python Type | JSON Schema |
|-------------|-------------|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list[T]` | `"array"` with `items` inferred from `T` |
| `dict` | `"object"` |
| `Optional[T]` | Inner type of `T` |
| `Literal[...]` | `enum` of the allowed values |
| `Enum` subclass | `enum` of the member values |

Typed lists and enums produce richer schemas, which improves tool-calling
accuracy:

```python
from typing import Literal

@tool
def set_status(tags: list[str], level: Literal["low", "medium", "high"]) -> dict:
    """Updates the status.

    Args:
        tags: Labels to attach.
        level: Priority level.
    """
    return {"tags": tags, "level": level}
```

`tags` becomes `{"type": "array", "items": {"type": "string"}}` and `level`
becomes `{"enum": ["low", "medium", "high"], "type": "string"}`.

## Using with Agents

```python
from agentify import Agent, tool

@tool
def search(query: str) -> dict:
    """Searches for information."""
    return {"results": [f"Result for: {query}"]}

agent = Agent(
    "You are a helpful assistant.",
    model="gpt-5.5",
    tools=[search],  # ← Use directly, no instantiation needed
)
```

## Comparison with Subclassing

### Subclassing:
```python
class MyTool(Tool):
    def __init__(self):
        schema = {
            "name": "my_tool",
            "description": "Does something",
            "parameters": {
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "..."}
                },
                "required": ["param"]
            }
        }
        super().__init__(schema, self._execute)
    
    def _execute(self, param: str):
        return {"result": param}

# Usage:
tools=[MyTool()]  # ← Needs to be instantiated
```

### Decorator:
```python
@tool
def my_tool(param: str) -> dict:
    """Does something.
    
    Args:
        param: Parameter description.
    """
    return {"result": param}

# Usage:
tools=[my_tool]  # ← Use directly
```

## When to Use Each Approach

**Use `@tool` when:**
- The tool is simple (stateless function)
- You want clean, concise code
- Parameters are basic types

**Use `Tool` (subclassing) when:**
- The tool needs internal state (attributes)
- Requires complex initialization logic
- You need custom hooks (on_init, etc.)



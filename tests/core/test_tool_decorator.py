"""Unit tests for the @tool decorator."""

import pytest
from agentify.core.tool import tool, Tool


def test_tool_decorator_basic():
    """Test basic tool creation with decorator."""
    @tool
    def simple_tool() -> str:
        """A simple test tool."""
        return "Hello"
    
    assert isinstance(simple_tool, Tool)
    assert simple_tool.name == "simple_tool"
    assert simple_tool.schema["description"] == "A simple test tool."
    assert simple_tool.schema["parameters"]["required"] == []


def test_tool_decorator_with_params():
    """Test tool creation with parameters."""
    @tool
    def add_numbers(a: int, b: int) -> int:
        """Adds two numbers.
        
        Args:
            a: First number.
            b: Second number.
        """
        return a + b
    
    assert add_numbers.name == "add_numbers"
    schema = add_numbers.schema
    
    # Check parameters schema
    assert "a" in schema["parameters"]["properties"]
    assert "b" in schema["parameters"]["properties"]
    assert schema["parameters"]["properties"]["a"]["type"] == "integer"
    assert schema["parameters"]["properties"]["b"]["type"] == "integer"
    assert schema["parameters"]["required"] == ["a", "b"]
    
    # Check parameter descriptions
    assert "First number" in schema["parameters"]["properties"]["a"]["description"]
    assert "Second number" in schema["parameters"]["properties"]["b"]["description"]


def test_tool_decorator_with_defaults():
    """Test tool with optional parameters (defaults)."""
    @tool
    def greet(name: str, greeting: str = "Hello") -> str:
        """Greets a person."""
        return f"{greeting}, {name}"
    
    schema = greet.schema
    
    # name is required, greeting is optional
    assert schema["parameters"]["required"] == ["name"]
    assert "name" in schema["parameters"]["properties"]
    assert "greeting" in schema["parameters"]["properties"]


def test_tool_decorator_custom_name_description():
    """Test tool with custom name and description."""
    @tool(name="custom_name", description="Custom description")
    def my_func():
        """Original docstring."""
        return "test"
    
    assert my_func.name == "custom_name"
    assert my_func.schema["description"] == "Custom description"


def test_tool_decorator_type_mapping():
    """Test that Python types map correctly to JSON Schema types."""
    from typing import List, Dict
    
    @tool
    def type_test(
        s: str,
        i: int,
        f: float,
        b: bool,
        l: List,
        d: Dict
    ):
        """Test various types."""
        pass
    
    props = type_test.schema["parameters"]["properties"]
    
    assert props["s"]["type"] == "string"
    assert props["i"]["type"] == "integer"
    assert props["f"]["type"] == "number"
    assert props["b"]["type"] == "boolean"
    assert props["l"]["type"] == "array"
    assert props["d"]["type"] == "object"


def test_tool_execution():
    """Test that decorated tools execute correctly."""
    @tool
    def add(a: int, b: int) -> dict:
        """Adds two numbers."""
        return {"sum": a + b}
    
    result = add(a=5, b=3)
    # Tool.__call__ returns JSON string
    assert '"sum": 8' in result or '"sum":8' in result


def test_tool_execution_with_error():
    """Test that tool errors are captured properly."""
    @tool
    def failing_tool() -> str:
        """Always fails."""
        raise ValueError("Intentional error")
    
    result = failing_tool()
    # Tool.__call__ returns error as JSON string
    assert "error" in result
    assert "Intentional error" in result


def test_tool_decorator_without_type_hints():
    """Test tool creation without type hints (should default to string)."""
    @tool
    def no_hints(x):
        """No type hints."""
        return x
    
    schema = no_hints.schema
    # Should default to string type
    assert schema["parameters"]["properties"]["x"]["type"] == "string"


def test_optional_parameter():
    """Test Optional[T] type hint handling."""
    from typing import Optional
    
    @tool
    def optional_param(name: str, age: Optional[int] = None) -> dict:
        """Test optional parameter."""
        return {"name": name, "age": age}
    
    schema = optional_param.schema
    
    # name is required, age is optional (has default)
    assert schema["parameters"]["required"] == ["name"]
    assert schema["parameters"]["properties"]["age"]["type"] == "integer"

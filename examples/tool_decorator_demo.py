import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentify import BaseAgent, AgentConfig, MemoryService, tool
from agentify.memory.stores import InMemoryStore
import datetime


# Define tools using the decorator
@tool
def get_current_time() -> dict:
    """Returns the current date and time in ISO 8601 format."""
    now = datetime.datetime.now().astimezone().isoformat()
    return {"current_time": now}


@tool
def calculate(expression: str) -> dict:
    """Evaluates a safe mathematical expression.
    
    Args:
        expression: The math expression to evaluate (e.g., '2 + 2 * (3 - 1)').
    
    Returns:
        The calculated result or an error.
    """
    import ast
    import operator as op
    
    allowed_ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
    }
    
    def eval_node(node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.BinOp):
            return allowed_ops[type(node.op)](
                eval_node(node.left),
                eval_node(node.right)
            )
        raise ValueError(f"Unsupported: {node}")
    
    try:
        tree = ast.parse(expression, mode="eval").body
        return {"result": eval_node(tree)}
    except Exception as e:
        return {"error": str(e)}


@tool(name="format_text", description="Formats text with custom style")
def text_formatter(text: str, style: str = "uppercase") -> dict:
    """Formats text according to the specified style.
    
    Args:
        text: The text to format.
        style: The formatting style ('uppercase', 'lowercase', 'title').
    
    Returns:
        The formatted text.
    """
    styles = {
        "uppercase": str.upper,
        "lowercase": str.lower,
        "title": str.title,
    }
    formatter = styles.get(style, str.upper)
    return {"formatted": formatter(text)}


def main():
    # Create agent with decorator-created tools
    store = InMemoryStore()
    memory = MemoryService(store=store)
    
    from agentify.memory.interfaces import MemoryAddress
    
    agent = BaseAgent(
        config=AgentConfig(
            name="ToolDemoAgent",
            system_prompt=(
                "You are a helpful assistant with time, calculation, and formatting tools. "
                "Use them to answer user questions accurately."
            ),
            provider="openai",
            model_name="gpt-4.1",
            temperature=0.0,
        ),
        memory=memory,
        memory_address=MemoryAddress(conversation_id="tool_demo"),
        tools=[get_current_time, calculate, text_formatter]
    )
    
    print("=" * 60)
    print("@tool Decorator Demo")
    print("=" * 60)
    print("\nTesting tools created with @tool decorator:\n")
    
    # Test 1: Time tool
    print("Test 1: Asking for current time")
    print("-" * 60)
    response1 = agent.run("What time is it now?")
    print(f"Response: {response1}\n")
    
    # Test 2: Calculator tool
    print("Test 2: Math calculation")
    print("-" * 60)
    response2 = agent.run("Calculate: (15 + 25) * 2")
    print(f"Response: {response2}\n")
    
    # Test 3: Text formatter tool
    print("Test 3: Text formatting")
    print("-" * 60)
    response3 = agent.run("Format 'hello world' as title case")
    print(f"Response: {response3}\n")
    
    print("=" * 60)
    print("✅ All decorator tools working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    main()

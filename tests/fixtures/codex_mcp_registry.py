from agentify.core.tool import Tool


def echo_tool(text: str) -> str:
    return f"ECHO_FROM_AGENTIFY: {text}"


def build_agentify_tools():
    return [
        Tool(
            schema={
                "name": "echo_tool",
                "description": "Echo text with a stable Agentify E2E prefix.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
            func=echo_tool,
        )
    ]

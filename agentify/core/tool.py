import enum
import json
import inspect
from typing import Dict, Any, Callable, List, Optional, get_type_hints, get_origin, get_args, Union

from typing import Literal
from dataclasses import dataclass


@dataclass(slots=True)
class Tool:
    """Wrapper for binding JSON-schema with its Python function."""

    schema: Dict[str, Any]
    func: Callable[..., Any]

    def __post_init__(self) -> None:
        if "name" not in self.schema:
            raise ValueError("Tool schema must include 'name'")

    @property
    def name(self) -> str:
        return self.schema["name"]

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Executes the function and returns JSON or string; captures generic errors."""
        try:
            result = self.func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)


def tool(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Union[Tool, Callable[[Callable[..., Any]], Tool]]:
    """Decorator to convert a function into a Tool with automatic schema generation.

    Automatically generates JSON Schema from function signature (type hints) and docstring.
    Supports basic Python types: str, int, float, bool, list, dict.

    Can be used with or without parentheses:
        @tool
        def my_function(): ...

        @tool(name="custom")
        def my_function(): ...

    Args:
        _func: The function being decorated (used when no parentheses).
        name: Optional custom name for the tool. Defaults to function name.
        description: Optional custom description. Defaults to function docstring.

    Returns:
        A Tool instance wrapping the decorated function.
    """

    def create_tool(func: Callable[..., Any]) -> Tool:
        # Extract function metadata
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip().split("\n")[0]

        # Get type hints
        sig = inspect.signature(func)
        type_hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

        # Build parameters schema
        properties: Dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            # Skip self, cls, *args, **kwargs
            if param_name in ("self", "cls") or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            param_type = type_hints.get(param_name, Any)
            param_schema = _type_to_json_schema(param_type)

            # Extract parameter description from docstring if available
            param_desc = _extract_param_description(func, param_name)
            if param_desc:
                param_schema["description"] = param_desc

            properties[param_name] = param_schema

            # Mark as required if no default value
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        # Build final schema
        schema = {
            "name": tool_name,
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

        return Tool(schema=schema, func=func)

    # Support both @tool and @tool() syntax
    if _func is None:
        # Called with parentheses: @tool() or @tool(name="x")
        return create_tool
    else:
        # Called without parentheses: @tool
        return create_tool(_func)


def _enum_schema(values: List[Any]) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"enum": values}
    json_type = _json_type_for_values(values)
    if json_type:
        schema["type"] = json_type
    return schema


def _type_to_json_schema(python_type: Any) -> Dict[str, Any]:
    """Convert a Python type hint to a JSON Schema fragment.

    Supports ``Optional[X]`` (nullable), ``Literal[...]`` / ``Enum`` (enum) and
    typed lists such as ``list[str]`` (with ``items``).
    """
    origin = get_origin(python_type)

    if origin is Literal:
        return _enum_schema(list(get_args(python_type)))

    if isinstance(python_type, type) and issubclass(python_type, enum.Enum):
        return _enum_schema([member.value for member in python_type])

    if origin is Union:
        non_none = [arg for arg in get_args(python_type) if arg is not type(None)]
        if len(non_none) == 1:
            return _type_to_json_schema(non_none[0])

    # bool must be checked before int, since bool is a subclass of int.
    if python_type is str or python_type == "str":
        return {"type": "string"}
    elif python_type is bool or python_type == "bool":
        return {"type": "boolean"}
    elif python_type is int or python_type == "int":
        return {"type": "integer"}
    elif python_type is float or python_type == "float":
        return {"type": "number"}
    elif origin is list or python_type is list:
        schema = {"type": "array"}
        args = get_args(python_type)
        if args and args[0] is not Any:
            schema["items"] = _type_to_json_schema(args[0])
        return schema
    elif origin is dict or python_type is dict:
        return {"type": "object"}
    else:
        return {"type": "string"}


def _json_type_for_values(values: List[Any]) -> Optional[str]:
    py_types = {type(v) for v in values}
    if not py_types:
        return None
    if py_types == {bool}:
        return "boolean"
    if py_types <= {int}:
        return "integer"
    if py_types <= {int, float}:
        return "number"
    if py_types <= {str}:
        return "string"
    return None


def _extract_param_description(func: Callable[..., Any], param_name: str) -> Optional[str]:
    """Extract parameter description from docstring (Google style)."""
    docstring = func.__doc__
    if not docstring:
        return None

    # Simple parser for "Args:" section in Google-style docstrings
    lines = docstring.split("\n")
    in_args_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args_section = True
            continue
        if in_args_section:
            if stripped and not line.startswith(" ") and stripped.endswith(":"):
                break
            if stripped.startswith(f"{param_name}:"):
                return stripped.split(":", 1)[1].strip()
            elif stripped.startswith(f"{param_name} (") or stripped.startswith(f"{param_name}("):
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
    return None

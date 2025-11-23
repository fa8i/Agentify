import json
from typing import Dict, Any, Callable
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

    def __call__(self, **kwargs: Any) -> str:
        """Executes the function and returns JSON or string; captures generic errors."""
        try:
            result = self.func(**kwargs)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

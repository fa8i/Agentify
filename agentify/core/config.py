from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ImageConfig:
    """Configuration for image processing."""
    max_side_px: int = 1024
    quality: int = 90
    detail: str = "auto"


@dataclass
class AgentConfig:
    """Configuration for the agent's behavior and model parameters."""
    name: str
    system_prompt: str
    provider: str
    model_name: str
    temperature: float = 1
    timeout: int = 60
    stream: bool = False
    max_retries: int = 3
    max_tool_iter: Optional[int] = 10
    reasoning_effort: Optional[str] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    client_config_override: Optional[Dict[str, Any]] = None
    callbacks: list = None

    def __post_init__(self):
        if self.callbacks is None:
            self.callbacks = []

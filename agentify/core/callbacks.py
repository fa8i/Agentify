from typing import Any, Dict, List, Optional, Protocol
import logging
from agentify.utils.style import Colors

logger = logging.getLogger(__name__)


class AgentCallbackHandler(Protocol):
    """Interface for agent callbacks to allow observability and side-effects."""

    def on_agent_start(self, agent_name: str, user_input: str) -> None:
        """Called when the agent starts processing a request."""
        ...

    def on_agent_finish(self, agent_name: str, response: str) -> None:
        """Called when the agent finishes processing a request."""
        ...

    def on_tool_start(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Called when a tool is about to be executed."""
        ...

    def on_tool_finish(self, tool_name: str, output: str) -> None:
        """Called when a tool finishes execution."""
        ...

    def on_llm_start(self, model_name: str, messages: List[Dict[str, Any]]) -> None:
        """Called when the LLM is about to be called."""
        ...

    def on_llm_new_token(self, token: str) -> None:
        """Called when a new token is generated (streaming only)."""
        ...

    def on_llm_end(self, response: Any) -> None:
        """Called when the LLM finishes generating a response."""
        ...

    def on_reasoning_step(self, content: str) -> None:
        """Called when the LLM generates a reasoning step."""
        ...

    def on_error(self, error: Exception, context: str) -> None:
        """Called when an error occurs."""
        ...


class LoggingCallbackHandler(AgentCallbackHandler):
    """Simple callback handler that logs events using the standard logging module."""

    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance or logger
        self.redact_keys = {"password", "api_key", "token", "secret", "key", "authorization"}

        # Auto-configure handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _mask_secrets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively mask sensitive keys in a dictionary."""
        masked = {}
        for k, v in data.items():
            if k.lower() in self.redact_keys:
                masked[k] = "******"
            elif isinstance(v, dict):
                masked[k] = self._mask_secrets(v)
            else:
                masked[k] = v
        return masked

    def on_agent_start(self, agent_name: str, user_input: str) -> None:
        self.logger.info(
            f"{Colors.BLUE}[VERBOSE] Agent '{agent_name}' started.{Colors.RESET} Input: {user_input[:100]}..."
        )

    def on_agent_finish(self, agent_name: str, response: str) -> None:
        self.logger.info(
            f"{Colors.BLUE}[VERBOSE] Agent '{agent_name}' finished.{Colors.RESET} Response: {response[:100]}..."
        )

    def on_tool_start(self, tool_name: str, args: Dict[str, Any]) -> None:
        # Redact sensitive values
        safe_args = self._mask_secrets(args)
        self.logger.info(
            f"{Colors.CYAN}[VERBOSE] Tool '{tool_name}' started.{Colors.RESET} Args: {safe_args}"
        )

    def on_tool_finish(self, tool_name: str, output: str) -> None:
        self.logger.info(
            f"{Colors.CYAN}[VERBOSE] Tool '{tool_name}' finished.{Colors.RESET} Output: {output[:100]}..."
        )

    def on_llm_start(self, model_name: str, messages: List[Dict[str, Any]]) -> None:
        self.logger.debug(f"LLM '{model_name}' started. Messages: {len(messages)}")

    def on_llm_new_token(self, token: str) -> None:
        pass  # Too verbose for standard logging

    def on_llm_end(self, response: Any) -> None:
        self.logger.debug("LLM finished.")

    def on_reasoning_step(self, content: str) -> None:
        self.logger.info(
            f"{Colors.GRAY}[Reasoning]{Colors.RESET} {Colors.GRAY}{content}{Colors.RESET}"
        )

    def on_error(self, error: Exception, context: str) -> None:
        self.logger.error(f"Error in {context}: {error}", exc_info=True)

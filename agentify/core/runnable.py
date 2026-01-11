from typing import Protocol, Any, Dict, Optional, Generator, AsyncGenerator, Union

class Runnable(Protocol):
    """Standard interface for any chainable unit/agent in Agentify."""

    def run(
        self,
        user_input: str,
        **kwargs: Any
    ) -> Union[str, Generator[str, None, None]]:
        """Synchronous execution."""
        ...

    async def arun(
        self,
        user_input: str,
        **kwargs: Any
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Asynchronous execution."""
        ...

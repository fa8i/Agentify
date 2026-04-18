from typing import List, Union, Any, AsyncGenerator, Optional, Dict, Generator
from agentify.core.runnable import Runnable
from agentify.core.sync_bridge import run_coro_blocking, stream_async_to_sync
from agentify.memory.interfaces import MemoryAddress

# Type alias for what can be a step in the pipeline
PipelineStep = Runnable


class SequentialPipeline(Runnable):
    """Executes a sequence of agents/teams/pipelines in order.

    The output of step N becomes the input of step N+1.
    """

    def __init__(self, steps: List[PipelineStep]):
        if not steps:
            raise ValueError("Pipeline must have at least one step.")
        self.steps = steps

    def _is_streaming_step(self, step: Any) -> bool:
        if hasattr(step, "config"):
            return bool(getattr(step.config, "stream", False))
        if hasattr(step, "supervisor") and hasattr(step.supervisor, "config"):
            return bool(getattr(step.supervisor.config, "stream", False))
        if hasattr(step, "root") and hasattr(step.root, "config"):
            return bool(getattr(step.root.config, "stream", False))
        if hasattr(step, "steps") and step.steps:
            return self._is_streaming_step(step.steps[-1])
        return False

    def run(
        self,
        user_input: str,
        session_id: str = "default_session",
        user_id: str = "default_user",
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[str, Generator[str, None, None]]:
        """Synchronous wrapper around arun()."""
        last_step = self.steps[-1] if self.steps else None
        is_streaming = self._is_streaming_step(last_step) if last_step is not None else False

        if is_streaming:
            return stream_async_to_sync(
                lambda: self.arun(
                    user_input=user_input,
                    session_id=session_id,
                    user_id=user_id,
                    context=context,
                    **kwargs,
                ),
                api_name="run",
                async_api_name="arun",
            )

        return run_coro_blocking(
            self.arun(
                user_input=user_input,
                session_id=session_id,
                user_id=user_id,
                context=context,
                **kwargs,
            ),
            api_name="run",
            async_api_name="arun",
        )

    async def arun(
        self,
        user_input: str,
        session_id: str = "default_session",
        user_id: str = "default_user",
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Run the pipeline sequentially."""

        current_input = user_input

        # Collect final result. Consume intermediate generators to pass string to next step.
        for i, step in enumerate(self.steps):
            is_last_step = i == len(self.steps) - 1

            # Determine step name for logging/memory
            step_name = getattr(step, "name", f"step_{i}")
            if hasattr(step, "config"):
                step_name = step.config.name

            response: Union[str, AsyncGenerator[str, None]]

            # Pass session info for unified Runnable execution.
            # Support legacy agents by constructing a MemoryAddress if applicable.
            run_kwargs = {
                "session_id": session_id, 
                "user_id": user_id,
                "context": context
            }
            # Add explicit memory address for BaseAgents (legacy support within Runnable)
            if hasattr(step, "config") and hasattr(step, "run"):
                 run_kwargs["memory_address"] = MemoryAddress(
                    user_id=user_id, conversation_id=session_id, agent_id=step_name
                )
            
            run_kwargs.update(kwargs) # Pass through any other kwargs

            response = await step.arun(user_input=current_input, **run_kwargs)

            # If not last step, consume output to pass to next step
            if not is_last_step:
                if hasattr(response, "__aiter__"):
                    parts = []
                    async for chunk in response:
                        parts.append(chunk)
                    current_input = "".join(parts)
                elif hasattr(response, "__iter__") and not isinstance(response, str):
                    current_input = "".join(list(response))
                else:
                    current_input = str(response)
            else:
                return response

        return current_input

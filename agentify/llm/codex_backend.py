import logging
from typing import Any, Dict, Optional, List, Union

try:
    from openai_codex import AsyncCodex
except ImportError:
    AsyncCodex = None

logger = logging.getLogger(__name__)

class DummyMessage:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = None
        self.role = "assistant"
        self.reasoning_content = None

class DummyChoice:
    def __init__(self, content: str):
        self.message = DummyMessage(content)
        self.delta = DummyMessage(content)

class DummyResponse:
    def __init__(self, content: str):
        self.choices = [DummyChoice(content)]

class CodexThreadBackend:
    """
    Native Codex Thread Backend mapping agentify sessions to codex threads.
    """
    is_native_thread_backend = True

    def __init__(self, config: Dict[str, Any], timeout: int):
        if AsyncCodex is None:
            raise ImportError("openai-codex is not installed. Please install it with `pip install agentify-core[codex]`.")
        self.config = config
        self.timeout = timeout
        self.threads = {}
        import shutil
        import os
        codex_path = shutil.which("codex")
        if not codex_path:
            # Fallback for common global install paths
            common_paths = [
                os.path.expanduser("~/.npm-global/bin/codex"),
                os.path.expanduser("~/.npm-global/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex/codex"),
                os.path.expanduser("~/.vscode/extensions/openai.chatgpt-26.5527.31454-linux-x64/bin/linux-x86_64/codex")
            ]
            for p in common_paths:
                if os.path.exists(p) and os.access(p, os.X_OK):
                    codex_path = p
                    break
                    
        if not codex_path:
            logger.warning("No global 'codex' executable found in PATH. AsyncCodex might fail to start if openai-codex-cli-bin is missing.")
            self.codex = AsyncCodex()
        else:
            try:
                from openai_codex import CodexConfig
                self.codex = AsyncCodex(config=CodexConfig(codex_bin=codex_path))
            except ImportError:
                self.codex = AsyncCodex()

    async def run_native(self, session_id: str, model: str, prompt: str, stream: bool = False, **kwargs) -> Any:
        if session_id not in self.threads:
            self.threads[session_id] = await self.codex.thread_start(model=model)
        
        thread = self.threads[session_id]
        
        # Tools and other kwargs can be passed here if the SDK supports them
        # For now, we assume simple text generation
        try:
            result = await thread.run(prompt)
        except AttributeError:
            logger.error("Codex Thread object does not have 'run' method. SDK might have changed.")
            raise

        response_content = getattr(result, "final_response", str(result))
        
        # If stream is requested, we yield a single chunk for simplicity unless the SDK exposes a stream generator
        if stream:
            async def stream_generator():
                yield DummyResponse(response_content)
            return stream_generator()
            
        return DummyResponse(response_content)

    # Fallback Adapter interface
    class Chat:
        class Completions:
            def __init__(self, backend: 'CodexThreadBackend'):
                self.backend = backend
            
            async def create(self, **kwargs) -> Any:
                messages = kwargs.get("messages", [])
                # If content is a list (multimodal), we extract text or serialize
                last_content = messages[-1]["content"] if messages else ""
                if isinstance(last_content, list):
                    prompt = " ".join([c.get("text", "") for c in last_content if c.get("type") == "text"])
                else:
                    prompt = last_content
                
                model = kwargs.get("model", "gpt-4")
                session_id = kwargs.get("session_id", "compat_session")
                stream = kwargs.get("stream", False)
                
                # Remove duplicate keys from kwargs before passing
                clean_kwargs = {k: v for k, v in kwargs.items() if k not in ("session_id", "model", "stream")}
                return await self.backend.run_native(session_id, model, prompt, stream, **clean_kwargs)
                
    @property
    def chat(self):
        if not hasattr(self, "_chat"):
            self._chat = self.Chat()
            self._chat.completions = self.Chat.Completions(self)
        return self._chat

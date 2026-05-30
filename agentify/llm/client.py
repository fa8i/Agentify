import os
from typing import Any, Callable, Dict, Optional, TypeAlias

from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI


LLMClientType: TypeAlias = OpenAI | AzureOpenAI
AsyncLLMClientType: TypeAlias = AsyncOpenAI | AsyncAzureOpenAI

ClientBuilder: TypeAlias = Callable[[Dict[str, Any], int], LLMClientType]
AsyncClientBuilder: TypeAlias = Callable[[Dict[str, Any], int], AsyncLLMClientType]


class LLMClientFactory:
    """Factory class to create LLM client instances for different providers."""

    SUPPORTED_PROVIDERS = [
        "azure",
        "openai",
        "deepseek",
        "gemini",
        "anthropic",
        "llama",
        "local",
        "codex",
    ]

    def __init__(self, default_timeout: int = 30):
        self.default_timeout = default_timeout

        self._builders: Dict[str, ClientBuilder] = {
            "openai": self._create_openai_client,
            "deepseek": self._create_deepseek_client,
            "gemini": self._create_gemini_client,
            "azure": self._create_azure_client,
            "anthropic": self._create_anthropic_client,
            "llama": self._create_llama_client,
            "local": self._create_local_client,
            "codex": self._create_codex_client,
        }

        self._async_builders: Dict[str, AsyncClientBuilder] = {
            "openai": self._create_openai_client_async,
            "deepseek": self._create_deepseek_client_async,
            "gemini": self._create_gemini_client_async,
            "azure": self._create_azure_client_async,
            "anthropic": self._create_anthropic_client_async,
            "llama": self._create_llama_client_async,
            "local": self._create_local_client_async,
            "codex": self._create_codex_client_async,
        }

    def _get_env_or_config(
        self,
        key: str,
        env_var_name: str,
        config: Dict[str, Any],
    ) -> str:
        value = config.get(key) or os.getenv(env_var_name)

        if not value:
            raise ValueError(
                f"Parameter '{key}' or environment variable "
                f"'{env_var_name}' is required but was not found."
            )

        return str(value)

    # -------------------------------------------------------------------------
    # Synchronous client builders
    # -------------------------------------------------------------------------

    def _create_openai_client(self, config: Dict[str, Any], timeout: int) -> OpenAI:
        api_key = self._get_env_or_config("api_key", "OPENAI_API_KEY", config)

        return OpenAI(
            api_key=api_key,
            timeout=timeout,
        )

    def _create_deepseek_client(self, config: Dict[str, Any], timeout: int) -> OpenAI:
        api_key = self._get_env_or_config("api_key", "DEEPSEEK_API_KEY", config)
        base_url = "https://api.deepseek.com"

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _create_anthropic_client(self, config: Dict[str, Any], timeout: int) -> OpenAI:
        api_key = self._get_env_or_config("api_key", "ANTHROPIC_API_KEY", config)
        base_url = "https://api.anthropic.com/v1/"

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _create_llama_client(self, config: Dict[str, Any], timeout: int) -> OpenAI:
        api_key = self._get_env_or_config("api_key", "LLAMA_API_KEY", config)
        base_url = "https://api.llama.com/compat/v1/"

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _create_local_client(self, config: Dict[str, Any], timeout: int) -> OpenAI:
        api_key = config.get("api_key", os.getenv("LOCAL_API_KEY", "api_key"))
        base_url = config.get(
            "base_url",
            os.getenv("LOCAL_API_BASE", "http://localhost:1234/v1"),
        )

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _create_gemini_client(self, config: Dict[str, Any], timeout: int) -> OpenAI:
        api_key = self._get_env_or_config("api_key", "GEMINI_API_KEY", config)
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _create_azure_client(self, config: Dict[str, Any], timeout: int) -> AzureOpenAI:
        api_key = self._get_env_or_config("api_key", "AZURE_OPENAI_KEY", config)
        api_version = self._get_env_or_config("api_version", "API_VERSION", config)
        azure_endpoint = self._get_env_or_config(
            "azure_endpoint",
            "AZURE_OPENAI_ENDPOINT",
            config,
        )

        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            timeout=timeout,
        )

    def _create_codex_client(self, config: Dict[str, Any], timeout: int) -> Any:
        from agentify.llm.codex_backend import CodexThreadBackend
        return CodexThreadBackend(config=config, timeout=timeout)

    # -------------------------------------------------------------------------
    # Asynchronous client builders
    # -------------------------------------------------------------------------

    def _create_openai_client_async(
        self,
        config: Dict[str, Any],
        timeout: int,
    ) -> AsyncOpenAI:
        api_key = self._get_env_or_config("api_key", "OPENAI_API_KEY", config)

        return AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
        )

    def _create_deepseek_client_async(
        self,
        config: Dict[str, Any],
        timeout: int,
    ) -> AsyncOpenAI:
        api_key = self._get_env_or_config("api_key", "DEEPSEEK_API_KEY", config)
        base_url = "https://api.deepseek.com"

        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _create_anthropic_client_async(
        self,
        config: Dict[str, Any],
        timeout: int,
    ) -> AsyncOpenAI:
        api_key = self._get_env_or_config("api_key", "ANTHROPIC_API_KEY", config)
        base_url = "https://api.anthropic.com/v1/"

        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _create_llama_client_async(
        self,
        config: Dict[str, Any],
        timeout: int,
    ) -> AsyncOpenAI:
        api_key = self._get_env_or_config("api_key", "LLAMA_API_KEY", config)
        base_url = "https://api.llama.com/compat/v1/"

        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _create_local_client_async(
        self,
        config: Dict[str, Any],
        timeout: int,
    ) -> AsyncOpenAI:
        api_key = config.get("api_key", os.getenv("LOCAL_API_KEY", "api_key"))
        base_url = config.get(
            "base_url",
            os.getenv("LOCAL_API_BASE", "http://localhost:1234/v1"),
        )

        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _create_gemini_client_async(
        self,
        config: Dict[str, Any],
        timeout: int,
    ) -> AsyncOpenAI:
        api_key = self._get_env_or_config("api_key", "GEMINI_API_KEY", config)
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _create_azure_client_async(
        self,
        config: Dict[str, Any],
        timeout: int,
    ) -> AsyncAzureOpenAI:
        api_key = self._get_env_or_config("api_key", "AZURE_OPENAI_KEY", config)
        api_version = self._get_env_or_config("api_version", "API_VERSION", config)
        azure_endpoint = self._get_env_or_config(
            "azure_endpoint",
            "AZURE_OPENAI_ENDPOINT",
            config,
        )

        return AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            timeout=timeout,
        )

    def _create_codex_client_async(
        self,
        config: Dict[str, Any],
        timeout: int,
    ) -> Any:
        from agentify.llm.codex_backend import CodexThreadBackend
        return CodexThreadBackend(config=config, timeout=timeout)

    # -------------------------------------------------------------------------
    # Public factory methods
    # -------------------------------------------------------------------------

    def create_client(
        self,
        provider: str,
        config_override: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> LLMClientType:
        provider_lower = provider.lower().strip()

        if provider_lower not in self._builders:
            raise ValueError(
                f"Provider '{provider}' not supported. "
                f"Supported: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )

        builder_func = self._builders[provider_lower]
        effective_timeout = timeout if timeout is not None else self.default_timeout
        effective_config = config_override or {}

        return builder_func(effective_config, effective_timeout)

    def create_async_client(
        self,
        provider: str,
        config_override: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> AsyncLLMClientType:
        provider_lower = provider.lower().strip()

        if provider_lower not in self._async_builders:
            raise ValueError(
                f"Provider '{provider}' not supported for async. "
                f"Supported: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )

        builder_func = self._async_builders[provider_lower]
        effective_timeout = timeout if timeout is not None else self.default_timeout
        effective_config = config_override or {}

        return builder_func(effective_config, effective_timeout)

import os
from dotenv import load_dotenv
from typing import Union, Dict, Any, Optional, Callable
from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI

load_dotenv()

LLMClientType = Union[OpenAI, AzureOpenAI]
AsyncLLMClientType = Union[AsyncOpenAI, AsyncAzureOpenAI]

ClientBuilder = Callable[[Dict[str, Any], int], LLMClientType]
AsyncClientBuilder = Callable[[Dict[str, Any], int], AsyncLLMClientType]


class LLMClientFactory:
    """Factory class to create LLM client instances for different providers."""

    SUPPORTED_PROVIDERS = ["azure", "openai", "deepseek", "gemini", "anthropic"]

    def __init__(self, default_timeout: int = 30):
        self.default_timeout = default_timeout
        self._builders: Dict[str, ClientBuilder] = {
            "openai": self._create_openai_client,
            "deepseek": self._create_deepseek_client,
            "gemini": self._create_gemini_client,
            "azure": self._create_azure_client,
            "anthropic": self._create_anthropic_client,
            "llama": self._create_llama_client,
        }
        self._async_builders: Dict[str, AsyncClientBuilder] = {
            "openai": self._create_openai_client_async,
            "deepseek": self._create_deepseek_client_async,
            "gemini": self._create_gemini_client_async,
            "azure": self._create_azure_client_async,
            "anthropic": self._create_anthropic_client_async,
            "llama": self._create_llama_client_async,
        }

    def _get_env_or_config(
        self, key: str, env_var_name: str, config: Dict[str, Any], required: bool = True
    ) -> Optional[str]:
        """Helper to obtain a value from config_override or an environment variable."""
        value = config.get(key, os.getenv(env_var_name))
        if required and not value:
            raise ValueError(
                f"Parameter '{key}' (or environment variable '{env_var_name}') is required but was not found."
            )
        return value

    # Synchronous client builders

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

    def _create_gemini_client(self, config: Dict[str, Any], timeout: int) -> OpenAI:
        api_key = self._get_env_or_config("api_key", "GEMINI_API_KEY", config)
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        client_args = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_args["base_url"] = base_url
        else:
            print(
                "Warning: No GEMINI_URL provided for Gemini. Assuming the OpenAI SDK handles it or it's not needed."
            )

        return OpenAI(**client_args)

    def _create_azure_client(self, config: Dict[str, Any], timeout: int) -> AzureOpenAI:
        api_key = self._get_env_or_config("api_key", "AZURE_OPENAI_KEY", config)
        api_version = self._get_env_or_config("api_version", "API_VERSION", config)
        azure_endpoint = self._get_env_or_config(
            "azure_endpoint", "AZURE_OPENAI_ENDPOINT", config
        )
        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            timeout=timeout,
        )

    # Asynchronous client builders

    def _create_openai_client_async(self, config: Dict[str, Any], timeout: int) -> AsyncOpenAI:
        api_key = self._get_env_or_config("api_key", "OPENAI_API_KEY", config)
        return AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
        )

    def _create_deepseek_client_async(self, config: Dict[str, Any], timeout: int) -> AsyncOpenAI:
        api_key = self._get_env_or_config("api_key", "DEEPSEEK_API_KEY", config)
        base_url = "https://api.deepseek.com"
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _create_anthropic_client_async(self, config: Dict[str, Any], timeout: int) -> AsyncOpenAI:
        api_key = self._get_env_or_config("api_key", "ANTHROPIC_API_KEY", config)
        base_url = "https://api.anthropic.com/v1/"
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _create_llama_client_async(self, config: Dict[str, Any], timeout: int) -> AsyncOpenAI:
        api_key = self._get_env_or_config("api_key", "LLAMA_API_KEY", config)
        base_url = "https://api.llama.com/compat/v1/"
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _create_gemini_client_async(self, config: Dict[str, Any], timeout: int) -> AsyncOpenAI:
        api_key = self._get_env_or_config("api_key", "GEMINI_API_KEY", config)
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        client_args = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_args["base_url"] = base_url

        return AsyncOpenAI(**client_args)

    def _create_azure_client_async(self, config: Dict[str, Any], timeout: int) -> AsyncAzureOpenAI:
        api_key = self._get_env_or_config("api_key", "AZURE_OPENAI_KEY", config)
        api_version = self._get_env_or_config("api_version", "API_VERSION", config)
        azure_endpoint = self._get_env_or_config(
            "azure_endpoint", "AZURE_OPENAI_ENDPOINT", config
        )
        return AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            timeout=timeout,
        )

    # Public factory methods
    # -------------------------------------------------------------------------

    def create_client(
        self,
        provider: str,
        config_override: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> LLMClientType:
        """Create a synchronous LLM client for the specified provider.

        Args:
            provider: Name of the provider (e.g., "openai", "azure").
            config_override: Optional dict to override or provide configuration parameters (e.g., api_key, base_url) instead of using environment variables.
            timeout: Specific timeout for this client; if not provided, the factory's default_timeout is used.

        Returns:
            An instance of the requested LLM client.

        Raises:
            ValueError: If the provider is not supported or required parameters are missing.
        """
        provider_lower = provider.lower()
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
        """Create an asynchronous LLM client for the specified provider.

        Args:
            provider: Name of the provider (e.g., "openai", "azure").
            config_override: Optional dict to override or provide configuration parameters.
            timeout: Specific timeout for this client; if not provided, the factory's default_timeout is used.

        Returns:
            An instance of the requested async LLM client (AsyncOpenAI or AsyncAzureOpenAI).

        Raises:
            ValueError: If the provider is not supported or required parameters are missing.
        """
        provider_lower = provider.lower()
        if provider_lower not in self._async_builders:
            raise ValueError(
                f"Provider '{provider}' not supported for async. "
                f"Supported: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )

        builder_func = self._async_builders[provider_lower]
        effective_timeout = timeout if timeout is not None else self.default_timeout
        effective_config = config_override or {}

        return builder_func(effective_config, effective_timeout)


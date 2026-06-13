"""Codex provider error types and classification."""

from __future__ import annotations


class CodexBackendError(RuntimeError):
    """Base Codex backend error with retry metadata."""

    non_retryable_provider_error = False


class NonRetryableCodexError(CodexBackendError):
    """Codex error that should not be retried by Agentify."""

    non_retryable_provider_error = True


class CodexAuthError(NonRetryableCodexError):
    """ChatGPT OAuth session is missing or expired. Run `codex login`."""


class CodexCLINotFoundError(NonRetryableCodexError):
    """The `codex` CLI binary could not be found or executed."""


class CodexModelNotSupportedError(NonRetryableCodexError):
    """The requested model is not available for this account or CLI version."""


class CodexUsageLimitError(NonRetryableCodexError):
    """The ChatGPT account hit its Codex usage limit."""


class CodexMCPStartupError(NonRetryableCodexError):
    """Codex could not start the Agentify runtime MCP server."""


class CodexToolNotFoundError(NonRetryableCodexError):
    """Codex tried to call an MCP tool that is not registered."""


class CodexEmptyTurnError(CodexBackendError):
    """The Codex turn completed without any reconstructible response text."""


class CodexStreamTimeoutError(CodexBackendError, TimeoutError):
    """Timed out waiting for the next Codex turn event."""


_CLASSIFIED_MARKERS: tuple[tuple[tuple[str, ...], type[CodexBackendError]], ...] = (
    (("usage limit exceeded", "usagelimitexceeded"), CodexUsageLimitError),
    (
        ("auth is not available", "unauthorized", "not logged in", "codex login"),
        CodexAuthError,
    ),
    (
        ("model is not supported", "not supported for this account"),
        CodexModelNotSupportedError,
    ),
    (("mcp server failed to start",), CodexMCPStartupError),
    (("mcp tool was not found",), CodexToolNotFoundError),
)

NON_RETRYABLE_MARKERS = (
    "usage limit exceeded",
    "usagelimitexceeded",
    "model is not supported",
    "not supported for this account",
    "mcp server failed to start",
    "mcp tool was not found",
)


def classify_codex_error(message: str) -> CodexBackendError:
    """Build the most specific Codex exception for an error message."""
    lowered = message.lower()
    for markers, error_cls in _CLASSIFIED_MARKERS:
        if any(marker in lowered for marker in markers):
            return error_cls(message)
    if any(marker in lowered for marker in NON_RETRYABLE_MARKERS):
        return NonRetryableCodexError(message)
    return CodexBackendError(message)


def raise_codex_errors(errors: list[str]) -> None:
    """Raise a retry-aware Codex exception for collected turn errors."""
    raise classify_codex_error("; ".join(errors))

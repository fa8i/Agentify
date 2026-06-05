"""Codex provider error types and classification."""

from __future__ import annotations


class CodexBackendError(RuntimeError):
    """Base Codex backend error with retry metadata."""

    non_retryable_provider_error = False


class NonRetryableCodexError(CodexBackendError):
    """Codex error that should not be retried by Agentify."""

    non_retryable_provider_error = True


NON_RETRYABLE_MARKERS = (
    "usage limit exceeded",
    "usagelimitexceeded",
    "model is not supported",
    "not supported for this account",
    "mcp server failed to start",
    "mcp tool was not found",
)


def raise_codex_errors(errors: list[str]) -> None:
    """Raise a retry-aware Codex exception for collected turn errors."""
    message = "; ".join(errors)
    lowered = message.lower()
    if any(marker in lowered for marker in NON_RETRYABLE_MARKERS):
        raise NonRetryableCodexError(message)
    raise CodexBackendError(message)

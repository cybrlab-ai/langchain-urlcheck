"""
Custom exceptions for the URLCheck LangChain adapter.

Exception hierarchy:
    McpClientError (base)
    ├── McpConnectionError - Network/HTTP errors
    ├── McpAuthenticationError - Invalid API key (401/403)
    ├── McpRateLimitError - Too many requests (HTTP 429 or JSON-RPC -32029)
    ├── McpValidationError - Input validation failures (-32602 or 2000-2099)
    └── McpToolError - Tool execution failures
"""

from __future__ import annotations

# =============================================================================
# Base Exception
# =============================================================================


class McpClientError(Exception):
    """Base exception for MCP client errors."""

    def __init__(
        self,
        message: str,
        code: int | None = None,
        retryable: bool | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable

    def __str__(self) -> str:
        return self.message


# =============================================================================
# Connection Errors
# =============================================================================


class McpConnectionError(McpClientError):
    """Raised when connection to the MCP server fails."""

    def __init__(
        self,
        message: str,
        code: int | None = None,
        retryable: bool | None = True,
    ):
        super().__init__(message=message, code=code, retryable=retryable)


class McpAuthenticationError(McpClientError):
    """Raised when API key authentication fails (HTTP 401/403)."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message=message, code=code, retryable=False)


# =============================================================================
# Rate Limit Error
# =============================================================================


class McpRateLimitError(McpClientError):
    """Raised when rate limit is exceeded (HTTP 429 or JSON-RPC -32029)."""

    def __init__(
        self,
        message: str,
        code: int | None = None,
        retry_after_seconds: float | None = None,
        retryable: bool = True,
    ):
        super().__init__(message=message, code=code, retryable=retryable)
        self.retry_after_seconds = retry_after_seconds


# =============================================================================
# Validation Error
# =============================================================================


class McpValidationError(McpClientError):
    """Raised for input validation failures (JSON-RPC -32602 or codes 2000-2099)."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message=message, code=code, retryable=False)


# =============================================================================
# Tool Error
# =============================================================================


class McpToolError(McpClientError):
    """Raised when an MCP tool returns an error."""

    def __init__(
        self,
        message: str,
        code: int | None = None,
        retryable: bool | None = None,
    ):
        super().__init__(message=message, code=code, retryable=retryable)

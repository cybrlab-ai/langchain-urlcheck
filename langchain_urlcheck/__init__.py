"""
URLCheck MCP LangChain Integration

A LangChain tool that lets agents verify URL safety before navigation,
powered by the URLCheck MCP server.

Stateless-only: session-based MCP mode is not supported currently.

Quick Start:
    >>> from langchain_urlcheck import SafeUrlScanTool
    >>>
    >>> # No API key required for free tier (up to 100 requests/day)
    >>> tool = SafeUrlScanTool()
    >>>
    >>> result = tool.invoke({"url": "https://example.com"})
    >>> print(result)

For more information, see:
    - GitHub: https://github.com/cybrlab-ai/langchain-urlcheck
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__author__ = "CybrLab.AI"

# Public API
from .adapter import get_mcp_server_config
from .client import AsyncUrlScannerMcpClient, UrlScannerMcpClient
from .exceptions import (
    McpAuthenticationError,
    McpClientError,
    McpConnectionError,
    McpRateLimitError,
    McpToolError,
    McpValidationError,
)
from .tool import (
    SafeUrlScanInput,
    SafeUrlScanTool,
    ScanFailure,
    ScanResult,
    create_url_scan_tool,
)

__all__ = [
    # Version
    "__version__",
    # Main tool
    "SafeUrlScanTool",
    "SafeUrlScanInput",
    "ScanResult",
    "ScanFailure",
    "create_url_scan_tool",
    # MCP adapter compatibility
    "get_mcp_server_config",
    # Low-level client
    "UrlScannerMcpClient",
    "AsyncUrlScannerMcpClient",
    # Exceptions
    "McpClientError",
    "McpConnectionError",
    "McpAuthenticationError",
    "McpRateLimitError",
    "McpToolError",
    "McpValidationError",
]

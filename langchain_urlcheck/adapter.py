"""MCP adapter compatibility helpers for langchain-mcp-adapters users."""

from __future__ import annotations

from typing import Any

__all__ = ["get_mcp_server_config"]


def get_mcp_server_config(
    api_key: str = "",
    mcp_url: str = "https://urlcheck.ai/mcp",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Get MCP server configuration for use with langchain-mcp-adapters.

    This helper returns a configuration dict compatible with MultiServerMCPClient.

    Warning:
        This path uses direct (synchronous) MCP execution. It does NOT support
        task-augmented execution or polling. For long-running scans, use
        SafeUrlScanTool instead.

    Args:
        api_key: Your urlcheck.ai API key. Optional for free tier (up to 100 req/day).
        mcp_url: MCP server endpoint URL.
        timeout_seconds: HTTP request timeout for direct calls.

    Returns:
        Configuration dict for MultiServerMCPClient.

    Example:
        >>> from langchain_mcp_adapters.client import MultiServerMCPClient
        >>> from langchain_urlcheck import get_mcp_server_config
        >>>
        >>> client = MultiServerMCPClient({
        ...     "urlcheck": get_mcp_server_config(api_key="sk-..."),
        ... })
        >>> tools = await client.get_tools()
    """
    headers: dict[str, str] = {
        "MCP-Protocol-Version": "2025-06-18",
    }
    if api_key:
        headers["X-API-Key"] = api_key

    return {
        "transport": "http",
        "url": mcp_url,
        "headers": headers,
        "timeout": timeout_seconds,
    }

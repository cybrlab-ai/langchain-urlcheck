"""Unit tests for MCP sync/async clients."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from langchain_urlcheck.client import AsyncUrlScannerMcpClient, UrlScannerMcpClient
from langchain_urlcheck.exceptions import (
    McpAuthenticationError,
    McpRateLimitError,
    McpToolError,
    McpValidationError,
)


class TestUrlScannerMcpClient:
    def test_init_sets_defaults(self) -> None:
        client = UrlScannerMcpClient(
            base_url="https://test.example.com/mcp",
            api_key="test-key",
        )
        try:
            assert client.base_url == "https://test.example.com/mcp"
            assert client.api_key == "test-key"
            assert client.protocol_version == "2025-06-18"
            assert client.http_timeout_seconds == 30
        finally:
            client.close()

    def test_init_free_tier_no_api_key(self) -> None:
        client = UrlScannerMcpClient(
            base_url="https://test.example.com/mcp",
        )
        try:
            assert client.api_key == ""
            assert "X-API-Key" not in client._http.headers
        finally:
            client.close()

    def test_init_strips_trailing_slash(self) -> None:
        client = UrlScannerMcpClient(
            base_url="https://test.example.com/mcp/",
            api_key="test-key",
        )
        try:
            assert client.base_url == "https://test.example.com/mcp"
        finally:
            client.close()

    @patch("httpx.Client.post")
    def test_authentication_error(self, mock_post: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_post.return_value = mock_response

        client = UrlScannerMcpClient(
            base_url="https://test.example.com/mcp",
            api_key="bad-key",
        )
        try:
            with pytest.raises(McpAuthenticationError):
                client.scan("https://example.com")
        finally:
            client.close()

    @patch("httpx.Client.post")
    def test_rate_limit_error(self, mock_post: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_post.return_value = mock_response

        client = UrlScannerMcpClient(
            base_url="https://test.example.com/mcp",
            api_key="test-key",
            max_retries=0,
        )
        try:
            with pytest.raises(McpRateLimitError) as exc_info:
                client.scan("https://example.com")
            assert "Rate limited" in str(exc_info.value)
            assert exc_info.value.retryable is True
        finally:
            client.close()

    @patch("httpx.Client.post")
    def test_parse_tool_response_success(self, mock_post: Mock) -> None:
        tool_response = Mock()
        tool_response.status_code = 200
        tool_response.headers = {}
        tool_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "risk_score": 0.05,
                                "confidence": 0.95,
                                "analysis_complete": True,
                                "agent_access_directive": "ALLOW",
                                "agent_access_reason": "clean",
                            }
                        ),
                    }
                ],
                "isError": False,
            },
        }
        mock_post.return_value = tool_response

        client = UrlScannerMcpClient(
            base_url="https://test.example.com/mcp",
            api_key="test-key",
        )
        try:
            result = client.scan("https://example.com")
            assert result["risk_score"] == 0.05
            assert result["confidence"] == 0.95
            assert result["analysis_complete"] is True
        finally:
            client.close()

    @patch("httpx.Client.post")
    def test_parse_tool_response_error(self, mock_post: Mock) -> None:
        tool_response = Mock()
        tool_response.status_code = 200
        tool_response.headers = {}
        tool_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"error": True, "message": "Queue is full"}),
                    }
                ],
                "isError": True,
            },
        }
        mock_post.return_value = tool_response

        client = UrlScannerMcpClient(
            base_url="https://test.example.com/mcp",
            api_key="test-key",
        )
        try:
            with pytest.raises(McpToolError) as exc_info:
                client.scan("https://example.com")
            assert "Queue is full" in str(exc_info.value)
        finally:
            client.close()

    @patch("httpx.Client.post")
    def test_retries_transient_rate_limit(self, mock_post: Mock) -> None:
        rate_limited = Mock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "0"}

        success = Mock()
        success.status_code = 200
        success.headers = {}
        success.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "risk_score": 0.1,
                                "confidence": 0.9,
                                "analysis_complete": True,
                                "agent_access_directive": "ALLOW",
                                "agent_access_reason": "clean",
                            }
                        ),
                    }
                ]
            },
        }
        mock_post.side_effect = [rate_limited, success]

        client = UrlScannerMcpClient(
            base_url="https://test.example.com/mcp",
            api_key="test-key",
            max_retries=1,
            retry_base_delay_seconds=0.0,
            retry_max_delay_seconds=0.01,
        )
        events: list[dict[str, object]] = []
        try:
            result = client.scan("https://example.com", retry_callback=events.append)
            assert result["agent_access_directive"] == "ALLOW"
            assert len(events) == 1
            assert events[0]["reason"] == "McpRateLimitError"
        finally:
            client.close()


class TestJsonRpcErrorMapping:
    @patch("httpx.Client.post")
    def test_code_32029_raises_rate_limit_error(self, mock_post: Mock) -> None:
        error_response = Mock()
        error_response.status_code = 200
        error_response.headers = {}
        error_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 2,
            "error": {
                "code": -32029,
                "message": "Rate limit exceeded. Retry after 5000 ms",
            },
        }
        mock_post.return_value = error_response

        client = UrlScannerMcpClient(
            base_url="https://test.example.com/mcp",
            api_key="test-key",
            max_retries=0,
        )
        try:
            with pytest.raises(McpRateLimitError) as exc_info:
                client.scan("https://example.com")
            assert exc_info.value.code == -32029
        finally:
            client.close()

    @patch("httpx.Client.post")
    def test_code_32602_raises_validation_error(self, mock_post: Mock) -> None:
        error_response = Mock()
        error_response.status_code = 200
        error_response.headers = {}
        error_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 2,
            "error": {
                "code": -32602,
                "message": "Invalid params: URL validation failed",
            },
        }
        mock_post.return_value = error_response

        client = UrlScannerMcpClient(
            base_url="https://test.example.com/mcp",
            api_key="test-key",
        )
        try:
            with pytest.raises(McpValidationError) as exc_info:
                client.scan("not-a-url")
            assert exc_info.value.code == -32602
        finally:
            client.close()


@pytest.mark.asyncio
class TestAsyncUrlScannerMcpClient:
    async def test_async_init_free_tier_no_api_key(self) -> None:
        async with AsyncUrlScannerMcpClient(
            base_url="https://test.example.com/mcp",
        ) as client:
            assert client.api_key == ""
            assert "X-API-Key" not in client._http.headers

    async def test_async_scan_success(self) -> None:
        response = Mock()
        response.status_code = 200
        response.headers = {}
        response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "risk_score": 0.04,
                                "confidence": 0.98,
                                "analysis_complete": True,
                                "agent_access_directive": "ALLOW",
                                "agent_access_reason": "clean",
                            }
                        ),
                    }
                ]
            },
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = response
            async with AsyncUrlScannerMcpClient(
                base_url="https://test.example.com/mcp",
                api_key="test-key",
            ) as client:
                result = await client.scan("https://example.com")

        assert result["risk_score"] == 0.04
        assert result["analysis_complete"] is True

    async def test_async_jsonrpc_error_mapping(self) -> None:
        error_response = Mock()
        error_response.status_code = 200
        error_response.headers = {}
        error_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"code": -32602, "message": "Invalid params"},
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = error_response
            async with AsyncUrlScannerMcpClient(
                base_url="https://test.example.com/mcp",
                api_key="test-key",
            ) as client:
                with pytest.raises(McpValidationError):
                    await client.scan("bad-url")

"""Unit tests for SafeUrlScanTool."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest

from langchain_urlcheck.exceptions import McpToolError
from langchain_urlcheck.tool import (
    SafeUrlScanInput,
    SafeUrlScanTool,
    ScanFailure,
    ScanResult,
    create_url_scan_tool,
)


class TestSafeUrlScanInput:
    def test_valid_input(self) -> None:
        input_data = SafeUrlScanInput(
            url="https://example.com",
            timeout_seconds=60,
        )
        assert input_data.url == "https://example.com"
        assert input_data.timeout_seconds == 60

    def test_valid_input_with_intent(self) -> None:
        input_data = SafeUrlScanInput(
            url="https://example.com/checkout",
            intent="purchase",
        )
        assert input_data.intent == "purchase"

    def test_default_timeout(self) -> None:
        input_data = SafeUrlScanInput(url="https://example.com")
        assert input_data.timeout_seconds == 120

    def test_timeout_minimum(self) -> None:
        with pytest.raises(ValueError):
            SafeUrlScanInput(url="https://example.com", timeout_seconds=25)

    def test_timeout_maximum(self) -> None:
        with pytest.raises(ValueError):
            SafeUrlScanInput(url="https://example.com", timeout_seconds=900)


class TestScanResult:
    def test_to_agent_response(self) -> None:
        result = ScanResult(
            risk_score=0.1,
            confidence=0.95,
            analysis_complete=True,
            agent_access_directive="ALLOW",
            agent_access_reason="clean",
        )
        parsed = json.loads(result.to_agent_response())
        assert parsed["risk_score"] == 0.1
        assert parsed["confidence"] == 0.95
        assert parsed["analysis_complete"] is True
        assert parsed["agent_access_directive"] == "ALLOW"
        assert parsed["agent_access_reason"] == "clean"


class TestScanFailure:
    def test_to_agent_response_minimal(self) -> None:
        failure = ScanFailure(error="Connection timeout")
        parsed = json.loads(failure.to_agent_response())
        assert parsed["error"] == "Connection timeout"
        assert "retryable" not in parsed

    def test_to_agent_response_full(self) -> None:
        failure = ScanFailure(
            error="Rate limited",
            numeric_code=-32029,
            retryable=True,
        )
        parsed = json.loads(failure.to_agent_response())
        assert parsed["numeric_code"] == -32029
        assert parsed["retryable"] is True


class TestSafeUrlScanTool:
    def test_free_tier_no_api_key(self) -> None:
        tool = SafeUrlScanTool()
        try:
            assert tool.api_key == ""
            assert tool.mcp_url == "https://urlcheck.ai/mcp"
        finally:
            tool.close()

    def test_from_server_json_uses_remote_url(self, tmp_path: Path) -> None:
        server_json = tmp_path / "server.json"
        server_json.write_text(
            json.dumps(
                {
                    "remotes": [
                        {"type": "streamable-http", "url": "https://custom.example.com/mcp"},
                    ]
                }
            ),
            encoding="utf-8",
        )

        tool = SafeUrlScanTool.from_server_json(
            api_key="test-key",
            server_json_path=server_json,
        )
        try:
            assert tool.mcp_url == "https://custom.example.com/mcp"
        finally:
            tool.close()

    def test_tool_metadata(self) -> None:
        tool = SafeUrlScanTool(
            mcp_url="https://test.example.com/mcp",
            api_key="test-key",
        )
        try:
            assert tool.name == "safe_url_scan"
            assert "malicious" in tool.description.lower()
            assert tool.args_schema == SafeUrlScanInput
            assert tool.wait_mode == "tasks_result"
        finally:
            tool.close()

    def test_calculate_poll_interval_uses_server_recommendation(self) -> None:
        tool = SafeUrlScanTool(
            mcp_url="https://test.example.com/mcp",
            api_key="test-key",
            min_poll_interval_seconds=1.0,
            max_poll_interval_seconds=5.0,
        )
        try:
            assert tool._calculate_poll_interval(2000) == 2.0
        finally:
            tool.close()

    def test_calculate_poll_interval_respects_minimum(self) -> None:
        tool = SafeUrlScanTool(
            mcp_url="https://test.example.com/mcp",
            api_key="test-key",
            min_poll_interval_seconds=1.0,
            max_poll_interval_seconds=5.0,
        )
        try:
            assert tool._calculate_poll_interval(500) == 1.0
        finally:
            tool.close()

    def test_calculate_poll_interval_respects_maximum(self) -> None:
        tool = SafeUrlScanTool(
            mcp_url="https://test.example.com/mcp",
            api_key="test-key",
            min_poll_interval_seconds=1.0,
            max_poll_interval_seconds=5.0,
        )
        try:
            assert tool._calculate_poll_interval(10000) == 5.0
        finally:
            tool.close()

    @patch.object(SafeUrlScanTool, "client", new_callable=PropertyMock)
    def test_run_successful_scan_tasks_result_mode(
        self,
        mock_client_property: PropertyMock,
    ) -> None:
        mock_client = Mock()
        mock_client_property.return_value = mock_client
        mock_client.scan.return_value = {"task": {"taskId": "test-task-123", "pollInterval": 1000}}
        mock_client.tasks_result.return_value = {
            "value": {
                "risk_score": 0.05,
                "confidence": 0.95,
                "analysis_complete": True,
                "agent_access_directive": "ALLOW",
                "agent_access_reason": "clean",
            }
        }

        tool = SafeUrlScanTool(
            mcp_url="https://test.example.com/mcp",
            api_key="test-key",
            wait_mode="tasks_result",
        )

        result = tool._run("https://example.com", timeout_seconds=60)
        parsed = json.loads(result)
        assert parsed["risk_score"] == 0.05
        assert parsed["agent_access_directive"] == "ALLOW"

    @patch.object(SafeUrlScanTool, "client", new_callable=PropertyMock)
    def test_tasks_result_timeout_falls_back_to_poll(
        self,
        mock_client_property: PropertyMock,
    ) -> None:
        mock_client = Mock()
        mock_client_property.return_value = mock_client
        mock_client.scan.return_value = {"task": {"taskId": "task-123", "pollInterval": 1000}}
        mock_client.tasks_result.side_effect = [
            McpToolError(
                "JSON-RPC error [-32603]: Task task-123 timed out after 300 seconds",
                code=-32603,
            ),
            {
                "value": {
                    "risk_score": 0.2,
                    "confidence": 0.8,
                    "analysis_complete": True,
                    "agent_access_directive": "ALLOW",
                    "agent_access_reason": "fallback poll success",
                }
            },
        ]
        mock_client.tasks_get.return_value = {
            "task": {"taskId": "task-123", "status": "completed", "pollInterval": 1000}
        }

        tool = SafeUrlScanTool(
            mcp_url="https://test.example.com/mcp",
            api_key="test-key",
            wait_mode="tasks_result",
        )
        result = tool._run("https://example.com", timeout_seconds=60)
        parsed = json.loads(result)
        assert parsed["agent_access_reason"] == "fallback poll success"
        assert mock_client.tasks_get.called

    @patch.object(SafeUrlScanTool, "client", new_callable=PropertyMock)
    def test_direct_mode_clamps_timeout(self, mock_client_property: PropertyMock) -> None:
        mock_client = Mock()
        mock_client_property.return_value = mock_client
        mock_client.scan.return_value = {
            "risk_score": 0.05,
            "confidence": 0.95,
            "analysis_complete": True,
            "agent_access_directive": "ALLOW",
            "agent_access_reason": "clean",
        }

        tool = SafeUrlScanTool(
            mcp_url="https://test.example.com/mcp",
            api_key="test-key",
            execution_mode="direct",
        )
        tool._run("https://example.com", timeout_seconds=600)
        _, kwargs = mock_client.scan.call_args
        assert kwargs["timeout_seconds"] == 300

    @pytest.mark.asyncio
    async def test_arun_uses_async_client_path(self) -> None:
        async_client = AsyncMock()
        async_client.__aenter__.return_value = async_client
        async_client.__aexit__.return_value = None
        async_client.scan.return_value = {
            "risk_score": 0.01,
            "confidence": 0.99,
            "analysis_complete": True,
            "agent_access_directive": "ALLOW",
            "agent_access_reason": "async clean",
        }

        with patch("langchain_urlcheck.tool.AsyncUrlScannerMcpClient", return_value=async_client):
            tool = SafeUrlScanTool(
                mcp_url="https://test.example.com/mcp",
                api_key="test-key",
                execution_mode="direct",
            )
            result = await tool._arun("https://example.com", timeout_seconds=60)

        parsed = json.loads(result)
        assert parsed["agent_access_reason"] == "async clean"
        async_client.scan.assert_awaited_once()


class TestCreateUrlScanTool:
    def test_creates_tool_with_defaults(self) -> None:
        tool = create_url_scan_tool(api_key="test-key")
        try:
            assert tool.mcp_url == "https://urlcheck.ai/mcp"
            assert tool.api_key == "test-key"
            assert tool.default_timeout_seconds == 120
            assert tool.execution_mode == "task"
            assert tool.wait_mode == "tasks_result"
            assert tool.cancel_on_timeout is False
        finally:
            tool.close()

    def test_creates_tool_free_tier(self) -> None:
        tool = create_url_scan_tool()
        try:
            assert tool.api_key == ""
            assert tool.mcp_url == "https://urlcheck.ai/mcp"
        finally:
            tool.close()

    def test_creates_tool_with_custom_values(self) -> None:
        tool = create_url_scan_tool(
            api_key="test-key",
            mcp_url="https://custom.example.com/mcp",
            timeout_seconds=120,
            execution_mode="direct",
            wait_mode="poll",
            task_ttl_ms=600000,
            cancel_on_timeout=True,
        )
        try:
            assert tool.mcp_url == "https://custom.example.com/mcp"
            assert tool.default_timeout_seconds == 120
            assert tool.execution_mode == "direct"
            assert tool.wait_mode == "poll"
            assert tool.task_ttl_ms == 600000
            assert tool.cancel_on_timeout is True
        finally:
            tool.close()

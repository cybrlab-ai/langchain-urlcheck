"""Standard LangChain tool integration tests.

Uses langchain-tests ToolsIntegrationTests to verify SafeUrlScanTool works
end-to-end through the LangChain BaseTool invoke/ainvoke interface.

Gated behind URLCHECK_RUN_LIVE_TESTS=1 because these tests hit the real MCP endpoint.
Free-tier access is used (no API key required).
"""

from __future__ import annotations

import os

import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_urlcheck import SafeUrlScanTool

LIVE_FLAG = os.environ.get("URLCHECK_RUN_LIVE_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE_FLAG,
    reason="Set URLCHECK_RUN_LIVE_TESTS=1 to run live integration tests",
)


class TestSafeUrlScanToolStandardIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[SafeUrlScanTool]:
        return SafeUrlScanTool

    @property
    def tool_constructor_params(self) -> dict:
        return {
            "mcp_url": "https://urlcheck.ai/mcp",
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "url": "https://example.com",
        }

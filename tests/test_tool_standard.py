"""Standard LangChain tool interface compliance tests.

Uses langchain-tests base classes to verify SafeUrlScanTool conforms
to the BaseTool contract expected by the LangChain ecosystem.
"""

from __future__ import annotations

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_urlcheck import SafeUrlScanTool


class TestSafeUrlScanToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[SafeUrlScanTool]:
        return SafeUrlScanTool

    @property
    def tool_constructor_params(self) -> dict:
        return {
            "mcp_url": "https://urlcheck.ai/mcp",
            "api_key": "test-key",
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "url": "https://example.com",
        }

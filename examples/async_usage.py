#!/usr/bin/env python3
"""
Example: async usage with SafeUrlScanTool.

Requirements:
    pip install langchain-urlcheck
"""

import asyncio
import json
import os
import sys

# Add parent directory for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_urlcheck import SafeUrlScanTool


async def main() -> None:
    api_key = os.environ.get("URLCHECK_API_KEY")
    if not api_key:
        print("Error: Set URLCHECK_API_KEY environment variable")
        sys.exit(1)

    tool = SafeUrlScanTool(
        mcp_url="https://urlcheck.ai/mcp",
        api_key=api_key,
        default_timeout_seconds=120,
    )

    result = await tool.ainvoke({"url": "https://example.com"})
    print(json.dumps(json.loads(result), indent=2))
    tool.close()


if __name__ == "__main__":
    asyncio.run(main())

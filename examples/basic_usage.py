#!/usr/bin/env python3
"""
Basic usage example for langchain-urlcheck.

This example shows how to use the SafeUrlScanTool to verify URL safety.
No API key is required for the free tier (up to 100 requests/day).
"""

import json
import os
import sys

# Add a parent directory to a path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_urlcheck import SafeUrlScanTool, create_url_scan_tool


def main():
    # Optional: set URLCHECK_API_KEY for higher volumes
    api_key = os.environ.get("URLCHECK_API_KEY", "")

    # Method 1: Free tier (no API key required)
    print("=" * 60)
    print("Method 1: Free tier (no API key)")
    print("=" * 60)

    tool = SafeUrlScanTool()

    print("\nScanning https://example.com...")
    result = tool.invoke({"url": "https://example.com"})
    print("Result:")
    print(json.dumps(json.loads(result), indent=2))

    # Method 2: With API key for higher volumes
    print("\n" + "=" * 60)
    print("Method 2: With API key (via factory function)")
    print("=" * 60)

    if api_key:
        tool2 = create_url_scan_tool(
            api_key=api_key,
            timeout_seconds=90,
        )

        print("\nScanning https://example.org...")
        result2 = tool2.invoke({"url": "https://example.org"})
        print("Result:")
        print(json.dumps(json.loads(result2), indent=2))
    else:
        print("\nSkipped (set URLCHECK_API_KEY for this example)")

    # Method 3: With a custom timeout per request
    print("\n" + "=" * 60)
    print("Method 3: Custom timeout per request")
    print("=" * 60)

    print("\nScanning with 60s timeout...")
    result3 = tool.invoke(
        {
            "url": "https://test.example.com/page",
            "timeout_seconds": 60,
        }
    )
    print("Result:")
    print(json.dumps(json.loads(result3), indent=2))


if __name__ == "__main__":
    main()

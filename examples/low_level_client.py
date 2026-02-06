#!/usr/bin/env python3
"""
Example: Using the low-level MCP client directly (stateless mode).

This example shows how to use the UrlScannerMcpClient for more
control over the scanning process.

No API key is required for the free tier (up to 100 requests/day).
"""

import os
import sys
import time

# Add a parent directory to a path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_urlcheck import (
    McpToolError,
    UrlScannerMcpClient,
)


def scan_url_with_progress(client: UrlScannerMcpClient, url: str, timeout: int = 120) -> dict:
    """
    Scan a URL with task-augmented polling.

    Args:
        client: MCP client instance
        url: URL to scan
        timeout: Maximum wait time in seconds

    Returns:
        Final scan result or error
    """
    print(f"\nStarting scan for: {url}")

    task_resp = client.scan(url, use_task=True)
    task = task_resp.get("task", {})
    task_id = task.get("taskId")
    if not task_id:
        raise McpToolError("No taskId returned from server")

    print(f"  Task ID: {task_id}")

    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        status = client.tasks_get(task_id)
        task_info = status.get("task", {})
        task_status = str(task_info.get("status", "")).lower()

        if task_status == "completed":
            result = client.tasks_result(task_id)
            return {
                "success": True,
                "url": url,
                "result": result.get("value"),
            }

        if task_status in ("failed", "cancelled"):
            # tasks_result will return a JSON-RPC error for terminal failures
            client.tasks_result(task_id)

        poll_ms = task_info.get("pollInterval", 2000)
        time.sleep(poll_ms / 1000)

    return {
        "success": False,
        "url": url,
        "error": f"Scan timed out after {timeout}s",
    }


def main():
    # Optional: set URLCHECK_API_KEY for higher volumes
    api_key = os.environ.get("URLCHECK_API_KEY", "")

    # Scan multiple URLs
    urls = [
        "https://example.com",
        "https://example.org",
    ]

    print("=" * 60)
    print("Scanning URLs")
    print("=" * 60)

    with UrlScannerMcpClient(
        base_url="https://urlcheck.ai/mcp",
        api_key=api_key,
        http_timeout_seconds=30,
    ) as client:
        results = []
        for url in urls:
            try:
                result = scan_url_with_progress(client, url)
                results.append(result)
            except McpToolError as e:
                print(f"  Tool error: {e}")
                results.append(
                    {
                        "success": False,
                        "url": url,
                        "error": str(e),
                    }
                )

    # Summary
    print("\n" + "=" * 60)
    print("Scan Results Summary")
    print("=" * 60)

    for result in results:
        url = result["url"]
        if result["success"]:
            r = result["result"]
            print(f"\n{url}")
            print(f"  Risk Score: {r.get('risk_score')}")
            print(f"  Confidence: {r.get('confidence')}")
        else:
            print(f"\n{url}")
            print(f"  Error: {result.get('error')}")


if __name__ == "__main__":
    main()

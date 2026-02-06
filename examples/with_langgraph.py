#!/usr/bin/env python3
"""
Example: SafeUrlScanTool in a simple LangGraph flow.

Requirements:
    pip install "langchain-urlcheck[langchain-full]"
"""

import json
import os
import sys
from typing import TypedDict


def main() -> None:
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        print("Error: Install LangGraph dependencies:")
        print('  pip install "langchain-urlcheck[langchain-full]"')
        sys.exit(1)

    from langchain_urlcheck import SafeUrlScanTool

    # Optional: set URLCHECK_API_KEY for higher volumes (free tier: 100 req/day)
    api_key = os.environ.get("URLCHECK_API_KEY", "")

    tool = SafeUrlScanTool(api_key=api_key)

    class UrlState(TypedDict):
        url: str
        scan_result: str

    def scan_node(state: UrlState) -> UrlState:
        result = tool.invoke({"url": state["url"]})
        return {"url": state["url"], "scan_result": result}

    graph = StateGraph(UrlState)
    graph.add_node("scan_url", scan_node)
    graph.set_entry_point("scan_url")
    graph.add_edge("scan_url", END)
    app = graph.compile()

    try:
        output = app.invoke({"url": "https://example.com", "scan_result": ""})
        print(json.dumps(json.loads(output["scan_result"]), indent=2))
    finally:
        tool.close()


if __name__ == "__main__":
    main()

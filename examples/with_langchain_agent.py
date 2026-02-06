#!/usr/bin/env python3
"""
Example: Using SafeUrlScanTool with a LangChain agent.

This example demonstrates how to integrate the URL scanner tool
into a LangChain agent that can analyze URLs when asked.

Requirements:
    pip install "langchain-urlcheck[langchain-full]"
"""

import os
import sys


def main():
    # Optional: set URLCHECK_API_KEY for higher volumes (free tier: 100 req/day)
    urlcheck_api_key = os.environ.get("URLCHECK_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Error: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    # Import LangChain components
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("Error: Install LangChain dependencies:")
        print('  pip install "langchain-urlcheck[langchain-full]"')
        sys.exit(1)

    from langchain_urlcheck import SafeUrlScanTool

    # Create the URL scanning tool (works without API key on free tier)
    url_scan_tool = SafeUrlScanTool(
        api_key=urlcheck_api_key,
        default_timeout_seconds=90,
    )

    # Create the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=openai_api_key,
    )

    # Create prompt template for the agent
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Always scan unknown URLs before answering."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create a LangChain agent with the tool
    agent = create_tool_calling_agent(llm, [url_scan_tool], prompt)
    executor = AgentExecutor(agent=agent, tools=[url_scan_tool], verbose=True)

    # Example queries
    queries = [
        "Is https://example.com safe to visit?",
        (
            "I received an email with a link to https://test.example.com/verify - "
            "should I click it?"
        ),
        "Check if https://example.org is a phishing site.",
    ]

    print("=" * 70)
    print("LangChain Agent with URL Security Scanner")
    print("=" * 70)

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print("=" * 70)

        try:
            response = executor.invoke({"input": query})
            print(f"\nAgent Response:\n{response['output']}")
        except Exception as e:
            print(f"\nError: {e}")

        print()


if __name__ == "__main__":
    main()

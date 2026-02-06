# langchain-urlcheck

LangChain integration that lets agents verify URL safety before navigation, powered by the [URLCheck](https://urlcheck.dev) MCP server.

`langchain-urlcheck` provides:
- `SafeUrlScanTool`: LangChain tool for URL safety checks
- `UrlScannerMcpClient`: low-level synchronous MCP JSON-RPC client
- `AsyncUrlScannerMcpClient`: low-level asynchronous MCP JSON-RPC client
- `get_mcp_server_config()`: helper for `langchain-mcp-adapters` compatibility

## What This Package Does

This package wraps URLCheck MCP tools so agents can evaluate links before navigation.
It supports:
- Task mode (default): submits a scan task, waits via `tasks/result`, falls back to polling
- Direct mode: single blocking `tools/call` request (bounded wait)
- Retry/backoff for transient failures
- Sync and async execution paths

Stateless streamable HTTP is supported. Stateful MCP session mode is not.

## Requirements

- Python 3.10 to 3.13
- URLCheck API key is optional (free tier: up to 100 requests/day without key; for higher volumes, contact contact@cybrlab.ai)

## Installation

```bash
pip install langchain-urlcheck
```

Optional extras:

```bash
# LangChain + OpenAI + LangGraph convenience set
pip install "langchain-urlcheck[langchain-full]"

# MCP adapter interoperability helper usage
pip install "langchain-urlcheck[mcp-adapters]"

# Local development and tests
pip install -e ".[dev]"
```

## Quick Start

### Basic Tool Usage (Free Tier)

```python
from langchain_urlcheck import SafeUrlScanTool

# No API key required for up to 100 requests/day
tool = SafeUrlScanTool()

result_json = tool.invoke({"url": "https://example.com"})
print(result_json)
```

### Scan with Intent

The `intent` parameter provides context about what the user intends to do at the URL,
enabling more targeted risk analysis via the `url_scanner_scan_with_intent` MCP tool:

```python
from langchain_urlcheck import SafeUrlScanTool

tool = SafeUrlScanTool()

# Intent helps the scanner assess context-specific risks
# (e.g., a login page is riskier for a "purchase" intent than a "read" intent)
result_json = tool.invoke({
    "url": "https://example.com/checkout",
    "intent": "purchase",
})
print(result_json)
```

### With API Key (Higher Volumes)

```python
from langchain_urlcheck import SafeUrlScanTool

tool = SafeUrlScanTool(api_key="your-api-key")

result_json = tool.invoke({"url": "https://example.com"})
print(result_json)
```

### LangChain Agent Usage

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_urlcheck import SafeUrlScanTool

tool = SafeUrlScanTool()  # or SafeUrlScanTool(api_key="your-api-key") for higher volumes
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Always scan unknown URLs before answering."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, [tool], prompt)
executor = AgentExecutor(agent=agent, tools=[tool], verbose=False)

response = executor.invoke({"input": "Is https://example.com safe to open?"})
print(response["output"])
```

### Async Tool Usage

```python
import asyncio
from langchain_urlcheck import SafeUrlScanTool

tool = SafeUrlScanTool()  # or SafeUrlScanTool(api_key="your-api-key")

async def main() -> None:
    result_json = await tool.ainvoke({"url": "https://example.com"})
    print(result_json)

asyncio.run(main())
```

## MCP Adapter Compatibility Path

If you already use `langchain-mcp-adapters`:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_urlcheck import get_mcp_server_config

client = MultiServerMCPClient(
    {
        "urlcheck": get_mcp_server_config(api_key="your-api-key"),
    }
)

tools = await client.get_tools()
```

Important:
- This path is for direct adapter interoperability.
- Managed task waiting/fallback guarantees are provided by `SafeUrlScanTool`.

## Output Contract

Successful tool response:

```json
{
  "risk_score": 0.15,
  "confidence": 0.92,
  "analysis_complete": true,
  "agent_access_directive": "ALLOW",
  "agent_access_reason": "clean"
}
```

Failure tool response:

```json
{
  "error": "JSON-RPC error [-32603]: Task ... timed out after 300 seconds",
  "numeric_code": -32603,
  "retryable": true
}
```

`agent_access_directive` values:
- `ALLOW`
- `DENY`
- `RETRY_LATER`
- `REQUIRE_CREDENTIALS`

## Configuration

### `SafeUrlScanTool` constructor options

| Parameter                       | Default                   | Description                             |
|---------------------------------|---------------------------|-----------------------------------------|
| `mcp_url`                       | `https://urlcheck.ai/mcp` | MCP endpoint                            |
| `api_key`                       | `""`                      | API key (optional for free tier)        |
| `default_timeout_seconds`       | `120`                     | End-to-end scan timeout budget          |
| `http_timeout_seconds`          | `30`                      | Per-request transport timeout           |
| `execution_mode`                | `"task"`                  | `"task"` or `"direct"`                  |
| `wait_mode`                     | `"tasks_result"`          | `"tasks_result"` or `"poll"`            |
| `task_ttl_ms`                   | `None`                    | Optional server task TTL                |
| `max_tasks_result_wait_seconds` | `300`                     | Max single wait call for `tasks/result` |
| `min_poll_interval_seconds`     | `2.0`                     | Poll floor                              |
| `max_poll_interval_seconds`     | `20.0`                    | Poll ceiling                            |
| `max_retries`                   | `2`                       | Retry attempts for transient errors     |
| `retry_base_delay_seconds`      | `0.5`                     | Retry base backoff                      |
| `retry_max_delay_seconds`       | `8.0`                     | Retry backoff cap                       |
| `cancel_on_timeout`             | `False`                   | Best-effort `tasks/cancel` on timeout   |
| `logger_name`                   | `None`                    | Optional lifecycle logger target        |

Invocation input schema (`tool.invoke` / `tool.ainvoke`):
- `url` (required)
- `intent` (optional, max 248 chars)
- `timeout_seconds` (optional, 30 to 720)

## Low-Level Client Usage

```python
from langchain_urlcheck import AsyncUrlScannerMcpClient, UrlScannerMcpClient

# Free tier (no key required)
client = UrlScannerMcpClient(base_url="https://urlcheck.ai/mcp")
result = client.scan("https://example.com", use_task=False)
client.close()

# Async with API key
async def run_async() -> None:
    async with AsyncUrlScannerMcpClient(
        base_url="https://urlcheck.ai/mcp",
        api_key="your-api-key",
    ) as async_client:
        result = await async_client.scan("https://example.com", use_task=False)
        print(result)
```

## Error Handling

```python
from langchain_urlcheck import (
    McpAuthenticationError,
    McpConnectionError,
    McpRateLimitError,
    SafeUrlScanTool,
)

tool = SafeUrlScanTool()  # or SafeUrlScanTool(api_key="your-api-key")

try:
    print(tool.invoke({"url": "https://example.com"}))
except McpAuthenticationError:
    print("Invalid API key")
except McpRateLimitError:
    print("Rate limited")
except McpConnectionError:
    print("Network/transport issue")
```

## Testing

Unit tests:

```bash
pytest tests/test_client.py tests/test_tool.py tests/test_adapter.py tests/test_tool_standard.py -v
```

## Security and Responsible Use

- Do not log API keys or auth headers.
- Run scans only on URLs/systems you are authorized to test.
- Follow applicable laws, policies, and platform terms.

## License

MIT. See `LICENSE`.

## Support

- [GitHub Issues](https://github.com/cybrlab-ai/langchain-urlcheck/issues)
- Email: contact@cybrlab.ai

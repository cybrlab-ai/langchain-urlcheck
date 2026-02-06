# Changelog

All notable changes to `langchain-urlcheck` will be documented in this file.

## [0.1.2] - 2026-02-06

### Added

- Standard `ToolsIntegrationTests` from `langchain-tests` for full LangChain integration compliance.

### Changed

- Restructured tests into `tests/unit_tests/` and `tests/integration_tests/` per LangChain conventions.
- CI now runs only unit tests explicitly (`tests/unit_tests/`).

## [0.1.1] - 2026-02-06

### Changed

- Migrated CI/CD to OIDC Trusted Publishers for PyPI publishing.
- Switched workflow to manual dispatch with branch guard on main.
- Fixed mypy strict type errors in client and tool modules.

## [0.1.0] - 2026-02-06

Initial public release.

### Added

- `SafeUrlScanTool` — LangChain `BaseTool` that lets agents verify URL safety before navigation, with task-augmented and direct execution modes.
- `create_url_scan_tool()` — factory function with common defaults.
- `SafeUrlScanTool.from_server_json()` — construct from MCP `server.json`.
- `UrlScannerMcpClient` — synchronous MCP JSON-RPC client over streamable HTTP.
- `AsyncUrlScannerMcpClient` — asynchronous MCP JSON-RPC client over streamable HTTP.
- `get_mcp_server_config()` — compatibility helper for `langchain-mcp-adapters`.
- Typed exception hierarchy: `McpClientError`, `McpConnectionError`, `McpAuthenticationError`, `McpRateLimitError`, `McpValidationError`, `McpToolError`.
- Task wait strategies: `tasks/result` with automatic fallback to `tasks/get` polling.
- Exponential backoff with jitter for transient failures (connection errors, HTTP 429, JSON-RPC -32029).
- Structured lifecycle events via LangChain callback manager and optional logger.
- Optional best-effort `tasks/cancel` on timeout.
- Python 3.10–3.13 support.
- CI/CD via GitHub Actions: lint, test (matrix), build, TestPyPI, PyPI, GitHub Release.
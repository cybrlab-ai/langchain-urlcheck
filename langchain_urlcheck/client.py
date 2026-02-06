"""
URLCheck MCP clients.

Low-level JSON-RPC clients for the URLCheck MCP server.
Stateless mode only (no sessions).

Protocol: MCP 2025-06-18 over Streamable HTTP
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import json
import random
import time
from collections.abc import Awaitable, Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx

from .exceptions import (
    McpAuthenticationError,
    McpClientError,
    McpConnectionError,
    McpRateLimitError,
    McpToolError,
    McpValidationError,
)

RetryCallback = Callable[[dict[str, Any]], None]
AsyncRetryCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


def _package_version() -> str:
    try:
        return version("langchain-urlcheck")
    except PackageNotFoundError:
        return "dev"


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


class _McpClientCommon:
    PROTOCOL_VERSION = "2025-06-18"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        protocol_version: str = PROTOCOL_VERSION,
        http_timeout_seconds: int = 30,
        max_retries: int = 2,
        retry_base_delay_seconds: float = 0.5,
        retry_max_delay_seconds: float = 8.0,
        user_agent: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.protocol_version = protocol_version
        self.http_timeout_seconds = http_timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_base_delay_seconds = max(0.0, retry_base_delay_seconds)
        self.retry_max_delay_seconds = max(self.retry_base_delay_seconds, retry_max_delay_seconds)
        self.user_agent = user_agent or f"langchain-urlcheck/{_package_version()}"
        self._request_id_counter = itertools.count(1)

    def _next_request_id(self) -> int:
        return next(self._request_id_counter)

    def _build_payload(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": method,
            "params": params or {},
        }

    @staticmethod
    def _retryable_exception(error: Exception) -> bool:
        if isinstance(error, McpRateLimitError):
            return True
        if isinstance(error, McpConnectionError):
            return error.retryable is not False
        return False

    def _next_retry_delay_seconds(
        self,
        attempt: int,
        error: Exception,
        deadline: float,
    ) -> float:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0:
            return 0.0

        exponential = self.retry_base_delay_seconds * (2**attempt)
        jitter = random.uniform(0.0, max(0.01, self.retry_base_delay_seconds))
        delay = min(exponential + jitter, self.retry_max_delay_seconds)

        retry_after_seconds = getattr(error, "retry_after_seconds", None)
        if retry_after_seconds is not None:
            delay = max(delay, float(retry_after_seconds))

        return min(delay, remaining)

    def _parse_jsonrpc_response(self, response: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(response, dict):
            raise McpClientError("Invalid JSON-RPC response type")

        if "error" in response:
            error = response["error"] if isinstance(response["error"], dict) else {}
            code = error.get("code")
            message = str(error.get("message", "Unknown error"))

            if code == -32029:
                raise McpRateLimitError(
                    f"Rate limit exceeded: {message}",
                    code=code,
                )

            if code == -32602 or (isinstance(code, int) and 2000 <= code <= 2099):
                raise McpValidationError(
                    f"Validation error: {message}",
                    code=code,
                )

            raise McpToolError(
                f"JSON-RPC error [{code}]: {message}",
                code=code if isinstance(code, int) else None,
            )

        result = response.get("result", {})
        if not isinstance(result, dict):
            raise McpClientError("Invalid JSON-RPC result payload")
        return result

    def _parse_tool_result(self, result: dict[str, Any]) -> dict[str, Any]:
        if not result:
            return {}

        if "task" in result:
            task = result.get("task")
            if isinstance(task, dict):
                return {"task": task}
            raise McpClientError("Unexpected task payload format")

        if result.get("isError") or result.get("is_error"):
            content = result.get("content", [])
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    error_text = first.get("text", "Unknown tool error")
                    if isinstance(error_text, str):
                        try:
                            error_json = json.loads(error_text)
                            message = str(error_json.get("message", error_text))
                            retryable = error_json.get("retryable")
                            raise McpToolError(
                                message,
                                retryable=bool(retryable) if isinstance(retryable, bool) else None,
                            )
                        except json.JSONDecodeError:
                            raise McpToolError(error_text) from None
            raise McpToolError("Tool execution failed")

        content = result.get("content", [])
        if not content:
            return {}

        first_content = content[0] if isinstance(content, list) else content
        if isinstance(first_content, dict) and first_content.get("type") == "text":
            text = first_content.get("text", "{}")
            if not isinstance(text, str):
                raise McpClientError("Unexpected tool response payload type")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise McpClientError(f"Failed to parse tool response: {exc}") from exc
            if not isinstance(parsed, dict):
                raise McpClientError("Tool response is not an object")
            return parsed

        raise McpClientError(f"Unexpected content format: {first_content}")


class UrlScannerMcpClient(_McpClientCommon):
    """Synchronous MCP JSON-RPC client for urlcheck.ai streamable-http transport."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        protocol_version: str = _McpClientCommon.PROTOCOL_VERSION,
        http_timeout_seconds: int = 30,
        max_retries: int = 2,
        retry_base_delay_seconds: float = 0.5,
        retry_max_delay_seconds: float = 8.0,
        user_agent: str | None = None,
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            protocol_version=protocol_version,
            http_timeout_seconds=http_timeout_seconds,
            max_retries=max_retries,
            retry_base_delay_seconds=retry_base_delay_seconds,
            retry_max_delay_seconds=retry_max_delay_seconds,
            user_agent=user_agent,
        )
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self.protocol_version,
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        self._http = httpx.Client(
            headers=headers,
            timeout=self.http_timeout_seconds,
            verify=True,
        )

    def _post_jsonrpc(
        self,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> httpx.Response:
        try:
            response = self._http.post(
                self.base_url,
                json=payload,
                headers=extra_headers,
                timeout=timeout_seconds or self.http_timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise McpConnectionError(
                f"Request timed out after {timeout_seconds or self.http_timeout_seconds:.2f}s",
                retryable=True,
            ) from exc
        except httpx.ConnectError as exc:
            raise McpConnectionError(f"Connection failed: {exc}", retryable=True) from exc
        except httpx.RequestError as exc:
            raise McpConnectionError(f"Request failed: {exc}", retryable=True) from exc

        if response.status_code == 401:
            raise McpAuthenticationError("Unauthorized", code=401)
        if response.status_code == 403:
            raise McpAuthenticationError("Access forbidden", code=403)
        if response.status_code == 429:
            raise McpRateLimitError(
                "Rate limited",
                code=429,
                retry_after_seconds=_parse_retry_after_seconds(response.headers.get("Retry-After")),
            )
        if response.status_code >= 500:
            raise McpConnectionError(
                f"HTTP {response.status_code}",
                code=response.status_code,
                retryable=True,
            )
        if response.status_code >= 400:
            raise McpConnectionError(
                f"HTTP {response.status_code}",
                code=response.status_code,
                retryable=False,
            )
        return response

    def call_method(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        retry_callback: RetryCallback | None = None,
    ) -> dict[str, Any]:
        payload = self._build_payload(method, params)
        budget_seconds = float(timeout_seconds or self.http_timeout_seconds)
        deadline = time.monotonic() + max(0.01, budget_seconds)
        attempt = 0

        while True:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                raise McpConnectionError(
                    f"Request timed out after {budget_seconds:.2f}s",
                    retryable=True,
                )

            try:
                response = self._post_jsonrpc(payload, timeout_seconds=remaining)
                try:
                    data = response.json()
                except ValueError as exc:
                    raise McpClientError(f"Invalid JSON response: {exc}") from exc
                return self._parse_jsonrpc_response(data)
            except (McpConnectionError, McpRateLimitError) as error:
                if attempt >= self.max_retries or not self._retryable_exception(error):
                    raise
                delay = self._next_retry_delay_seconds(attempt, error, deadline)
                if delay <= 0:
                    raise
                if retry_callback is not None:
                    retry_callback(
                        {
                            "method": method,
                            "attempt": attempt + 1,
                            "delay_seconds": delay,
                            "reason": type(error).__name__,
                        }
                    )
                time.sleep(delay)
                attempt += 1

    def tools_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        task: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        retry_callback: RetryCallback | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "name": tool_name,
            "arguments": arguments,
        }
        if task is not None:
            params["task"] = task

        result = self.call_method(
            "tools/call",
            params=params,
            timeout_seconds=timeout_seconds,
            retry_callback=retry_callback,
        )
        return self._parse_tool_result(result)

    def scan(
        self,
        url: str,
        intent: str | None = None,
        use_task: bool = False,
        task_ttl_ms: int | None = None,
        timeout_seconds: int | None = None,
        retry_callback: RetryCallback | None = None,
    ) -> dict[str, Any]:
        tool_name = "url_scanner_scan_with_intent" if intent else "url_scanner_scan"
        arguments: dict[str, Any] = {"url": url}
        if intent:
            arguments["intent"] = intent

        task_param: dict[str, Any] | None = None
        if use_task:
            task_param = {}
            if task_ttl_ms is not None:
                task_param["ttl"] = task_ttl_ms

        effective_timeout = timeout_seconds
        if not use_task:
            effective_timeout = min(timeout_seconds or self.http_timeout_seconds, 300)

        return self.tools_call(
            tool_name,
            arguments,
            task=task_param,
            timeout_seconds=effective_timeout,
            retry_callback=retry_callback,
        )

    def tasks_get(
        self,
        task_id: str,
        timeout_seconds: int | None = None,
        retry_callback: RetryCallback | None = None,
    ) -> dict[str, Any]:
        return self.call_method(
            "tasks/get",
            {"taskId": task_id},
            timeout_seconds=timeout_seconds,
            retry_callback=retry_callback,
        )

    def tasks_result(
        self,
        task_id: str,
        timeout_seconds: int | None = None,
        retry_callback: RetryCallback | None = None,
    ) -> dict[str, Any]:
        return self.call_method(
            "tasks/result",
            {"taskId": task_id},
            timeout_seconds=timeout_seconds,
            retry_callback=retry_callback,
        )

    def tasks_cancel(
        self,
        task_id: str,
        timeout_seconds: int | None = None,
        retry_callback: RetryCallback | None = None,
    ) -> dict[str, Any]:
        return self.call_method(
            "tasks/cancel",
            {"taskId": task_id},
            timeout_seconds=timeout_seconds,
            retry_callback=retry_callback,
        )

    def tasks_list(
        self,
        timeout_seconds: int | None = None,
        retry_callback: RetryCallback | None = None,
    ) -> dict[str, Any]:
        return self.call_method(
            "tasks/list",
            {},
            timeout_seconds=timeout_seconds,
            retry_callback=retry_callback,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> UrlScannerMcpClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class AsyncUrlScannerMcpClient(_McpClientCommon):
    """Asynchronous MCP JSON-RPC client for urlcheck.ai streamable-http transport."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        protocol_version: str = _McpClientCommon.PROTOCOL_VERSION,
        http_timeout_seconds: int = 30,
        max_retries: int = 2,
        retry_base_delay_seconds: float = 0.5,
        retry_max_delay_seconds: float = 8.0,
        user_agent: str | None = None,
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            protocol_version=protocol_version,
            http_timeout_seconds=http_timeout_seconds,
            max_retries=max_retries,
            retry_base_delay_seconds=retry_base_delay_seconds,
            retry_max_delay_seconds=retry_max_delay_seconds,
            user_agent=user_agent,
        )
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self.protocol_version,
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        self._http = httpx.AsyncClient(
            headers=headers,
            timeout=self.http_timeout_seconds,
            verify=True,
        )

    async def _post_jsonrpc(
        self,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> httpx.Response:
        try:
            response = await self._http.post(
                self.base_url,
                json=payload,
                headers=extra_headers,
                timeout=timeout_seconds or self.http_timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise McpConnectionError(
                f"Request timed out after {timeout_seconds or self.http_timeout_seconds:.2f}s",
                retryable=True,
            ) from exc
        except httpx.ConnectError as exc:
            raise McpConnectionError(f"Connection failed: {exc}", retryable=True) from exc
        except httpx.RequestError as exc:
            raise McpConnectionError(f"Request failed: {exc}", retryable=True) from exc

        if response.status_code == 401:
            raise McpAuthenticationError("Unauthorized", code=401)
        if response.status_code == 403:
            raise McpAuthenticationError("Access forbidden", code=403)
        if response.status_code == 429:
            raise McpRateLimitError(
                "Rate limited",
                code=429,
                retry_after_seconds=_parse_retry_after_seconds(response.headers.get("Retry-After")),
            )
        if response.status_code >= 500:
            raise McpConnectionError(
                f"HTTP {response.status_code}",
                code=response.status_code,
                retryable=True,
            )
        if response.status_code >= 400:
            raise McpConnectionError(
                f"HTTP {response.status_code}",
                code=response.status_code,
                retryable=False,
            )
        return response

    async def call_method(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        retry_callback: AsyncRetryCallback | None = None,
    ) -> dict[str, Any]:
        payload = self._build_payload(method, params)
        budget_seconds = float(timeout_seconds or self.http_timeout_seconds)
        deadline = time.monotonic() + max(0.01, budget_seconds)
        attempt = 0

        while True:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                raise McpConnectionError(
                    f"Request timed out after {budget_seconds:.2f}s",
                    retryable=True,
                )

            try:
                response = await self._post_jsonrpc(payload, timeout_seconds=remaining)
                try:
                    data = response.json()
                except ValueError as exc:
                    raise McpClientError(f"Invalid JSON response: {exc}") from exc
                return self._parse_jsonrpc_response(data)
            except (McpConnectionError, McpRateLimitError) as error:
                if attempt >= self.max_retries or not self._retryable_exception(error):
                    raise
                delay = self._next_retry_delay_seconds(attempt, error, deadline)
                if delay <= 0:
                    raise
                if retry_callback is not None:
                    maybe_awaitable = retry_callback(
                        {
                            "method": method,
                            "attempt": attempt + 1,
                            "delay_seconds": delay,
                            "reason": type(error).__name__,
                        }
                    )
                    if inspect.isawaitable(maybe_awaitable):
                        await maybe_awaitable
                await asyncio.sleep(delay)
                attempt += 1

    async def tools_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        task: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        retry_callback: AsyncRetryCallback | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "name": tool_name,
            "arguments": arguments,
        }
        if task is not None:
            params["task"] = task

        result = await self.call_method(
            "tools/call",
            params=params,
            timeout_seconds=timeout_seconds,
            retry_callback=retry_callback,
        )
        return self._parse_tool_result(result)

    async def scan(
        self,
        url: str,
        intent: str | None = None,
        use_task: bool = False,
        task_ttl_ms: int | None = None,
        timeout_seconds: int | None = None,
        retry_callback: AsyncRetryCallback | None = None,
    ) -> dict[str, Any]:
        tool_name = "url_scanner_scan_with_intent" if intent else "url_scanner_scan"
        arguments: dict[str, Any] = {"url": url}
        if intent:
            arguments["intent"] = intent

        task_param: dict[str, Any] | None = None
        if use_task:
            task_param = {}
            if task_ttl_ms is not None:
                task_param["ttl"] = task_ttl_ms

        effective_timeout = timeout_seconds
        if not use_task:
            effective_timeout = min(timeout_seconds or self.http_timeout_seconds, 300)

        return await self.tools_call(
            tool_name,
            arguments,
            task=task_param,
            timeout_seconds=effective_timeout,
            retry_callback=retry_callback,
        )

    async def tasks_get(
        self,
        task_id: str,
        timeout_seconds: int | None = None,
        retry_callback: AsyncRetryCallback | None = None,
    ) -> dict[str, Any]:
        return await self.call_method(
            "tasks/get",
            {"taskId": task_id},
            timeout_seconds=timeout_seconds,
            retry_callback=retry_callback,
        )

    async def tasks_result(
        self,
        task_id: str,
        timeout_seconds: int | None = None,
        retry_callback: AsyncRetryCallback | None = None,
    ) -> dict[str, Any]:
        return await self.call_method(
            "tasks/result",
            {"taskId": task_id},
            timeout_seconds=timeout_seconds,
            retry_callback=retry_callback,
        )

    async def tasks_cancel(
        self,
        task_id: str,
        timeout_seconds: int | None = None,
        retry_callback: AsyncRetryCallback | None = None,
    ) -> dict[str, Any]:
        return await self.call_method(
            "tasks/cancel",
            {"taskId": task_id},
            timeout_seconds=timeout_seconds,
            retry_callback=retry_callback,
        )

    async def tasks_list(
        self,
        timeout_seconds: int | None = None,
        retry_callback: AsyncRetryCallback | None = None,
    ) -> dict[str, Any]:
        return await self.call_method(
            "tasks/list",
            {},
            timeout_seconds=timeout_seconds,
            retry_callback=retry_callback,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncUrlScannerMcpClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

"""
LangChain tool that lets agents verify URL safety before navigation.

Provides a LangChain tool that supports both direct and task-augmented MCP scans,
with sync and async execution paths.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

try:
    from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
    from langchain_core.tools import BaseTool
except ImportError:
    try:
        from langchain.callbacks.manager import (
            AsyncCallbackManagerForToolRun,
            CallbackManagerForToolRun,
        )
        from langchain.tools import BaseTool
    except ImportError as e:
        raise ImportError("LangChain not found. Install with: pip install langchain-core") from e

from .client import AsyncUrlScannerMcpClient, UrlScannerMcpClient
from .exceptions import (
    McpClientError,
    McpRateLimitError,
    McpToolError,
    McpValidationError,
)


class SafeUrlScanInput(BaseModel):
    """Input schema for the SafeUrlScanTool."""

    url: str = Field(
        ...,
        description="The URL to scan for malicious content before navigation.",
        examples=["https://example.com", "https://example.org/login"],
    )
    intent: str | None = Field(
        default=None,
        description="Optional user intent context (e.g., login, purchase, booking).",
        max_length=248,
    )
    timeout_seconds: int = Field(
        default=120,
        description=(
            "Maximum time to wait for scan completion (queue + scan time). "
            "Direct calls are capped at 300 seconds."
        ),
        ge=30,
        le=720,
    )

    model_config = ConfigDict(extra="forbid")


class ScanResult(BaseModel):
    """Successful scan result returned to the agent."""

    risk_score: float = Field(description="Risk score from 0.0 (safe) to 1.0 (dangerous)")
    confidence: float = Field(description="Confidence in the score (0.0-1.0)")
    analysis_complete: bool = Field(description="Whether full analysis completed")
    agent_access_directive: str = Field(
        description="ALLOW, DENY, RETRY_LATER, or REQUIRE_CREDENTIALS guidance for agents"
    )
    agent_access_reason: str = Field(description="Reason for the access directive")

    def to_agent_response(self) -> str:
        return json.dumps(self.model_dump(), indent=2)


class ScanFailure(BaseModel):
    """Failed scan result with MCP error context."""

    error: str = Field(description="Human-readable error message")
    numeric_code: int | None = Field(default=None, description="Numeric JSON-RPC error code")
    retryable: bool | None = Field(default=None, description="Whether caller may retry this error")

    def to_agent_response(self) -> str:
        return json.dumps(self.model_dump(exclude_none=True), indent=2)


class SafeUrlScanTool(BaseTool):
    """
    LangChain tool that lets agents verify URL safety before navigation.

    Managed task mode is recommended and enabled by default.
    """

    name: str = "safe_url_scan"
    description: str = (
        "Scans a URL for malicious content (phishing, malware) before navigation. "
        "Returns risk score (0-1), confidence, and agent access guidance."
    )
    args_schema: type[BaseModel] = SafeUrlScanInput

    mcp_url: str = Field(default="https://urlcheck.ai/mcp", exclude=True)
    api_key: str = Field(default="", exclude=True)
    default_timeout_seconds: int = Field(default=120, exclude=True)
    http_timeout_seconds: int = Field(default=30, exclude=True)
    min_poll_interval_seconds: float = Field(default=2.0, exclude=True)
    max_poll_interval_seconds: float = Field(default=20.0, exclude=True)
    execution_mode: Literal["task", "direct"] = Field(default="task", exclude=True)
    wait_mode: Literal["poll", "tasks_result"] = Field(default="tasks_result", exclude=True)
    task_ttl_ms: int | None = Field(default=None, exclude=True)
    max_tasks_result_wait_seconds: int = Field(default=300, exclude=True)
    cancel_on_timeout: bool = Field(default=False, exclude=True)
    max_retries: int = Field(default=2, exclude=True)
    retry_base_delay_seconds: float = Field(default=0.5, exclude=True)
    retry_max_delay_seconds: float = Field(default=8.0, exclude=True)
    logger_name: str | None = Field(default=None, exclude=True)

    _client: UrlScannerMcpClient | None = PrivateAttr(default=None)
    _logger: logging.Logger | None = PrivateAttr(default=None)

    def __init__(
        self,
        mcp_url: str = "https://urlcheck.ai/mcp",
        api_key: str = "",
        default_timeout_seconds: int = 120,
        http_timeout_seconds: int = 30,
        min_poll_interval_seconds: float = 2.0,
        max_poll_interval_seconds: float = 20.0,
        execution_mode: Literal["task", "direct"] = "task",
        wait_mode: Literal["poll", "tasks_result"] = "tasks_result",
        task_ttl_ms: int | None = None,
        max_tasks_result_wait_seconds: int = 300,
        cancel_on_timeout: bool = False,
        max_retries: int = 2,
        retry_base_delay_seconds: float = 0.5,
        retry_max_delay_seconds: float = 8.0,
        logger_name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            mcp_url=mcp_url,
            api_key=api_key,
            default_timeout_seconds=default_timeout_seconds,
            http_timeout_seconds=http_timeout_seconds,
            min_poll_interval_seconds=min_poll_interval_seconds,
            max_poll_interval_seconds=max_poll_interval_seconds,
            execution_mode=execution_mode,
            wait_mode=wait_mode,
            task_ttl_ms=task_ttl_ms,
            max_tasks_result_wait_seconds=max_tasks_result_wait_seconds,
            cancel_on_timeout=cancel_on_timeout,
            max_retries=max_retries,
            retry_base_delay_seconds=retry_base_delay_seconds,
            retry_max_delay_seconds=retry_max_delay_seconds,
            logger_name=logger_name,
            **kwargs,
        )
        self._client = self._new_sync_client()
        self._logger = logging.getLogger(logger_name) if logger_name else None

    @classmethod
    def from_server_json(
        cls,
        server_json_path: str | Path,
        api_key: str = "",
        **kwargs: Any,
    ) -> SafeUrlScanTool:
        """Create a tool from MCP server.json defaults."""
        path = Path(server_json_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        remotes = data.get("remotes", [])

        mcp_url = "https://urlcheck.ai/mcp"
        if isinstance(remotes, list):
            for remote in remotes:
                if isinstance(remote, dict) and remote.get("type") == "streamable-http":
                    url = remote.get("url")
                    if isinstance(url, str) and url:
                        mcp_url = url
                        break

        return cls(
            mcp_url=mcp_url,
            api_key=api_key,
            **kwargs,
        )

    def _new_sync_client(self) -> UrlScannerMcpClient:
        return UrlScannerMcpClient(
            base_url=self.mcp_url,
            api_key=self.api_key,
            http_timeout_seconds=self.http_timeout_seconds,
            max_retries=self.max_retries,
            retry_base_delay_seconds=self.retry_base_delay_seconds,
            retry_max_delay_seconds=self.retry_max_delay_seconds,
        )

    def _new_async_client(self) -> AsyncUrlScannerMcpClient:
        return AsyncUrlScannerMcpClient(
            base_url=self.mcp_url,
            api_key=self.api_key,
            http_timeout_seconds=self.http_timeout_seconds,
            max_retries=self.max_retries,
            retry_base_delay_seconds=self.retry_base_delay_seconds,
            retry_max_delay_seconds=self.retry_max_delay_seconds,
        )

    @property
    def client(self) -> UrlScannerMcpClient:
        if self._client is None:
            self._client = self._new_sync_client()
        return self._client

    @staticmethod
    def _hash_url(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _calculate_poll_interval_static(
        server_recommended_ms: int | None,
        min_poll_interval_seconds: float,
        max_poll_interval_seconds: float,
    ) -> float:
        if server_recommended_ms is None or server_recommended_ms <= 0:
            return min_poll_interval_seconds
        server_secs = server_recommended_ms / 1000.0
        return max(min_poll_interval_seconds, min(server_secs, max_poll_interval_seconds))

    def _calculate_poll_interval(self, server_recommended_ms: int | None) -> float:
        return self._calculate_poll_interval_static(
            server_recommended_ms=server_recommended_ms,
            min_poll_interval_seconds=self.min_poll_interval_seconds,
            max_poll_interval_seconds=self.max_poll_interval_seconds,
        )

    def _emit_event(
        self,
        run_manager: CallbackManagerForToolRun | None,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        message = json.dumps({"event": event, **payload}, sort_keys=True)
        if self._logger is not None:
            self._logger.info(message)
        if run_manager is None:
            return
        try:
            run_manager.on_text(message)
        except Exception:
            return

    async def _emit_event_async(
        self,
        run_manager: AsyncCallbackManagerForToolRun | None,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        message = json.dumps({"event": event, **payload}, sort_keys=True)
        if self._logger is not None:
            self._logger.info(message)
        if run_manager is None:
            return
        try:
            maybe_awaitable = run_manager.on_text(message)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        except Exception:
            return

    def _sync_retry_callback(
        self,
        run_manager: CallbackManagerForToolRun | None,
    ) -> Any:
        def _callback(event: dict[str, Any]) -> None:
            self._emit_event(
                run_manager,
                "retry_scheduled",
                {
                    "reason": event.get("reason"),
                    "delay_seconds": round(float(event.get("delay_seconds", 0.0)), 3),
                    "attempt": event.get("attempt"),
                    "method": event.get("method"),
                },
            )

        return _callback

    def _async_retry_callback(
        self,
        run_manager: AsyncCallbackManagerForToolRun | None,
    ) -> Any:
        async def _callback(event: dict[str, Any]) -> None:
            await self._emit_event_async(
                run_manager,
                "retry_scheduled",
                {
                    "reason": event.get("reason"),
                    "delay_seconds": round(float(event.get("delay_seconds", 0.0)), 3),
                    "attempt": event.get("attempt"),
                    "method": event.get("method"),
                },
            )

        return _callback

    def _format_success(self, result: dict[str, Any]) -> str:
        risk_score = result.get("risk_score")
        confidence = result.get("confidence")
        analysis_complete = result.get("analysis_complete")
        agent_access_directive = result.get("agent_access_directive")
        agent_access_reason = result.get("agent_access_reason")

        missing = [
            name
            for name, value in [
                ("risk_score", risk_score),
                ("confidence", confidence),
                ("analysis_complete", analysis_complete),
                ("agent_access_directive", agent_access_directive),
                ("agent_access_reason", agent_access_reason),
            ]
            if value is None
        ]
        if missing:
            raise McpClientError(f"Missing fields in scan result: {', '.join(missing)}")

        return ScanResult(
            risk_score=float(risk_score),
            confidence=float(confidence),
            analysis_complete=bool(analysis_complete),
            agent_access_directive=str(agent_access_directive),
            agent_access_reason=str(agent_access_reason),
        ).to_agent_response()

    def _format_failure(
        self,
        error: str,
        numeric_code: int | None = None,
        retryable: bool | None = None,
    ) -> str:
        return ScanFailure(
            error=error,
            numeric_code=numeric_code,
            retryable=retryable,
        ).to_agent_response()

    def _extract_task_result(self, task_result: dict[str, Any]) -> dict[str, Any]:
        value = task_result.get("value")
        if isinstance(value, dict):
            return value
        raise McpClientError("Unexpected tasks/result payload (missing value)")

    @staticmethod
    def _is_tasks_result_wait_timeout(error: McpToolError) -> bool:
        if error.code != -32603:
            return False
        message = str(error).lower()
        return "task" in message and "timed out" in message

    def _remaining_timeout(self, deadline: float, max_wait_seconds: int | None = None) -> int:
        remaining = max(0.0, deadline - time.monotonic())
        if max_wait_seconds is not None:
            remaining = min(remaining, float(max_wait_seconds))
        return max(1, int(remaining))

    def _cancel_task_if_needed(
        self,
        task_id: str,
        deadline: float,
        run_manager: CallbackManagerForToolRun | None,
        retry_callback: Any,
    ) -> None:
        if not self.cancel_on_timeout:
            return
        try:
            self.client.tasks_cancel(
                task_id,
                timeout_seconds=self._remaining_timeout(deadline),
                retry_callback=retry_callback,
            )
            self._emit_event(run_manager, "task_cancel_requested", {"task_id": task_id})
        except Exception:
            self._emit_event(run_manager, "task_cancel_failed", {"task_id": task_id})

    async def _cancel_task_if_needed_async(
        self,
        client: AsyncUrlScannerMcpClient,
        task_id: str,
        deadline: float,
        run_manager: AsyncCallbackManagerForToolRun | None,
        retry_callback: Any,
    ) -> None:
        if not self.cancel_on_timeout:
            return
        try:
            await client.tasks_cancel(
                task_id,
                timeout_seconds=self._remaining_timeout(deadline),
                retry_callback=retry_callback,
            )
            await self._emit_event_async(run_manager, "task_cancel_requested", {"task_id": task_id})
        except Exception:
            await self._emit_event_async(run_manager, "task_cancel_failed", {"task_id": task_id})

    def _wait_for_task_result(
        self,
        task_id: str,
        deadline: float,
        run_manager: CallbackManagerForToolRun | None,
        retry_callback: Any,
    ) -> dict[str, Any]:
        self._emit_event(run_manager, "task_wait_strategy", {"strategy": "tasks_result"})
        try:
            wait_timeout = self._remaining_timeout(deadline, self.max_tasks_result_wait_seconds)
            result = self.client.tasks_result(
                task_id,
                timeout_seconds=wait_timeout,
                retry_callback=retry_callback,
            )
            return self._extract_task_result(result)
        except McpToolError as error:
            if self._is_tasks_result_wait_timeout(error):
                self._emit_event(
                    run_manager,
                    "task_wait_strategy",
                    {"strategy": "poll", "reason": "tasks_result_timeout"},
                )
                return self._poll_for_task_result(task_id, deadline, run_manager, retry_callback)
            raise
        except McpClientError as error:
            if "timed out" in str(error).lower():
                self._emit_event(
                    run_manager,
                    "task_wait_strategy",
                    {"strategy": "poll", "reason": "client_timeout"},
                )
                return self._poll_for_task_result(task_id, deadline, run_manager, retry_callback)
            raise

    async def _wait_for_task_result_async(
        self,
        client: AsyncUrlScannerMcpClient,
        task_id: str,
        deadline: float,
        run_manager: AsyncCallbackManagerForToolRun | None,
        retry_callback: Any,
    ) -> dict[str, Any]:
        await self._emit_event_async(
            run_manager,
            "task_wait_strategy",
            {"strategy": "tasks_result"},
        )
        try:
            wait_timeout = self._remaining_timeout(deadline, self.max_tasks_result_wait_seconds)
            result = await client.tasks_result(
                task_id,
                timeout_seconds=wait_timeout,
                retry_callback=retry_callback,
            )
            return self._extract_task_result(result)
        except McpToolError as error:
            if self._is_tasks_result_wait_timeout(error):
                await self._emit_event_async(
                    run_manager,
                    "task_wait_strategy",
                    {"strategy": "poll", "reason": "tasks_result_timeout"},
                )
                return await self._poll_for_task_result_async(
                    client, task_id, deadline, run_manager, retry_callback
                )
            raise
        except McpClientError as error:
            if "timed out" in str(error).lower():
                await self._emit_event_async(
                    run_manager,
                    "task_wait_strategy",
                    {"strategy": "poll", "reason": "client_timeout"},
                )
                return await self._poll_for_task_result_async(
                    client, task_id, deadline, run_manager, retry_callback
                )
            raise

    def _poll_for_task_result(
        self,
        task_id: str,
        deadline: float,
        run_manager: CallbackManagerForToolRun | None,
        retry_callback: Any,
    ) -> dict[str, Any]:
        last_task: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            task_info = self.client.tasks_get(
                task_id,
                timeout_seconds=self._remaining_timeout(deadline),
                retry_callback=retry_callback,
            )
            task = task_info.get("task") if isinstance(task_info, dict) else None
            if not isinstance(task, dict):
                raise McpClientError("Unexpected tasks/get response (missing task)")
            last_task = task

            status = str(task.get("status", "")).lower()
            if status == "completed":
                wait_timeout = self._remaining_timeout(deadline, self.max_tasks_result_wait_seconds)
                result = self.client.tasks_result(
                    task_id,
                    timeout_seconds=wait_timeout,
                    retry_callback=retry_callback,
                )
                return self._extract_task_result(result)

            if status in ("failed", "cancelled"):
                wait_timeout = self._remaining_timeout(deadline, self.max_tasks_result_wait_seconds)
                self.client.tasks_result(
                    task_id,
                    timeout_seconds=wait_timeout,
                    retry_callback=retry_callback,
                )
                raise McpToolError(f"Task {task_id} ended with status {status}")

            poll_interval_ms = task.get("pollInterval") or task.get("poll_interval_ms")
            time.sleep(self._calculate_poll_interval(poll_interval_ms))

        last_status = last_task.get("status") if last_task else None
        self._cancel_task_if_needed(task_id, deadline, run_manager, retry_callback)
        raise McpClientError(
            f"Client timed out waiting for task (last_status={last_status})",
            retryable=True,
        )

    async def _poll_for_task_result_async(
        self,
        client: AsyncUrlScannerMcpClient,
        task_id: str,
        deadline: float,
        run_manager: AsyncCallbackManagerForToolRun | None,
        retry_callback: Any,
    ) -> dict[str, Any]:
        last_task: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            task_info = await client.tasks_get(
                task_id,
                timeout_seconds=self._remaining_timeout(deadline),
                retry_callback=retry_callback,
            )
            task = task_info.get("task") if isinstance(task_info, dict) else None
            if not isinstance(task, dict):
                raise McpClientError("Unexpected tasks/get response (missing task)")
            last_task = task

            status = str(task.get("status", "")).lower()
            if status == "completed":
                wait_timeout = self._remaining_timeout(deadline, self.max_tasks_result_wait_seconds)
                result = await client.tasks_result(
                    task_id,
                    timeout_seconds=wait_timeout,
                    retry_callback=retry_callback,
                )
                return self._extract_task_result(result)

            if status in ("failed", "cancelled"):
                wait_timeout = self._remaining_timeout(deadline, self.max_tasks_result_wait_seconds)
                await client.tasks_result(
                    task_id,
                    timeout_seconds=wait_timeout,
                    retry_callback=retry_callback,
                )
                raise McpToolError(f"Task {task_id} ended with status {status}")

            poll_interval_ms = task.get("pollInterval") or task.get("poll_interval_ms")
            await asyncio.sleep(self._calculate_poll_interval(poll_interval_ms))

        last_status = last_task.get("status") if last_task else None
        await self._cancel_task_if_needed_async(
            client,
            task_id,
            deadline,
            run_manager,
            retry_callback,
        )
        raise McpClientError(
            f"Client timed out waiting for task (last_status={last_status})",
            retryable=True,
        )

    def _run(
        self,
        url: str,
        intent: str | None = None,
        timeout_seconds: int = 120,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        timeout = timeout_seconds or self.default_timeout_seconds
        start_time = time.monotonic()
        deadline = start_time + timeout
        retry_callback = self._sync_retry_callback(run_manager)

        self._emit_event(
            run_manager,
            "scan_started",
            {
                "mode": self.execution_mode,
                "wait_mode": self.wait_mode,
                "url_hash": self._hash_url(url),
                "timeout_seconds": timeout,
            },
        )

        try:
            if self.execution_mode == "direct":
                direct_timeout = min(timeout, 300)
                result = self.client.scan(
                    url,
                    intent=intent,
                    use_task=False,
                    timeout_seconds=direct_timeout,
                    retry_callback=retry_callback,
                )
                response = self._format_success(result)
            else:
                start_resp = self.client.scan(
                    url,
                    intent=intent,
                    use_task=True,
                    task_ttl_ms=self.task_ttl_ms,
                    timeout_seconds=self._remaining_timeout(deadline),
                    retry_callback=retry_callback,
                )
                task = start_resp.get("task") if isinstance(start_resp, dict) else None
                task_id = task.get("taskId") if isinstance(task, dict) else None
                if not task_id:
                    self._emit_event(
                        run_manager,
                        "scan_failed",
                        {"category": "task_submission", "code": None},
                    )
                    return self._format_failure("Failed to start scan - no taskId returned")

                self._emit_event(run_manager, "task_submitted", {"task_id": task_id})
                if self.wait_mode == "tasks_result":
                    result = self._wait_for_task_result(
                        task_id, deadline, run_manager, retry_callback
                    )
                else:
                    self._emit_event(run_manager, "task_wait_strategy", {"strategy": "poll"})
                    result = self._poll_for_task_result(
                        task_id, deadline, run_manager, retry_callback
                    )
                response = self._format_success(result)

            latency_ms = int((time.monotonic() - start_time) * 1000)
            directive = json.loads(response).get("agent_access_directive")
            self._emit_event(
                run_manager,
                "scan_completed",
                {"directive": directive, "latency_ms": latency_ms},
            )
            return response

        except (McpRateLimitError, McpValidationError, McpToolError, McpClientError) as error:
            self._emit_event(
                run_manager,
                "scan_failed",
                {
                    "category": type(error).__name__,
                    "code": error.code,
                    "retryable": error.retryable,
                },
            )
            return self._format_failure(
                error=str(error),
                numeric_code=error.code,
                retryable=error.retryable,
            )
        except Exception as error:
            self._emit_event(
                run_manager,
                "scan_failed",
                {"category": "UnexpectedError", "code": None},
            )
            return self._format_failure(
                error=f"Unexpected error: {error}",
                retryable=False,
            )

    async def _arun(
        self,
        url: str,
        intent: str | None = None,
        timeout_seconds: int = 120,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        timeout = timeout_seconds or self.default_timeout_seconds
        start_time = time.monotonic()
        deadline = start_time + timeout
        retry_callback = self._async_retry_callback(run_manager)

        await self._emit_event_async(
            run_manager,
            "scan_started",
            {
                "mode": self.execution_mode,
                "wait_mode": self.wait_mode,
                "url_hash": self._hash_url(url),
                "timeout_seconds": timeout,
            },
        )

        try:
            async with self._new_async_client() as client:
                if self.execution_mode == "direct":
                    direct_timeout = min(timeout, 300)
                    result = await client.scan(
                        url,
                        intent=intent,
                        use_task=False,
                        timeout_seconds=direct_timeout,
                        retry_callback=retry_callback,
                    )
                    response = self._format_success(result)
                else:
                    start_resp = await client.scan(
                        url,
                        intent=intent,
                        use_task=True,
                        task_ttl_ms=self.task_ttl_ms,
                        timeout_seconds=self._remaining_timeout(deadline),
                        retry_callback=retry_callback,
                    )
                    task = start_resp.get("task") if isinstance(start_resp, dict) else None
                    task_id = task.get("taskId") if isinstance(task, dict) else None
                    if not task_id:
                        await self._emit_event_async(
                            run_manager,
                            "scan_failed",
                            {"category": "task_submission", "code": None},
                        )
                        return self._format_failure("Failed to start scan - no taskId returned")

                    await self._emit_event_async(
                        run_manager, "task_submitted", {"task_id": task_id}
                    )
                    if self.wait_mode == "tasks_result":
                        result = await self._wait_for_task_result_async(
                            client, task_id, deadline, run_manager, retry_callback
                        )
                    else:
                        await self._emit_event_async(
                            run_manager, "task_wait_strategy", {"strategy": "poll"}
                        )
                        result = await self._poll_for_task_result_async(
                            client, task_id, deadline, run_manager, retry_callback
                        )
                    response = self._format_success(result)

            latency_ms = int((time.monotonic() - start_time) * 1000)
            directive = json.loads(response).get("agent_access_directive")
            await self._emit_event_async(
                run_manager,
                "scan_completed",
                {"directive": directive, "latency_ms": latency_ms},
            )
            return response

        except (McpRateLimitError, McpValidationError, McpToolError, McpClientError) as error:
            await self._emit_event_async(
                run_manager,
                "scan_failed",
                {
                    "category": type(error).__name__,
                    "code": error.code,
                    "retryable": error.retryable,
                },
            )
            return self._format_failure(
                error=str(error),
                numeric_code=error.code,
                retryable=error.retryable,
            )
        except Exception as error:
            await self._emit_event_async(
                run_manager,
                "scan_failed",
                {"category": "UnexpectedError", "code": None},
            )
            return self._format_failure(
                error=f"Unexpected error: {error}",
                retryable=False,
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        self.close()


def create_url_scan_tool(
    api_key: str = "",
    mcp_url: str = "https://urlcheck.ai/mcp",
    timeout_seconds: int = 120,
    execution_mode: Literal["task", "direct"] = "task",
    wait_mode: Literal["poll", "tasks_result"] = "tasks_result",
    task_ttl_ms: int | None = None,
    cancel_on_timeout: bool = False,
) -> SafeUrlScanTool:
    """Factory function to create a SafeUrlScanTool with common defaults."""
    return SafeUrlScanTool(
        mcp_url=mcp_url,
        api_key=api_key,
        default_timeout_seconds=timeout_seconds,
        execution_mode=execution_mode,
        wait_mode=wait_mode,
        task_ttl_ms=task_ttl_ms,
        cancel_on_timeout=cancel_on_timeout,
    )

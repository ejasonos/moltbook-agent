from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import requests
from urllib.parse import urlparse

from .messages import Message, ModelResponse
from .parser import ModelParser
from utils.logging import log_event, log_payloads_enabled


class NVIDIAClient:
    """
    Client for NVIDIA's OpenAI-compatible inference API.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: int = 600,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not configured."
            )

    def chat(
        self,
        messages: list[Message | dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> ModelResponse:

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._normalize_messages(messages),
            "temperature": temperature,
            "stream": False,
        }

        if tools:
            payload["tools"] = tools
            payload["parallel_tool_calls"] = False

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        request_id = uuid.uuid4().hex
        endpoint = f"{self.base_url}/chat/completions"
        started = time.perf_counter()
        log_event(
            "model.request.sent",
            provider="nvidia",
            request_id=request_id,
            model=self.model,
            endpoint=endpoint,
            message_count=len(payload["messages"]),
            tool_count=len(tools or []),
        )
        if log_payloads_enabled():
            log_event(
                "model.request.payload",
                level=logging.DEBUG,
                provider="nvidia",
                request_id=request_id,
                payload=payload,
            )

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            log_event(
                "model.response.received",
                provider="nvidia",
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            if log_payloads_enabled():
                log_event(
                    "model.response.payload",
                    level=logging.DEBUG,
                    provider="nvidia",
                    request_id=request_id,
                    payload=response.text,
                )
            response.raise_for_status()
            data = response.json()
            parsed_response = self._parse_response(data)
            parsed_response.usage["request_duration_ms"] = duration_ms
            log_event(
                "model.response.parsed",
                provider="nvidia",
                request_id=request_id,
                content_length=len(parsed_response.content or ""),
                tool_calls=[call.tool_name for call in parsed_response.tool_calls],
                finish_reason=parsed_response.finish_reason,
                usage=parsed_response.usage,
            )
            if log_payloads_enabled():
                log_event(
                    "model.response.details",
                    level=logging.DEBUG,
                    provider="nvidia",
                    request_id=request_id,
                    content=parsed_response.content,
                    tool_calls=[call.to_dict() for call in parsed_response.tool_calls],
                    raw=data,
                )
            return parsed_response
        except Exception as exc:
            log_event(
                "model.request.failed",
                level=logging.ERROR,
                provider="nvidia",
                request_id=request_id,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                error=str(exc),
                exc_info=True,
            )
            raise

    def health_check(self) -> tuple[bool, str]:
        """Check that the configured NVIDIA endpoint lists the selected model."""
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            return False, "nvidia_endpoint_must_use_https"
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
                allow_redirects=False,
            )
            if 300 <= response.status_code < 400:
                return False, "nvidia_model_endpoint_redirect_rejected"
            if not response.ok:
                return False, f"nvidia_model_endpoint_http_{response.status_code}"
            payload = response.json()
            model_ids = {
                str(item.get("id"))
                for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            } if isinstance(payload, dict) else set()
            if not model_ids:
                return False, "nvidia_model_catalog_unavailable"
            if self.model not in model_ids:
                return False, "configured_nvidia_model_not_available"
            return True, "available"
        except (requests.RequestException, ValueError):
            return False, "nvidia_model_endpoint_unreachable"

    def _normalize_messages(
        self,
        messages: list[Message | dict[str, Any]],
    ) -> list[dict[str, Any]]:

        normalized: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, Message):
                normalized.append(message.to_dict())

            elif isinstance(message, dict):
                normalized.append(message)

            else:
                raise TypeError(
                    f"Unsupported message type: {type(message).__name__}"
                )

        return self._split_multi_call_turns(normalized)

    @staticmethod
    def _split_multi_call_turns(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Serialize completed tool batches as single-call turns."""
        result: list[dict[str, Any]] = []
        index = 0

        while index < len(messages):
            message = messages[index]
            tool_calls = message.get("tool_calls")

            if (
                message.get("role") != "assistant"
                or not isinstance(tool_calls, list)
                or len(tool_calls) < 2
            ):
                result.append(message)
                index += 1
                continue

            call_ids = [
                call.get("id")
                for call in tool_calls
                if isinstance(call, dict)
            ]
            if (
                len(call_ids) != len(tool_calls)
                or any(not call_id for call_id in call_ids)
                or len(set(call_ids)) != len(call_ids)
            ):
                result.append(message)
                index += 1
                continue

            responses: dict[str, dict[str, Any]] = {}
            cursor = index + 1
            while cursor < len(messages):
                tool_message = messages[cursor]
                if tool_message.get("role") != "tool":
                    break
                tool_call_id = tool_message.get("tool_call_id")
                if tool_call_id not in call_ids:
                    break
                responses[tool_call_id] = tool_message
                cursor += 1

            if any(call_id not in responses for call_id in call_ids):
                result.append(message)
                index += 1
                continue

            for tool_call, call_id in zip(tool_calls, call_ids):
                result.append(
                    {
                        **message,
                        "tool_calls": [tool_call],
                    }
                )
                result.append(responses[call_id])

            index = cursor

        return result

    def _parse_response(
        self,
        data: dict[str, Any],
    ) -> ModelResponse:

        return ModelParser.parse(data)

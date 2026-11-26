from __future__ import annotations

import os
import re
import threading
import time
from collections import deque
from typing import Any
from urllib.parse import urlparse

import requests


BASE_URL = "https://www.moltbook.com/api/v1"
ALLOWED_HOST = "www.moltbook.com"
READ_METHODS = {"GET", "HEAD"}


class MoltbookAPIError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: Any = None,
        *,
        outcome_unknown: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data
        self.outcome_unknown = outcome_unknown


class MoltbookClient:
    """Authenticated client restricted to the official Moltbook HTTPS API."""

    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = os.getenv("MOLTBOOK_API_KEY", "") if api_key is None else api_key
        self.timeout = timeout
        try:
            configured_limit = int(os.getenv("MOLTBOOK_MAX_REQUESTS_PER_MINUTE", "100"))
        except ValueError as exc:
            raise ValueError("MOLTBOOK_MAX_REQUESTS_PER_MINUTE must be an integer.") from exc
        if configured_limit < 1:
            raise ValueError("MOLTBOOK_MAX_REQUESTS_PER_MINUTE must be positive.")
        self.max_requests_per_minute = min(configured_limit, 100)
        self._request_times: deque[float] = deque()
        self._rate_lock = threading.Lock()
        self.session = requests.Session()
        # Credentials are attached per validated request, never stored in a
        # session-wide header that could follow an accidental URL change.
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Molt-Agent/1.0",
        })

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != ALLOWED_HOST
            or parsed.port is not None
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise MoltbookAPIError("Request blocked: untrusted Moltbook host.")
        if not parsed.path.startswith("/api/v1/"):
            raise MoltbookAPIError("Request blocked: invalid API path.")

    def _safe_message(self, message: object) -> str:
        text = str(message)
        if self.api_key:
            text = text.replace(self.api_key, "[REDACTED]")
        text = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
        return text[:1000]

    def _reserve_rate_slot(self) -> None:
        while True:
            with self._rate_lock:
                now = time.monotonic()
                while self._request_times and now - self._request_times[0] >= 60:
                    self._request_times.popleft()
                if len(self._request_times) < self.max_requests_per_minute:
                    self._request_times.append(now)
                    return
                wait_seconds = max(0.05, 60 - (now - self._request_times[0]))
            time.sleep(wait_seconds)

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        method = method.upper()
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        url = f"{BASE_URL}{endpoint}"
        self._validate_url(url)

        if authenticated and not self.api_key:
            raise MoltbookAPIError("MOLTBOOK_API_KEY is not configured.")

        headers = {"Authorization": f"Bearer {self.api_key}"} if authenticated else {}
        retryable_read = method in READ_METHODS
        max_attempts = 3 if retryable_read else 1

        for attempt in range(max_attempts):
            try:
                self._reserve_rate_slot()
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=False,
                )
            except requests.Timeout as exc:
                if not retryable_read:
                    raise MoltbookAPIError(
                        "Moltbook write timed out; its outcome is unknown.",
                        outcome_unknown=True,
                    ) from exc
                if attempt + 1 < max_attempts:
                    time.sleep(2 ** attempt)
                    continue
                raise MoltbookAPIError("Moltbook read request timed out.") from exc
            except requests.RequestException as exc:
                raise MoltbookAPIError(
                    "Moltbook connection failed; write outcome is unknown."
                    if not retryable_read
                    else "Moltbook connection failed.",
                    outcome_unknown=not retryable_read,
                ) from exc

            if 300 <= response.status_code < 400:
                raise MoltbookAPIError(
                    "Redirect rejected for security.",
                    response.status_code,
                    outcome_unknown=not retryable_read,
                )

            if response.status_code == 429:
                if retryable_read and attempt + 1 < max_attempts:
                    try:
                        retry_after = float(response.headers.get("Retry-After", "5"))
                    except ValueError:
                        retry_after = 5
                    time.sleep(min(max(0, retry_after), 30))
                    continue
                raise MoltbookAPIError(
                    "Moltbook rate limit reached; the request was not retried.",
                    429,
                    outcome_unknown=False,
                )

            try:
                data = response.json()
            except ValueError:
                data = {"raw_response": self._safe_message(response.text)}

            if not response.ok:
                detail = data.get("error", "Moltbook request failed.") if isinstance(data, dict) else "Moltbook request failed."
                raise MoltbookAPIError(
                    self._safe_message(detail),
                    response.status_code,
                    outcome_unknown=(not retryable_read and response.status_code >= 500),
                )

            return data if isinstance(data, dict) else {"data": data}

        raise MoltbookAPIError("Moltbook request failed.")

    def get(self, endpoint: str, params: dict | None = None):
        return self._request("GET", endpoint, params=params)

    def post(self, endpoint: str, payload: dict | None = None):
        return self._request("POST", endpoint, json=payload)

    def delete(self, endpoint: str):
        return self._request("DELETE", endpoint)

    def patch(self, endpoint: str, payload: dict):
        return self._request("PATCH", endpoint, json=payload)

    def register(self, name: str, description: str):
        return self._request(
            "POST",
            "/agents/register",
            json={"name": name, "description": description},
            authenticated=False,
        )

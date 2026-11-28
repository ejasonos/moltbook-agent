from __future__ import annotations

import os
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from config.api_limits import max_api_requests_per_minute
from memory.database import resolve_database_path


BASE_URL = "https://www.moltbook.com/api/v1"
ALLOWED_HOST = "www.moltbook.com"
READ_METHODS = {"GET", "HEAD"}


class SharedAPIRateLimiter:
    """SQLite-backed rolling-window limiter shared by all local processes."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        limit: int | None = None,
        window_seconds: float = 60,
    ) -> None:
        self.database_path = resolve_database_path(database_path)
        if self.database_path == ":memory:":
            raise ValueError("The shared API limiter requires a file-backed database.")
        self.limit = limit or max_api_requests_per_minute()
        if self.limit < 1:
            raise ValueError("API request limit must be positive.")
        self.window_seconds = max(1.0, float(window_seconds))
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS moltbook_api_requests "
                "(requested_at REAL NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_moltbook_api_requests_time "
                "ON moltbook_api_requests(requested_at)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS moltbook_api_limit_state "
                "(state_key TEXT PRIMARY KEY, state_value REAL NOT NULL)"
            )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.database_path), timeout=30)
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        finally:
            connection.close()

    def reserve(self) -> None:
        """Atomically reserve a slot, counting every attempted HTTP request."""
        while True:
            wait_seconds = 0.0
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                now = time.time()
                connection.execute(
                    "DELETE FROM moltbook_api_requests WHERE requested_at <= ?",
                    (now - self.window_seconds,),
                )
                blocked = connection.execute(
                    "SELECT state_value FROM moltbook_api_limit_state "
                    "WHERE state_key = 'blocked_until'"
                ).fetchone()
                if blocked and blocked[0] > now:
                    wait_seconds = blocked[0] - now
                else:
                    count, oldest = connection.execute(
                        "SELECT COUNT(*), MIN(requested_at) "
                        "FROM moltbook_api_requests"
                    ).fetchone()
                    if count < self.limit:
                        connection.execute(
                            "INSERT INTO moltbook_api_requests(requested_at) VALUES (?)",
                            (now,),
                        )
                        connection.commit()
                        return
                    wait_seconds = max(
                        0.05,
                        self.window_seconds - (now - float(oldest)),
                    )
                connection.rollback()
            time.sleep(max(0.05, wait_seconds))

    def observe_response(self, headers: Any, status_code: int) -> float | None:
        """Apply server-provided retry/reset timing to all local processes."""
        retry_after = self._retry_after(headers)
        remaining = headers.get("X-RateLimit-Remaining") if headers else None
        if status_code == 429 and retry_after is None:
            retry_after = 5.0
        if remaining is not None:
            try:
                depleted = float(remaining) <= 0
            except (TypeError, ValueError):
                depleted = False
            if depleted and retry_after is None:
                reset = headers.get("X-RateLimit-Reset")
                try:
                    reset_value = float(reset)
                    now = time.time()
                    retry_after = max(0.0, reset_value - now) if reset_value > now else max(0.0, reset_value)
                except (TypeError, ValueError):
                    retry_after = None
        if retry_after is None or retry_after <= 0:
            return retry_after
        blocked_until = time.time() + retry_after
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO moltbook_api_limit_state(state_key, state_value) "
                "VALUES ('blocked_until', ?) ON CONFLICT(state_key) "
                "DO UPDATE SET state_value = MAX(state_value, excluded.state_value)",
                (blocked_until,),
            )
            connection.commit()
        return retry_after

    @staticmethod
    def _retry_after(headers: Any) -> float | None:
        if not headers:
            return None
        value = headers.get("Retry-After")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                parsed = parsedate_to_datetime(str(value))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None


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

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 30,
        *,
        rate_limiter: SharedAPIRateLimiter | None = None,
        database_path: str | Path | None = None,
    ):
        self.api_key = os.getenv("MOLTBOOK_API_KEY", "") if api_key is None else api_key
        self.timeout = timeout
        self.rate_limiter = rate_limiter or SharedAPIRateLimiter(database_path)
        self.max_requests_per_minute = self.rate_limiter.limit
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
                self.rate_limiter.reserve()
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

            retry_after = self.rate_limiter.observe_response(
                response.headers, response.status_code
            )

            if response.status_code == 429:
                if retryable_read and attempt + 1 < max_attempts:
                    time.sleep(min(max(0.0, retry_after or 5.0), 30.0))
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

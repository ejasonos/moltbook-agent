from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os
import time
from typing import Any
from urllib.parse import urlparse

import requests


BASE_URL = "https://www.moltbook.com/api/v1"
ALLOWED_HOST = "www.moltbook.com"


class MoltbookAPIError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: Any = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class MoltbookClient:
    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.getenv("MOLTBOOK_API_KEY")
        self.timeout = timeout
        self.session = requests.Session()

        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Molt-Agent/1.0",
        })

        if self.api_key:
            self.session.headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)

        if parsed.scheme != "https":
            raise MoltbookAPIError("HTTPS is required.")

        if parsed.hostname != ALLOWED_HOST:
            raise MoltbookAPIError(
                "Request blocked: untrusted Moltbook host."
            )

        if not parsed.path.startswith("/api/v1/"):
            raise MoltbookAPIError(
                "Request blocked: invalid API path."
            )

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:

        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"

        url = f"{BASE_URL}{endpoint}"
        self._validate_url(url)

        if authenticated and not self.api_key:
            raise MoltbookAPIError(
                "MOLTBOOK_API_KEY is not configured."
            )

        headers = {}

        if not authenticated:
            headers["Authorization"] = ""

        for attempt in range(3):
            try:
                response = self.session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=json,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=False,
                )

                if 300 <= response.status_code < 400:
                    raise MoltbookAPIError(
                        "Redirect rejected for security.",
                        response.status_code,
                    )

                if response.status_code == 429:
                    if method.upper() not in {"GET", "HEAD"}:
                        raise MoltbookAPIError(
                            "Rate limited. Write was not retried.",
                            429,
                            response.text,
                        )

                    retry_after = response.headers.get(
                        "Retry-After", "5"
                    )

                    if attempt < 2:
                        time.sleep(min(float(retry_after), 30))
                        continue

                try:
                    data = response.json()
                except ValueError:
                    data = {"raw_response": response.text}

                if not response.ok:
                    raise MoltbookAPIError(
                        data.get("error", "Moltbook request failed."),
                        response.status_code,
                        data,
                    )

                if isinstance(data, dict):
                    return data

                return {"data": data}

            except requests.Timeout as exc:
                if attempt == 2:
                    raise MoltbookAPIError(
                        "Moltbook request timed out."
                    ) from exc

                time.sleep(2 ** attempt)

            except requests.RequestException as exc:
                raise MoltbookAPIError(
                    f"Network error: {exc}"
                ) from exc

        raise MoltbookAPIError("Request failed.")

    def get(self, endpoint: str, params: dict | None = None):
        return self._request("GET", endpoint, params=params)

    def post(
        self,
        endpoint: str,
        payload: dict | None = None,
    ):
        return self._request("POST", endpoint, json=payload)

    def delete(self, endpoint: str):
        return self._request("DELETE", endpoint)

    def patch(self, endpoint: str, payload: dict):
        return self._request("PATCH", endpoint, json=payload)

    def register(self, name: str, description: str):
        return self._request(
            "POST",
            "/agents/register",
            json={
                "name": name,
                "description": description,
            },
            authenticated=False,
        )
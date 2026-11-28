from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import sqlite3
import time

import pytest

from integrations.moltbook import client as client_module
from integrations.moltbook.client import MoltbookAPIError, MoltbookClient, SharedAPIRateLimiter


def _reserve_in_child(database_path, output):
    limiter = SharedAPIRateLimiter(database_path, limit=2, window_seconds=1)
    limiter.reserve()
    output.put(time.time())


class FakeResponse:
    def __init__(self, status=200, headers=None):
        self.status_code = status
        self.headers = headers or {}
        self.ok = 200 <= status < 300
        self.text = "{}"

    def json(self):
        return {"success": True}


class FakeSession:
    def __init__(self, responses=None):
        self.responses = list(responses or [FakeResponse()])
        self.calls = []
        self.headers = {}

    def request(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0) if self.responses else FakeResponse()


def test_get_and_post_each_reserve_a_shared_api_slot(tmp_path):
    database_path = tmp_path / "shared.db"
    limiter = SharedAPIRateLimiter(database_path, limit=10)
    client = MoltbookClient(api_key="test-key", rate_limiter=limiter)
    client.session = FakeSession()

    client.get("/feed")
    client.post("/posts", {"title": "test"})

    with sqlite3.connect(database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM moltbook_api_requests").fetchone()[0]
    assert count == 2
    assert [call["method"] for call in client.session.calls] == ["GET", "POST"]


def test_separate_limiter_instances_share_the_rolling_window(tmp_path):
    database_path = tmp_path / "shared.db"
    first = SharedAPIRateLimiter(database_path, limit=2, window_seconds=1)
    second = SharedAPIRateLimiter(database_path, limit=2, window_seconds=1)

    with ThreadPoolExecutor(max_workers=4) as pool:
        timestamps = list(pool.map(lambda index: (first if index % 2 else second).reserve() or time.time(), range(4)))

    timestamps.sort()
    assert timestamps[2] - timestamps[0] >= 0.9


def test_separate_processes_cannot_bypass_the_shared_limit(tmp_path):
    database_path = str(tmp_path / "shared.db")
    SharedAPIRateLimiter(database_path, limit=2, window_seconds=1)
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(target=_reserve_in_child, args=(database_path, output))
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    timestamps = [output.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    timestamps.sort()
    assert timestamps[2] - timestamps[0] >= 0.9


def test_get_retries_429_a_bounded_number_of_times(tmp_path, monkeypatch):
    limiter = SharedAPIRateLimiter(tmp_path / "shared.db", limit=10)
    client = MoltbookClient(api_key="test-key", rate_limiter=limiter)
    client.session = FakeSession([
        FakeResponse(429, {"Retry-After": "0"}),
        FakeResponse(200),
    ])
    sleeps = []
    monkeypatch.setattr(client_module.time, "sleep", lambda delay: sleeps.append(delay))

    assert client.get("/feed")["success"] is True
    assert len(client.session.calls) == 2
    assert len(sleeps) == 1


def test_post_does_not_retry_429(tmp_path):
    limiter = SharedAPIRateLimiter(tmp_path / "shared.db", limit=10)
    client = MoltbookClient(api_key="test-key", rate_limiter=limiter)
    client.session = FakeSession([FakeResponse(429, {"Retry-After": "0"})])

    with pytest.raises(MoltbookAPIError) as raised:
        client.post("/posts", {"title": "one attempt"})

    assert raised.value.outcome_unknown is False
    assert len(client.session.calls) == 1


def test_failed_http_attempt_still_consumes_a_rate_slot(tmp_path):
    database_path = tmp_path / "shared.db"
    limiter = SharedAPIRateLimiter(database_path, limit=10)
    client = MoltbookClient(api_key="test-key", rate_limiter=limiter)
    client.session = FakeSession([FakeResponse(400)])

    with pytest.raises(MoltbookAPIError):
        client.get("/feed")

    with sqlite3.connect(database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM moltbook_api_requests").fetchone()[0]
    assert count == 1


def test_retry_after_header_blocks_other_client_instances(tmp_path):
    database_path = tmp_path / "shared.db"
    limiter_one = SharedAPIRateLimiter(database_path, limit=10)
    limiter_two = SharedAPIRateLimiter(database_path, limit=10)
    delay = limiter_one.observe_response({"Retry-After": "0.05"}, 429)
    started = time.monotonic()

    limiter_two.reserve()

    assert delay == pytest.approx(0.05)
    assert time.monotonic() - started >= 0.04

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from memory import SocialMemory
from moltbook.heartbeat import MoltHeartbeat
from runtime.health import RuntimeHealth


@pytest.fixture(scope="module")
def web_app(tmp_path_factory):
    path = tmp_path_factory.mktemp("health-app") / "memory.db"
    old = os.environ.get("MOLTBOOK_MEMORY_DB")
    os.environ["MOLTBOOK_MEMORY_DB"] = str(path)
    try:
        return importlib.import_module("app")
    finally:
        if old is None:
            os.environ.pop("MOLTBOOK_MEMORY_DB", None)
        else:
            os.environ["MOLTBOOK_MEMORY_DB"] = old


def test_health_http_routes_expose_safe_runtime_state(tmp_path, web_app, monkeypatch):
    memory = SocialMemory(tmp_path / "health.db")
    try:
        memory.store.set_state(
            "runtime_health_components",
            json.dumps({
                "nvidia_model": {"status": "healthy", "detail": "configured"},
                "tool_discovery": {"status": "healthy", "detail": "available"},
            }),
        )
        memory.store.set_state("moltbook_auth_status", "authenticated")
        memory.store.set_state("heartbeat_status", "running")
        memory.store.set_state("last_error", "failed token test-secret-value")
        monkeypatch.setenv("MOLTBOOK_API_KEY", "test-secret-value")
        monkeypatch.setattr(web_app, "get_memory", lambda: memory)

        from starlette.routing import Route
        routes = {route.path: route.endpoint for route in web_app.app.routes if isinstance(route, Route)}
        response = routes["/health"](None)
        details_response = routes["/health/details"](None)
        code = response.status_code
        detail_code = details_response.status_code
        body = json.loads(response.body)
        details = json.loads(details_response.body)

        assert code == detail_code == 200
        assert body["status"] in {"healthy", "degraded"}
        assert "heartbeat" in details["components"]
        assert details["outstanding_uncertain_actions"] == 0
        assert "test-secret-value" not in json.dumps(details)
    finally:
        memory.close()


def test_health_reports_database_mcp_and_authentication_failures(tmp_path):
    memory = SocialMemory(tmp_path / "health.db")
    memory.store.set_state("moltbook_auth_status", "failed")
    runtime = SimpleNamespace(
        memory=memory,
        agent_loop=SimpleNamespace(model=SimpleNamespace(model="test-model")),
        registry=SimpleNamespace(discover=lambda: (_ for _ in ()).throw(ConnectionError("offline"))),
    )
    health = RuntimeHealth(runtime)
    report = health.check()
    assert report["components"]["mcp_connectivity"]["status"] == "degraded"
    assert report["components"]["moltbook_authentication"]["status"] == "unhealthy"

    memory.close()
    unavailable = RuntimeHealth(memory=memory).check()
    assert unavailable["components"]["sqlite"]["status"] == "unhealthy"


@pytest.mark.parametrize(
    "outcome, action_status, success",
    [
        ("participation_succeeded", "success", True),
        ("abstained", "abstained", True),
        ("authentication_failure", "not_selected", False),
        ("failed_social_action", "failed", False),
        ("uncertain_action", "unknown", False),
        ("infrastructure_failure", "not_selected", False),
    ],
)
def test_heartbeat_records_cycle_classification(tmp_path, outcome, action_status, success):
    memory = SocialMemory(tmp_path / "heartbeat.db")
    result = SimpleNamespace(
        success=success,
        outcome=outcome,
        action_status=action_status,
        unresolved_errors=[] if success else [outcome],
        error=None,
    )
    heartbeat = MoltHeartbeat(lambda: result, memory=memory)
    try:
        actual, _error = heartbeat._run_cycle()
        assert actual is success
        assert memory.store.get_state("cycle_outcome") == outcome
        assert memory.store.get_state("cycle_action_status") == action_status
        assert memory.store.get_state("cycle_status") == ("success" if success else "failed")
    finally:
        memory.close()


def test_heartbeat_failure_backoff_survives_restart_and_caps_at_30_seconds(tmp_path):
    memory = SocialMemory(tmp_path / "heartbeat.db")
    result = SimpleNamespace(
        success=False,
        outcome="uncertain_action",
        action_status="unknown",
        unresolved_errors=["write uncertain"],
        error=None,
    )
    heartbeat = None

    def callback():
        heartbeat.stop()
        return result

    heartbeat = MoltHeartbeat(callback, memory=memory, max_backoff_seconds=300)
    try:
        heartbeat.run()
        assert memory.store.get_state("consecutive_failure_count") == "1"
        resumed = MoltHeartbeat(lambda: result, memory=memory)._initial_schedule()
        from datetime import datetime, timezone
        assert resumed is not None
        assert 0 <= (resumed - datetime.now(timezone.utc)).total_seconds() <= 5.2
        assert heartbeat.max_backoff_seconds == 30
        assert memory.store.get_state("heartbeat_status") == "stopped"
    finally:
        memory.close()


def test_heartbeat_database_failure_is_reported_as_failure(tmp_path):
    memory = SocialMemory(tmp_path / "heartbeat.db")
    memory.close()
    heartbeat = MoltHeartbeat(
        lambda: SimpleNamespace(success=True, outcome="abstained", action_status="abstained"),
        memory=memory,
    )
    success, error = heartbeat._run_cycle()
    assert success is False
    assert "database persistence failed" in error


def test_source_package_excludes_credentials_databases_and_runtime_files(tmp_path):
    from build_package import build_archive

    root = tmp_path / "project"
    (root / "agent").mkdir(parents=True)
    (root / ".config" / "moltbook").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "logs").mkdir()
    (root / "README.md").write_text("readme", encoding="utf-8")
    (root / "requirements.txt").write_text("requests", encoding="utf-8")
    (root / "agent" / "loop.py").write_text("source", encoding="utf-8")
    (root / ".env").write_text("do-not-package", encoding="utf-8")
    (root / ".config" / "moltbook" / "credentials.json").write_text("secret", encoding="utf-8")
    (root / "data" / "memory.db").write_bytes(b"database")
    (root / "logs" / "molt.log").write_text("log", encoding="utf-8")

    archive_path, count = build_archive(root, root / "dist" / "project.zip")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()

    assert count == len(names) == 3
    assert set(names) == {"README.md", "requirements.txt", "agent/loop.py"}

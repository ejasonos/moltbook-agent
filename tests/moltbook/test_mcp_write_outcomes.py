from __future__ import annotations

import importlib

import pytest

from integrations.moltbook.client import MoltbookAPIError
from memory import SocialMemory


@pytest.fixture
def tool_server(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLTBOOK_MEMORY_DB", str(tmp_path / "server-limiter.db"))
    module = importlib.import_module("moltbook_tools_http")
    memory = SocialMemory(tmp_path / "memory.db")
    module._memory = memory
    yield module, memory
    module._memory = None
    memory.close()


def test_confirmed_post_and_comment_persist_resource_ids(tool_server):
    module, memory = tool_server
    post = module._execute_write(
        "create_post", "agents", {"title": "A post", "content": "body"},
        lambda: {"post": {"id": "post-new"}},
    )
    comment = module._execute_write(
        "create_comment", "post-target", {"content": "A comment"},
        lambda: {"comment": {"id": "comment-new"}},
    )

    assert post["status"] == comment["status"] == "success"
    records = memory.database.fetch_all(
        "SELECT interaction_type, status, resource_id FROM interactions ORDER BY id"
    )
    assert [(row["interaction_type"], row["status"], row["resource_id"]) for row in records] == [
        ("create_post", "success", "post-new"),
        ("create_comment", "success", "comment-new"),
    ]


def test_api_rejection_is_persisted_as_failed(tool_server):
    module, memory = tool_server

    response = module._execute_write(
        "create_comment", "post-failed", {"content": "rejected"},
        lambda: (_ for _ in ()).throw(MoltbookAPIError("rejected", status_code=400)),
    )

    assert response["status"] == "failed"
    assert memory.database.fetch_one(
        "SELECT status FROM interactions WHERE target_id = 'post-failed'"
    )["status"] == "failed"


def test_unknown_write_is_reconciled_once_and_never_resubmitted(tool_server, monkeypatch):
    module, memory = tool_server
    operation_calls = []
    reconciliation_calls = []
    monkeypatch.setattr(
        module,
        "_reconcile_uncertain_write",
        lambda *_args: reconciliation_calls.append("checked") or None,
    )

    def timeout_after_send():
        operation_calls.append("submitted")
        raise MoltbookAPIError("timeout", outcome_unknown=True)

    response = module._execute_write(
        "create_comment", "post-unknown", {"content": "maybe sent"}, timeout_after_send
    )

    assert response["status"] == "unknown"
    assert operation_calls == ["submitted"]
    assert reconciliation_calls == ["checked"]
    assert memory.database.fetch_one(
        "SELECT status FROM interactions WHERE target_id = 'post-unknown'"
    )["status"] == "unknown"


def test_verification_challenge_remains_unconfirmed(tool_server):
    module, memory = tool_server
    response = module._execute_write(
        "create_comment", "post-challenge", {"content": "challenge"},
        lambda: {"verification_required": True, "challenge": "challenge-id"},
    )

    assert response["success"] is False
    assert response["status"] == "verification_required"
    assert memory.database.fetch_one(
        "SELECT status FROM interactions WHERE target_id = 'post-challenge'"
    )["status"] == "verification_required"

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from agent.state import AgentState, ToolExecution
from integrations.moltbook.client import MoltbookAPIError
from memory import SocialMemory
from moltbook.runner import MoltRunner


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLTBOOK_MEMORY_DB", str(tmp_path / "limiter.db"))
    tools = importlib.import_module("moltbook_tools_http")
    memory = SocialMemory(tmp_path / "memory.db")
    tools._memory = memory
    feed = {"posts": [{
        "id": "pipeline-post",
        "title": "A useful engineering discussion",
        "content": "Evidence and implementation details.",
        "submolt": "engineering",
        "author": {"id": "pipeline-author", "username": "Author"},
    }]}
    post = {"post": feed["posts"][0]}
    comments = {"comments": [{
        "id": "pipeline-comment",
        "post_id": "pipeline-post",
        "content": "The discussion has a complete answer.",
        "author": {"id": "pipeline-commenter", "username": "Commenter"},
    }]}

    class MCP:
        def call_tool(self, _name, _arguments):
            return SimpleNamespace(
                success=True,
                structured_content={"success": True, "profile": {"username": "Penelope"}},
                content=None,
                error=None,
            )

    class PipelineLoop:
        config = SimpleNamespace(max_candidate_posts_per_run=5)
        mcp_client = MCP()

        def __init__(self, mode):
            self.mode = mode

        def run(self, _task, **_kwargs):
            state = AgentState(task="mocked autonomous cycle")
            feed_response = tools._read(lambda: feed, remember_posts=True)
            post_response = tools._read(lambda: post, remember_posts=True)
            comments_response = tools._read(
                lambda: comments,
                on_result=lambda value: tools._remember_comments("pipeline-post", value),
            )
            state.tool_executions.extend([
                ToolExecution("molt_get_feed", {}, json.dumps(feed_response), True),
                ToolExecution("molt_get_post", {"post_id": "pipeline-post"}, json.dumps(post_response), True),
                ToolExecution("molt_get_comments", {"post_id": "pipeline-post"}, json.dumps(comments_response), True),
            ])
            if self.mode == "abstain":
                state.finish("ABSTAIN_REASON: The inspected discussion already has a complete answer.")
                state.metadata["last_evaluation"] = {"next_action": "synthesize", "goal_complete": True}
                return state

            def operation():
                if self.mode == "failed":
                    raise MoltbookAPIError("Moltbook rejected the comment", status_code=400)
                if self.mode == "unknown":
                    raise MoltbookAPIError("write timed out", outcome_unknown=True)
                return {"comment": {"id": "pipeline-comment-created"}}

            response = tools._execute_write(
                "create_comment",
                "pipeline-post",
                {"content": "A useful contribution."},
                operation,
            )
            state.metadata["social_actions"] = [{
                "tool": "molt_create_comment",
                "status": response.get("status"),
                "result": json.dumps(response),
                "arguments": {"post_id": "pipeline-post", "content": "A useful contribution."},
            }]
            if not response.get("success"):
                state.unresolved_tool_failures["molt_create_comment"] = response.get("error", "write failed")
            state.metadata["last_evaluation"] = {"next_action": "finish", "goal_complete": True}
            state.finish("Cycle completed.")
            return state

    monkeypatch.setattr(tools, "_reconcile_uncertain_write", lambda *_args: None)
    yield tools, memory, PipelineLoop
    tools._memory = None
    memory.close()


@pytest.mark.parametrize("mode", ["abstain", "success", "failed", "unknown"])
def test_full_mocked_cycle_persists_investigation_and_classifies_action(pipeline, tmp_path, mode):
    tools, memory, PipelineLoop = pipeline
    result = MoltRunner(PipelineLoop(mode), memory=memory).run_cycle()

    assert result.posts_inspected == 1
    assert result.discussions_evaluated == 1
    assert memory.get_recent_posts()[0]["post_id"] == "pipeline-post"
    assert memory.get_recent_comments_for_post("pipeline-post")[0]["comment_id"] == "pipeline-comment"
    if mode == "abstain":
        assert result.success is True
        assert result.outcome == "abstained"
        assert result.action_succeeded is False
    elif mode == "success":
        assert result.success is True
        assert result.action_succeeded is True
        assert result.resource_id == "pipeline-comment-created"
        assert result.outcome == "participation_succeeded"
        assert memory.database.fetch_one(
            "SELECT status FROM interactions WHERE target_id = 'pipeline-post'"
        )["status"] == "success"
    elif mode == "failed":
        assert result.success is False
        assert result.action_status == "failed"
        assert result.outcome == "failed_social_action"
    else:
        assert result.success is False
        assert result.action_status == "unknown"
        assert result.outcome == "uncertain_action"
        assert memory.database.fetch_one(
            "SELECT status FROM interactions WHERE target_id = 'pipeline-post'"
        )["status"] == "unknown"

    snapshot = tmp_path / f"{mode}-snapshot.db"
    memory.database.backup_to(snapshot)
    restored = SocialMemory(snapshot)
    try:
        assert restored.get_recent_posts()[0]["post_id"] == "pipeline-post"
        assert restored.get_recent_comments_for_post("pipeline-post")
    finally:
        restored.close()

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.state import AgentState, ToolExecution
from memory import SocialMemory
from moltbook.runner import MoltCycleResult, MoltRunner


def completed_state():
    state = AgentState(task="cycle")
    state.finish("Done.")
    return state


def cycle_result(**overrides):
    values = dict(
        cycle_id="cycle-test",
        start_time="2026-01-01T00:00:00+00:00",
        completion_time="2026-01-01T00:00:01+00:00",
        authentication_status="authenticated",
        discussions_evaluated=1,
        action_selected="molt_create_comment",
        action_attempted=True,
        action_succeeded=True,
        action_status="success",
        action_result={"success": True, "resource_id": "comment-1"},
        resource_id="comment-1",
        persistence_succeeded=True,
        agent_state=completed_state(),
    )
    values.update(overrides)
    return MoltCycleResult(**values)


@pytest.mark.parametrize("action", ["molt_create_post", "molt_create_comment"])
def test_confirmed_social_actions_are_successful(action):
    result = cycle_result(action_selected=action)
    assert result.success is True
    assert result.outcome == "participation_succeeded"
    assert result.action_succeeded is True
    assert result.action_verified is False


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        ("failed", "failed_social_action"),
        ("unknown", "uncertain_action"),
        ("verification_required", "verification_required"),
        ("pending", "infrastructure_failure"),
    ],
)
def test_unconfirmed_action_statuses_never_count_as_success(status, outcome):
    result = cycle_result(
        action_succeeded=False,
        action_status=status,
        action_result={"success": False, "status": status},
    )
    assert result.success is False
    assert result.outcome == outcome


def test_legitimate_abstention_is_healthy_but_not_a_social_action():
    result = cycle_result(
        discussions_evaluated=2,
        action_selected="abstain",
        action_attempted=False,
        action_succeeded=False,
        action_status="abstained",
        action_result=None,
        abstention_reason="The inspected discussion already has a complete answer.",
    )
    assert result.success is True
    assert result.outcome == "abstained"
    assert result.action_succeeded is False


def test_abstention_without_investigated_discussion_is_not_success():
    result = cycle_result(
        discussions_evaluated=0,
        action_selected="abstain",
        action_attempted=False,
        action_succeeded=False,
        action_status="abstained",
        abstention_reason="Nothing useful to add.",
    )
    assert result.success is False


def test_authentication_failure_and_database_failure_are_classified():
    unauthenticated = cycle_result(authentication_status="failed")
    assert unauthenticated.success is False
    assert unauthenticated.outcome == "authentication_failure"

    persistence_failure = cycle_result(persistence_succeeded=False)
    assert persistence_failure.success is False
    assert persistence_failure.outcome == "infrastructure_failure"


class FakeMCP:
    def call_tool(self, _name, _arguments):
        return SimpleNamespace(
            success=True,
            structured_content={"success": True, "profile": {"username": "Penelope"}},
            content=None,
            error=None,
        )


class FakeAgentLoop:
    config = SimpleNamespace(max_candidate_posts_per_run=5)
    mcp_client = FakeMCP()

    def __init__(self, action_status="success"):
        self.action_status = action_status

    def run(self, _task, **_kwargs):
        state = completed_state()
        state.tool_executions.extend([
            ToolExecution("molt_get_feed", {}, result=json.dumps({"posts": [{"id": "p1", "title": "t", "content": "c"}]}), success=True),
            ToolExecution("molt_get_post", {"post_id": "p1"}, result=json.dumps({"id": "p1", "title": "t", "content": "c"}), success=True),
            ToolExecution("molt_get_comments", {"post_id": "p1"}, result=json.dumps({"comments": []}), success=True),
        ])
        if self.action_status == "abstained":
            state.final_answer = "ABSTAIN_REASON: The discussion was already resolved."
        else:
            state.metadata["social_actions"] = [{
                "tool": "molt_create_comment",
                "status": self.action_status,
                "arguments": {"post_id": "p1", "content": "Useful comment"},
                "result": json.dumps({
                    "success": self.action_status == "success",
                    "status": self.action_status,
                    "resource_id": "c1" if self.action_status == "success" else None,
                    "reconciled": False,
                }),
            }]
            if self.action_status != "success":
                state.unresolved_tool_failures["molt_create_comment"] = "write not confirmed"
        return state


@pytest.mark.parametrize("status, expected", [("success", True), ("failed", False), ("unknown", False)])
def test_runner_end_to_end_classifies_mocked_social_write(tmp_path, status, expected):
    memory = SocialMemory(tmp_path / "runner.db")
    try:
        runner = MoltRunner(FakeAgentLoop(status), memory=memory)
        result = runner.run_cycle()
        assert result.success is expected
        assert result.action_status == status
        assert result.action_attempted is True
        assert result.action_succeeded is (status == "success")
        assert json.loads(memory.store.get_state("last_cycle_result"))["outcome"] == result.outcome
        if expected:
            assert result.resource_id == "c1"
            assert result.outcome == "participation_succeeded"
        elif status == "unknown":
            assert result.outcome == "uncertain_action"
    finally:
        memory.close()


def test_runner_end_to_end_abstention_is_healthy(tmp_path):
    memory = SocialMemory(tmp_path / "runner.db")
    try:
        result = MoltRunner(FakeAgentLoop("abstained"), memory=memory).run_cycle()
        assert result.success
        assert result.action_status == "abstained"
        assert result.action_succeeded is False
        assert result.outcome == "abstained"
    finally:
        memory.close()


def test_cycle_result_persistence_exception_fails_result():
    class Store:
        def set_state(self, *_args):
            raise OSError("disk unavailable")

    memory = SimpleNamespace(store=Store())
    runner = MoltRunner(FakeAgentLoop(), memory=memory)
    result = cycle_result()

    runner._persist_cycle(result)

    assert result.persistence_succeeded is False
    assert result.success is False
    assert result.outcome == "infrastructure_failure"

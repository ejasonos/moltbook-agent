from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.evaluator import Evaluation, NextAction
from agent.loop import AgentLoop, AgentLoopConfig
from agent.recovery import RecoveryManager
from mcp_client.schemas import MCPTool, MCPToolResult
from model.messages import ModelResponse, ToolCall


class FakeRegistry:
    def __init__(self, has_feed=False):
        self.tools = [MCPTool("molt_get_feed")] if has_feed else []

    def discover(self):
        return self.tools

    def get(self, name):
        return next((tool for tool in self.tools if tool.name == name), None)

    def to_openai_tools(self):
        if not self.tools:
            return []
        return [{
            "type": "function",
            "function": {"name": "molt_get_feed", "parameters": {"type": "object"}},
        }]


class FakeModel:
    model = "test-model"

    def chat(self, **_kwargs):
        return ModelResponse(content="Final answer.")


class FakePlanner:
    def __init__(self):
        self.calls = 0

    def plan(self, _state):
        self.calls += 1
        return SimpleNamespace(response=ModelResponse(content=f"plan-{self.calls}"))


class FakeExecutor:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def execute(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(response=self.responses.pop(0))


class FakeEvaluator:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def evaluate(self, **_kwargs):
        return Evaluation(
            evidence_sufficient=False,
            task_complete=False,
            goal_complete=False,
            reason="Continue with the next phase.",
            next_action=self.outcomes.pop(0),
        )


class FakeMCPClient:
    def call_tool(self, _name, _arguments):
        return MCPToolResult(
            tool_name="molt_get_feed",
            success=True,
            structured_content={"success": True, "posts": [{"id": "post-1"}]},
        )


def make_loop(outcomes, executor_responses, *, max_iterations=5, has_feed=False):
    loop = AgentLoop.__new__(AgentLoop)
    loop.model = FakeModel()
    loop.mcp_client = FakeMCPClient()
    loop.registry = FakeRegistry(has_feed=has_feed)
    loop.workspace = SimpleNamespace()
    loop.config = AgentLoopConfig(
        max_iterations=max_iterations,
        max_tool_calls=10,
        enable_subagents=False,
        max_runtime_seconds=30,
    )
    from agent.context import ContextManager
    loop.context = ContextManager()
    loop.recovery = RecoveryManager(max_retries_per_tool=1)
    loop.planner = FakePlanner()
    loop.executor = FakeExecutor(executor_responses)
    loop.evaluator = FakeEvaluator(outcomes)
    return loop


@pytest.mark.parametrize("phase", [NextAction.INVESTIGATE, NextAction.SOCIAL_ACTION])
def test_phase_transition_replans_before_next_executor_turn(phase):
    loop = make_loop(
        [phase, NextAction.SYNTHESIZE],
        [ModelResponse(content="Need the next phase."), ModelResponse(content="Investigated.")],
    )

    state = loop._run("complete one bounded Moltbook cycle")

    assert loop.planner.calls == 2
    assert loop.executor.calls == 2
    assert state.completed
    assert state.final_answer == "Final answer."


def test_continue_reuses_executor_plan_without_replanning():
    loop = make_loop(
        [NextAction.CONTINUE],
        [
            ModelResponse(tool_calls=[ToolCall("molt_get_feed", {}, "call-1")]),
            ModelResponse(content="Continue with the existing plan."),
        ],
        has_feed=True,
    )

    state = loop._run("read the feed")

    assert loop.planner.calls == 1
    assert loop.executor.calls == 2
    assert state.tool_executions[0].tool_name == "molt_get_feed"
    assert state.completed


def test_synthesize_is_terminal_for_planner_and_social_action_phases():
    loop = make_loop(
        [NextAction.SYNTHESIZE],
        [ModelResponse(content="Enough evidence.")],
    )

    state = loop._run("summarize")

    assert loop.planner.calls == 1
    assert loop.executor.calls == 1
    assert state.completed


def test_iteration_limit_still_applies_after_phase_transition():
    loop = make_loop(
        [NextAction.INVESTIGATE],
        [ModelResponse(content="Feed retrieved.")],
        max_iterations=1,
    )

    state = loop._run("investigate")

    assert loop.planner.calls == 1
    assert state.iteration == 1
    assert state.error == "Maximum agent iterations reached."

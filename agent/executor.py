from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.context import ContextManager
from agent.state import AgentState
from model.client import ModelClient
from model.messages import Message, ModelResponse
from utils.logging import log_context, log_event


@dataclass
class Execution:
    """Contains the executor's requested actions."""

    response: ModelResponse

    @property
    def should_use_tools(self) -> bool:
        return self.response.has_tool_calls

    @property
    def should_finish(self) -> bool:
        return not self.response.has_tool_calls


class Executor:
    """
    Converts the planner's intentions into executable tool calls.

    IMPORTANT:
    The executor does not dispatch MCP tools itself.

    It produces tool calls which AgentLoop subsequently validates
    and dispatches through its controlled execution runtime.
    """

    def __init__(
        self,
        model: ModelClient,
        context: ContextManager | None = None,
    ) -> None:
        self.model = model
        self.context = context or ContextManager()

    def execute(
        self,
        state: AgentState,
        tools: list[dict[str, Any]] | None = None,
    ) -> Execution:

        available_tools = tools or []

        executor_instructions = Message(
            role="system",
            content=(
                "You are the EXECUTOR model in an AI agent loop\n\n"

                "The planner has already determined what needs "
                "to happen next.\n\n"

                "YOUR RESPONSIBILITY:\n"
                "Your reesponsibility is to act with available tools; "
                "actual tool calls using the available tools.\n\n"

                "EXECUTION RULES:\n"
                "1. Read and follow the latest planner instructions.\n"
                "2. Respect the evaluator's required execution phase.\n"
                "3. Use the available tools to perform the requested "
                "action.\n"
                "4. Never pretend that a tool has been executed.\n"
                "5. Never report a tool operation as successful "
                "before receiving its result.\n"
                "6. Use delegate_task only for independent "
                "read-only investigations.\n"

                "TOOL EXECUTION REQUIREMENT:\n"
                "If the planner has requested an action that requires "
                "a tool, you MUST request the appropriate tool call.\n\n"
            ),
        )

        reserved_tokens = (
            self.context.estimate_tokens_for_value(
                available_tools
            )
            + self.context.token_counter(
                executor_instructions
            )
        )

        messages = self.context.build_messages(
            state,
            reserved_tokens=reserved_tokens,
        )

        messages.insert(
            0,
            executor_instructions,
        )

        log_event(
            "agent.executor.requested",
            message_count=len(messages),
            tool_count=len(available_tools),
            iteration=state.iteration,
        )

        with log_context(model_purpose="executor"):
            response = self.model.chat(
                messages=messages,
                tools=available_tools or None,
            )

        log_event(
            "agent.executor.responded",
            content_length=len(response.content or ""),
            has_tool_calls=response.has_tool_calls,
            tool_calls=[
                call.tool_name
                for call in response.tool_calls
            ],
        )

        return Execution(
            response=response,
        )

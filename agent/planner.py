from __future__ import annotations

from dataclasses import dataclass

from agent.context import ContextManager
from agent.state import AgentState
from model.client import ModelClient
from model.messages import Message, ModelResponse
from utils.logging import log_context, log_event


@dataclass
class Plan:
    """A proposed next action for the agent."""

    response: ModelResponse

    @property
    def content(self) -> str:
        return (self.response.content or "").strip()


class Planner:
    """
    Determines what the agent should do next.

    The planner is deliberately tool-free.

    It produces intentions, not executions.
    """

    def __init__(
        self,
        model: ModelClient,
        context: ContextManager | None = None,
    ) -> None:
        self.model = model
        self.context = context or ContextManager()

    def plan(
        self,
        state: AgentState,
    ) -> Plan:

        planner_instructions = Message(
            role="system",
            content=(
                "You are the PLANNER of a multi-stage autonomous "
                "AI agent.\n\n"

                "Your responsibility is to determine the next "
                "useful action required to accomplish the user's "
                "original objective.\n\n"

                "IMPORTANT ARCHITECTURAL RULES:\n"
                "1. You do not execute tools.\n"
                "2. You do not claim that an action has been performed "
                "merely because you planned it.\n"
                "3. The executor will receive your plan and perform "
                "the necessary actions using available tools.\n"
                "4. The evaluator determines whether the objective "
                "has been accomplished.\n\n"

                "PLANNING REQUIREMENTS:\n"
                "- Understand the original user task.\n"
                "- Examine previous tool execution results.\n"
                "- Respect the evaluator's required next phase.\n"
                "- Respect Moltbook policy decisions and unresolved write outcomes.\n"
                "- Avoid repeating successful or failed operations "
                "without a justified reason.\n"
                "- Prefer specific, actionable steps.\n"
                "- Break complex tasks into manageable actions.\n"
                "- If sufficient evidence exists, plan synthesis "
                "rather than unnecessary investigation.\n\n"

                "YOUR RESPONSE MUST CONTAIN:\n"
                "OBJECTIVE: What needs to be accomplished.\n"
                "CURRENT_STATUS: What is already established.\n"
                "NEXT_ACTION: The specific next action.\n"
                "REASON: Why this action is necessary.\n\n"

                "AVAILABLE TOOLS: \n"
                "Use only tools provided in the current tool definitions. \n"
                "These tools provide Moltbook interaction and persistent memory.\n"
                "Never invent tool names or assume workspace tools are available.\n\n"

                "Do not provide a final answer unless the next "
                "appropriate action is synthesis."
            ),
        )

        messages = self.context.build_messages(
            state,
            reserved_tokens=self.context.token_counter(
                planner_instructions
            ),
        )

        messages.insert(0, planner_instructions)

        log_event(
            "agent.planner.requested",
            message_count=len(messages),
            iteration=state.iteration,
        )

        with log_context(model_purpose="planner"):
            response = self.model.chat(
                messages=messages,
            )

        log_event(
            "agent.planner.responded",
            content_length=len(response.content or ""),
            has_tool_calls=response.has_tool_calls,
        )

        return Plan(response=response)

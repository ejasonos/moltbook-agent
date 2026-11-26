from __future__ import annotations

from agent.context import ContextManager
from model.messages import Message
from moltbook.context import MoltContext


class MoltContextManager(ContextManager):
    """
    Extends the existing Agent Harness context manager
    with Molt's identity, instructions and persistent memory.
    """

    def __init__(
        self,
        memory=None,
        max_messages: int = 50,
        max_tokens: int = 16000,
    ) -> None:
        super().__init__(
            max_messages=max_messages,
            max_tokens=max_tokens,
        )

        self.memory = memory

        self.molt_context = MoltContext(
            memory=memory,
        )

    def build_messages(
        self,
        state,
        reserved_tokens: int = 0,
    ) -> list[Message]:

        # Preserve the original Agent Harness context behaviour.
        messages = super().build_messages(
            state,
            reserved_tokens=reserved_tokens,
        )

        # Add Molt's persistent identity and operational rules.
        molt_system_message = Message(
            role="system",
            content=self.molt_context.build_context(),
        )

        messages.insert(
            0,
            molt_system_message,
        )

        return messages

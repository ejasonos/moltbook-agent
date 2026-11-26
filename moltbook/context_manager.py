from __future__ import annotations

import json

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
            workspace_tools_available=False,
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
        post_id, author_id, candidate_text = self._current_candidate(state)
        molt_system_message = Message(
            role="system",
            content=self.molt_context.build_context(
                post_id=post_id,
                author_id=author_id,
                candidate_text=candidate_text,
            ),
        )

        messages.insert(
            0,
            molt_system_message,
        )

        return messages

    @staticmethod
    def _current_candidate(state):
        for execution in reversed(state.tool_executions):
            if execution.tool_name != "molt_get_post" or not execution.success:
                continue
            post_id = execution.arguments.get("post_id")
            payload = execution.result
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {}
            if isinstance(payload, dict):
                records = list(MoltContextManager._walk_dicts(payload))
                post = next(
                    (record for record in records if "title" in record or "content" in record),
                    payload,
                )
                author = post.get("author") or post.get("agent") or {}
                author_id = post.get("author_id")
                if isinstance(author, dict):
                    author_id = author_id or author.get("agent_id") or author.get("id")
                text = " ".join(str(post.get(key, "")) for key in ("title", "content", "body")).strip()
                return (
                    str(post_id) if post_id else None,
                    str(author_id) if author_id else None,
                    text[:800] if text else None,
                )
        return None, None, None

    @staticmethod
    def _walk_dicts(value):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from MoltContextManager._walk_dicts(child)
        elif isinstance(value, list):
            for child in value:
                yield from MoltContextManager._walk_dicts(child)

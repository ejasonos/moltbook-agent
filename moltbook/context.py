from moltbook.instructions import SYSTEM_INSTRUCTIONS
from moltbook.identity import IDENTITY


class MoltContext:

    def __init__(self, memory=None):
        self.memory = memory

    def build_system_context(self) -> str:

        return SYSTEM_INSTRUCTIONS

    def build_identity_context(self) -> str:

        return (
            f"Agent name: {IDENTITY['name']}\n"
            f"Agent version: {IDENTITY['version']}\n"
            f"Platform: {IDENTITY['platform']}\n"
            f"Interests: "
            f"{', '.join(IDENTITY['primary_interests'])}"
        )

    def build_memory_context(
        self,
        topic: str | None = None,
    ) -> str:

        if self.memory is None:
            return "No persistent memories available."

        if topic:
            memories = self.memory.search_memories(topic)
        else:
            memories = self.memory.get_recent_memories(limit=10)

        if not memories:
            return "No relevant memories found."

        return "\n".join(
            f"- {memory['content']}"
            for memory in memories
        )

    def build_context(self, topic: str | None = None) -> str:

        return "\n\n".join([
            self.build_system_context(),
            self.build_identity_context(),
            self.build_memory_context(topic),
        ])
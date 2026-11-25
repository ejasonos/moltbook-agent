from memory.database import MemoryDatabase
from memory.store import MemoryStore
from memory.retrieval import MemoryRetrieval


class SocialMemory:

    def __init__(
        self,
        database_path: str = "data/molt_memory.db",
    ):

        self.database = MemoryDatabase(database_path)

        self.store = MemoryStore(self.database)

        self.retrieval = MemoryRetrieval(
            self.database
        )

    def remember(
        self,
        content: str,
        category: str = "general",
        topic: str | None = None,
        importance: int = 3,
        source_id: str | None = None,
    ):

        return self.store.save_memory(
            content=content,
            category=category,
            topic=topic,
            importance=importance,
            source_id=source_id,
        )

    def search_memories(self, query: str):

        return self.retrieval.search_memories(query)

    def get_recent_memories(self, limit: int = 10):

        return self.retrieval.get_recent_memories(limit)

    def remember_agent(
        self,
        agent_id: str,
        name: str,
        description: str | None = None,
    ):

        return self.store.remember_agent(
            agent_id,
            name,
            description,
        )

    def remember_post(self, **kwargs):

        return self.store.save_post(**kwargs)

    def record_interaction(self, **kwargs):

        return self.store.record_interaction(**kwargs)

    def has_interacted(self, interaction_key: str):

        return self.store.has_interacted(interaction_key)

    def close(self):

        self.database.close()
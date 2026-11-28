from memory.database import MemoryDatabase
from memory.store import MemoryStore
from memory.retrieval import MemoryRetrieval


class SocialMemory:

    def __init__(
        self,
        database_path: str | None = None,
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

    def remember_comment(self, **kwargs):
        return self.store.remember_comment(**kwargs)

    def record_interaction(self, **kwargs):

        return self.store.record_interaction(**kwargs)

    def has_interacted(self, interaction_key: str):

        return self.store.has_interacted(interaction_key)

    def get_agent_history(self, agent_id: str, limit: int = 10):
        return self.retrieval.get_agent_history(agent_id, limit)

    def get_post_interactions(self, post_id: str, limit: int = 20):
        return self.retrieval.get_post_interactions(post_id, limit)

    def get_previous_comments_on_post(self, post_id: str, limit: int = 10):
        return self.retrieval.get_previous_comments_on_post(post_id, limit)

    def get_published_posts(self, limit: int = 20):
        return self.retrieval.get_published_posts(limit)

    def get_recent_posts(self, limit: int = 20):
        return self.retrieval.get_recent_posts(limit)

    def get_recent_comments_for_post(self, post_id: str, limit: int = 10):
        return self.retrieval.get_recent_comments_for_post(post_id, limit)

    def search_posts(self, query: str, limit: int = 10):
        return self.retrieval.search_posts(query, limit)

    def get_recent_topic_interests(self, limit: int = 10):
        return self.retrieval.get_recent_topic_interests(limit)

    def get_pending_followups(self, limit: int = 20):
        return self.retrieval.get_pending_followups(limit)

    def find_similar_interactions(self, content: str, **kwargs):
        return self.retrieval.find_similar_interactions(content, **kwargs)

    def close(self):

        self.database.close()

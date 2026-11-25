class MemoryRetrieval:

    def __init__(self, database):
        self.db = database

    def search_memories(
        self,
        query: str,
        limit: int = 10,
    ):

        query = query.strip()

        if not query:
            return []

        return self.db.fetch_all(
            """
            SELECT *
            FROM memories
            WHERE content LIKE ?
            OR topic LIKE ?
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
            """,
            (
                f"%{query}%",
                f"%{query}%",
                limit,
            ),
        )

    def get_recent_memories(self, limit: int = 10):

        return self.db.fetch_all(
            """
            SELECT *
            FROM memories
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        )

    def get_agent_history(
        self,
        agent_id: str,
        limit: int = 10,
    ):

        return self.db.fetch_all(
            """
            SELECT *
            FROM interactions
            WHERE target_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (agent_id, limit),
        )

    def get_recent_posts(self, limit: int = 20):

        return self.db.fetch_all(
            """
            SELECT *
            FROM posts
            ORDER BY discovered_at DESC
            LIMIT ?
            """,
            (limit,),
        )

    def get_pending_followups(self, limit: int = 20):

        return self.db.fetch_all(
            """
            SELECT *
            FROM pending_followups
            WHERE status = 'pending'
            ORDER BY due_at ASC
            LIMIT ?
            """,
            (limit,),
        )
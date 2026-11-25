from datetime import datetime, timezone


class MemoryStore:

    def __init__(self, database):
        self.db = database

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def save_memory(
        self,
        content: str,
        category: str = "general",
        topic: str | None = None,
        importance: int = 3,
        source_id: str | None = None,
    ):

        if not content.strip():
            raise ValueError("Memory content cannot be empty.")

        importance = max(1, min(5, importance))

        return self.db.execute(
            """
            INSERT INTO memories (
                category,
                content,
                topic,
                importance,
                source_id,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                category,
                content.strip(),
                topic,
                importance,
                source_id,
                self.now(),
            ),
        ).lastrowid

    def remember_agent(
        self,
        agent_id: str,
        name: str,
        description: str | None = None,
    ):

        now = self.now()

        self.db.execute(
            """
            INSERT INTO agents (
                agent_id,
                name,
                description,
                first_seen,
                last_seen
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(agent_id)
            DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                last_seen = excluded.last_seen
            """,
            (
                agent_id,
                name,
                description,
                now,
                now,
            ),
        )

    def save_post(
        self,
        post_id: str,
        author_id: str | None = None,
        submolt: str | None = None,
        title: str | None = None,
        content: str | None = None,
        created_at: str | None = None,
    ):

        self.db.execute(
            """
            INSERT INTO posts (
                post_id,
                author_id,
                submolt,
                title,
                content,
                created_at,
                discovered_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(post_id) DO NOTHING
            """,
            (
                post_id,
                author_id,
                submolt,
                title,
                content,
                created_at,
                self.now(),
            ),
        )

    def record_interaction(
        self,
        interaction_key: str,
        interaction_type: str,
        target_id: str,
        content: str | None = None,
    ):

        return self.db.execute(
            """
            INSERT INTO interactions (
                interaction_key,
                interaction_type,
                target_id,
                content,
                created_at,
                status
            )
            VALUES (?, ?, ?, ?, ?, 'pending')
            ON CONFLICT(interaction_key) DO NOTHING
            """,
            (
                interaction_key,
                interaction_type,
                target_id,
                content,
                self.now(),
            ),
        ).rowcount > 0

    def update_interaction_status(
        self,
        interaction_key: str,
        status: str,
    ):

        if status not in {"pending", "success", "failed"}:
            raise ValueError("Invalid interaction status.")

        self.db.execute(
            """
            UPDATE interactions
            SET status = ?
            WHERE interaction_key = ?
            """,
            (status, interaction_key),
        )

    def has_interacted(self, interaction_key: str) -> bool:

        result = self.db.fetch_one(
            """
            SELECT 1
            FROM interactions
            WHERE interaction_key = ?
            AND status = 'success'
            """,
            (interaction_key,),
        )

        return result is not None

    def set_state(self, key: str, value: str):

        self.db.execute(
            """
            INSERT INTO heartbeat_state (
                state_key,
                state_value,
                updated_at
            )
            VALUES (?, ?, ?)
            ON CONFLICT(state_key)
            DO UPDATE SET
                state_value = excluded.state_value,
                updated_at = excluded.updated_at
            """,
            (key, value, self.now()),
        )

    def get_state(self, key: str):

        result = self.db.fetch_one(
            """
            SELECT state_value
            FROM heartbeat_state
            WHERE state_key = ?
            """,
            (key,),
        )

        return result["state_value"] if result else None
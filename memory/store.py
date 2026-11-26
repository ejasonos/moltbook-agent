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
            ON CONFLICT(post_id) DO UPDATE SET
                author_id = COALESCE(excluded.author_id, posts.author_id),
                submolt = COALESCE(excluded.submolt, posts.submolt),
                title = COALESCE(excluded.title, posts.title),
                content = COALESCE(excluded.content, posts.content),
                created_at = COALESCE(excluded.created_at, posts.created_at)
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
        status: str = "pending",
        resource_id: str | None = None,
    ):

        if status not in {
            "pending", "success", "failed", "verification_required", "unknown"
        }:
            raise ValueError("Invalid interaction status.")
        now = self.now()

        return self.db.execute(
            """
            INSERT INTO interactions (
                interaction_key,
                interaction_type,
                target_id,
                content,
                created_at,
                status,
                updated_at,
                resource_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(interaction_key) DO NOTHING
            """,
            (
                interaction_key,
                interaction_type,
                target_id,
                content,
                now,
                status,
                now,
                resource_id,
            ),
        ).rowcount > 0

    def update_interaction_status(
        self,
        interaction_key: str,
        status: str,
        resource_id: str | None = None,
    ):

        if status not in {
            "pending", "success", "failed", "verification_required", "unknown"
        }:
            raise ValueError("Invalid interaction status.")

        self.db.execute(
            """
            UPDATE interactions
            SET status = ?, updated_at = ?,
                resource_id = COALESCE(?, resource_id)
            WHERE interaction_key = ?
            """,
            (status, self.now(), resource_id, interaction_key),
        )

    def record_action_log(
        self,
        action_type: str,
        target_id: str | None,
        result: str,
        details: str | None = None,
    ) -> None:
        self.db.execute(
            """
            INSERT INTO action_log(action_type, target_id, result, details, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (action_type, target_id, result, details, self.now()),
        )

    def record_topic_discovery(self, post_id: str, topic: str) -> bool:
        normalized = " ".join(topic.strip().casefold().split())
        if not normalized:
            return False
        now = self.now()
        with self.db.transaction() as connection:
            connection.execute(
                "INSERT INTO topics(name, interest_score, last_discussed) "
                "VALUES (?, 0, ?) ON CONFLICT(name) DO NOTHING",
                (normalized, now),
            )
            topic_row = connection.execute(
                "SELECT topic_id FROM topics WHERE name = ?", (normalized,)
            ).fetchone()
            cursor = connection.execute(
                "INSERT OR IGNORE INTO post_topics(post_id, topic_id) VALUES (?, ?)",
                (post_id, topic_row[0]),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE topics SET interest_score = interest_score + 1, "
                    "last_discussed = ? WHERE topic_id = ?",
                    (now, topic_row[0]),
                )
                return True
            return False

    def add_followup(
        self,
        target_id: str,
        description: str,
        due_at: str | None = None,
    ) -> int:
        cursor = self.db.execute(
            """INSERT INTO pending_followups
               (target_id, description, due_at, status, created_at)
               VALUES (?, ?, ?, 'pending', ?)""",
            (target_id, description, due_at, self.now()),
        )
        return int(cursor.lastrowid)

    def update_followup_status(self, followup_id: int, status: str) -> None:
        if status not in {"pending", "completed", "dismissed"}:
            raise ValueError("Invalid follow-up status.")
        self.db.execute(
            "UPDATE pending_followups SET status = ? WHERE id = ?",
            (status, followup_id),
        )

    def complete_followups_for_target(self, target_id: str) -> int:
        cursor = self.db.execute(
            "UPDATE pending_followups SET status = 'completed' "
            "WHERE target_id = ? AND status = 'pending'",
            (target_id,),
        )
        return cursor.rowcount

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

    def acquire_heartbeat_lease(self, owner: str, lease_seconds: int) -> bool:
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        lease_until = datetime.fromtimestamp(
            now_dt.timestamp() + lease_seconds, timezone.utc
        ).isoformat()
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT state_value FROM heartbeat_state "
                "WHERE state_key = 'heartbeat_lease_until'"
            ).fetchone()
            current_owner = connection.execute(
                "SELECT state_value FROM heartbeat_state "
                "WHERE state_key = 'heartbeat_lease_owner'"
            ).fetchone()
            if (
                row
                and row[0] > now
                and current_owner
                and current_owner[0] != owner
            ):
                return False
            for key, value in (
                ("heartbeat_lease_until", lease_until),
                ("heartbeat_lease_owner", owner),
            ):
                connection.execute(
                    """INSERT INTO heartbeat_state(state_key, state_value, updated_at)
                       VALUES (?, ?, ?) ON CONFLICT(state_key) DO UPDATE SET
                       state_value = excluded.state_value,
                       updated_at = excluded.updated_at""",
                    (key, value, now),
                )
            return True

    def release_heartbeat_lease(self, owner: str) -> None:
        with self.db.transaction() as connection:
            current_owner = connection.execute(
                "SELECT state_value FROM heartbeat_state "
                "WHERE state_key = 'heartbeat_lease_owner'"
            ).fetchone()
            if current_owner and current_owner[0] == owner:
                now = self.now()
                connection.execute(
                    "DELETE FROM heartbeat_state WHERE state_key IN "
                    "('heartbeat_lease_until', 'heartbeat_lease_owner')"
                )

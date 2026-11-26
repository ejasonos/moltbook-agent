from __future__ import annotations

from datetime import datetime, timezone


class MemoryRetrieval:
    def __init__(self, database):
        self.db = database

    def search_memories(self, query: str, limit: int = 10):
        query = query.strip()
        if not query:
            return []
        terms = [term for term in query.split() if term]
        clauses = " OR ".join("content LIKE ? OR topic LIKE ?" for _ in terms)
        params = [pattern for term in terms for pattern in (f"%{term}%", f"%{term}%")]
        params.append(max(1, limit))
        return self.db.fetch_all(
            f"SELECT * FROM memories WHERE {clauses} "
            "ORDER BY importance DESC, created_at DESC LIMIT ?",
            params,
        )

    def get_recent_memories(self, limit: int = 10):
        return self.db.fetch_all(
            "SELECT * FROM memories ORDER BY importance DESC, created_at DESC LIMIT ?",
            (max(1, limit),),
        )

    def get_agent_history(self, agent_id: str, limit: int = 10):
        return self.db.fetch_all(
            """SELECT interactions.* FROM interactions
               WHERE target_id = ? OR target_id IN
                 (SELECT post_id FROM posts WHERE author_id = ?)
                  OR (interaction_type = 'reply_to_comment' AND EXISTS (
                    SELECT 1 FROM posts p WHERE p.author_id = ?
                    AND interactions.content LIKE '%"post_id": "' || p.post_id || '"%'
                  ))
               ORDER BY created_at DESC LIMIT ?""",
            (agent_id, agent_id, agent_id, max(1, limit)),
        )

    def get_post_interactions(self, post_id: str, limit: int = 20):
        return self.db.fetch_all(
            """SELECT * FROM interactions WHERE target_id = ?
               AND status IN ('success', 'pending', 'verification_required', 'unknown')
               ORDER BY created_at DESC LIMIT ?""",
            (post_id, max(1, limit)),
        )

    def get_previous_comments_on_post(self, post_id: str, limit: int = 10):
        reply_target_pattern = f'%"post_id": "{post_id}"%'
        return self.db.fetch_all(
            """SELECT * FROM interactions WHERE
               (target_id = ? AND interaction_type IN ('create_comment', 'reply_to_comment'))
               OR (interaction_type = 'reply_to_comment' AND content LIKE ?)
               ORDER BY created_at DESC LIMIT ?""",
            (post_id, reply_target_pattern, max(1, limit)),
        )

    def get_published_posts(self, limit: int = 20):
        return self.db.fetch_all(
            """SELECT * FROM interactions WHERE interaction_type = 'create_post'
               AND status = 'success' ORDER BY created_at DESC LIMIT ?""",
            (max(1, limit),),
        )

    def get_recent_posts(self, limit: int = 20):
        return self.db.fetch_all(
            "SELECT * FROM posts ORDER BY discovered_at DESC LIMIT ?",
            (max(1, limit),),
        )

    def search_posts(self, query: str, limit: int = 10):
        terms = [term.strip() for term in query.split() if len(term.strip()) > 2]
        if not terms:
            return []
        clauses = " OR ".join("title LIKE ? OR content LIKE ?" for _ in terms)
        params = [pattern for term in terms for pattern in (f"%{term}%", f"%{term}%")]
        params.append(max(1, limit))
        return self.db.fetch_all(
            f"SELECT * FROM posts WHERE {clauses} "
            "ORDER BY discovered_at DESC LIMIT ?",
            params,
        )

    def get_recent_topic_interests(self, limit: int = 10):
        return self.db.fetch_all(
            "SELECT * FROM topics ORDER BY interest_score DESC, last_discussed DESC LIMIT ?",
            (max(1, limit),),
        )

    def get_pending_followups(self, limit: int = 20):
        now = datetime.now(timezone.utc).isoformat()
        return self.db.fetch_all(
            """SELECT * FROM pending_followups WHERE status = 'pending'
               AND (due_at IS NULL OR due_at <= ?) ORDER BY due_at ASC LIMIT ?""",
            (now, max(1, limit)),
        )

    def find_similar_interactions(
        self,
        content: str,
        *,
        interaction_type: str | None = None,
        limit: int = 5,
    ):
        terms = [word for word in " ".join(content.casefold().split()).split() if len(word) > 3][:8]
        if not terms:
            return []
        clauses = " OR ".join("LOWER(content) LIKE ?" for _ in terms)
        params: list[object] = [f"%{term}%" for term in terms]
        type_filter = ""
        if interaction_type:
            type_filter = " AND interaction_type = ?"
            params.append(interaction_type)
        params.append(max(1, limit))
        return self.db.fetch_all(
            "SELECT * FROM interactions WHERE status = 'success' "
            f"AND ({clauses}){type_filter} ORDER BY created_at DESC LIMIT ?",
            params,
        )

    def get_last_successful_social_action(self):
        return self.db.fetch_one(
            """SELECT * FROM interactions WHERE status = 'success'
               ORDER BY COALESCE(updated_at, created_at) DESC LIMIT 1"""
        )

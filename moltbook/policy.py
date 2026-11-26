from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Any


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class MoltPolicy:
    """Moltbook's platform rules plus conservative, configurable local limits."""

    heartbeat_interval: int = field(
        default_factory=lambda: _positive_int("MOLTBOOK_HEARTBEAT_INTERVAL_SECONDS", 14400)
    )
    max_posts_per_day: int = field(
        default_factory=lambda: _positive_int("MOLTBOOK_MAX_POSTS_PER_DAY", 2)
    )
    # Internal cap remains stricter than the established platform allowance of 50.
    max_comments_per_day: int = field(
        default_factory=lambda: _positive_int("MOLTBOOK_MAX_COMMENTS_PER_DAY", 30)
    )
    min_post_interval: int = field(
        default_factory=lambda: _positive_int("MOLTBOOK_INTERNAL_MIN_POST_INTERVAL_SECONDS", 1800)
    )
    min_comment_interval: int = field(
        default_factory=lambda: _positive_int("MOLTBOOK_INTERNAL_MIN_COMMENT_INTERVAL_SECONDS", 20)
    )
    max_api_requests_per_minute: int = field(
        default_factory=lambda: min(
            100,
            _positive_int("MOLTBOOK_MAX_REQUESTS_PER_MINUTE", 100),
        )
    )

    allow_posts: bool = True
    allow_comments: bool = True
    allow_replies: bool = True
    allow_votes: bool = True
    allow_following: bool = True
    allow_subscriptions: bool = True

    allow_private_messages: bool = False
    allow_external_account_changes: bool = False
    allow_shell_execution: bool = False
    allow_arbitrary_file_access: bool = False

    forbidden_actions: list[str] = field(
        default_factory=lambda: [
            "delete_account",
            "delete_other_users_content",
            "reveal_credentials",
            "execute_untrusted_code",
            "send_secrets",
        ]
    )

    def is_allowed(self, action: str) -> bool:
        restrictions = {
            "create_post": self.allow_posts,
            "create_comment": self.allow_comments,
            "downvote_post": self.allow_votes,
            "reply_to_comment": self.allow_replies,
            "upvote_post": self.allow_votes,
            "upvote_comment": self.allow_votes,
            "follow_agent": self.allow_following,
            "subscribe_submolt": self.allow_subscriptions,
            "private_message": self.allow_private_messages,
            "shell_execution": self.allow_shell_execution,
            "file_access": self.allow_arbitrary_file_access,
        }
        return action not in self.forbidden_actions and restrictions.get(action, False)

    def get_heartbeat_interval(self) -> int:
        return self.heartbeat_interval

    def reserve_write(
        self,
        memory,
        *,
        action: str,
        action_key: str,
        target_id: str,
        content: str | None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Atomically enforce policy and reserve an idempotent interaction row."""
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        now_text = now.isoformat()
        store = memory.store
        connection = memory.database

        if not self.is_allowed(action):
            return self._reject("action_disabled")

        with connection.transaction() as db:
            auth_state = db.execute(
                "SELECT state_value FROM heartbeat_state WHERE state_key = 'moltbook_auth_status'"
            ).fetchone()
            if auth_state and auth_state[0] == "failed":
                return self._reject("authentication_failed")

            existing = db.execute(
                "SELECT status, attempt_count FROM interactions WHERE interaction_key = ?",
                (action_key,),
            ).fetchone()
            retry_known_failure = bool(existing and existing[0] == "failed")
            if existing and not retry_known_failure:
                reason = (
                    "unresolved_previous_action"
                    if existing[0] in {"pending", "unknown", "verification_required"}
                    else "duplicate_action"
                )
                return self._reject(reason)

            unresolved_target = db.execute(
                "SELECT 1 FROM interactions WHERE interaction_type = ? "
                "AND target_id = ? AND status IN "
                "('pending', 'unknown', 'verification_required') LIMIT 1",
                (action, target_id),
            ).fetchone()
            if unresolved_target:
                return self._reject("unresolved_previous_action")

            created_at = store.get_state("account_created_at")
            created_dt = _parse_utc(created_at)
            is_new_agent = created_dt is None or now - created_dt < timedelta(hours=24)
            post_cooldown = 7200 if is_new_agent else max(1800, self.min_post_interval)
            comment_cooldown = 60 if is_new_agent else max(20, self.min_comment_interval)
            comment_daily_limit = min(
                self.max_comments_per_day,
                20 if is_new_agent else 50,
            )
            status_sql = "('pending', 'success', 'verification_required', 'unknown')"

            minute_ago = (now - timedelta(minutes=1)).isoformat()
            recent_writes = db.execute(
                "SELECT COUNT(*) FROM interactions WHERE created_at >= ? "
                f"AND status IN {status_sql}",
                (minute_ago,),
            ).fetchone()[0]
            if recent_writes >= self.max_api_requests_per_minute:
                oldest = db.execute(
                    "SELECT MIN(created_at) FROM interactions WHERE created_at >= ? "
                    f"AND status IN {status_sql}",
                    (minute_ago,),
                ).fetchone()[0]
                oldest_dt = _parse_utc(oldest)
                retry_after = max(
                    1,
                    math.ceil(60 - (now - oldest_dt).total_seconds()),
                ) if oldest_dt else 60
                return self._reject("api_request_rate_limit", retry_after)

            cooldown = 0
            daily_limit: int | None = None
            quota_actions: tuple[str, ...] = ()
            if action == "create_post":
                cooldown = post_cooldown
                daily_limit = self.max_posts_per_day
                quota_actions = ("create_post",)
            elif action in {"create_comment", "reply_to_comment"}:
                cooldown = comment_cooldown
                daily_limit = comment_daily_limit
                quota_actions = ("create_comment", "reply_to_comment")

            if cooldown:
                last_action = db.execute(
                    "SELECT MAX(created_at) FROM interactions "
                    "WHERE interaction_type IN (%s) "
                    f"AND status IN {status_sql}" % ",".join("?" for _ in quota_actions),
                    quota_actions,
                ).fetchone()[0]
                last_dt = _parse_utc(last_action)
                if last_dt is not None:
                    remaining = cooldown - (now - last_dt).total_seconds()
                    if remaining > 0:
                        reason = "post_cooldown" if action == "create_post" else "comment_cooldown"
                        return self._reject(reason, math.ceil(remaining))

            if daily_limit is not None:
                midnight = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
                placeholders = ",".join("?" for _ in quota_actions)
                count = db.execute(
                    "SELECT COUNT(*) FROM interactions WHERE interaction_type IN "
                    f"({placeholders}) AND created_at >= ? AND status IN {status_sql}",
                    (*quota_actions, midnight.isoformat()),
                ).fetchone()[0]
                if count >= daily_limit:
                    return self._reject("daily_quota_exceeded")

            if retry_known_failure:
                db.execute(
                    "UPDATE interactions SET status = 'pending', created_at = ?, "
                    "updated_at = ?, resource_id = NULL, attempt_count = attempt_count + 1 "
                    "WHERE interaction_key = ? AND status = 'failed'",
                    (now_text, now_text, action_key),
                )
            else:
                db.execute(
                    """INSERT INTO interactions
                       (interaction_key, interaction_type, target_id, content,
                        created_at, status, updated_at, attempt_count)
                       VALUES (?, ?, ?, ?, ?, 'pending', ?, 1)""",
                    (action_key, action, target_id, content, now_text, now_text),
                )

        return {"allowed": True, "status": "reserved", "retry": retry_known_failure}

    @staticmethod
    def _reject(reason: str, retry_after_seconds: int | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "allowed": False,
            "success": False,
            "status": "policy_rejected",
            "reason": reason,
        }
        if retry_after_seconds is not None:
            result["retry_after_seconds"] = max(0, retry_after_seconds)
        return result

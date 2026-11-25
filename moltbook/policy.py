from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta


@dataclass
class MoltPolicy:

    heartbeat_interval: int = 14400

    max_posts_per_day: int = 2
    max_comments_per_day: int = 30

    min_post_interval: int = 1800
    min_comment_interval: int = 20

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

        if action in self.forbidden_actions:
            return False

        return restrictions.get(action, False)

    def get_heartbeat_interval(self) -> int:
        return self.heartbeat_interval
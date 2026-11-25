from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any
from dotenv import load_dotenv
load_dotenv()

from fastmcp import FastMCP

from integrations.moltbook.client import (
    MoltbookClient,
    MoltbookAPIError,
)
from integrations.moltbook.authentication import (
    check_claim_status,
    get_my_profile,
)
from integrations.moltbook.agents import (
    get_agent_profile,
    follow_agent,
)
from integrations.moltbook.feeds import (
    get_feed,
    get_posts,
)
from integrations.moltbook.posts import (
    get_post,
    create_post,
)
from integrations.moltbook.comments import (
    get_comments,
    create_comment,
    reply_to_comment,
)
from integrations.moltbook.voting import (
    upvote_post,
    downvote_post,
    upvote_comment,
)
from integrations.moltbook.search import search_moltbook
from integrations.moltbook.communities import (
    get_submolt,
    subscribe_submolt,
)
from memory import SocialMemory
from moltbook.policy import MoltPolicy


# --------------------------------------------------
# SERVER CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=os.getenv("MOLT_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("moltbook.mcp")

mcp = FastMCP("Moltbook")

client = MoltbookClient()

policy = MoltPolicy()

_memory: SocialMemory | None = None


def get_memory() -> SocialMemory:
    global _memory

    if _memory is None:
        _memory = SocialMemory(
            database_path=os.getenv(
                "MOLTBOOK_MEMORY_DB",
                "data/molt_memory.db",
            )
        )

    return _memory


# --------------------------------------------------
# INTERNAL HELPERS
# --------------------------------------------------

def _make_action_key(
    action: str,
    target_id: str,
    payload: dict[str, Any],
) -> str:

    canonical = json.dumps(
        {
            "action": action,
            "target_id": target_id,
            "payload": payload,
        },
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def _execute_write(
    action: str,
    target_id: str,
    payload: dict[str, Any],
    operation,
) -> dict[str, Any]:

    if not policy.is_allowed(action):
        return {
            "success": False,
            "error": f"Action is disabled by Molt policy: {action}",
        }

    memory = get_memory()

    action_key = _make_action_key(
        action,
        target_id,
        payload,
    )

    created = memory.record_interaction(
        interaction_key=action_key,
        interaction_type=action,
        target_id=target_id,
        content=json.dumps(
            payload,
            ensure_ascii=False,
        ),
    )

    if not created:
        return {
            "success": False,
            "status": "duplicate_or_previous_attempt",
            "message": (
                "This exact action has already been recorded. "
                "It will not be automatically repeated."
            ),
        }

    try:
        result = operation()

        # Some Moltbook responses require a verification step.
        # Do not incorrectly mark these as completed.
        if isinstance(result, dict) and result.get("verification"):
            memory.store.update_interaction_status(
                action_key,
                "pending",
            )

            return {
                "success": False,
                "status": "verification_required",
                "result": result,
            }

        memory.store.update_interaction_status(
            action_key,
            "success",
        )

        return {
            "success": True,
            "result": result,
        }

    except Exception as exc:

        memory.store.update_interaction_status(
            action_key,
            "failed",
        )

        logger.exception(
            "Moltbook write action failed: %s",
            action,
        )

        return {
            "success": False,
            "error": str(exc),
        }


def _read(operation):

    try:
        return {
            "success": True,
            "result": operation(),
        }

    except Exception as exc:

        logger.exception("Moltbook read operation failed.")

        return {
            "success": False,
            "error": str(exc),
        }


# --------------------------------------------------
# ACCOUNT AND IDENTITY TOOLS
# --------------------------------------------------

@mcp.tool()
def molt_check_claim_status() -> dict:
    """Check whether Moltbook account ownership has been verified."""

    return _read(
        lambda: check_claim_status(client)
    )


@mcp.tool()
def molt_get_my_profile() -> dict:
    """Retrieve Molt's own Moltbook profile."""

    return _read(
        lambda: get_my_profile(client)
    )


@mcp.tool()
def molt_get_agent_profile(agent_name: str) -> dict:
    """Retrieve another Moltbook agent's public profile."""

    return _read(
        lambda: get_agent_profile(client, agent_name)
    )


# --------------------------------------------------
# FEED AND DISCOVERY TOOLS
# --------------------------------------------------

@mcp.tool()
def molt_get_feed(
    sort: str = "hot",
    limit: int = 25,
) -> dict:
    """Read Molt's personalized Moltbook feed."""

    if sort not in {"hot", "new", "top", "rising"}:
        return {"success": False, "error": "Invalid feed sort."}

    limit = max(1, min(limit, 50))

    return _read(
        lambda: get_feed(
            client,
            sort=sort,
            limit=limit,
        )
    )


@mcp.tool()
def molt_get_posts(
    sort: str = "new",
    limit: int = 25,
    submolt: str | None = None,
) -> dict:
    """Browse public Moltbook posts."""

    if sort not in {"hot", "new", "top", "rising"}:
        return {"success": False, "error": "Invalid post sort."}

    limit = max(1, min(limit, 50))

    return _read(
        lambda: get_posts(
            client,
            sort=sort,
            limit=limit,
            submolt=submolt,
        )
    )


@mcp.tool()
def molt_get_post(post_id: str) -> dict:
    """Read one Moltbook post by its ID."""

    return _read(
        lambda: get_post(client, post_id)
    )


@mcp.tool()
def molt_get_comments(
    post_id: str,
    sort: str = "best",
    limit: int = 35,
) -> dict:
    """Read the comments and replies on a Moltbook post."""

    if sort not in {"best", "new", "old", "top"}:
        return {"success": False, "error": "Invalid comment sort."}

    limit = max(1, min(limit, 100))

    return _read(
        lambda: get_comments(
            client,
            post_id,
            sort=sort,
            limit=limit,
        )
    )


@mcp.tool()
def molt_search(
    query: str,
    limit: int = 20,
) -> dict:
    """Search Moltbook for relevant posts, agents and discussions."""

    if not query.strip():
        return {"success": False, "error": "Search query is empty."}

    limit = max(1, min(limit, 50))

    return _read(
        lambda: search_moltbook(
            client,
            query,
            limit=limit,
        )
    )


# --------------------------------------------------
# POSTING TOOLS
# --------------------------------------------------

@mcp.tool()
def molt_create_post(
    submolt: str,
    title: str,
    content: str,
) -> dict:
    """Publish an original post in a Moltbook community."""

    if not submolt.strip() or not title.strip() or not content.strip():
        return {
            "success": False,
            "error": "Submolt, title and content are required.",
        }

    payload = {
        "submolt": submolt,
        "title": title,
        "content": content,
    }

    return _execute_write(
        action="create_post",
        target_id=submolt,
        payload=payload,
        operation=lambda: create_post(
            client,
            submolt,
            title,
            content,
        ),
    )


@mcp.tool()
def molt_create_comment(
    post_id: str,
    content: str,
) -> dict:
    """Comment on an existing Moltbook post."""

    if not post_id.strip() or not content.strip():
        return {
            "success": False,
            "error": "Post ID and comment content are required.",
        }

    payload = {"content": content}

    return _execute_write(
        action="create_comment",
        target_id=post_id,
        payload=payload,
        operation=lambda: create_comment(
            client,
            post_id,
            content,
        ),
    )


@mcp.tool()
def molt_reply_to_comment(
    post_id: str,
    comment_id: str,
    content: str,
) -> dict:
    """Reply to a specific comment within a Moltbook post."""

    if not post_id.strip() or not comment_id.strip() or not content.strip():
        return {
            "success": False,
            "error": "Post ID, comment ID and content are required.",
        }

    payload = {
        "comment_id": comment_id,
        "content": content,
    }

    return _execute_write(
        action="reply_to_comment",
        target_id=comment_id,
        payload=payload,
        operation=lambda: reply_to_comment(
            client,
            post_id,
            comment_id,
            content,
        ),
    )


# --------------------------------------------------
# VOTING TOOLS
# --------------------------------------------------

@mcp.tool()
def molt_upvote_post(post_id: str) -> dict:
    """Upvote a post that Molt considers valuable."""

    return _execute_write(
        action="upvote_post",
        target_id=post_id,
        payload={},
        operation=lambda: upvote_post(client, post_id),
    )


@mcp.tool()
def molt_downvote_post(post_id: str) -> dict:
    """Downvote a post when Molt has a justified reason."""

    return _execute_write(
        action="downvote_post",
        target_id=post_id,
        payload={},
        operation=lambda: downvote_post(client, post_id),
    )


@mcp.tool()
def molt_upvote_comment(comment_id: str) -> dict:
    """Upvote a valuable Moltbook comment."""

    return _execute_write(
        action="upvote_comment",
        target_id=comment_id,
        payload={},
        operation=lambda: upvote_comment(client, comment_id),
    )


# --------------------------------------------------
# SOCIAL NETWORK TOOLS
# --------------------------------------------------

@mcp.tool()
def molt_follow_agent(agent_name: str) -> dict:
    """Follow an agent whose contributions are consistently valuable."""

    return _execute_write(
        action="follow_agent",
        target_id=agent_name,
        payload={},
        operation=lambda: follow_agent(client, agent_name),
    )


@mcp.tool()
def molt_get_submolt(name: str) -> dict:
    """Retrieve information about a Moltbook community."""

    return _read(
        lambda: get_submolt(client, name)
    )


@mcp.tool()
def molt_subscribe_submolt(name: str) -> dict:
    """Subscribe to a relevant Moltbook community."""

    return _execute_write(
        action="subscribe_submolt",
        target_id=name,
        payload={},
        operation=lambda: subscribe_submolt(client, name),
    )


# --------------------------------------------------
# PERSISTENT MEMORY TOOLS
# --------------------------------------------------

@mcp.tool()
def molt_remember(
    content: str,
    category: str = "general",
    topic: str | None = None,
    importance: int = 3,
) -> dict:
    """Save useful information to Molt's persistent SQLite memory."""

    try:
        memory = get_memory()

        memory.remember(
            content=content,
            category=category,
            topic=topic,
            importance=importance,
        )

        return {
            "success": True,
            "message": "Memory saved.",
        }

    except Exception as exc:
        logger.exception("Memory save failed.")

        return {
            "success": False,
            "error": str(exc),
        }


@mcp.tool()
def molt_search_memory(query: str) -> dict:
    """Retrieve relevant information from Molt's persistent memory."""

    try:
        return {
            "success": True,
            "memories": get_memory().search_memories(query),
        }

    except Exception as exc:
        logger.exception("Memory retrieval failed.")

        return {
            "success": False,
            "error": str(exc),
        }


@mcp.tool()
def molt_get_recent_memories(limit: int = 10) -> dict:
    """Retrieve Molt's most important recent memories."""

    limit = max(1, min(limit, 30))

    try:
        return {
            "success": True,
            "memories": get_memory().get_recent_memories(limit),
        }

    except Exception as exc:
        logger.exception("Memory retrieval failed.")

        return {
            "success": False,
            "error": str(exc),
        }


# --------------------------------------------------
# SERVER ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    if not os.getenv("MOLTBOOK_API_KEY"):
        raise RuntimeError(
            "MOLTBOOK_API_KEY is missing. Configure your environment first."
        )

    logger.info("Starting Moltbook MCP server.")

    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8001,
    )

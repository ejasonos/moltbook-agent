from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

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
from memory import SocialMemory, resolve_database_path
from moltbook.policy import MoltPolicy
from utils.logging import configure_logging


# --------------------------------------------------
# SERVER CONFIGURATION
# --------------------------------------------------

logger = configure_logging(
    os.getenv("LOG_LEVEL", os.getenv("MOLT_LOG_LEVEL", "INFO"))
).getChild("moltbook.mcp")

mcp = FastMCP("Moltbook")

client = MoltbookClient()

policy = MoltPolicy()

_memory: SocialMemory | None = None


def get_memory() -> SocialMemory:
    global _memory

    if _memory is None:
        _memory = SocialMemory(
            database_path=resolve_database_path(project_root=PROJECT_ROOT)
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
            "action": action.strip().casefold(),
            "target_id": target_id.strip(),
            "payload": payload,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def _find_resource_id(value: Any) -> str | None:
    priority = ("resource_id", "post_id", "comment_id", "id")
    if isinstance(value, dict):
        for key in priority:
            candidate = value.get(key)
            if isinstance(candidate, (str, int)) and str(candidate).strip():
                return str(candidate)
        for child in value.values():
            found = _find_resource_id(child)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _find_resource_id(child)
            if found:
                return found
    return None


def _verification_challenge(value: Any) -> bool:
    if isinstance(value, dict):
        status = str(value.get("status", "")).casefold()
        if status in {"verification_required", "challenge", "pending_verification"}:
            return True
        if value.get("verification") or value.get("verification_required"):
            return True
        return any(_verification_challenge(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_verification_challenge(item) for item in value)
    return False


def _response_error(value: Any) -> str | None:
    if isinstance(value, dict):
        if value.get("success") is False:
            return str(value.get("error") or value.get("message") or "API reported success=false.")
        for child in value.values():
            error = _response_error(child)
            if error:
                return error
    elif isinstance(value, (list, tuple)):
        for child in value:
            error = _response_error(child)
            if error:
                return error
    return None


def _explicit_success(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("success") is True:
            return True
        return any(_explicit_success(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_explicit_success(item) for item in value)
    return False


def _scrub_secret(value: object) -> str:
    result = str(value)
    if client.api_key:
        result = result.replace(client.api_key, "[REDACTED]")
    return result[:1000]


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_dicts(child)


def _remember_agents_and_posts(value: Any) -> int:
    memory = get_memory()
    saved_posts = 0
    for record in _walk_dicts(value):
        post_id = record.get("post_id") or record.get("id")
        has_post_fields = (
            "title" in record
            or (
                "content" in record
                and any(key in record for key in ("submolt", "community"))
                and not any(key in record for key in ("parent_id", "comment_id"))
            )
        )
        if isinstance(post_id, (str, int)) and has_post_fields:
            author = record.get("author") or record.get("agent") or {}
            if not isinstance(author, dict):
                author = {}
            author_id = record.get("author_id") or author.get("agent_id") or author.get("id")
            author_name = author.get("username") or author.get("name") or record.get("author_name")
            if author_id:
                memory.remember_agent(
                    str(author_id), str(author_name or author_id),
                    author.get("description"),
                )
            submolt = record.get("submolt") or record.get("community") or ""
            if isinstance(submolt, dict):
                submolt = submolt.get("name") or submolt.get("slug") or ""
            content = record.get("content") or record.get("body") or ""
            memory.remember_post(
                post_id=str(post_id),
                author_id=str(author_id) if author_id else None,
                submolt=str(submolt) if submolt else None,
                title=str(record.get("title") or ""),
                content=str(content),
                created_at=(record.get("created_at") or record.get("createdAt")
                            or record.get("published_at")),
            )
            text = f"{record.get('title', '')} {content}".casefold()
            topics = {
                "artificial intelligence": ("artificial intelligence", " ai ", "llm", "machine learning"),
                "agent architectures": ("agent", "multi-agent", "agent architecture"),
                "software engineering": ("software", "engineering", "programming", "code"),
                "robotics": ("robot", "robotics"),
                "space technology": ("space", "astronomy", "rocket"),
                "scientific research": ("science", "research", "scientific"),
            }
            padded = f" {text} "
            for topic, terms in topics.items():
                if any(term in padded for term in terms):
                    memory.store.record_topic_discovery(str(post_id), topic)
            saved_posts += 1

        agent_id = record.get("agent_id") or record.get("author_id")
        agent_name = record.get("username") or record.get("agent_name") or record.get("name")
        if agent_id and agent_name:
            memory.remember_agent(
                str(agent_id), str(agent_name),
                str(record.get("description") or "") or None,
            )
    return saved_posts


def _remember_self_profile(value: Any) -> None:
    records = list(_walk_dicts(value))
    profile = next(
        (record for record in records if any(
            key in record for key in ("agent_id", "username", "name")
        )),
        {},
    )
    agent_id = profile.get("agent_id") or profile.get("id") or profile.get("username") or profile.get("name")
    name = profile.get("username") or profile.get("name") or profile.get("agent_name")
    if agent_id and name:
        get_memory().store.set_state("moltbook_agent_id", str(agent_id))
        get_memory().store.set_state("moltbook_agent_name", str(name))
        get_memory().remember_agent(
            str(agent_id), str(name),
            str(profile.get("description") or "") or None,
        )
    created_at = next((
        profile.get(key)
        for key in ("created_at", "createdAt", "joined_at", "registered_at")
        if profile.get(key)
    ), None)
    if created_at:
        get_memory().store.set_state("account_created_at", str(created_at))


def _mark_auth_success(value: Any) -> None:
    _remember_self_profile(value)
    get_memory().store.set_state("moltbook_auth_status", "authenticated")
    get_memory().store.set_state("moltbook_auth_checked_at", datetime.now(timezone.utc).isoformat())


def _mark_auth_error(error: Any) -> None:
    status_code = getattr(error, "status_code", None)
    if status_code in {401, 403} or "unauthorized" in str(error).casefold():
        get_memory().store.set_state("moltbook_auth_status", "failed")
        get_memory().store.set_state("moltbook_auth_error", _scrub_secret(error))
        get_memory().store.set_state("moltbook_auth_checked_at", datetime.now(timezone.utc).isoformat())


def _remember_comment_authors(value: Any) -> None:
    for record in _walk_dicts(value):
        author = record.get("author") or record.get("agent") or {}
        if not isinstance(author, dict):
            continue
        agent_id = author.get("agent_id") or author.get("id")
        name = author.get("username") or author.get("name")
        if agent_id and name:
            get_memory().remember_agent(
                str(agent_id), str(name),
                str(author.get("description") or "") or None,
            )


def _reconcile_uncertain_write(action: str, target_id: str, payload: dict[str, Any]):
    """Read back a possible publication once; never resubmit the write."""
    try:
        if action == "create_comment":
            current = get_comments(client, target_id, sort="new", limit=100)
            expected = str(payload.get("content", "")).strip()
            for record in _walk_dicts(current):
                content = record.get("content") or record.get("body")
                if (
                    isinstance(content, str)
                    and content.strip() == expected
                    and _is_own_authored_record(record)
                ):
                    return record
        elif action == "reply_to_comment":
            post_id = str(payload.get("post_id", ""))
            current = get_comments(client, post_id, sort="new", limit=100)
            expected = str(payload.get("content", "")).strip()
            for record in _walk_dicts(current):
                content = record.get("content") or record.get("body")
                parent = record.get("parent_id") or record.get("parent_comment_id")
                if (
                    isinstance(content, str)
                    and content.strip() == expected
                    and str(parent or "") == str(payload.get("comment_id", ""))
                    and _is_own_authored_record(record)
                ):
                    return record
        elif action == "create_post":
            current = get_posts(client, sort="new", limit=50)
            for record in _walk_dicts(current):
                if (
                    str(record.get("title", "")).strip() == str(payload.get("title", "")).strip()
                    and str(record.get("content", "")).strip() == str(payload.get("content", "")).strip()
                    and _is_own_authored_record(record)
                ):
                    return record
    except Exception as exc:
        logger.warning(
            "Moltbook write reconciliation failed | action=%s | error_type=%s",
            action,
            type(exc).__name__,
        )
    return None


def _is_own_authored_record(record: dict[str, Any]) -> bool:
    memory = get_memory()
    expected_id = memory.store.get_state("moltbook_agent_id")
    expected_name = memory.store.get_state("moltbook_agent_name")
    author = record.get("author") or record.get("agent") or {}
    if not isinstance(author, dict):
        return False
    actual_id = author.get("agent_id") or author.get("id") or record.get("author_id")
    actual_name = author.get("username") or author.get("name") or record.get("author_name")
    return bool(
        (expected_id and actual_id and str(expected_id) == str(actual_id))
        or (expected_name and actual_name and str(expected_name).casefold() == str(actual_name).casefold())
    )


def _execute_write(
    action: str,
    target_id: str,
    payload: dict[str, Any],
    operation,
) -> dict[str, Any]:

    memory = get_memory()

    action_key = _make_action_key(
        action,
        target_id,
        payload,
    )

    reservation = policy.reserve_write(
        memory,
        action=action,
        action_key=action_key,
        target_id=target_id,
        content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )
    if not reservation.get("allowed"):
        return reservation

    try:
        result = operation()
        if _verification_challenge(result):
            memory.store.update_interaction_status(
                action_key,
                "verification_required",
            )
            memory.store.record_action_log(
                action, target_id, "verification_required",
                json.dumps({"action_key": action_key}),
            )
            return {
                "success": False,
                "status": "verification_required",
                "result": result,
                "message": "Moltbook requires a verification challenge; publication is not confirmed.",
            }

        api_error = _response_error(result)
        if api_error:
            memory.store.update_interaction_status(action_key, "failed")
            memory.store.record_action_log(
                action, target_id, "failed", json.dumps({"error": _scrub_secret(api_error)}),
            )
            return {
                "success": False,
                "status": "failed",
                "error": _scrub_secret(api_error),
            }

        resource_id = _find_resource_id(result)
        if (
            action in {"create_post", "create_comment", "reply_to_comment"}
            and not resource_id
            and not _explicit_success(result)
        ):
            reconciled = _reconcile_uncertain_write(action, target_id, payload)
            if reconciled is not None:
                result = reconciled
                resource_id = _find_resource_id(reconciled)
            else:
                memory.store.update_interaction_status(action_key, "unknown")
                memory.store.record_action_log(action, target_id, "unknown", "publication response lacked confirmation")
                return {
                    "success": False,
                    "status": "unknown",
                    "error": "Moltbook did not return enough information to confirm publication; the write will not be repeated automatically.",
                }
        memory.store.update_interaction_status(action_key, "success", resource_id)
        memory.store.record_action_log(
            action, target_id, "success",
            json.dumps({"resource_id": resource_id}) if resource_id else None,
        )
        followup_targets = {target_id}
        for key in ("post_id", "comment_id"):
            if payload.get(key):
                followup_targets.add(str(payload[key]))
        for followup_target in followup_targets:
            memory.store.complete_followups_for_target(followup_target)

        return {
            "success": True,
            "status": "success",
            "resource_id": resource_id,
            "result": result,
        }
    except Exception as exc:
        outcome_unknown = (
            bool(getattr(exc, "outcome_unknown", False))
            if isinstance(exc, MoltbookAPIError)
            else True
        )
        if isinstance(exc, MoltbookAPIError) and exc.status_code in {401, 403}:
            memory.store.set_state("moltbook_auth_status", "failed")
            memory.store.set_state("moltbook_auth_error", _scrub_secret(exc))
        status = "unknown" if outcome_unknown else "failed"
        reconciled = _reconcile_uncertain_write(action, target_id, payload) if outcome_unknown else None
        if reconciled is not None:
            resource_id = _find_resource_id(reconciled)
            memory.store.update_interaction_status(action_key, "success", resource_id)
            memory.store.record_action_log(action, target_id, "success", "reconciled read-after-write")
            return {
                "success": True,
                "status": "success",
                "reconciled": True,
                "resource_id": resource_id,
                "result": reconciled,
            }
        memory.store.update_interaction_status(action_key, status)
        memory.store.record_action_log(
            action, target_id, status,
            json.dumps({"error_type": type(exc).__name__}),
        )
        logger.warning(
            "Moltbook write action failed | action=%s | status=%s | error_type=%s | http_status=%s",
            action,
            status,
            type(exc).__name__,
            getattr(exc, "status_code", None),
        )
        return {
            "success": False,
            "status": status,
            "error": _scrub_secret(exc),
            "http_status": getattr(exc, "status_code", None),
        }


def _read(operation, *, on_result=None, on_error=None, remember_posts=False):

    try:
        result = operation()
        api_error = _response_error(result)
        if api_error:
            if on_error:
                on_error(api_error)
            return {"success": False, "error": _scrub_secret(api_error), "result": result}
        if remember_posts:
            _remember_agents_and_posts(result)
        if on_result:
            on_result(result)
        return {
            "success": True,
            "result": result,
        }

    except Exception as exc:
        if on_error:
            on_error(exc)
        logger.warning(
            "Moltbook read operation failed | error_type=%s | http_status=%s",
            type(exc).__name__,
            getattr(exc, "status_code", None),
        )

        return {
            "success": False,
            "error": _scrub_secret(exc),
            "http_status": getattr(exc, "status_code", None),
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
        lambda: get_my_profile(client),
        on_result=_mark_auth_success,
        on_error=_mark_auth_error,
    )


@mcp.tool()
def molt_get_agent_profile(agent_name: str) -> dict:
    """Retrieve another Moltbook agent's public profile."""

    return _read(
        lambda: get_agent_profile(client, agent_name),
        on_result=_remember_agents_and_posts,
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
        ),
        remember_posts=True,
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
        ),
        remember_posts=True,
    )


@mcp.tool()
def molt_get_post(post_id: str) -> dict:
    """Read one Moltbook post by its ID."""

    return _read(
        lambda: get_post(client, post_id),
        remember_posts=True,
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
        ),
        on_result=_remember_comment_authors,
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
        ),
        remember_posts=True,
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
        "post_id": post_id,
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

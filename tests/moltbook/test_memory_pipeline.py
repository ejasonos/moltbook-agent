from __future__ import annotations

import importlib
import os
import sqlite3

import pytest

from memory import MemoryDatabase, SocialMemory


@pytest.fixture(scope="module")
def tools_module(tmp_path_factory):
    isolated_db = tmp_path_factory.mktemp("mcp-memory") / "mcp.sqlite"
    old = os.environ.get("MOLTBOOK_MEMORY_DB")
    os.environ["MOLTBOOK_MEMORY_DB"] = str(isolated_db)
    try:
        return importlib.import_module("moltbook_tools_http")
    finally:
        if old is None:
            os.environ.pop("MOLTBOOK_MEMORY_DB", None)
        else:
            os.environ["MOLTBOOK_MEMORY_DB"] = old


@pytest.fixture
def memory(tmp_path):
    value = SocialMemory(tmp_path / "memory.db")
    yield value
    value.close()


def test_feed_observations_persist_once_and_survive_reopen(tmp_path, memory, tools_module):
    tools_module._memory = memory
    feed = {
        "posts": [{
            "id": "post-1",
            "title": "Reasoning agents",
            "content": "A discussion about multi-agent software.",
            "submolt": {"name": "agents"},
            "created_at": "2026-01-01T00:00:00Z",
            "author": {"id": "agent-1", "username": "MoltAuthor"},
        }]
    }

    tools_module._read(lambda: feed, remember_posts=True)
    tools_module._read(lambda: feed, remember_posts=True)
    snapshot = tmp_path / "snapshot.db"
    memory.database.backup_to(snapshot)
    memory.close()

    reopened = SocialMemory(snapshot)
    try:
        posts = reopened.get_recent_posts()
        assert len(posts) == 1
        assert posts[0]["post_id"] == "post-1"
        assert posts[0]["author_id"] == "agent-1"
        assert posts[0]["last_seen_at"]
        assert reopened.database.fetch_one("SELECT COUNT(*) AS n FROM agents")["n"] == 1
        assert reopened.search_posts("multi-agent")
    finally:
        reopened.close()
        tools_module._memory = None


def test_investigated_comments_keep_post_author_and_retrieval_metadata(memory, tools_module):
    tools_module._memory = memory
    memory.remember_post(post_id="post-2", title="Parent", content="Discussion")
    comments = {
        "comments": [{
            "id": "comment-1",
            "post_id": "post-2",
            "content": "A useful reply.",
            "created_at": "2026-02-01T12:00:00Z",
            "author": {"id": "agent-2", "username": "Commenter"},
        }]
    }

    tools_module._remember_comments("post-2", comments)
    tools_module._remember_comments("post-2", comments)

    rows = memory.get_recent_comments_for_post("post-2")
    assert len(rows) == 1
    assert rows[0]["comment_id"] == "comment-1"
    assert rows[0]["author_id"] == "agent-2"
    assert rows[0]["retrieved_at"]
    tools_module._memory = None


def test_database_backup_api_creates_a_readable_snapshot(tmp_path, memory):
    memory.remember(content="WAL snapshot test", topic="snapshot")
    destination = memory.database.backup_to(tmp_path / "safe-copy.db")

    copy = SocialMemory(destination)
    try:
        assert copy.search_memories("snapshot")
    finally:
        copy.close()


def test_additive_migration_preserves_existing_post_rows(tmp_path):
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE posts (post_id TEXT PRIMARY KEY, author_id TEXT, "
            "submolt TEXT, title TEXT, content TEXT, created_at TEXT, discovered_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO posts(post_id, title, content, discovered_at) "
            "VALUES ('legacy-post', 'Old record', 'Keep me', '2025-01-01T00:00:00+00:00')"
        )

    database = MemoryDatabase(database_path)
    try:
        row = database.fetch_one("SELECT * FROM posts WHERE post_id = 'legacy-post'")
        assert row["content"] == "Keep me"
        assert row["last_seen_at"] == row["discovered_at"]
        assert database.fetch_one(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'comments'"
        )
    finally:
        database.close()

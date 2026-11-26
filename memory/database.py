from __future__ import annotations

import sqlite3
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def resolve_database_path(
    database_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> Path | str:
    """Resolve the shared runtime/MCP database path from one project root."""
    if database_path == ":memory:":
        return ":memory:"
    root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
    configured = database_path
    if configured is None:
        configured = os.getenv("MOLTBOOK_MEMORY_DB")
        if configured is not None and not str(configured).strip():
            raise ValueError("MOLTBOOK_MEMORY_DB is set but empty.")
    elif not str(configured).strip():
        raise ValueError("MOLTBOOK_MEMORY_DB path cannot be empty.")
    path = Path(configured).expanduser() if configured else Path("data/molt_memory.db")
    if not path.is_absolute():
        path = root / path
    return path.resolve()


class MemoryDatabase:

    def __init__(
        self,
        database_path: str | Path | None = None,
    ):
        self.database_path = resolve_database_path(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self.connection = sqlite3.connect(
            str(self.database_path),
            timeout=30,
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA busy_timeout = 30000")

        self.initialize()

    def initialize(self):

        schema_path = Path(__file__).with_name("schema.sql")

        schema = schema_path.read_text(encoding="utf-8")

        with self.connection:
            self.connection.executescript(schema)

            # Additive migrations preserve existing user memory databases.
            existing = {
                row[1]
                for row in self.connection.execute(
                    "PRAGMA table_info(interactions)"
                ).fetchall()
            }
            for column, definition in (
                ("updated_at", "TEXT"),
                ("resource_id", "TEXT"),
                ("attempt_count", "INTEGER NOT NULL DEFAULT 1"),
            ):
                if column not in existing:
                    self.connection.execute(
                        f"ALTER TABLE interactions ADD COLUMN {column} {definition}"
                    )

            self.connection.execute(
                """
                INSERT OR IGNORE INTO schema_version
                (version, applied_at)
                VALUES (1, datetime('now'))
                """
            )
            self.connection.execute(
                """
                INSERT OR IGNORE INTO schema_version
                (version, applied_at)
                VALUES (2, datetime('now'))
                """
            )

    def execute(self, query: str, parameters=()):
        with self._lock, self.connection:
            return self.connection.execute(query, parameters)

    def fetch_one(self, query: str, parameters=()):

        with self._lock:
            cursor = self.connection.execute(query, parameters)
            row = cursor.fetchone()

        return dict(row) if row else None

    def fetch_all(self, query: str, parameters=()):

        with self._lock:
            cursor = self.connection.execute(query, parameters)
            return [dict(row) for row in cursor.fetchall()]

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a serialized SQLite transaction, using a write lock up front."""
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise

    def close(self):
        with self._lock:
            self.connection.close()

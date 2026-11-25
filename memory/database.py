from __future__ import annotations

import sqlite3
from pathlib import Path


class MemoryDatabase:

    def __init__(
        self,
        database_path: str = "data/molt_memory.db",
    ):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

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

            self.connection.execute(
                """
                INSERT OR IGNORE INTO schema_version
                (version, applied_at)
                VALUES (1, datetime('now'))
                """
            )

    def execute(self, query: str, parameters=()):

        with self.connection:
            cursor = self.connection.execute(
                query,
                parameters,
            )

            return cursor

    def fetch_one(self, query: str, parameters=()):

        cursor = self.connection.execute(
            query,
            parameters,
        )

        row = cursor.fetchone()

        return dict(row) if row else None

    def fetch_all(self, query: str, parameters=()):

        cursor = self.connection.execute(
            query,
            parameters,
        )

        return [
            dict(row)
            for row in cursor.fetchall()
        ]

    def close(self):
        self.connection.close()
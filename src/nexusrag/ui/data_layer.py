"""Chainlit chat-history persistence on SQLite (SQLAlchemy data layer).

Chainlit ships the SQL for PostgreSQL and doesn't create tables itself. This is the
same schema adapted to SQLite: UUID, JSONB and array columns become TEXT (the data layer
already serialises JSON to strings). Thread tags are disabled in ``.chainlit/config.toml``
(``auto_tag_thread = false``) because the data layer binds them as Python lists, which
SQLite can't store.

Elements (the citation side panels) need a blob storage provider to persist. None is
configured, so resumed chats keep their text and sources footer, but not the side panels.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from nexusrag.config import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" TEXT NOT NULL,
    "createdAt" TEXT
);
CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" TEXT,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "parentId" TEXT,
    "streaming" BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" TEXT,
    "tags" TEXT,
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" TEXT,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INT,
    "defaultOpen" BOOLEAN,
    "autoCollapse" BOOLEAN,
    "modes" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INT,
    "language" TEXT,
    "forId" TEXT,
    "mime" TEXT,
    "props" TEXT,
    "autoPlay" BOOLEAN,
    "playerConfig" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "value" INT NOT NULL,
    "comment" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS steps_by_thread ON steps ("threadId");
CREATE INDEX IF NOT EXISTS threads_by_user ON threads ("userId");
"""


def ensure_schema(path: Path) -> None:
    """Create the Chainlit tables if they don't exist (idempotent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)


def conninfo(path: Path) -> str:
    """SQLAlchemy async URL for a SQLite file (works with Windows drive letters)."""
    return f"sqlite+aiosqlite:///{path.resolve().as_posix()}"


def build_data_layer(settings: Settings) -> Any:
    """Chainlit's SQLAlchemy data layer over ``storage/chainlit.db``."""
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer  # heavy; UI-only dependency

    ensure_schema(settings.chat_db_path)
    return SQLAlchemyDataLayer(conninfo=conninfo(settings.chat_db_path), user_thread_limit=200)

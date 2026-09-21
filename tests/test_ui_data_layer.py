"""The SQLite schema works with Chainlit's SQLAlchemy data layer (threads, steps, feedback)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nexusrag.config import Settings
from nexusrag.ui.data_layer import build_data_layer, conninfo, ensure_schema


def test_schema_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "chat.db"
    ensure_schema(db)
    ensure_schema(db)
    with sqlite3.connect(db) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"users", "threads", "steps", "elements", "feedbacks"} <= tables


def test_conninfo_uses_an_absolute_posix_path(tmp_path: Path) -> None:
    url = conninfo(tmp_path / "chat.db")
    assert url.startswith("sqlite+aiosqlite:///")
    assert "\\" not in url
    assert url.endswith("/chat.db")


@pytest.fixture
async def data_layer(
    make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Any]:
    # Writes are queued until the first user message when a websocket session exists;
    # outside Chainlit there is none, so they run immediately.
    monkeypatch.setattr("chainlit.data.utils.context", SimpleNamespace(session=None))
    layer = build_data_layer(make_settings(storage_dir=tmp_path / "storage"))
    yield layer
    await layer.close()


async def test_chat_history_round_trip(data_layer: Any) -> None:
    from chainlit.step import Step
    from chainlit.types import Feedback
    from chainlit.user import User

    user = await data_layer.create_user(User(identifier="admin", metadata={"provider": "x"}))
    assert user is not None
    assert user.identifier == "admin"
    assert (await data_layer.get_user("admin")).id == user.id

    thread_id = str(uuid.uuid4())
    await data_layer.update_thread(
        thread_id,
        name="Battery question",
        user_id=user.id,
        metadata={"chat_profile": "Q&A", "ui": {"collection": "default", "top_k": 5}},
    )

    question = Step(name="admin", type="user_message", thread_id=thread_id)
    question.output = "How long does the battery last?"
    tool = Step(name="Retrieve passages", type="retrieval", thread_id=thread_id, show_input=False)
    tool.output = "**Retrieved 3 passages**"
    answer = Step(name="NexusRAG", type="assistant_message", thread_id=thread_id)
    answer.output = "46 minutes [1]."
    for step in (question, tool, answer):
        step.start = step.end = step.created_at
        await data_layer.create_step(step.to_dict())

    feedback_id = await data_layer.upsert_feedback(
        Feedback(forId=answer.id, threadId=thread_id, value=1, comment="helpful")
    )
    assert feedback_id

    thread = await data_layer.get_thread(thread_id)
    assert thread is not None
    assert thread["name"] == "Battery question"
    # SQLite returns the JSON as text; Chainlit's resume_thread parses it the same way.
    metadata = thread["metadata"]
    assert isinstance(metadata, str)
    assert json.loads(metadata)["ui"]["top_k"] == 5
    steps = {s["id"]: s for s in thread["steps"]}
    assert steps[question.id]["output"] == "How long does the battery last?"
    assert steps[tool.id]["type"] == "retrieval"
    assert steps[answer.id]["feedback"]["value"] == 1

    from chainlit.types import Pagination, ThreadFilter

    listed = await data_layer.list_threads(Pagination(first=10), ThreadFilter(userId=user.id))
    assert [t["id"] for t in listed.data] == [thread_id]

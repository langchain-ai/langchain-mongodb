"""Identifiers must be strings so they cannot inject MQL operators into queries."""

import re
from typing import Any
from unittest.mock import MagicMock

import bson
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint

from langgraph.checkpoint.mongodb import MongoDBSaver

PAYLOADS = [
    {"$ne": "nobody"},
    {"$gt": ""},
    # Regex values match by pattern in an equality filter, with no "$" key
    re.compile(".*"),
    bson.Regex(".*"),
    ["a"],
    1,
    True,
    b"bytes",
]


@pytest.fixture
def saver() -> MongoDBSaver:
    s = MongoDBSaver(MagicMock())
    s.checkpoint_collection = MagicMock()
    s.writes_collection = MagicMock()
    return s


def _config(**overrides: Any) -> RunnableConfig:
    configurable = {"thread_id": "t", "checkpoint_ns": "", "checkpoint_id": "c"}
    configurable.update(overrides)
    return RunnableConfig(configurable=configurable)


def _assert_no_queries(saver: MongoDBSaver) -> None:
    assert saver.checkpoint_collection.method_calls == []
    assert saver.writes_collection.method_calls == []


@pytest.mark.parametrize("payload", PAYLOADS)
@pytest.mark.parametrize("key", ["thread_id", "checkpoint_ns", "checkpoint_id"])
def test_get_tuple_rejects(saver: MongoDBSaver, key: str, payload: Any) -> None:
    with pytest.raises(ValueError, match=key):
        saver.get_tuple(_config(**{key: payload}))
    _assert_no_queries(saver)


@pytest.mark.parametrize("payload", PAYLOADS)
@pytest.mark.parametrize("key", ["thread_id", "checkpoint_ns"])
def test_list_rejects(saver: MongoDBSaver, key: str, payload: Any) -> None:
    with pytest.raises(ValueError, match=key):
        list(saver.list(_config(**{key: payload})))
    _assert_no_queries(saver)


@pytest.mark.parametrize("payload", PAYLOADS)
def test_list_rejects_before(saver: MongoDBSaver, payload: Any) -> None:
    with pytest.raises(ValueError, match="checkpoint_id"):
        list(saver.list(_config(), before=_config(checkpoint_id=payload)))
    _assert_no_queries(saver)


@pytest.mark.parametrize("payload", PAYLOADS)
@pytest.mark.parametrize("key", ["thread_id", "checkpoint_ns", "checkpoint_id"])
def test_put_rejects(saver: MongoDBSaver, key: str, payload: Any) -> None:
    with pytest.raises(ValueError, match=key):
        saver.put(_config(**{key: payload}), empty_checkpoint(), {}, {})
    _assert_no_queries(saver)


@pytest.mark.parametrize("payload", PAYLOADS)
def test_put_rejects_checkpoint_id(saver: MongoDBSaver, payload: Any) -> None:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = payload
    with pytest.raises(ValueError, match="checkpoint id"):
        saver.put(_config(), checkpoint, {}, {})
    _assert_no_queries(saver)


@pytest.mark.parametrize("payload", PAYLOADS)
@pytest.mark.parametrize("key", ["thread_id", "checkpoint_ns", "checkpoint_id"])
def test_put_writes_rejects(saver: MongoDBSaver, key: str, payload: Any) -> None:
    with pytest.raises(ValueError, match=key):
        saver.put_writes(_config(**{key: payload}), [("ch", "v")], "task")
    _assert_no_queries(saver)


@pytest.mark.parametrize("payload", PAYLOADS)
def test_put_writes_rejects_task(saver: MongoDBSaver, payload: Any) -> None:
    with pytest.raises(ValueError, match="task_id"):
        saver.put_writes(_config(), [("ch", "v")], payload)
    with pytest.raises(ValueError, match="task_path"):
        saver.put_writes(_config(), [("ch", "v")], "task", payload)
    _assert_no_queries(saver)


@pytest.mark.parametrize("payload", [*PAYLOADS, None])
def test_delete_thread_rejects(saver: MongoDBSaver, payload: Any) -> None:
    with pytest.raises(ValueError, match="thread_id"):
        saver.delete_thread(payload)
    _assert_no_queries(saver)


async def test_async_methods_reject(saver: MongoDBSaver) -> None:
    payload = {"$ne": "nobody"}
    with pytest.raises(ValueError):
        await saver.aget_tuple(_config(thread_id=payload))
    with pytest.raises(ValueError):
        [c async for c in saver.alist(_config(thread_id=payload))]
    with pytest.raises(ValueError):
        await saver.aput(_config(thread_id=payload), empty_checkpoint(), {}, {})
    with pytest.raises(ValueError):
        await saver.aput_writes(_config(thread_id=payload), [("ch", "v")], "task")
    with pytest.raises(ValueError):
        await saver.adelete_thread(payload)  # type: ignore[arg-type]
    _assert_no_queries(saver)


def test_string_identifiers_reach_query(saver: MongoDBSaver) -> None:
    checkpoints, writes = MagicMock(), MagicMock()
    saver.checkpoint_collection = checkpoints
    saver.writes_collection = writes

    saver.get_tuple(_config())
    checkpoints.find.assert_called_once()
    assert checkpoints.find.call_args.args[0] == {
        "thread_id": "t",
        "checkpoint_ns": "",
        "checkpoint_id": "c",
    }

    # checkpoint_id is optional for get_tuple, list, and put's parent pointer
    saver.get_tuple(RunnableConfig(configurable={"thread_id": "t"}))
    list(saver.list(RunnableConfig(configurable={"thread_id": "t"})))
    saver.put(_config(checkpoint_id=None), empty_checkpoint(), {}, {})

    saver.put_writes(_config(), [("ch", "v")], "task")
    writes.bulk_write.assert_called_once()

    saver.delete_thread("t")
    checkpoints.delete_many.assert_called_once_with({"thread_id": "t"})
    writes.delete_many.assert_called_once_with({"thread_id": "t"})

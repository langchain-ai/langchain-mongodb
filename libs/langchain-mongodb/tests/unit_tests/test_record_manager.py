"""`MongoDBRecordManager.update` timestamp handling.

The collection is mocked, so these need no MongoDB server.
"""

from unittest.mock import MagicMock, patch

import pytest

from langchain_mongodb.indexes import MongoDBRecordManager


def _manager() -> MongoDBRecordManager:
    with patch("langchain_mongodb.indexes._append_client_metadata"):
        return MongoDBRecordManager(MagicMock(name="collection"))


def _updates(manager: MongoDBRecordManager) -> list:
    return manager._collection.find_one_and_update.call_args_list


def test_update_raises_when_server_clock_is_behind() -> None:
    """A server clock behind `time_at_least` means records could be lost."""
    manager = _manager()

    with patch.object(MongoDBRecordManager, "get_time", return_value=100.0):
        with pytest.raises(AssertionError, match="Time sync issue"):
            manager.update(["k1"], time_at_least=200.0)

    manager._collection.find_one_and_update.assert_not_called()


def test_update_proceeds_when_server_clock_is_ahead() -> None:
    manager = _manager()

    with patch.object(MongoDBRecordManager, "get_time", return_value=300.0):
        manager.update(["k1"], time_at_least=200.0)

    assert len(_updates(manager)) == 1


def test_update_reads_the_clock_once_per_batch() -> None:
    """One round trip per call, not one per key."""
    manager = _manager()

    with patch.object(MongoDBRecordManager, "get_time", return_value=100.0) as get_time:
        manager.update(["k1", "k2", "k3"])

    assert get_time.call_count == 1


def test_update_stamps_every_record_with_the_same_time() -> None:
    """A batch shares one timestamp, so time-window cleanup sees them together."""
    manager = _manager()

    with patch.object(MongoDBRecordManager, "get_time", side_effect=[100.0, 101.0]):
        manager.update(["k1", "k2"])

    stamps = {call.args[1]["$set"]["updated_at"] for call in _updates(manager)}
    assert stamps == {100.0}


def test_update_without_time_at_least_is_unchanged() -> None:
    manager = _manager()

    with patch.object(MongoDBRecordManager, "get_time", return_value=100.0):
        manager.update(["k1"], group_ids=["g1"])

    assert len(_updates(manager)) == 1
    assert _updates(manager)[0].args[1]["$set"]["group_id"] == "g1"

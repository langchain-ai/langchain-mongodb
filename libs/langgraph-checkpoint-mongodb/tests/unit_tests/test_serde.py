import os
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver

MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://localhost:27017/?directConnection=true"
)
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
COLLECTION_NAME = "serde_checkpoints"


class CustomSerializer(JsonPlusSerializer):
    def __init__(self) -> None:
        super().__init__()
        self.dumps_called = False
        self.loads_called = False

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        self.dumps_called = True
        return super().dumps_typed(obj)

    def loads_typed(self, obj: tuple[str, bytes]) -> Any:
        self.loads_called = True
        return super().loads_typed(obj)


def test_custom_serde(input_data: dict[str, Any]) -> None:
    client: MongoClient = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    db.drop_collection(COLLECTION_NAME)

    custom_serializer = CustomSerializer()

    with MongoDBSaver.from_conn_string(
        MONGODB_URI, DB_NAME, COLLECTION_NAME, serde=custom_serializer
    ) as saver:
        put_config = saver.put(
            input_data["config_1"],
            input_data["chkpnt_1"],
            input_data["metadata_1"],
            {},
        )

        assert custom_serializer.dumps_called

        retrieved_checkpoint_tuple = saver.get_tuple(put_config)

        assert custom_serializer.loads_called

        assert retrieved_checkpoint_tuple is not None
        assert retrieved_checkpoint_tuple.checkpoint == input_data["chkpnt_1"]
        assert retrieved_checkpoint_tuple.metadata == input_data["metadata_1"]

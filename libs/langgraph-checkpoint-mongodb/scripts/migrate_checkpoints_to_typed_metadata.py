"""

Script to migrate metadata member of checkpoint collections
- from <=v0.2.1 which is json
- to >=v0.2.2 which is typed (defaulting to msgpack)

Data that was created on <v0.2.2 cannot be read by newer langgraph-checkpoint-mongodb.

Notes:
    - writes_collections is not in scope as it has always used serde.dumps_typed / serde.loads_typed

# TODO:
    - Add test of consistency through MongoDBSaver API (v>=0.2.2)
    - Add durability to stop and continue halfway
    - Add argparse for database and collections
    - Extend out to N collections and databases
    - Add batching
"""

import logging
import os
import sys
from typing import Any, Union

from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

from langgraph.checkpoint.mongodb import MongoDBSaver

logging.basicConfig(level=logging.INFO)

serde = JsonPlusSerializer()


def loads_metadata_orig(metadata: dict[str, Any]) -> CheckpointMetadata:
    """Deserialize metadata document

    The CheckpointMetadata class itself cannot be stored directly in MongoDB,
    but as a dictionary it can. For efficient filtering in MongoDB,
    we keep dict keys as strings.

    metadata is stored in MongoDB collection with string keys and
    serde serialized keys.
    """
    if isinstance(metadata, dict):
        output = dict()
        for key, value in metadata.items():
            output[key] = loads_metadata_orig(value)
        return output
    else:
        return serde.loads_typed(("json", metadata))


def dumps_metadata_new(
    metadata: Union[CheckpointMetadata, Any],
) -> Union[bytes, dict[str, Any]]:
    """Serialize all values in metadata dictionary.

    Keep dict keys as strings for efficient filtering in MongoDB
    """
    if isinstance(metadata, dict):
        output = dict()
        for key, value in metadata.items():
            output[key] = dumps_metadata_new(value)
        return output
    else:
        return serde.dumps_typed(metadata)


def insert_non_duplicates(clxn, buffer):
    try:
        clxn.insert_many(buffer, ordered=False)
    except BulkWriteError as e:
        # Ignore duplicate key errors, re-raise anything else
        write_errors = e.details.get("writeErrors", [])
        non_dupe_errors = [err for err in write_errors if err.get("code") != 11000]
        if non_dupe_errors:
            raise
    finally:
        buffer.clear()


def main():
    MONGODB_URI = os.environ.get(
        "MONGODB_URI", "mongodb://localhost:27017/?directConnection=true"
    )
    DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
    COLLECTION_NAME = os.environ.get("CHECKPOINTS", "checkpoints")

    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    clxn_orig = db[COLLECTION_NAME]
    clxn_new = db[COLLECTION_NAME + "-new"]

    clxn_new.delete_many({})  # todo - once complete. remove this or add to argparse

    logging.info("Beginning checkpoint data migration.")
    logging.info(f"{MONGODB_URI=}")
    logging.info(f"{DB_NAME=}")
    logging.info(f"Original {COLLECTION_NAME=}")
    logging.info(f"Migrated COLLECTION_NAME={clxn_new.name}")

    n_orig = clxn_orig.count_documents({})
    logging.info(f"The collection contains {n_orig} documents.")

    if n_orig == 0:
        logging.critical(f"The provided {COLLECTION_NAME=} was empty or did not exist.")
        sys.exit(1)

    BATCH_SIZE = os.environ.get("BATCH_SIZE", 1000)
    buffer = []
    processed = 0

    cursor = clxn_orig.find({}, batch_size=BATCH_SIZE)
    for doc in cursor:
        if "metadata" in doc:
            load_meta_orig = loads_metadata_orig(doc["metadata"])
            doc["metadata"] = dumps_metadata_new(load_meta_orig)

        buffer.append(doc)
        processed += 1

        if len(buffer) >= BATCH_SIZE:
            insert_non_duplicates(clxn_new, buffer)

            if processed % (10 * BATCH_SIZE) == 0:
                logging.info(f"Migrated {processed}/{n_orig} documents")

    if buffer:
        insert_non_duplicates(clxn_new, buffer)

    logging.info(f"Migration complete. Total documents migrated: {processed}")

    n_new = clxn_new.count_documents({})
    assert n_orig == n_new == processed

    saver_orig = MongoDBSaver(
        client=client, db_name=DB_NAME, checkpoint_collection_name=COLLECTION_NAME
    )
    saver_new = MongoDBSaver(
        client=client,
        db_name=DB_NAME,
        checkpoint_collection_name=COLLECTION_NAME + "-new",
    )

    # Demonstrate that the latest code version works on the new data.
    checkpoints_new = saver_new.list(config=None, limit=2)
    sample_thread = next(checkpoints_new).config["configurable"]["thread_id"]
    sample_checkpoint = saver_new.get_tuple(
        config={"configurable": {"thread_id": sample_thread}}
    )
    try:
        import json

        logging.info(f"{json.dumps(sample_checkpoint.metadata)=}")
    except Exception as e:
        logging.error(
            f"Unable to json dump sample checkpoint metadata. This reveal an issue in the script. Error:{e}"
        )
    # Demonstrate that the old data, as expected, cannot be read with the latest langgraph-checkpoint-mongodb
    try:
        list(saver_orig.list(config=None, limit=2))
    except ValueError as e:
        logging.info(
            f"Attempts to read OLD data format with NEW code (saver_orig.list(None) raise ValueError {e}"
        )


if __name__ == "__main__":
    main()

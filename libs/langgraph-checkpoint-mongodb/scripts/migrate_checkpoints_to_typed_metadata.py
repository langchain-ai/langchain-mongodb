"""Script to migrate metadata of checkpoint collections
- from <=v0.2.1 which is json
- to >=v0.2.2 which is typed (defaulting to msgpack)

Data that was created on <v0.2.2 cannot be read by newer langgraph-checkpoint-mongodb.

Notes:
    - writes_collections is not in scope as it has always used serde.dumps_typed / serde.loads_typed
"""

import argparse
import logging
from typing import Any, Union

from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

from langgraph.checkpoint.mongodb import MongoDBSaver

logging.basicConfig(level=logging.INFO)

serde = JsonPlusSerializer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate langgraph checkpoint metadata to typed format (>= v0.2.2)."
    )

    parser.add_argument(
        "--mongodb-uri",
        default="mongodb://localhost:27017/?directConnection=true",
        help="MongoDB connection URI",
    )

    parser.add_argument(
        "--db",
        required=True,
        help="Database name containing checkpoint collections",
    )

    parser.add_argument(
        "--collections",
        nargs="+",
        required=True,
        help=(
            "One or more checkpoint collection names to migrate. "
            "Each will be copied to <name>-new."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of documents per insert batch",
    )

    parser.add_argument(
        "--clear-destination",
        action="store_true",
        help="Delete destination (<collection>-new) before migrating",
    )

    return parser.parse_args()


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
    args = parse_args()

    client = MongoClient(args.mongodb_uri)
    db = client[args.db]

    logging.info("Beginning checkpoint data migration.")
    logging.info(f"mongodb_uri={args.mongodb_uri}")
    logging.info(f"db={args.db}")
    logging.info(f"collections={args.collections}")
    logging.info(f"batch_size={args.batch_size}")

    for collection_name in args.collections:
        logging.info(f"--- Migrating collection: {collection_name} ---")

        clxn_orig = db[collection_name]
        clxn_new = db[f"{collection_name}-new"]

        if args.clear_destination:
            logging.warning(f"Clearing destination collection {clxn_new.name}")
            clxn_new.delete_many({})

        n_orig = clxn_orig.count_documents({})
        logging.info(f"Source collection contains {n_orig} documents")

        if n_orig == 0:
            logging.warning(f"Skipping empty or missing collection: {collection_name}")
            continue

        buffer = []
        processed = 0

        cursor = clxn_orig.find({}, batch_size=args.batch_size)
        for doc in cursor:
            if "metadata" in doc:
                load_meta_orig = loads_metadata_orig(doc["metadata"])
                doc["metadata"] = dumps_metadata_new(load_meta_orig)

            buffer.append(doc)
            processed += 1

            if len(buffer) >= args.batch_size:
                insert_non_duplicates(clxn_new, buffer)

                if processed % (10 * args.batch_size) == 0:
                    logging.info(
                        f"[{collection_name}] Migrated {processed}/{n_orig} documents"
                    )

        if buffer:
            insert_non_duplicates(clxn_new, buffer)

        logging.info(
            f"[{collection_name}] Migration complete. Total documents migrated: {processed}"
        )

        n_new = clxn_new.count_documents({})
        assert n_new == processed

        # Validation via MongoDBSaver
        saver_new = MongoDBSaver(
            client=client,
            db_name=args.db,
            checkpoint_collection_name=f"{collection_name}-new",
        )

        checkpoints_new = saver_new.list(config=None, limit=2)
        sample_thread = next(checkpoints_new).config["configurable"]["thread_id"]
        sample_checkpoint = saver_new.get_tuple(
            config={"configurable": {"thread_id": sample_thread}}
        )

        try:
            import json

            logging.info(
                f"[{collection_name}] sample metadata={json.dumps(sample_checkpoint.metadata)}"
            )
        except Exception as e:
            logging.error(f"[{collection_name}] Unable to json dump metadata: {e}")

        # Demonstrate expected failure on old data
        saver_orig = MongoDBSaver(
            client=client,
            db_name=args.db,
            checkpoint_collection_name=collection_name,
        )
        try:
            list(saver_orig.list(config=None, limit=2))
        except ValueError as e:
            logging.info(
                f"[{collection_name}] Old format correctly fails with new code: {e}"
            )


if __name__ == "__main__":
    main()

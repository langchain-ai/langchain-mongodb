"""Script to migrate metadata of checkpoint collections
- from <=v0.2.1 which is json
- to >=v0.2.2 which is typed (defaulting to msgpack)

Data that was created on <v0.2.2 cannot be read by newer langgraph-checkpoint-mongodb.

Notes:
    - writes_collections is not in scope as it has always used serde.dumps_typed / serde.loads_typed
"""

import argparse
import logging
import time
from typing import Any, Union

from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

from langgraph.checkpoint.mongodb import MongoDBSaver

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
        help="One or more checkpoint collection names to migrate",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of documents per insert batch",
    )

    parser.add_argument(
        "--suffix",
        default="-new",
        help="Suffix for migrated collections (default: -new)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run migration without writing any data",
    )

    parser.add_argument(
        "--clear-destination",
        action="store_true",
        help="Delete destination collection before migrating",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )

    return parser.parse_args()


def loads_metadata_orig(metadata: dict[str, Any]) -> CheckpointMetadata:
    if isinstance(metadata, dict):
        return {k: loads_metadata_orig(v) for k, v in metadata.items()}
    return serde.loads_typed(("json", metadata))


def dumps_metadata_new(
    metadata: Union[CheckpointMetadata, Any],
) -> Union[bytes, dict[str, Any]]:
    if isinstance(metadata, dict):
        return {k: dumps_metadata_new(v) for k, v in metadata.items()}
    return serde.dumps_typed(metadata)


def insert_non_duplicates(clxn, buffer):
    try:
        clxn.insert_many(buffer, ordered=False)
    except BulkWriteError as e:
        write_errors = e.details.get("writeErrors", [])
        non_dupe_errors = [err for err in write_errors if err.get("code") != 11000]
        if non_dupe_errors:
            raise
    finally:
        buffer.clear()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    start_time = time.time()

    client = MongoClient(args.mongodb_uri)
    db = client[args.db]

    logging.info("Beginning checkpoint data migration")
    logging.info(f"mongodb_uri={args.mongodb_uri}")
    logging.info(f"db={args.db}")
    logging.info(f"collections={args.collections}")
    logging.info(f"batch_size={args.batch_size}")
    logging.info(f"suffix={args.suffix}")
    logging.info(f"dry_run={args.dry_run}")

    total_scanned = 0
    total_migrated = 0

    for collection_name in args.collections:
        logging.info(f"--- Migrating collection: {collection_name} ---")

        clxn_orig = db[collection_name]
        dest_name = f"{collection_name}{args.suffix}"
        clxn_new = db[dest_name]

        if args.clear_destination and not args.dry_run:
            logging.warning(f"Clearing destination collection {dest_name}")
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
            total_scanned += 1

            if "metadata" in doc:
                doc["metadata"] = dumps_metadata_new(
                    loads_metadata_orig(doc["metadata"])
                )

            buffer.append(doc)
            processed += 1

            if len(buffer) >= args.batch_size:
                if not args.dry_run:
                    insert_non_duplicates(clxn_new, buffer)
                else:
                    buffer.clear()

        if buffer:
            if not args.dry_run:
                insert_non_duplicates(clxn_new, buffer)
            else:
                buffer.clear()

        total_migrated += processed

        logging.info(
            f"[{collection_name}] Migration complete. Documents processed: {processed}"
        )

        if not args.dry_run:
            n_new = clxn_new.count_documents({})
            assert n_new == processed

            saver_new = MongoDBSaver(
                client=client,
                db_name=args.db,
                checkpoint_collection_name=dest_name,
            )

            checkpoints_new = saver_new.list(config=None, limit=1)
            sample_thread = next(checkpoints_new).config["configurable"]["thread_id"]
            sample_checkpoint = saver_new.get_tuple(
                config={"configurable": {"thread_id": sample_thread}}
            )

            logging.debug(
                f"[{collection_name}] Sample metadata: {sample_checkpoint.metadata}"
            )

    elapsed = time.time() - start_time
    rate = total_migrated / elapsed if elapsed > 0 else 0

    logging.info("=== Migration Summary ===")
    logging.info(f"Collections processed: {len(args.collections)}")
    logging.info(f"Documents scanned:    {total_scanned}")
    logging.info(f"Documents migrated:   {total_migrated}")
    logging.info(f"Elapsed time:         {elapsed:.2f}s")
    logging.info(f"Throughput:           {rate:.2f} docs/sec")

    if args.dry_run:
        logging.info("Dry-run mode enabled: no data was written")


if __name__ == "__main__":
    main()

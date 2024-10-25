import os

from langchain_core.documents import Document
from pymongo import MongoClient

from langchain_mongodb.docstores import MongoDBDocStore

CONNECTION_STRING = os.environ.get("MONGODB_URI", "localhost:27017")
DB_NAME = "langchain_test_db"
COLLECTION_NAME = "docstore"


def test_docstore(technical_report_pages):
    clxn_name = COLLECTION_NAME
    client = MongoClient(CONNECTION_STRING)
    db = client[DB_NAME]
    db.drop_collection(COLLECTION_NAME)
    clxn = db[COLLECTION_NAME]

    n_docs = len(technical_report_pages)
    assert clxn.count_documents({}) == 0
    docstore = MongoDBDocStore(collection=clxn)

    docstore.mset((str(i), technical_report_pages[i]) for i in range(n_docs))
    assert clxn.count_documents({}) == n_docs

    twenties = list(docstore.yield_keys(prefix="2"))
    assert len(twenties) == 11  # includes 2, 20, 21, ..., 29

    docstore.mdelete([str(i) for i in range(20, 30)] + ["2"])
    assert clxn.count_documents({}) == n_docs - 11
    assert set(docstore.mget(twenties)) == {None}

    sample = docstore.mget(["8", "16", "24", "36"])
    assert sample[2] is None
    assert all(isinstance(sample[i], Document) for i in [0, 1, 3])

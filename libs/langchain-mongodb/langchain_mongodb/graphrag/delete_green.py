"""
# Greener Build


Hi team, IHAC who is using `count_documents({ext_type: 4, ext_lastmod: { $gt:1758102623, $lte:1758124036 }})` method
and internally it's converting it into
`[{ $match: { ext_type:4, ext_lastmod: { $gt:1758102623, $lte:1758124036 } } }, { $group: { _id:1, n: { $sum:1 } } }]`
and query results looks like: `keysExamined:3720495 docsExamined:3720491`.
I try the same query in mongosh and it looks like: `keysExamined:3720495 docsExamined:0`,
am I missing something here? The customer concern is like it looks like a `COLLSCAN` due to the number of docs examined. Any ideas?


"""

import os
from turtledemo.chaos import coosys

from langchain_mongodb.graphrag.graph import MongoDBGraphStore

CONNECTION_STRING = os.environ.get("MONGODB_URI", "")  # Use Prakul's Cluster 0
DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_graphrag"
CONNECTION_STRING = "mongodb+srv://prakul_test:x8Qyc2CyUGxcWgpc@cluster0.8by45wo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

'''
graph_store = MongoDBGraphStore(
    connection_string=CONNECTION_STRING,
    database_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    entity_extraction_model=None,
)
coll = graph_store.collection
'''




filter1 = {"_id": {"$in": ['ACME Corporation', 'GreenTech Ltd.']}}
filter = {"$match": filter1}

n = coll.count_documents(filter1)

filter_gt = {"relationships.attributes.since[0]": {"$gt": "2000"}}

coll.count_documents(filter_gt)


# TODO - Need something that includes $gt, $lte


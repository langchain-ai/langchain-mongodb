import os
from typing import List

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from pymongo import MongoClient
from pymongo.collection import Collection

extraction_context = """
# CONTEXT:
## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
You will be provided a text document as INPUT.
Try to capture as much information from the text as possible without sacrificing accuracy.
Do not add any information that is not explicitly mentioned in the text. The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.

## 2. General Specification for a Knowledge Graph Schema in a MongoDB Collection.
Each Entity will be represented by a single JSON Document. It will have the following fields.
* ID: A unique identifier for the entity (e.g., UUID, name).
* type: A string specifying the type of the entity (e.g., “Person”, “Organization”).
* properties: A dictionary containing key-value pairs of attributes describing the entity.
* relationships: Stored as embedded key-value pairs. Keys are relationship types, values are lists of target entity IDs, along with additional metadata describing the relationship to that entity.

## 3. Example Entity structure
{{
  "ID": "12345",
  "name": "Alice Palace",
  "type": "Person",
  "properties": {{
    "position": "CEO",
    "startDate": "2018-01-01"
  }},
  "relationships": {{
    "EMPLOYEE": [
      {{
        "target": "MongoDB"
      }}
    ],
    "FRIEND": [
      {{
        "target": "Jarnail Singh",
        "properties": {{
          "since": "2019-05-01"
        }}
      }},
      {{
        "target": "Jasbinder Kaur",
        "properties": {{
          "since": "2015-05-01"
        }}
      }}
    ]
  }}
}}

## 3. Entity Resolution and Consistency
**Maintain Entity Consistency when extracting entities.** If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he") always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID. Define required fields (e.g., ID, name,, type) and allow optional properties.

## 4. Relationships
Ensure that every relationship target points to a valid entity ID. Ensure consistency and generality in relationship types when constructing knowledge schemas. Instead of using specific and momentary types such as 'became_professor', use more general and timeless relationship types like 'professor'. Make sure to use general and timeless relationship types!

Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
"""

entity_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(extraction_context),
            HumanMessagePromptTemplate.from_template("Input: {input_document}"),
        ]
    )


def traverse_graph(collection: Collection, start_name: str, relationship: str):
    """Follow a single relationship type

    This assumes one already knows the relationships.
    It will be easy to ascertain all from the starting entity document
    """


    pipeline = [
        {
            "$match": {"name": start_name}  # Starting document
        },
        {
            "$graphLookup": {
                "from": collection.name,  # Collection to search in
                "startWith": "$name",  # Starting field for the traversal
                "connectFromField": f"relationships.{relationship}.target",  # Field to match in the related documents
                "connectToField": "name",  # Field to match on the current document
                "as": "connections",  # Name of the output array containing the related entities
                "maxDepth": 3,  # Optional: limit recursion depth
                "depthField": "depth"  # Optional: include depth information
            }
        },
        {
            "$project": {'connections': 1}
        }
    ]

    results = list(collection.aggregate(pipeline))
    connections = results[0]["connections"]
    from pprint import pprint
    for entity in connections:
        pprint(entity)
    return connections


if __name__ == '__main__':
    # A simple example
    from langchain_openai import ChatOpenAI
    import json

    # Define an LLM for Entity Extraction. >>> Requires OPENAI_API_KEY in os.environ <<<
    entity_model = ChatOpenAI(model="gpt-4o", temperature=0.0)
    # Combine it with the prompt template
    entity_chain = entity_prompt | entity_model
    # Invoke on a document to extract entities and relationships
    example_document = "Casey Clements is a Senior Python Engineer working at MongoDB, a NoSQL Database company."
    entity_output = entity_chain.invoke(dict(input_document=example_document))

    # Post-Process output string into list of entity json documents
    # Strip the ```json prefix and trailing ```
    json_string = entity_output.content.lstrip("```json").rstrip("```").strip()
    if json_string.startswith("{"):
        json_string = f"[{json_string}]"

    extracted_entities = json.loads(json_string)
    if isinstance(extracted_entities, dict):
        extracted_entities = [extracted_entities]  # TODO - Unnecessary?

    # Insert them into a Collection
    from pymongo import MongoClient
    client = MongoClient("mongodb://localhost:27017/")

    db = client["delete_this_database"]
    collection_name = "delete_this_collection"
    collection = db[collection_name]
    collection.delete_many({})

    collection.insert_many(extracted_entities)
    print(f"{collection.count_documents({})} documents inserted")
    print(f"{collection.find_one({})=}")


    connections = traverse_graph(collection, start_name="Casey Clements", relationship="EMPLOYEE")

# TODO - Continue prototype
#   Turn this into MongoDBGraphStore.add_documents
#   Create an integration test
#   Add traverse_graph to MongoDBGraphStore
#   - Start from entity and follow its relationships
#   - Run Extraction on Question. Run traversal from each entity
#   - multiple pipelines. Loops over entities *and* relationships?
#       - modify connectToField to point to a broader range using $unwind before traversal.
#       - Use $facet to perform different $graphLookup traversals for different subsets of documents in parallel, if needed.
#       - Combine results afterward using $merge or $unionWith if you need a single consolidated view.
#   Formulate questions for the Aggregation Experts
#   - Post-process outputs
#   Rough prototype for Query Prompt. Then another chain for querying


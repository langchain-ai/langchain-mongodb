import json
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pymongo import UpdateOne
from pymongo.collection import Collection

from langchain_mongodb.graphrag import prompts


class MongoDBGraphStore:
    """GraphRAG DataStore

    GraphRAG is a ChatModel that provides responses to semantic queries.
    As in Vector RAG, we augment the Chat Model's training data
    with relevant information that we collect from  documents.

    In Vector RAG, one uses an "Embedding" model that converts both
    the query, and the potentially relevant documents, into vectors,
    which can then be compared, and the most similar supplied to the
    Chat Model as context to the query.

    In Graph RAG, one uses an Entity-Extraction model that converts both
    the query, and the potentially relevant documents, into graphs. These are
    composed of nodes that are entities (nouns) and edges that are relationships.
    The idea is that the graph can find connections between entities and
    hence answer questions that require more than one connection.

    It is also about finding common entities in documents,
    combining the properties found and hence providing richer context than Vector RAG,
    especially in certain cases.

    When a document is extracted, each entity is represented by a single
    MongoDB Document, and relationships are defined in a nested field named
    'relationships'. The schema, and an example, are described in the
    :data:`~langchain_mongodb.graphrag.prompts.entity_context` prompts module.

    When a query is made, the model extracts the entities and relationships from it,
    then traverses the graph starting from each of the entities found.
    The connected entities and relationships form the context
    that is included with the query to the Chat Model.

    Requirements:
        Documents
        Entity Extraction Model
        Prompt Template
        Query
        Chat Model

    Example Query: "Does Casey Clements work at MongoDB?"
        ==> Entities: [(name=Casey Clements, type=person), (name=MongoDB, type=Corporation)]
        ==> Relationships: [(name=EMPLOYEE), ]
    Example document ingested into Collection:
        "MongoDB employees everyone."

    From this information, one would be able to deduce that Casey Clements DOES work at MongoDB.
    The Graph is there to provide context/information, not answer the question.
    The LLM will know that everyone contains every person.
    But how will the LLM know to categorize Casey Clements as a person? It could be an organization, for example.
        - One would add hints or examples in their prompt to the Entity Extraction Model.
        - This is the right track. Even if we're not focussing on Entity Extraction, we care about providing an API to allow it
         ==> e.g. Could give specific types to include, and that one must pick one, or if ambiguous, throw away information.
    """

    def __init__(
        self,
        collection: Collection[Dict[str, Any]],
        entity_extraction_model: BaseChatModel,
        entity_prompt: ChatPromptTemplate = prompts.entity_prompt,
        query_prompt: ChatPromptTemplate = prompts.query_prompt,
    ):
        """
        Args:
            collection: Collection representing an Entity Graph
            entity_extraction_model: LLM for converting documents into Graph of Entities and Relationships
            entity_prompt: Prompt to fill graph store with entities following schema
            query_prompt: Prompt extracts entities and relationships as search starting points.
        """
        self.collection = collection
        self.entity_extraction_model = entity_extraction_model
        self.entity_prompt = entity_prompt
        self.query_prompt = query_prompt

    def add_documents(self, documents: Union[Document, List[Document]]):
        """Extract entities and upsert into the collection.

        Each MongoDB document represents a single entity.
        Its relationships and properties are treated consistently.
        """
        documents = [documents] if isinstance(documents, Document) else documents
        for doc in documents:
            entities = self.extract_entities(doc.page_content)

            # Create update operations for each entity
            operations = []
            for entity in entities:
                operations.append(
                    UpdateOne(
                        filter={"ID": entity["ID"]},  # Match on ID
                        update={
                            "$setOnInsert": {  # Set if upsert
                                "ID": entity["ID"],
                                "type": entity["type"],
                            },
                            "$addToSet": {  # Update without overwriting
                                **{
                                    f"properties.{k}": v
                                    for k, v in entity.get("properties", {}).items()
                                },
                                **{
                                    f"relationships.{k}": {"$each": v}
                                    for k, v in entity.get("relationships", {}).items()
                                },
                            },
                        },
                        upsert=True,  # Insert if document doesn't exist
                    )
                )

            # Execute bulk write for the entities
            if operations:
                self.collection.bulk_write(operations)
            # TODO - $return ids.

    def extract_entities(
        self, raw_document: str, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Extract entities and their relations using chosen prompt and LLM."""
        # Combine the llm with the prompt template to form a chain
        chain: RunnableSequence = self.entity_prompt | self.entity_extraction_model
        # Invoke on a document to extract entities and relationships
        response: AIMessage = chain.invoke(dict(input_document=raw_document))
        # Post-Process output string into list of entity json documents
        # Strip the ```json prefix and trailing ```
        json_string = (
            response.content.removeprefix("```json").removesuffix("```").strip()
        )
        extracted = json.loads(json_string)
        return extracted["entities"]

    def extract_entity_names(self, raw_document: str, **kwargs: Any) -> List[str]:
        """Extract entity names from a document for similarity_search.

        The second entity extraction has a different form and purpose than
        the first as we are looking for starting points of our search and
        paths to follow. We aim to find source nodes,  but no target nodes or edges.
        """
        # Combine the llm with the prompt template to form a chain
        chain: RunnableSequence = self.query_prompt | self.entity_extraction_model
        # Invoke on a document to extract entities and relationships
        response: AIMessage = chain.invoke(dict(input_document=raw_document))
        # Post-Process output string into list of entity json documents
        # Strip the ```json prefix and suffix
        json_string = (
            response.content.removeprefix("```json").removesuffix("```").strip()
        )
        return json.loads(json_string)

    def find_entity_by_name(self, name: str) -> Optional[List[dict]]:
        return list(self.collection.find({"ID": name}))

    def related_entities(
        self, starting_entities: List[str], depth=3
    ) -> List[Dict[str, Any]]:
        """Find connections to an entity by following a given relationship."""

        pipeline = [
            # Match the starting entity by ID
            {"$match": {"ID": {"$in": starting_entities}}},
            {
                "$graphLookup": {
                    "from": self.collection.name,
                    "startWith": {
                        # Extract all relationship targets dynamically
                        "$reduce": {
                            "input": {"$objectToArray": "$relationships"},
                            "initialValue": [],
                            "in": {"$concatArrays": ["$$value", "$$this.v.target"]},
                        }
                    },
                    "connectFromField": "ID",  # Match on entity ID
                    "connectToField": "ID",  # Use the target entity's ID
                    "as": "connections",  # Output field for related entities
                    "maxDepth": depth,  # Adjust based on traversal needs
                }
            },
            # Unwind the connections array to process each connection as its own document
            {"$unwind": "$connections"},
            # Replace the root with the connection document
            {"$replaceRoot": {"newRoot": "$connections"}},
            # Exclude MongoDB's internal `_id` field
            {"$project": {"_id": 0}},
        ]
        return list(self.collection.aggregate(pipeline))

    def respond_to_query(self, query, related_entities):
        """Respond to query given information found in related entities."""
        # TODO Examine API of this (and other) methods
        from .prompts import entity_schema, rag_prompt

        # Combine the llm with the prompt template to form a chain
        chain: RunnableSequence = rag_prompt | self.entity_extraction_model
        # Invoke with query
        response: AIMessage = chain.invoke(
            dict(
                query=query,
                related_entities=related_entities,
                entity_schema=entity_schema,
            )
        )
        return response.content

    def similarity_search(self, query: str, depth: Optional[int] = None) -> List:
        """Find entities related to the input string
        1. Use LLM & Prompt to find entities.
            - What can we do at this point with the other semantics of the query (e.g. verbs)?
        2. For each entity, use $match to find them in collection.
            For each result get its relationships.
            For each relationship, call $graphLookup to find related entities.
        3. With all the found entities provided as context, send the query to the chatbot.

        Args:
            query:
            depth:

        Returns:

        """
        raise NotImplementedError


# TODO
#   - Merge entities extracted from different documents
#   - Change ID to 'name' or ID to _id
#   - Update Design Doc
#   - Add: Constraints
#       - relationships
#       - entity types
#   - Add schema validation
#   - Should I add an Entity class?
#   - similarity_search
#   - Retriever
#   - GraphChain

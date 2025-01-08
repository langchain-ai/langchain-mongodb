import json
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
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

    When a auery is made, the model extracts the entities and relationships from it,
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
        query_prompt: ChatPromptTemplate = prompts.retrieval_prompt,
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

    def add_documents(self, documents: Document | List[Document]):
        """Extract entities and insert into collection.

        Each mongodb document represents a single entity.
        Its relationships are embedded fields pointing to other entities.
        """
        documents = [documents] if isinstance(documents, Document) else documents
        for doc in documents:
            entities = self.extract_entities(doc.page_content)
            self.collection.insert_many(entities)

    def extract_entities(self, raw_document: str) -> List[Dict[str, Any]]:
        """Extract entities and their relations using chosen prompt and LLM."""
        # Combine the llm with the prompt template to form a chain
        chain: RunnableSequence = self.entity_prompt | self.entity_extraction_model
        # Invoke on a document to extract entities and relationships
        response: AIMessage = chain.invoke(dict(input_document=raw_document))
        # Post-Process output string into list of entity json documents
        # Strip the ```json prefix and trailing ```
        json_string = response.content.lstrip("```json").rstrip("```").strip()
        extracted = json.loads(json_string)
        return extracted["entities"]

    def extract_entities_from_query(self, raw_document: str) -> Dict[str, List[str]]:
        """Extract entities and relationships for similarity_search.

        The second entity extraction has a different form and purpose than
        the first as we are looking for starting points of our search and
        paths to follow. We aim to find source nodes and edges, but no target nodes.
        This form is common in questions.
        For example, "Where is MongoDB located?" provides a source entity, "MongoDB"
        and a relationship "location" but the question mark is effectively the target.
        """

        # Combine the llm with the prompt template to form a chain
        chain: RunnableSequence = self.query_prompt | self.entity_extraction_model
        # Invoke on a document to extract entities and relationships
        response: AIMessage = chain.invoke(dict(input_document=raw_document))
        # Post-Process output string into list of entity json documents
        # Strip the ```json prefix and suffix
        json_string = response.content.lstrip("```json").rstrip("```").strip()
        return json.loads(json_string)

    def related_entities(self, entity, relationships, depth=2) -> Dict[str, Dict[str, Any]]:
        """Find connections to an entity by following a given relationship."""

        related = {}
        for relation in relationships:
            pipeline = [
                {"$match": {"ID": entity}},
                {
                    "$graphLookup": {
                        "from": self.collection.name,
                        "startWith": "$ID",
                        "connectFromField": "ID",
                        "connectToField": f"relationships.{relation}.target",
                        "as": "connections",
                        "maxDepth": depth,
                    }
                },
                # {"$project": {"_id": False}},
            ]
            result = list(self.collection.aggregate(pipeline))
            if result:
                result = result[0]
                related.update({e["_id"]: e for e in result.pop("connections")})
                related.update({result['_id']: result})
        return related


# TODO
#   - similarity_search
#   - Retriever
#   - GraphChain
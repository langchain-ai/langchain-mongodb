"""

GraphRAG - ChatModel that provides responses to semantic queries.
    As in Vector RAG, we augment the Chat Model's training data
    with relevant information that we collect from  documents.

    In Vector RAG, one uses an "Embedding" model that converts both
    the query, and the potentially relevant documents, into vectors,
    which can then be compared, and the most similar supplied to the
    Chat Model as context to the query.

    In Graph RAG, one uses an Entity-Extraction model that converts both
    the query, and the potentially relevant documents, into graphs
    composed of nodes that are entities (nouns) and edges that are relationships.
    The Graph representation of each document is presented as a list of triplets
    ( source node / entity, edge / relationship, target node / entity).
    This is a common representation, and LLMs have no trouble interpreting it.

    The idea is that the Graph can find connections between entities and
     hence answer questions that require more than one connection.

     It is also about finding common entities in documents,
     combining the properties found and hence provide richer context than Vector RAG,
    especially in certain cases.

    We first ingest all of our documents by entity extraction, and place these
    into a single graph where each individual document now is a subgraph.

    When a auery is made, the model extracts the entities from it,
    then traverses the graph starting from each of the entities found.
    The connected entities and relationships form the context
    that is included with the query to the Chat Model.


    Requirements:
        Documents
        Entity Extraction Model
        Prompt Template
        Query
        Chat Model


    Query => List[Triple]
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



    - NOTES
        - Query has a clear subject-object-predicate structure.  There is asymmetry.
        - There is also asymmetry in the graph lookup as it requires a relationship, from, and to fields.
        - So what do we search for to find our information?
            - I am looking for documents with Casey Clements.
                -

    Example Triple: ["Casey Clements", "EMPLOYEE", "MongoDB"]



    So many questions
        - If scope includes add_triplets, do we keep track of properties as they are added?
            - e.g. Entity types
        - If someone asks about Casey Clements, do we want to know whether he tied his shoes?
            ==> - With $graphLookup, we need a node, and a relationship.  <=====


"""
import json
from typing import Any, Dict, List
from pymongo.collection import Collection
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from .prompts import entity_prompt

class MongoDBGraphStore(BaseModel):

    collection: Collection[Dict[str, Any]]
    """Collection representing an Entity Graph"""

    entity_extraction_model: BaseChatModel
    """LLM for converting documents into Graph of Entities and Relationships"""

    entity_prompt: ChatPromptTemplate
    """Context prompt specifically requests schema of entities"""
    
    def add_documents(self, documents: List[Document]):
        # TODO
        for doc in documents:
            entities = self.extract_entities(doc)
            self.collection.insert_many(entities)
        
    
    def extract_entities(self, raw_document: str) -> List[Dict[str, Any]]:
        # Combine the llm with the prompt template to form a chain
        chain: RunnableSequence = entity_prompt | self.entity_extraction_model
        # Invoke on a document to extract entities and relationships
        response: AIMessage = chain.invoke(dict(input_document=raw_document))
        # Post-Process output string into list of entity json documents
        # Strip the ```json prefix and trailing ```
        json_string = response.content.lstrip("```json").rstrip("```").strip()
        if json_string.startswith("{"):
            json_string = f"[{json_string}]"
        extracted_entities = json.loads(json_string)
        return extracted_entities
       
    
    
     
    
    def related_entities(self, entity, relationship, depth=2):
        """ Find nodes connected to a starting id
        This will add a list of triplets to a new field 'related_nodes' in the start_id document.
        """
        
        # TODO
        #   - Pass in entities. Extract relationships from these
        #   - LEVEL-UP: unwind, merge, unionWith, facet

        start_id = entity["name"]
        
        pipeline = [
            {"$match": {"_id": start_id}},
            {
                "$graphLookup": {
                    "from": self.collection.name,
                    "startWith": "$_id",
                    "connectFromField": "_id",
                    "connectToField": relationship,  # Adjust based on your schema
                    "as": "relatedNodes",
                    "maxDepth": depth,
                }
            }
        ]
        return list(self.collection.aggregate(pipeline))

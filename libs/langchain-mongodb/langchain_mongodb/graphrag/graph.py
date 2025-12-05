from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeAlias, Union

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from pymongo.results import BulkWriteResult

from langchain_mongodb.graphrag import example_templates, prompts

from ..utils import DRIVER_METADATA, _append_client_metadata
from .prompts import rag_prompt
from .schema import entity_schema

if TYPE_CHECKING:
    Entity: TypeAlias = Dict[str, Any]
    """Represents an Entity in the knowledge graph with specific schema. See .schema"""

    import holoviews  # type: ignore[import-untyped]
    import networkx

logger = logging.getLogger(__name__)


class MongoDBGraphStore:
    """GraphRAG DataStore

    GraphRAG is a ChatModel that provides responses to semantic queries
    based on a Knowledge Graph that an LLM is used to create.
    As in Vector RAG, we augment the Chat Model's training data
    with relevant information that we collect from documents.

    In Vector RAG, one uses an "Embedding" model that converts both
    the query, and the potentially relevant documents, into vectors,
    which can then be compared, and the most similar supplied to the
    Chat Model as context to the query.

    In Graph RAG, one uses an "Entity-Extraction" model that converts
    text into Entities and their relationships, a Knowledge Graph.
    Comparison is done by Graph traversal, finding entities connected
    to the query prompts. These are then supplied to the Chat Model as context.
    The main difference is that GraphRAG's output is typically in a structured format.

    GraphRAG excels in finding links and common entities,
    even if these come from different articles. It can combine information from
    distinct sources providing richer context than Vector RAG in certain cases.

    Here are a few examples of so-called multi-hop questions where GraphRAG excels:
    - What is the connection between ACME Corporation and GreenTech Ltd.?
    - Who is leading the SolarGrid Initiative, and what is their role?
    - Which organizations are participating in the SolarGrid Initiative?
    - What is John Doe’s role in ACME’s renewable energy projects?
    - Which company is headquartered in San Francisco and involved in the SolarGrid Initiative?

    In Graph RAG, one uses an Entity-Extraction model that interprets
    text documents that it is given and extracting the query,
    and the potentially relevant documents, into graphs. These are
    composed of nodes that are entities (nouns) and edges that are relationships.
    The idea is that the graph can find connections between entities and
    hence answer questions that require more than one connection.

    In MongoDB, Knowledge Graphs are stored in a single Collection.
    Each MongoDB Document represents a single entity (node),
    and its relationships (edges) are defined in a nested field named
    "relationships". The schema, and an example, are described in the
    :data:`~langchain_mongodb.graphrag.prompts.entity_context` prompts module.

    When a query is made, the model extracts the entities in it,
    then traverses the graph to find connections.
    The closest entities and their relationships form the context
    that is included with the query to the Chat Model.

    Consider this example Query: "Does John Doe work at MongoDB?"
    GraphRAG can answer this question even if the following two statements come
    from completely different sources.
    - "Jane Smith works with John Doe."
    - "Jane Smith works at MongoDB."
    """

    def __init__(
        self,
        *,
        connection_string: Optional[str] = None,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        collection: Optional[Collection] = None,
        entity_extraction_model: BaseChatModel,
        entity_prompt: Optional[ChatPromptTemplate] = None,
        query_prompt: Optional[ChatPromptTemplate] = None,
        max_depth: int = 3,
        allowed_entity_types: Optional[List[str]] = None,
        allowed_relationship_types: Optional[List[str]] = None,
        entity_examples: Optional[str] = None,
        entity_name_examples: str = "",
        validate: bool = False,
        validation_action: str = "warn",
    ):
        """
        Args:
            connection_string: A valid MongoDB connection URI.
            database_name: The name of the database to connect to.
            collection_name: The name of the collection to connect to.
            collection: A Collection that will represent a Knowledge Graph.
                ** One may pass a Collection in lieu of connection_string, database_name, and collection_name.
            entity_extraction_model: LLM for converting documents into Graph of Entities and Relationships.
            entity_prompt: Prompt to fill graph store with entities following schema.
                Defaults to .prompts.ENTITY_EXTRACTION_INSTRUCTIONS
            query_prompt: Prompt extracts entities and relationships as search starting points.
                Defaults to .prompts.NAME_EXTRACTION_INSTRUCTIONS
            max_depth: Maximum recursion depth in graph traversal.
            allowed_entity_types: If provided, constrains search to these types.
            allowed_relationship_types: If provided, constrains search to these types.
            entity_examples: A string containing any number of additional examples to provide as context for entity extraction.
            entity_name_examples: A string appended to prompts.NAME_EXTRACTION_INSTRUCTIONS containing examples.
            validate: If True, entity schema will be validated on every insert or update.
            validation_action: One of {"warn", "error"}.
              - If "warn", the default, documents will be inserted but errors logged.
              - If "error", an exception will be raised if any document does not match the schema.
        """
        self._schema = deepcopy(entity_schema)
        collection_existed = True
        if connection_string and collection is not None:
            raise ValueError(
                "Pass one of: connection_string, database_name, and collection_name"
                "OR a MongoDB Collection."
            )
        if collection is None:  # collection is specified by uri and names
            assert collection_name is not None
            assert database_name is not None
            client: MongoClient = MongoClient(
                connection_string,
                driver=DRIVER_METADATA,
            )
            db = client[database_name]
            if collection_name not in db.list_collection_names(
                authorizedCollections=True
            ):
                validator = {"$jsonSchema": self._schema} if validate else None
                collection = client[database_name].create_collection(
                    collection_name,
                    validator=validator,
                    validationAction=validation_action,
                )
                collection_existed = False
            else:
                collection = db[collection_name]
        else:
            if not isinstance(collection, Collection):
                raise ValueError(
                    "collection must be a MongoDB Collection. "
                    "Consider using connection_string, database_name, and collection_name."
                )

        if validate and collection_existed:
            # first check for existing validator
            collection_info = collection.database.command(
                "listCollections", filter={"name": collection.name}
            )
            collection_options = collection_info.get("cursor", {}).get("firstBatch", [])
            validator = collection_options[0].get("options", {}).get("validator", None)
            if not validator:
                try:
                    collection.database.command(
                        "collMod",
                        collection.name,
                        validator={"$jsonSchema": self._schema},
                        validationAction=validation_action,
                    )
                except OperationFailure:
                    logger.warning(
                        "Validation will NOT be performed. "
                        "User must be DB Admin to add validation **after** a Collection is created. \n"
                        "Please add validator when you create collection: "
                        "db.create_collection.(coll_name, validator={'$jsonSchema': schema.entity_schema})"
                    )
        self.collection = collection

        # append_metadata was added in PyMongo 4.14.0, but is a valid database name on earlier versions
        _append_client_metadata(collection.database.client)

        self.entity_extraction_model = entity_extraction_model
        self.entity_prompt = (
            prompts.entity_prompt if entity_prompt is None else entity_prompt
        )
        self.query_prompt = (
            prompts.query_prompt if query_prompt is None else query_prompt
        )
        self.entity_examples = (
            example_templates.entity_extraction
            if entity_examples is None
            else entity_examples
        )
        self.entity_name_examples = entity_name_examples

        self.max_depth = max_depth
        self._schema = deepcopy(entity_schema)
        if allowed_entity_types:
            self.allowed_entity_types = allowed_entity_types
            self._schema["properties"]["type"]["enum"] = allowed_entity_types  # type:ignore[index]
        else:
            self.allowed_entity_types = []
        if allowed_relationship_types:
            # Update Prompt
            self.allowed_relationship_types = allowed_relationship_types
            # Update schema. Disallow other keys..
            self._schema["properties"]["relationships"]["properties"]["types"][  # type:ignore[index]
                "enum"
            ] = allowed_relationship_types
        else:
            self.allowed_relationship_types = []

    @property
    def entity_schema(self) -> dict[str, Any]:
        """JSON Schema Object of Entities. Will be applied if validate is True.

        See Also:
            `$jsonSchema <https://www.mongodb.com/docs/manual/reference/operator/query/jsonSchema/>`_
        """
        return self._schema

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        database_name: str,
        collection_name: str,
        entity_extraction_model: BaseChatModel,
        entity_prompt: ChatPromptTemplate = prompts.entity_prompt,
        query_prompt: ChatPromptTemplate = prompts.query_prompt,
        max_depth: int = 3,
        allowed_entity_types: Optional[List[str]] = None,
        allowed_relationship_types: Optional[List[str]] = None,
        entity_examples: Optional[str] = None,
        entity_name_examples: str = "",
        validate: bool = False,
        validation_action: str = "warn",
    ) -> MongoDBGraphStore:
        """Construct a `MongoDB KnowLedge Graph for RAG`
        from a MongoDB connection URI.

        Args:
            connection_string: A valid MongoDB connection URI.
            database_name: The name of the database to connect to.
            collection_name: The name of the collection to connect to.
            entity_extraction_model: LLM for converting documents into Graph of Entities and Relationships.
            entity_prompt: Prompt to fill graph store with entities following schema.
            query_prompt: Prompt extracts entities and relationships as search starting points.
            max_depth: Maximum recursion depth in graph traversal.
            allowed_entity_types: If provided, constrains search to these types.
            allowed_relationship_types: If provided, constrains search to these types.
            entity_examples: A string containing any number of additional examples to provide as context for entity extraction.
            entity_name_examples: A string appended to prompts.NAME_EXTRACTION_INSTRUCTIONS containing examples.
            validate: If True, entity schema will be validated on every insert or update.
            validation_action: One of {"warn", "error"}.
              - If "warn", the default, documents will be inserted but errors logged.
              - If "error", an exception will be raised if any document does not match the schema.

        Returns:
            A new MongoDBGraphStore instance.
        """
        client: MongoClient = MongoClient(
            connection_string,
            driver=DRIVER_METADATA,
        )
        collection = client[database_name].create_collection(collection_name)
        return cls(
            collection=collection,
            entity_extraction_model=entity_extraction_model,
            entity_prompt=entity_prompt,
            query_prompt=query_prompt,
            max_depth=max_depth,
            allowed_entity_types=allowed_entity_types,
            allowed_relationship_types=allowed_relationship_types,
            entity_examples=entity_examples,
            entity_name_examples=entity_name_examples,
            validate=validate,
            validation_action=validation_action,
        )

    def close(self) -> None:
        """Close the resources used by the MongoDBGraphStore."""
        self.collection.database.client.close()

    def _write_entities(self, entities: List[Entity]) -> BulkWriteResult | None:
        """Isolate logic to insert and aggregate entities."""
        operations = []
        for entity in entities:
            relationships = entity.get("relationships", {})
            target_ids = relationships.get("target_ids", [])
            types = relationships.get("types", [])
            attributes = relationships.get("attributes", [])

            # Ensure the lengths of target_ids, types, and attributes align
            if not (len(target_ids) == len(types) == len(attributes)):
                logger.warning(
                    f"Targets, types, and attributes do not have the same length for {entity['_id']}!"
                )

            operations.append(
                UpdateOne(
                    filter={"_id": entity["_id"]},  # Match on _id
                    update={
                        "$setOnInsert": {  # Set if upsert
                            "_id": entity["_id"],
                            "type": entity["type"],
                        },
                        "$addToSet": {  # Update without overwriting
                            **{
                                f"attributes.{k}": {"$each": v}
                                for k, v in entity.get("attributes", {}).items()
                            },
                        },
                        "$push": {  # Push new entries into arrays
                            "relationships.target_ids": {"$each": target_ids},
                            "relationships.types": {"$each": types},
                            "relationships.attributes": {"$each": attributes},
                        },
                    },
                    upsert=True,
                )
            )

        # Execute bulk write for the entities
        if operations:
            return self.collection.bulk_write(operations)
        return None

    def add_documents(
        self, documents: Union[Document, List[Document]]
    ) -> List[BulkWriteResult]:
        """Extract entities and upsert into the collection.

        Each entity is represented by a single MongoDB Document.
        Existing entities identified in documents will be updated.

        Args:
            documents: list of textual documents and associated metadata.
        Returns:
            List containing metadata on entities inserted and updated, one value for each input document.
        """
        documents = [documents] if isinstance(documents, Document) else documents
        results = []
        for doc in documents:
            # Call LLM to find all Entities in doc
            entities = self.extract_entities(doc.page_content)
            logger.debug(f"Entities found: {[e['_id'] for e in entities]}")
            # Insert new or combine with existing entities
            new_results = self._write_entities(entities)
            assert new_results is not None
            results.append(new_results)
        return results

    def extract_entities(self, raw_document: str, **kwargs: Any) -> List[Entity]:
        """Extract entities and their relations using chosen prompt and LLM.

        Args:
            raw_document: A single text document as a string. Typically prose.
        Returns:
            List of Entity dictionaries.
        """
        # Combine the LLM with the prompt template to form a chain
        chain = self.entity_prompt | self.entity_extraction_model
        # Invoke on a document to extract entities and relationships
        response = chain.invoke(
            dict(
                input_document=raw_document,
                entity_schema=self.entity_schema,
                entity_examples=self.entity_examples,
                allowed_entity_types=self.allowed_entity_types,
                allowed_relationship_types=self.allowed_relationship_types,
            )
        )
        # Post-Process output string into list of entity json documents
        # Strip the ```json prefix and trailing ```
        assert isinstance(response.content, str)
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

        Args:
            raw_document: A single text document as a string. Typically prose.
        Returns:
            List of entity names / _ids.
        """
        # Combine the llm with the prompt template to form a chain
        chain = self.query_prompt | self.entity_extraction_model
        # Invoke on a document to extract entities and relationships
        response = chain.invoke(
            dict(
                input_document=raw_document,
                entity_name_examples=self.entity_name_examples,
                allowed_entity_types=self.allowed_entity_types,
            )
        )
        # Post-Process output string into list of entity json documents
        # Strip the ```json prefix and suffix
        assert isinstance(response.content, str)
        json_string = (
            response.content.removeprefix("```json").removesuffix("```").strip()
        )
        return json.loads(json_string)

    def find_entity_by_name(self, name: str) -> Optional[Entity]:
        """Utility to get Entity dict from Knowledge Graph / Collection.
        Args:
            name: _id string to look for.
        Returns:
            List of Entity dicts if any match name.
        """
        return self.collection.find_one({"_id": name})

    def related_entities(
        self,
        starting_entities: List[str],
        max_depth: Optional[int] = None,
    ) -> List[Entity]:
        """Traverse Graph along relationship edges to find connected entities.

        Args:
            starting_entities: Traversal begins with documents whose _id fields match these strings.
            max_depth: Recursion continues until no more matching documents are found,
                or until the operation reaches a recursion depth specified by this parameter.

        Returns:
            List of connected entities.
        """
        pipeline = [
            # Match starting entities
            {"$match": {"_id": {"$in": starting_entities}}},
            {
                "$graphLookup": {
                    "from": self.collection.name,
                    "startWith": "$relationships.target_ids",  # Start traversal with relationships.target_ids
                    "connectFromField": "relationships.target_ids",  # Traverse via relationships.target_ids
                    "connectToField": "_id",  # Match to entity _id field
                    "as": "connections",  # Store connections
                    "maxDepth": max_depth or self.max_depth,  # Limit traversal depth
                    "depthField": "depth",  # Track depth
                }
            },
            # Exclude connections from the original document
            {
                "$project": {
                    "_id": 0,
                    "original": {
                        "_id": "$_id",
                        "type": "$type",
                        "attributes": "$attributes",
                        "relationships": "$relationships",
                    },
                    "connections": 1,  # Retain connections for deduplication
                }
            },
            # Combine original and connections into one array
            {
                "$project": {
                    "combined": {
                        "$concatArrays": [
                            ["$original"],  # Include original as an array
                            "$connections",  # Include connections
                        ]
                    }
                }
            },
            # Unwind the combined array into individual documents
            {"$unwind": "$combined"},
            # Remove duplicates by grouping on `_id` and keeping the first document
            {
                "$group": {
                    "_id": "$combined._id",  # Group by entity _id
                    "entity": {"$first": "$combined"},  # Keep the first occurrence
                }
            },
            # Format the final output
            {
                "$replaceRoot": {
                    "newRoot": "$entity"  # Use the deduplicated document as the root
                }
            },
        ]
        return list(self.collection.aggregate(pipeline))  # type:ignore[arg-type]

    def similarity_search(self, input_document: str) -> List[Entity]:
        """Retrieve list of connected Entities found via traversal of KnowledgeGraph.

        1. Use LLM & Prompt to find entities within the input_document itself.
        2. Find Entity Nodes that match those found in the input_document.
        3. Traverse the graph using these as starting points.

        Args:
            input_document: String to find relevant documents for.
        Returns:
            List of connected Entity dictionaries.
        """
        starting_ids: List[str] = self.extract_entity_names(input_document)
        return self.related_entities(starting_ids)

    def chat_response(
        self,
        query: str,
        chat_model: Optional[BaseChatModel] = None,
        prompt: Optional[ChatPromptTemplate] = None,
    ) -> BaseMessage:
        """Responds to a query given information found in Knowledge Graph.

        Args:
            query: Prompt before it is augmented by Knowledge Graph.
            chat_model: ChatBot. Defaults to entity_extraction_model.
            prompt: Alternative Prompt Template. Defaults to prompts.rag_prompt.
        Returns:
            Response Message. response.content contains text.
        """
        if chat_model is None:
            chat_model = self.entity_extraction_model
        if prompt is None:
            prompt = rag_prompt

        # Perform Retrieval on knowledge graph
        related_entities = self.similarity_search(query)
        # Combine the LLM with the prompt template to form a chain
        chain = prompt | chat_model
        # Invoke with query
        return chain.invoke(
            dict(
                query=query,
                related_entities=related_entities,
                entity_schema=entity_schema,
            )
        )

    def to_networkx(
        self,
        nx_opts: Optional[dict] = None,
        json_opts: Optional[dict] = None,
        **kwargs: Any,
    ) -> networkx.DiGraph:
        """Utility converts Entity Collection to `NetworkX DiGraph <https://networkx.org/documentation/stable/index.html>`_

        NOTE: Requires optional-dependency "viz", i.e. `pip install "langchain-mongodb[viz]"`.

        Args:
            nx_opts: Keyword arguments for networkx calls.
            json_opts: Keyword arguments for printing of node attributes and types.
            **kwargs: Keyword arguments available for compatibility.

        Returns: networkx.DiGraph
        """

        try:
            import json

            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "Install optional-dependency `viz` for networkx or to view in Holoviews"
            ) from e

        def _safe_get(lst: list, i: int, default: Any = "") -> Any:
            return lst[i] if i < len(lst) else default

        nx_opts = {} if nx_opts is None else nx_opts
        json_opts = {} if json_opts is None else json_opts

        # First pass: Add all nodes with their attributes
        nx_graph = nx.DiGraph(**nx_opts)
        for doc in self.collection.find({}, {"type": 1, "attributes": 1}):
            # Add node with all attributes
            node_id = doc["_id"]
            node_attrs = {}
            node_attrs["type"] = json.dumps(doc.get("type", ""), **json_opts)
            node_attrs["attributes"] = json.dumps(
                doc.get("attributes", {}), **json_opts
            )
            nx_graph.add_node(node_id, **node_attrs, **json_opts)

        # Second pass: Add edges based on relationships
        for doc in self.collection.find({}, {"_id": 1, "relationships": 1}):
            source_id = doc["_id"]
            relationships = doc.get("relationships", {})
            # relationships can contain numerous target_ids, each with type and attributes
            target_ids = relationships.get("target_ids", [])
            n_targets = len(target_ids)
            types = relationships.get("types", [])
            attrs = relationships.get("attributes", [])

            for t in range(n_targets):
                # Add edge and attributes
                edge_attrs = {}
                edge_attrs["type"] = json.dumps(_safe_get(types, t), **json_opts)
                edge_attrs["attributes"] = json.dumps(_safe_get(attrs, t), **json_opts)
                if nx_graph.has_node(target_ids[t]):
                    nx_graph.add_edge(source_id, target_ids[t], **edge_attrs, **nx_opts)
                else:
                    logger.warning(
                        f"{source_id=} references {target_ids[t]=} not found in collection"
                    )

        return nx_graph

    def view(
        self,
        layout: Optional[Callable] = None,
        nx_opts: Optional[dict] = None,
        json_opts: Optional[dict] = None,
        edge_opts: Optional[dict] = None,
        node_opts: Optional[dict] = None,
        **kwargs: Any,
    ) -> holoviews.Graph:
        """Draws a Knowledge Graph as Holoviews/Bokeh interactive plot.

        We first convert the entity collection to a NetworkX Graph,
        and then convert it to a Holoviews Graph via their API.

        The default layout chosen is the spring_layout.
        This maximizes the distance between nodes. As our entities have a type field,
        however, another good layout choice might be
        `layout=nx.multipartite_layout, nx_opts["subset_key"]= "type"`
        as multipartite layout positions nodes in straight lines by subset key.

        NOTE: Requires optional-dependency "viz", i.e. `pip install "langchain-mongodb[viz]"`.

        You can save the view as any HoloViews object with `.save`.
        The type will be inferred from the filename's suffix,
        (e.g., hv.save(graph, "graph.html")) or by clicking the download widget
        on the Bokeh plot from a Jupyter notebook.

        Args:
            layout: `networkx layout. <https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout>`_
                Defaults to networkx.spring_layout.
            nx_opts: Keyword arguments for to_networkx function.
            json_opts: Keyword arguments for printing of node attributes and types.
            edge_opts: Keyword arguments to draw edges.
            node_opts: Keyword arguments to draw nodes.
            **kwargs: Keyword arguments available for compatibility.

        Returns: `holoviews.Graph <https://holoviews.org/user_guide/Network_Graphs.html>`_

        """
        try:
            import holoviews as hv
            import networkx as nx

            hv.extension("bokeh")
        except ImportError as e:
            raise ImportError("To view graph, install optional-dependency `viz`") from e

        hv.opts.defaults(
            hv.opts.Graph(xaxis=None, yaxis=None), hv.opts.Nodes(xaxis=None, yaxis=None)
        )

        if layout is None:
            layout = nx.spring_layout
        # Convert entity collection to NetworkX graph.
        nx_opts = {} if nx_opts is None else nx_opts
        json_opts = {} if json_opts is None else json_opts
        nx_graph = self.to_networkx(**nx_opts, **json_opts)
        # Convert to HoloViews Graph
        hv_graph = hv.Graph.from_networkx(nx_graph, layout, **nx_opts)
        # Display with hover tools over edges and nodes
        edge_opts = {} if edge_opts is None else edge_opts
        node_opts = {} if node_opts is None else node_opts
        return hv_graph.opts(
            inspection_policy="edges", **edge_opts
        ) * hv_graph.nodes.opts(**node_opts)

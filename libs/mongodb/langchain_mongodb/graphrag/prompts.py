import os
from typing import List

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pymongo import MongoClient
from pymongo.collection import Collection

# TODO - Parameterize this to constrain relationships.ChatPromptTemplate.from_messages([extraction_context]) a la https://github.com/datastax/ragstack-ai/blob/main/libs/knowledge-graph/ragstack_knowledge_graph/runnables.py
# TODO - Examine whether relationships with a list of dicts makes sense. It's there to add properties to the relationship but will it work with graphLookup?

entity_schema = """
A valid json document with a single top-level key 'entities'.
It's value should be an array of the entities inferred. 
 
Each Entity will be represented by a single JSON Document. It will have the following fields.
* ID: A unique identifier for the entity (e.g., UUID, name).
* type: A string specifying the type of the entity (e.g., “Person”, “Organization”).
* relationships: Stored as embedded key-value pairs. Keys are relationship types, values are lists of target entity IDs, along with additional metadata describing the relationship to that entity.
* properties: A dictionary containing key-value pairs of attributes describing the entity. Properties should not include things that could be entities. When in doubt, make something an entity.

## Example Entity structure
{{
  "ID": "Alice Palace",
  "type": "Person",
  "properties": {{
    "position": "CEO",
    "startDate": "2018-01-01"
  }},
  "relationships": {{
    "employer": [  
      {{
        "target": "MongoDB"
      }}
    ],
    "friend": [
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
"""

extraction_context = """
# CONTEXT:
## Overview
You are a top-tier algorithm designed for extracting information to build a knowledge graph in structured formats 
of entities and their relationships.
INPUT: You will be provided a text document.
Try to capture as much information from the text as possible without sacrificing accuracy.
Do not add any information that is not explicitly mentioned in the text. The aim is to achieve simplicity and clarity in the knowledge graph.
OUTPUT: You will produce valid json. It will have a single top-level key 'entities', with its value an array
 of the entities inferred. Each should follows the schema below. 

## General Schema Specification for a Knowledge Graph Entity in a MongoDB Collection.
Each Entity will be represented by a single JSON Document. It will have the following fields.
* ID: A unique identifier for the entity (e.g., UUID, name).
* type: A string specifying the type of the entity (e.g., “Person”, “Organization”).
* relationships: Stored as embedded key-value pairs. Keys are relationship types, values are lists of target entity IDs, along with additional metadata describing the relationship to that entity.
* properties: A dictionary containing key-value pairs of attributes describing the entity. Properties should not include things that could be entities. When in doubt, make something an entity.

## Example Entity structure
{{
  "ID": "12345",
  "name": "Alice Palace",
  "type": "Person",
  "properties": {{
    "position": "CEO",
    "startDate": "2018-01-01"
  }},
  "relationships": {{
    "employer": [  
      {{
        "target": "MongoDB"
      }}
    ],
    "friend": [
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

## Entity Extraction Rules
1. **Persons**: Extract all individuals mentioned, using their full names as unique IDs when available.
2. **Organizations**: Treat all named organizations (e.g., companies, schools, or groups) as distinct entities with the type "Organization." 
   - If an organization is associated with a person (e.g., as an employer), capture the relationship explicitly (e.g., "employee").
   - Do not nest organizations as properties of another entity; they should be separate entities with their own unique IDs.
3. **Relationships**: Capture all relationships inferred from the text and ensure target entity IDs are consistent.
4. **Places**: Extract named locations as entities with the type "location".

## Entity Resolution and Consistency
**Maintain Entity Consistency when extracting entities.** If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he") always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID. Define required fields (e.g., ID, name,, type) and allow optional properties.

## Relationships
Ensure that every relationship target points to a valid entity ID. Ensure consistency and generality in relationship types when constructing knowledge schemas. Instead of using specific and momentary types such as 'became_professor', use more general and timeless relationship types like 'professor'. Make sure to use general and timeless relationship types!

Remember, the knowledge graph should be coherent, consistent, and facilitate clear navigation of entities and their connections. 
Avoid mixing properties and entities for clarity. Maintaining consistency in entity references is crucial.
"""

# entity_prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessagePromptTemplate.from_template(extraction_context),
#         HumanMessagePromptTemplate.from_template("INPUT: {input_document}"),
#     ]
# )



retrieval_context = """
You are a top-tier algorithm designed for extracting information in the form of knowledge graphs
comprised of entities (nodes) and their relationships (ships). 

INPUT: You will be provided a short document (query) 
from which you infer the entities as names or human-readable identifiers found in the text,
and probable relationships. These will form the starting points for graph traversal
to find similar entities. Hence, we are not looking for directed graph triples 
 like (source node, relationship edge, target node). We are looking for source entity id,
 and relationships implied by the input text. This form is common in questions.
 For example, "Where is MongoDB located?" provides a source entity, "MongoDB"
 and a relationship "location" but the question mark is effectively the target.

OUTPUT: Provide your response as valid json where 
keys are entity IDs and values are lists of relationships.
"""



query_schema = """
A valid json document where keys are entity IDs and values are lists of relationships.

The outputs will form the starting points for graph traversal
to find similar entities. Hence, we are not looking for directed graph triples 
like (source node, relationship edge, target node). We are looking for source entity ids,
and relationships implied by the input text. This form is common in questions.
For example, "Where is MongoDB located?" provides a source entity, "MongoDB"
and a relationship "location" but the question mark is effectively the target.

Example:
{{
    "MongoDB": ["industry", "employee", "event"],
    "Alice Palace": ["employer", "friend", "event", "activity"]
}}
"""

# retrieval_prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessagePromptTemplate.from_template(retrieval_context),
#         HumanMessagePromptTemplate.from_template("INPUT: {input_document}"),
#     ]
# )

extraction_template = """
## Overview
You are a top-tier algorithm designed to extract information from unstructured text 
to build a knowledge graph in structured format of entities (nodes) and their relationships (edges). 
The graph will be stored in a MongoDB Collection and traversed using $graphLookup
from starting points of entity IDs and relationship types.

Use the following as guidelines.

- Simplicity: The graph should have as few entities and relationship types as needed to convey the information in the input.
- Consistency: Connections can only be made if entities and relationships use consistent naming
- Generality: The graph should be useful for describing the concepts in not just this document but other similar documents.
- Accuracy: Do not add any information that is not explicitly mentioned in the text.

INPUT: You will be provided a text document.
OUTPUT: You will produce valid json according the "Output Schema" section below.
 
## Entities

An entity in a knowledge graph is a uniquely identifiable object or concept, 
such as a person, organization, location, object, or event, 
represented as a node with attributes (properties) and relationships to other entities, 
enabling structured and meaningful connections within the graph.

Extract all entities mentioned, using their full names as unique IDs when available.

Maintain Entity Consistency when extracting entities. If an entity, such as "John Doe", 
is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
always use the most complete identifier for that entity throughout the knowledge graph. 
In this example, use "John Doe" as the entity ID. 
Define required fields (e.g., ID, name,, type) and allow optional properties.

Do not nest organizations as properties of another entity. they should be separate entities with their own unique IDs.

## Relationships

Relationships represent edges in the knowledge graph. Relationships describe a specific edge type. 
Ensure consistency and generality in relationship names when constructing knowledge schemas. 
Instead of using specific and momentary types such as 'worked_at', use more general and timeless relationship types 
like 'employee'. Add details as properties. Make sure to use general and timeless relationship types!

If synonyms are found in the document, choose the most general and use consistently.

If a relationship is bidirectional, each entity should contain the relationship with the other entity as target. 
For example, if Casey works at MongoDB, MongoDB is an employer of Casey, and Casey is an employee of MongoDB.

## Output Schema
{output_schema}
"""

entity_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(extraction_template.format(output_schema=entity_schema)),
        HumanMessagePromptTemplate.from_template("INPUT: {input_document}"),
    ]
)

retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(extraction_template.format(output_schema=query_schema)),
        HumanMessagePromptTemplate.from_template("INPUT: {input_document}"),
    ]
)

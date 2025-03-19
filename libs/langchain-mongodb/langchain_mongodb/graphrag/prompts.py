from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

ENTITY_EXTRACTION_INSTRUCTIONS = """
## Overview
You are a meticulous analyst tasked with extracting information from unstructured text
to build a knowledge graph in a structured json format of entities (nodes) and their relationships (edges).
The graph will be stored in a MongoDB Collection and traversed using $graphLookup
from starting points of entity nodes matching names found in a query, and follow their relationships.

Use the following as guidelines.

- Simplicity: The graph should have as few entities and relationship types as needed to convey the information in the input.
- Consistency: Connections can only be made if entities and relationships use consistent naming.
- Generality: The graph should be useful for describing the concepts in not just this document but other similar documents.
- Accuracy: Do not add any information that is not explicitly mentioned in the text.

INPUT: You will be provided a text document.
OUTPUT:
- You will produce valid json according the "Output Schema" section below.
- Your response **must be** a **valid JSON document** with NO extra text, explanations, or markdown formatting.
- The extracted entities and relationships **MUST STRICTLY CONFORM** to the constraints outlined below.
- Any entities or relationships not matching the allowed types must be **EXCLUDED**.


## Entities
An entity in a knowledge graph is a uniquely identifiable object or concept
(such as a person, organization, location, object, or event),
represented as a node with attributes (properties) and relationships to other entities.

Use the reserved field name `_id` for the name. It will be a unique primary key,
and MongoDB automatically creates an index for the `_id` field.

Maintain Entity Consistency when extracting entities. If an entity, such as "John Doe",
is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "John", "Mr Doe", "he"),
always use the most complete identifier for that entity throughout the knowledge graph.
In this example, use "John Doe" as the entity `_id.`

**Allowed Entity Types**:
- Extract ONLY entities whose `type` matches one of the following: {allowed_entity_types}.
- NOTE: If this list is empty, ANY `type` is permitted.

### Examples of Exclusions:
- If `allowed_entity_types` is `["Person", "Organization"]`, and the text mentions "Event" or "Location",
  these entities must **NOT** be included in the output.

## Relationships
Relationships represent edges in the knowledge graph. Relationships describe a specific edge type.
Relationships MUST include a target entity, but Entities can be extracted that DO NOT have relationships!
Ensure consistency and generality in relationship names when constructing knowledge schemas.
Instead of using specific and momentary types such as 'worked_at', use more general and timeless relationship types
like 'employee'. Add details as attributes. Make sure to use general and timeless relationship types!

**Allowed Relationship Types**:
- Extract ONLY relationships whose `type` matches one of the following: {allowed_relationship_types}.
- If this list is empty, ANY relationship type is permitted.
- Map synonymous or related terms to the closest matching allowed type. For example:
	-	“works for” or “employed by” → employee
	-	“manages” or “supervises” → manager
- If a relationship cannot be named with one of the allowed keys, **DO NOT include it**.
- An entity need not have a relationships object if no relationship is found that matches the allowed relation types.

### Examples of Exclusions:
- If `allowed_relationship_types` is `["employs", "friend"]` and the text implies a "partner" relationship,
  the entities can be added, but the "partner" relationship must **NOT** be included.

## Validation
Before producing the final output:
1. Validate that all extracted entities have an `_id` and `type`.
2. Validate that all `type` values are in {allowed_entity_types}.
3. Validate that all relationships use keys in {allowed_relationship_types}.
4. Exclude any entities or relationships failing validation.

## Output Schema
Output a valid JSON document with a single top-level key, `entities`, as an array of objects.
Each object must conform to the following schema:
{entity_schema}

{entity_examples}
"""


NAME_EXTRACTION_INSTRUCTIONS = """
You are a meticulous analyst tasked with extracting information from documents to form
knowledge graphs of entities (nodes) and their relationships (edges).

You will be provided a short document (query) from which you infer the entity names.
You need not think about relationships between the entities. You only need names.

Provide your response as a valid JSON Array of entity names
or human-readable identifiers, found in the text.

**Allowed Entity Types**:
- Extract ONLY entities whose `type` matches one of the following: {allowed_entity_types}.
- NOTE: If this list is empty, ANY `type` is permitted.

### Examples of Exclusions:
- If `allowed_entity_types` is `["Person", "Organization"]`, and the text mentions "Event" or "Location",
  these entities must **NOT** be included in the output.

 ## Examples:
 Example 1: `allowed_entity_types` is `[]`
    input: "John Doe works at ACME in New York"
    output: ["John Doe", "ACME", "New York"]

 In this example, you would identify 3 entities:
 John Doe of type person; ACME of type organization; New York of type place.

Example 2: `allowed_entity_types` is `[organization, place]`
    input: "John Doe works at ACME in New York"
    output: ["ACME", "New York"]

 In this example, you would identify only 2 entities:
 ACME of type organization; New York of type place.
 John Doe, of type person, would be excluded.

 2. input: "In what continent is Brazil?
    output: ["Brazil"]

This example is in the form of a question. There is one entity,

3. input: "For legal and operational purposes, many governments and organizations adopt specific definitions."
   output: []

In the final example, there are no entities.
Though there are concepts and nouns that might be types or attributes of entities,
there is nothing here that could be seen as being a unique identifier or name.

### (Optional) Additional Examples

{entity_name_examples}
"""


RAG_INSTRUCTIONS = """
## Context
You are a meticulous analyst tasked with extracting information in the form of knowledge graphs
comprised of entities (nodes) and their relationships (edges).

Based on the user input (query) that will be provided, you have already retrieved information
from the knowledge graph in the form of a list of entities known to be related to those in the Query.

From the context retrieved alone, please respond to the Query.
Your response should be a string of concise prose.

## Entity Schema
The entities have the following schema matching MongoDB's $jsonSchema style used for MongoDB validation.

{entity_schema}

## Entities Found to be Related to Query
{related_entities}
"""

entity_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(ENTITY_EXTRACTION_INSTRUCTIONS),
        HumanMessagePromptTemplate.from_template("{input_document}"),
    ]
)

query_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(NAME_EXTRACTION_INSTRUCTIONS),
        HumanMessagePromptTemplate.from_template("{input_document}"),
    ]
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(RAG_INSTRUCTIONS),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

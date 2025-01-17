"""
The following contains the JSON Schema for the Entities as entered in the Collection
representing the Knowledge Graph.
If validate is set to True, the schema is enforced upon insert and update.
See `$jsonSchema <https://www.mongodb.com/docs/manual/reference/operator/query/jsonSchema/>`_

The following defines the default entity_schema.
It allows all possible values of "type" and "relationship".

If allowed_entity_types: List[str] is given to MongoDBGraphStore's constructor,
then `self._schema["properties"]["type"]["enum"] = allowed_entity_types` is added.

If allowed_relationship_types: List[str] is given to MongoDBGraphStore's constructor,
additionalProperties is set to False, and relationship schema is provided for each key.
"""

"""Validation schema for relationships when inserting into Collection."""
relationship_schema = {
    "bsonType": "array",
    "items": {
        "bsonType": "object",
        "required": ["target"],
        "properties": {
            "target": {
                "bsonType": "string",
                "description": "ID of the target entity",
            },
            "attributes": {
                "bsonType": "object",
                "description": "Metadata describing the relationship",
                "additionalProperties": {"bsonType": "string"},
            },
        },
    },
}

"""Validation Schema for Enrtity used when inserting into Collection"""
entity_schema = {
    "bsonType": "object",
    "required": ["ID", "type", "attributes", "relationships"],
    "properties": {
        "ID": {
            "bsonType": "string",
            "description": "Unique identifier for the entity",
        },
        "type": {
            "bsonType": "string",
            "description": "Type of the entity (e.g., 'Person', 'Organization')",
        },
        "attributes": {
            "bsonType": "object",
            "description": "Key-value pairs describing the entity",
            "additionalProperties": {
                "anyOf": [
                    {"bsonType": "string"},  # Single string value
                    {
                        "bsonType": "array",
                        "items": {"bsonType": "string"},  # Array of strings
                    },
                ]
            },
        },
        "relationships": {
            "bsonType": "object",
            "description": "Key-value pairs of relationships",
            "additionalProperties": relationship_schema,
        },
    },
}

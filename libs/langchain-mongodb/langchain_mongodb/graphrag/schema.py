"""
The following contains the JSON Schema for the Entities in the Collection representing the Knowledge Graph.
If validate is set to True, the schema is enforced upon insert and update.
See `$jsonSchema <https://www.mongodb.com/docs/manual/reference/operator/query/jsonSchema/>`_

The following defines the default entity_schema.
It allows all possible values of "type" and "relationship".

If allowed_entity_types: List[str] is given to MongoDBGraphStore's constructor,
then `self._schema["properties"]["type"]["enum"] = allowed_entity_types` is added.

If allowed_relationship_types: List[str] is given to MongoDBGraphStore's constructor,
additionalProperties is set to False, and relationship schema is provided for each key.
"""

entity_schema = {
    "bsonType": "object",
    "required": ["_id", "type"],
    "properties": {
        "_id": {
            "bsonType": "string",
            "description": "Unique identifier for the entity",
        },
        "type": {
            "bsonType": "string",
            "description": "Type of the entity (e.g., 'Person', 'Organization')",
            # Note: When constrained, predefined types are added. For example:
            # "enum": ["Person", "Organization", "Location", "Event"],
        },
        "attributes": {
            "bsonType": "object",
            "description": "Key-value pairs describing the entity",
            "additionalProperties": {
                "bsonType": "array",
                "items": {"bsonType": "string"},  # Enforce array of strings
            },
        },
        "relationships": {
            "bsonType": "object",
            "description": "Key-value pairs of relationships",
            "required": ["target_ids"],
            "properties": {
                "target_ids": {
                    "bsonType": "array",
                    "description": "name/_id values of the target entities",
                    "items": {"bsonType": "string"},
                },
                "types": {
                    "bsonType": "array",
                    "description": "An array of relationships to corresponding target_ids (in same array position).",
                    "items": {"bsonType": "string"},
                    # Note: When constrained, predefined types are added. For example:
                    # "enum": ["used_in", "owns", "written_by", "located_in"],  # Predefined types
                },
                "attributes": {
                    "bsonType": "array",
                    "description": "An array of attributes describing the relationships to corresponding target_ids (in same array position). Each element is an object containing key-value pairs, where values are arrays of strings.",
                    "items": {
                        "bsonType": "object",
                        "additionalProperties": {
                            "bsonType": "array",
                            "items": {"bsonType": "string"},
                        },
                    },
                },
            },
            "additionalProperties": False,
        },
    },
}

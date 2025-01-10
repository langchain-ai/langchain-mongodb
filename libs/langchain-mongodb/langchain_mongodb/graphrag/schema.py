"""
https://www.mongodb.com/docs/manual/reference/operator/query/jsonSchema/

TODO - Finalize this
"""

entity_schema = {
    "bsonType": "object",
    "required": ["ID", "type", "relationships"],
    "properties": {
        "ID": {
            "bsonType": "string",
            "description": "A unique identifier for the entity.",
        },
        "type": {"bsonType": "string", "description": "The type of entity."},
        "properties": {
            "bsonType": "object",
            "description": "Key-value attributes describing the entity.",
        },
        "relationships": {
            "bsonType": "object",
            "description": "Embedded relationships to other entities.",
            "additionalProperties": {
                "bsonType": "array",
                "items": {
                    "bsonType": "object",
                    "required": ["target"],
                    "properties": {
                        "target": {
                            "bsonType": "string",
                            "description": "The target entity ID.",
                        },
                        "properties": {
                            "bsonType": "object",
                            "description": "Metadata about the relationship.",
                        },
                    },
                },
            },
        },
    },
}

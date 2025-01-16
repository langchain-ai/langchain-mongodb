"""
https://www.mongodb.com/docs/manual/reference/operator/query/jsonSchema/
"""

entity_schema = {
    "bsonType": "object",
    "required": ["ID", "type", "relationships"],  #  "properties",
    "properties": {
        "ID": {
            "bsonType": "string",
            "description": "Unique identifier for the entity",
        },
        "type": {
            "bsonType": "string",
            "description": "Type of the entity (e.g., 'Person', 'Organization')",
        },
        "properties": {
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
            "additionalProperties": {
                "bsonType": "array",
                "items": {
                    "bsonType": "object",
                    "required": ["target"],
                    "properties": {
                        "target": {
                            "bsonType": "string",
                            "description": "ID of the target entity",
                        },
                        "properties": {
                            "bsonType": "object",
                            "description": "Metadata describing the relationship",
                            "additionalProperties": {"bsonType": "string"},
                        },
                    },
                },
            },
        },
    },
}

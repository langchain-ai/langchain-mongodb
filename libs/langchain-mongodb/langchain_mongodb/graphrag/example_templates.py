"""These prompts serve as defaults, and templates when you wish to override them.

LLM performance can be greatly increased by adding domain-specific examples.
"""

entity_extraction = """
## Examples
Use the following examples to guide your work.

### Example 1: Constrained entity and relationship types: Person, Friend
#### Input
Alice Palace has been the CEO of MongoDB since January 1, 2018.
She maintains close friendships with Jarnail Singh, whom she has known since May 1, 2019,
and Jasbinder Kaur, who she has been seeing weekly since May 1, 2015.

#### Output
(If `allowed_entity_types` is ["Person"] and `allowed_relationship_types` is ["Friend"])
{{
  "entities": [
    {{
      "_id": "Alice Palace",
      "type": "Person",
      "attributes": {{
        "job": ["CEO of MongoDB"],
        "startDate": ["2018-01-01"]
      }},
      "relationships": {{
        "targets": ["Jasbinder Kaur", "Jarnail Singh"],
        "types": ["Friend", "Friend"],
        "attributes": [
          {{ "since": ["2019-05-01"] }},
          {{ "since": ["2015-05-01"], "frequency": ["weekly"] }}
        ]
      }}
    }},
    {{
      "_id": "Jarnail Singh",
      "type": "Person",
      "relationships": {{
        "targets": ["Alice Palace"],
        "types": ["Friend"],
        "attributes": [{{ "since": ["2019-05-01"] }}]
      }}
    }},
    {{
      "_id": "Jasbinder Kaur",
      "type": "Person",
      "relationships": {{
        "targets": ["Alice Palace"],
        "types": ["Friend"],
        "attributes": [{{ "since": ["2015-05-01"], "frequency": ["weekly"] }}]
      }}
    }}
  ]
}}

### Example 2: Event Extraction
#### Input
The 2022 OpenAI Developer Conference took place on October 10, 2022, in San Francisco.
Elon Musk and Sam Altman were keynote speakers at the event.

#### Output
(If `allowed_entity_types` is ["Event", "Person"] and `allowed_relationship_types` is ["Speaker"])
{{
  "entities": [
    {{
      "_id": "2022 OpenAI Developer Conference",
      "type": "Event",
      "attributes": {{
        "date": ["2022-10-10"],
        "location": ["San Francisco"]
      }},
      "relationships": {{
        "targets": ["Elon Musk", "Sam Altman"],
        "types": ["Speaker", "Speaker"]
      }}
    }},
    {{ "_id": "Elon Musk", "type": "Person" }},
    {{ "_id": "Sam Altman", "type": "Person" }}
  ]
}}

### Example 3: Concept Relationship
#### Input
Quantum computing is a field of study that focuses on developing computers based on the principles of quantum mechanics.

#### Output
(If `allowed_entity_types` is ["Concept"] and `allowed_relationship_types` is ["Based On"])
{{
  "entities": [
    {{
      "_id": "Quantum Computing",
      "type": "Concept",
      "relationships": {{
        "targets": ["Quantum Mechanics"],
        "types": ["Based On"]
      }}
    }},
    {{ "_id": "Quantum Mechanics", "type": "Concept" }}
  ]
}}

### Example 4: News Article
#### Input
On March 1, 2023, NASA successfully launched the Artemis II mission, sending astronauts to orbit the Moon.
NASA Administrator Bill Nelson praised the historic achievement.

#### Output
(If `allowed_entity_types` is ["Organization", "Event", "Person"] and `allowed_relationship_types` is ["Managed By", "Praised By"])
{{
  "entities": [
    {{
      "_id": "Artemis II Mission",
      "type": "Event",
      "attributes": {{ "date": ["2023-03-01"] }},
      "relationships": {{
        "targets": ["NASA"],
        "types": ["Managed By"]
      }}
    }},
    {{
      "_id": "NASA",
      "type": "Organization"
    }},
    {{
      "_id": "Bill Nelson",
      "type": "Person",
      "relationships": {{
        "targets": ["Artemis II Mission"],
        "types": ["Praised By"]
      }}
    }}
  ]
}}

### Example 5: Technical Article
#### Input
Rust is a programming language that guarantees memory safety without requiring garbage collection.
It is known for its strong ownership model, which prevents data races.

#### Output
(If `allowed_entity_types` is ["Programming Language", "Concept"] and `allowed_relationship_types` is ["Ensures", "Uses"])
{{
  "entities": [
    {{
      "_id": "Rust",
      "type": "Programming Language",
      "relationships": {{
        "targets": ["Memory Safety"],
        "types": ["Ensures"]
      }}
    }},
    {{
      "_id": "Memory Safety",
      "type": "Concept",
      "relationships": {{
        "targets": ["Ownership Model"],
        "types": ["Uses"]
      }}
    }},
    {{ "_id": "Ownership Model", "type": "Concept" }}
  ]
}}
"""

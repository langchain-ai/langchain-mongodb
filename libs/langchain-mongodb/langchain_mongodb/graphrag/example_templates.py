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

"""
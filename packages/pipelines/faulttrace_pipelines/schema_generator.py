from typing import Any, Dict
from faulttrace_core.models import FactSpec, AggregationSpec

class SchemaGenerator:
    """
    WP3: Query-Specific Extraction Schemas.
    Generates a minimal JSON Schema from FactSpec and AggregationSpec so the model
    extracts only required fields.
    """

    @classmethod
    def generate_extraction_schema(cls, fact_spec: FactSpec, agg_spec: AggregationSpec, expected_records: list[str]) -> Dict[str, Any]:
        """
        Generates a strict JSON schema that ensures:
        - include record_id and scope_decision for every row
        - reject invented fields
        """
        
        properties: Dict[str, Any] = {
            "record_id": {
                "type": "string",
                "description": "Must exactly match one of the provided record IDs."
            },
            "scope_decision": {
                "type": "string",
                "enum": ["in_scope", "ambiguous", "missing_evidence"],
                "description": "Whether the record is in-scope for the query, ambiguous, or lacks evidence."
            }
        }
        
        required_fields = ["record_id", "scope_decision"]
        
        # Add dynamic fields from fact_spec
        for field in fact_spec.fields:
            properties[field] = {
                # We use a broad type here because it could be string/number/boolean.
                # In a real implementation we would type-match with the database schema, 
                # but for LLMs we can accept primitive types or null.
                "type": ["string", "number", "boolean", "null"],
                "description": f"Extracted value for {field}."
            }
            required_fields.append(field)
            
        # The schema dictates an array of extractions
        schema = {
            "type": "object",
            "properties": {
                "extracted_records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": properties,
                        "required": required_fields,
                        "additionalProperties": False
                    },
                    "description": "List of extracted records corresponding to the batch."
                }
            },
            "required": ["extracted_records"],
            "additionalProperties": False
        }
        
        return schema

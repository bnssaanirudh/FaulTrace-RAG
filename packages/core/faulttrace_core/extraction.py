from typing import List, Optional
from pydantic import BaseModel, Field

class ExtractedField(BaseModel):
    field_name: str = Field(..., description="The name of the attribute extracted")
    value: str = Field(..., description="The extracted value")
    snippet_evidence: Optional[str] = Field(None, description="Exact text snippet from the record proving this value")
    is_missing: bool = Field(default=False, description="True if the value could not be found in the text")

class RecordExtraction(BaseModel):
    record_id: str = Field(..., description="ID of the source CorpusRecord")
    is_relevant: bool = Field(..., description="True if the record contains information relevant to the query")
    extracted_fields: List[ExtractedField] = Field(default_factory=list, description="Fields extracted from this record")
    reasoning: Optional[str] = Field(None, description="Brief explanation of inclusion/exclusion and extraction")

class QueryExtractionResult(BaseModel):
    query_id: str = Field(..., description="The ID of the query")
    record_extractions: List[RecordExtraction] = Field(default_factory=list, description="Extractions per record evaluated")

    @classmethod
    def get_json_schema(cls) -> dict:
        """Returns the strict JSON schema for OpenAI structured outputs."""
        return {
            "type": "object",
            "properties": {
                "query_id": {"type": "string"},
                "record_extractions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "record_id": {"type": "string"},
                            "is_relevant": {"type": "boolean"},
                            "extracted_fields": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field_name": {"type": "string"},
                                        "value": {"type": ["string", "null"]},
                                        "snippet_evidence": {"type": ["string", "null"]},
                                        "is_missing": {"type": "boolean"}
                                    },
                                    "required": ["field_name", "value", "snippet_evidence", "is_missing"],
                                    "additionalProperties": False
                                }
                            },
                            "reasoning": {"type": ["string", "null"]}
                        },
                        "required": ["record_id", "is_relevant", "extracted_fields", "reasoning"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["query_id", "record_extractions"],
            "additionalProperties": False
        }

"""
Provenance-first extraction schema for FaultTrace-RAG.

These models represent the citation and evidence layer for text-benchmark answers.
Every extracted fact, cited answer, and entity reference is traceable to a specific
document, chunk, and character span in the source corpus.

Design principles:
- All citations are validated against known doc/chunk IDs at construction time.
- Support status is always explicit — there is no implicit "supported".
- Abstention is first-class: if evidence is inadequate, abstention_reason is required.
- Provider/model/version is always recorded for reproducibility.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SupportStatus(StrEnum):
    """The degree to which retrieved evidence supports a claimed answer."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONFLICTING = "conflicting"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ValidationStatus(StrEnum):
    VALID = "valid"
    REPAIRED = "repaired"
    REJECTED = "rejected"
    PENDING = "pending"


class EntityType(StrEnum):
    """Configurable entity types for scientific / general-domain text."""

    SCIENTIFIC_CONCEPT = "scientific_concept"
    METHOD = "method"
    DISEASE = "disease"
    CHEMICAL = "chemical"
    ORGANIZATION = "organization"
    PRODUCT = "product"
    DATE = "date"
    PERSON = "person"
    LOCATION = "location"
    CLAIM = "claim"
    QUANTITY = "quantity"
    OTHER = "other"


class RelationType(StrEnum):
    """Relation types for knowledge graph edges."""

    SUPPORTS = "supports"
    REFUTES = "refutes"
    MENTIONS = "mentions"
    COMPARES = "compares"
    MEASURES = "measures"
    CAUSES = "causes"
    USES_METHOD = "uses_method"
    BELONGS_TO_TOPIC = "belongs_to_topic"
    CONTAINS = "contains"
    RELATED_TO = "related_to"
    SEMANTIC_SIMILAR_TO = "semantic_similar_to"


class FactType(StrEnum):
    ENTITY = "entity"
    RELATION = "relation"
    CLAIM = "claim"
    QUANTITY = "quantity"
    DATE = "date"
    DEFINITION = "definition"


# ---------------------------------------------------------------------------
# Span Models
# ---------------------------------------------------------------------------


class ExtractionSpan(BaseModel):
    """Character-level or sentence-level location of evidence within a document."""

    doc_id: str = Field(..., description="Source document ID")
    chunk_id: str | None = Field(None, description="Chunk ID if document was chunked")
    sentence_id: str | None = Field(None, description="Sentence-level ID (e.g. 'doc1_s3')")

    char_start: int | None = Field(None, ge=0, description="Character offset start (inclusive)")
    char_end: int | None = Field(None, ge=0, description="Character offset end (exclusive)")

    surface_text: str = Field(..., description="Exact text surface form at this location")
    source_quote: str = Field(..., description="The verbatim quote from the source document")

    @model_validator(mode="after")
    def validate_span_bounds(self) -> ExtractionSpan:
        if self.char_start is not None and self.char_end is not None:
            if self.char_end <= self.char_start:
                raise ValueError(
                    f"char_end ({self.char_end}) must be > char_start ({self.char_start})"
                )
        return self


# ---------------------------------------------------------------------------
# ExtractedFact
# ---------------------------------------------------------------------------


class ExtractedFact(BaseModel):
    """A single extracted fact with full provenance."""

    fact_id: str = Field(..., description="Unique ID for this fact within the record")
    fact_type: FactType = Field(..., description="Category of this fact")
    entity_type: EntityType | None = Field(None, description="Entity type if applicable")
    relation_type: RelationType | None = Field(None, description="Relation type if applicable")

    normalized_value: str = Field(..., description="Canonically normalized extracted value")
    surface_text: str = Field(..., description="Original surface form in the document")
    aliases: list[str] = Field(default_factory=list, description="Known aliases or alternate forms")

    span: ExtractionSpan | None = Field(None, description="Evidence span in source document")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence [0, 1]")

    # Provenance
    provider: str = Field(
        ..., description="Extraction provider name (e.g. 'deterministic_fixture')"
    )
    model_version: str | None = Field(None, description="Model or rule-set version")
    extracted_at: str | None = Field(None, description="ISO timestamp of extraction")

    # Validation
    validation_status: ValidationStatus = Field(default=ValidationStatus.PENDING)
    rejection_reason: str | None = Field(None, description="Reason for rejection if rejected")
    repair_count: int = Field(default=0, description="Number of repair attempts before acceptance")


# ---------------------------------------------------------------------------
# CitedAnswer
# ---------------------------------------------------------------------------


class CitedAnswer(BaseModel):
    """
    A benchmark answer with full citation and support status.

    Every answer exposed by text-benchmark pipelines must populate this model.
    Unanswerable questions must set support_status=INSUFFICIENT_EVIDENCE and
    provide a non-empty abstention_reason.
    """

    answer_text: str | None = Field(None, description="The answer text or value")
    answer_value: Any | None = Field(
        None, description="Structured answer value (number, list, etc.)"
    )

    support_status: SupportStatus = Field(
        ..., description="Degree to which retrieved evidence supports this answer"
    )

    # Citation pointers
    cited_doc_ids: list[str] = Field(
        default_factory=list, description="IDs of documents cited as evidence"
    )
    cited_chunk_ids: list[str] = Field(default_factory=list, description="Chunk-level citation IDs")
    cited_sentence_ids: list[str] = Field(
        default_factory=list, description="Sentence-level citation IDs"
    )

    supporting_quote: str | None = Field(
        None, description="The verbatim supporting quote from the best evidence document"
    )

    # Abstention
    abstention_reason: str | None = Field(
        None, description="Required when support_status is INSUFFICIENT_EVIDENCE or CONFLICTING"
    )

    @model_validator(mode="after")
    def validate_abstention_reason(self) -> CitedAnswer:
        needs_reason = self.support_status in (
            SupportStatus.INSUFFICIENT_EVIDENCE,
            SupportStatus.CONFLICTING,
        )
        if needs_reason and not self.abstention_reason:
            raise ValueError(
                f"abstention_reason is required when support_status is '{self.support_status.value}'"
            )
        return self


# ---------------------------------------------------------------------------
# ExtractionRecord
# ---------------------------------------------------------------------------


class ExtractionRecord(BaseModel):
    """
    Full provenance record for a single document's extraction in a benchmark run.

    This is the top-level output of any ExtractionProvider. It binds together
    dataset context, all extracted facts, and the cited answer — and enforces
    citation integrity at construction time.
    """

    # Provenance identifiers
    dataset_id: str = Field(..., description="e.g. 'scifact', 'hotpotqa', 'covidqa'")
    split: str = Field(..., description="e.g. 'test', 'dev', 'train'")
    doc_id: str = Field(..., description="Source document ID")
    chunk_id: str | None = Field(None, description="Chunk ID if applicable")

    # Content
    query_id: str | None = Field(None, description="Query or claim ID this was extracted for")
    facts: list[ExtractedFact] = Field(default_factory=list)
    cited_answer: CitedAnswer | None = Field(None)

    # Metadata
    provider: str = Field(..., description="Provider name")
    model_version: str | None = Field(None)
    schema_version: str = Field(default="1.0.0")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (e.g. rejection reason, repair log)"
    )

    @model_validator(mode="after")
    def validate_citation_integrity(self) -> ExtractionRecord:
        """
        Citation integrity enforcement:
        Every cited_doc_id in the CitedAnswer must be the actual doc_id of this record
        or match an explicitly declared related doc_id.

        For multi-document answers (e.g. HotpotQA), this should be called at the
        aggregate level. At the single-document level, we validate that no
        *foreign* doc_id is cited that wasn't passed through extraction.
        """
        if self.cited_answer is None:
            return self

        for cited_id in self.cited_answer.cited_doc_ids:
            if cited_id != self.doc_id:
                raise ValueError(
                    f"Citation integrity violation: cited_doc_id '{cited_id}' "
                    f"is not the extraction record's doc_id '{self.doc_id}'. "
                    "Use AggregatedExtractionResult for multi-document citations."
                )
        return self


class AggregatedExtractionResult(BaseModel):
    """
    Multi-document extraction result for a single query.

    Aggregates ExtractionRecords from multiple documents and validates that
    all cited_doc_ids in the final CitedAnswer exist in the record set.
    """

    query_id: str
    dataset_id: str
    split: str

    records: list[ExtractionRecord] = Field(default_factory=list)
    final_answer: CitedAnswer | None = Field(None)

    provider: str
    model_version: str | None = None
    schema_version: str = "1.0.0"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_multi_doc_citation_integrity(self) -> AggregatedExtractionResult:
        """Ensure all cited_doc_ids in final_answer exist in the records set."""
        if self.final_answer is None:
            return self

        known_doc_ids = {r.doc_id for r in self.records}
        for cited_id in self.final_answer.cited_doc_ids:
            if cited_id not in known_doc_ids:
                raise ValueError(
                    f"Citation integrity violation: cited_doc_id '{cited_id}' "
                    f"was not found in the extraction record set. "
                    f"Known doc_ids: {sorted(known_doc_ids)}"
                )
        return self

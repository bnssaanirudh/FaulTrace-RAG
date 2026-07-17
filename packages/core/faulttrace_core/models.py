"""
Core domain models for FaultTrace-RAG.

All models use Pydantic v2 with strict validation.
Models are independent of FastAPI and persistence layers.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Literal, Optional, Union
from uuid import UUID, uuid4

import orjson
from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RecordCategory(str, Enum):
    ELECTRONICS = "Electronics"
    BOOKS = "Books"
    HOME_KITCHEN = "Home & Kitchen"
    SPORTS = "Sports & Outdoors"
    CLOTHING = "Clothing"
    TOYS = "Toys & Games"
    BEAUTY = "Beauty"
    AUTOMOTIVE = "Automotive"
    FOOD = "Food & Grocery"
    OFFICE = "Office Products"


class QueryFamily(str, Enum):
    COUNT = "count"
    MEAN = "mean"
    PROPORTION = "proportion"
    COMPARISON = "comparison"
    TOP_K = "top_k"
    TREND = "trend"


class AggregationKind(str, Enum):
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    PROPORTION = "proportion"
    COMPARISON = "comparison"
    TOP_K = "top_k"
    TREND = "trend"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class TraceEventType(str, Enum):
    QUERY_LOAD = "query_load"
    SCOPE_ENUMERATE = "scope_enumerate"
    FACT_EXTRACT = "fact_extract"
    AGGREGATE = "aggregate"
    VALIDATE = "validate"
    PERSIST = "persist"
    ERROR = "error"


class AgreementStatus(str, Enum):
    AGREED = "agreed"
    DISAGREED = "disagreed"
    SINGLE_ENGINE = "single_engine"
    NOT_COMPUTED = "not_computed"


class ReasonCode(str, Enum):
    SCOPE_NOT_ENUMERATED = "SCOPE_NOT_ENUMERATED"
    SCOPE_COVERAGE_BELOW_REQUIRED = "SCOPE_COVERAGE_BELOW_REQUIRED"
    SCOPE_COVERAGE_UNKNOWN = "SCOPE_COVERAGE_UNKNOWN"
    CONTEXT_TRUNCATED = "CONTEXT_TRUNCATED"
    EXTRACTION_ROWS_MISSING = "EXTRACTION_ROWS_MISSING"
    EXTRACTION_AMBIGUOUS = "EXTRACTION_AMBIGUOUS"
    REQUIRED_FIELD_MISSING = "REQUIRED_FIELD_MISSING"
    DENOMINATOR_INCOMPLETE = "DENOMINATOR_INCOMPLETE"
    RANKING_DOMAIN_INCOMPLETE = "RANKING_DOMAIN_INCOMPLETE"
    TIE_BOUNDARY_UNRESOLVED = "TIE_BOUNDARY_UNRESOLVED"
    TIME_BUCKET_INCOMPLETE = "TIME_BUCKET_INCOMPLETE"
    AGGREGATION_INVALID = "AGGREGATION_INVALID"
    GOLD_NOT_AVAILABLE_FOR_EVALUATION = "GOLD_NOT_AVAILABLE_FOR_EVALUATION"
    CERTIFIED = "CERTIFIED"


class CoverageDecision(str, Enum):
    CERTIFIED = "certified"
    ABSTAIN = "abstain"
    PARTIAL = "partial"
    UNCERTIFIED = "uncertified"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"


class NullPolicy(str, Enum):
    EXCLUDE = "exclude"
    INCLUDE_AS_ZERO = "include_as_zero"
    INCLUDE_AS_NULL = "include_as_null"


class TiePolicy(str, Enum):
    FIRST = "first"
    ALL = "all"
    RANDOM_STABLE = "random_stable"


class FailureCode(str, Enum):
    SCOPE_COMPILATION_FAILURE = "scope_compilation_failure"
    UNSUPPORTED_SEMANTIC_PREDICATE = "unsupported_semantic_predicate"
    RENDERING_FAILURE = "rendering_failure"
    PROVIDER_TIMEOUT = "provider_timeout"
    INVALID_STRUCTURED_OUTPUT = "invalid_structured_output"
    INVENTED_RECORD_ID = "invented_record_id"
    MISSING_RECORD_ID = "missing_record_id"
    FIELD_VALIDATION_FAILURE = "field_validation_failure"
    UNRESOLVED_AMBIGUITY = "unresolved_ambiguity"
    INCOMPLETE_MAP_COVERAGE = "incomplete_map_coverage"
    DETERMINISTIC_REDUCER_FAILURE = "deterministic_reducer_failure"
    GOLD_COMPARISON_UNAVAILABLE = "gold_comparison_unavailable"
    ARTIFACT_INTEGRITY_FAILURE = "artifact_integrity_failure"


class RepairReason(str, Enum):
    INVALID_JSON = "invalid_json"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    WRONG_PRIMITIVE_TYPE = "wrong_primitive_type"
    INVENTED_RECORD_ID = "invented_record_id"
    AMBIGUOUS_EXTRACTION = "ambiguous_extraction"
    NONE = "none"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_hash(data: Any) -> str:
    """Compute a stable SHA-256 hash of JSON-serializable data."""
    serialized = orjson.dumps(data, option=orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS)
    return hashlib.sha256(serialized).hexdigest()[:32]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# CorpusRecord
# ---------------------------------------------------------------------------


class CorpusRecord(BaseModel):
    """A single record in the Track M corpus (Amazon review metadata structure)."""

    model_config = {"frozen": True}

    record_id: str = Field(..., description="Unique record identifier")
    source: str = Field(default="track_m_synthetic", description="Data source identifier")
    source_record_id: str = Field(..., description="Original ID in the source dataset")
    world_id: str = Field(..., description="Corpus world this record belongs to")
    product_id: str = Field(..., description="Product ASIN or synthetic equivalent")
    parent_id: Optional[str] = Field(None, description="Parent ASIN for product variants")
    category: RecordCategory = Field(..., description="Top-level product category")
    title: str = Field(..., description="Product title")
    brand: str = Field(..., description="Brand name")
    rating: float = Field(..., ge=1.0, le=5.0, description="Star rating 1-5")
    helpful_votes: int = Field(default=0, ge=0, description="Number of helpful votes")
    verified_purchase: bool = Field(..., description="Whether purchase was verified")
    event_time: datetime = Field(..., description="Review timestamp")
    price: Optional[Decimal] = Field(None, description="Product price (may be missing)")
    attributes: dict[str, Any] = Field(default_factory=dict, description="Extensible attributes")
    text: str = Field(default="", description="Review text stub")
    raw_payload_hash: str = Field(..., description="SHA-256 hash of original payload")
    schema_version: str = Field(default=SCHEMA_VERSION)

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: float) -> float:
        # Round to 1 decimal place for stability
        rounded = round(v, 1)
        if not (1.0 <= rounded <= 5.0):
            raise ValueError(f"Rating {rounded} out of range [1.0, 5.0]")
        return rounded

    def content_hash(self) -> str:
        """Stable hash of this record's content (excluding record_id and world_id)."""
        data = {
            "source_record_id": self.source_record_id,
            "product_id": self.product_id,
            "category": self.category.value,
            "title": self.title,
            "brand": self.brand,
            "rating": float(self.rating),
            "event_time": self.event_time.isoformat(),
            "verified_purchase": self.verified_purchase,
        }
        return _stable_hash(data)

    def model_dump_json_bytes(self) -> bytes:
        return orjson.dumps(self.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# CorpusWorld
# ---------------------------------------------------------------------------


class CorpusWorld(BaseModel):
    """A deterministic corpus world at a given scale."""

    model_config = {"frozen": True}

    world_id: str = Field(..., description="Unique world identifier")
    dataset_id: str = Field(default="track_m", description="Originating dataset")
    seed: int = Field(..., description="RNG seed for determinism")
    scale_n: int = Field(..., gt=0, description="Number of records")
    parent_world_id: Optional[str] = Field(
        None, description="Parent world at smaller N (for nestedness)"
    )
    creation_policy: str = Field(
        default="deterministic_superset",
        description="Policy guaranteeing nestedness: records in N are first N of larger world",
    )
    record_ids_hash: str = Field(..., description="Stable hash of sorted record IDs")
    manifest_path: str = Field(..., description="Path to manifest JSON file")
    created_at: datetime = Field(default_factory=_utcnow)
    schema_version: str = Field(default=SCHEMA_VERSION)

    def content_hash(self) -> str:
        return _stable_hash(
            {
                "world_id": self.world_id,
                "seed": self.seed,
                "scale_n": self.scale_n,
                "record_ids_hash": self.record_ids_hash,
            }
        )


# ---------------------------------------------------------------------------
# ScopePredicate (Safe Declarative AST)
# ---------------------------------------------------------------------------


class EqPredicate(BaseModel):
    """field == value"""

    kind: Literal["eq"] = "eq"
    field: str
    value: Any

    def validate_field(self, allowed_fields: set[str]) -> None:
        if self.field not in allowed_fields:
            raise ValueError(f"Field '{self.field}' not in allowed fields: {allowed_fields}")


class NeqPredicate(BaseModel):
    """field != value"""

    kind: Literal["neq"] = "neq"
    field: str
    value: Any


class InPredicate(BaseModel):
    """field in [values]"""

    kind: Literal["in"] = "in"
    field: str
    values: list[Any]

    @field_validator("values")
    @classmethod
    def validate_values(cls, v: list[Any]) -> list[Any]:
        if len(v) == 0:
            raise ValueError("InPredicate.values must not be empty")
        return v


class NotInPredicate(BaseModel):
    """field not in [values]"""

    kind: Literal["not_in"] = "not_in"
    field: str
    values: list[Any]


class RangePredicate(BaseModel):
    """low <= field <= high (inclusive, half-open supported via None)"""

    kind: Literal["range"] = "range"
    field: str
    low: Optional[Any] = None
    high: Optional[Any] = None
    low_inclusive: bool = True
    high_inclusive: bool = True

    @model_validator(mode="after")
    def validate_range(self) -> "RangePredicate":
        if self.low is None and self.high is None:
            raise ValueError("RangePredicate must have at least one of low or high")
        return self


class IsNullPredicate(BaseModel):
    """field is null"""

    kind: Literal["is_null"] = "is_null"
    field: str


class IsNotNullPredicate(BaseModel):
    """field is not null"""

    kind: Literal["is_not_null"] = "is_not_null"
    field: str


class AndPredicate(BaseModel):
    """Conjunction of predicates"""

    kind: Literal["and"] = "and"
    operands: list["ScopePredicate"]

    @field_validator("operands")
    @classmethod
    def validate_operands(cls, v: list[Any]) -> list[Any]:
        if len(v) < 2:
            raise ValueError("AndPredicate requires at least 2 operands")
        return v


class OrPredicate(BaseModel):
    """Disjunction of predicates"""

    kind: Literal["or"] = "or"
    operands: list["ScopePredicate"]

    @field_validator("operands")
    @classmethod
    def validate_operands(cls, v: list[Any]) -> list[Any]:
        if len(v) < 2:
            raise ValueError("OrPredicate requires at least 2 operands")
        return v


# Discriminated union for the safe predicate AST
ScopePredicate = Union[
    EqPredicate,
    NeqPredicate,
    InPredicate,
    NotInPredicate,
    RangePredicate,
    IsNullPredicate,
    IsNotNullPredicate,
    AndPredicate,
    OrPredicate,
]

# Update forward references for recursive types
AndPredicate.model_rebuild()
OrPredicate.model_rebuild()


# ---------------------------------------------------------------------------
# FactSpec
# ---------------------------------------------------------------------------


class DerivedField(BaseModel):
    """A derived field computed before aggregation."""

    name: str
    expression_kind: Literal["identity", "year", "month", "quarter", "log1p"] = "identity"
    source_field: str


class FactSpec(BaseModel):
    """Specification for fact extraction from scoped records."""

    fields: list[str] = Field(..., description="Fields to extract from records")
    derived_fields: list[DerivedField] = Field(
        default_factory=list, description="Deterministically derived fields"
    )
    null_policy: NullPolicy = Field(
        default=NullPolicy.EXCLUDE, description="How to handle null values"
    )

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("FactSpec.fields must not be empty")
        return v


# ---------------------------------------------------------------------------
# AggregationSpec
# ---------------------------------------------------------------------------


class CountSpec(BaseModel):
    kind: Literal["count"] = "count"
    distinct: bool = False
    field: Optional[str] = None


class SumSpec(BaseModel):
    kind: Literal["sum"] = "sum"
    field: str
    null_policy: NullPolicy = NullPolicy.EXCLUDE


class MeanSpec(BaseModel):
    kind: Literal["mean"] = "mean"
    field: str
    null_policy: NullPolicy = NullPolicy.EXCLUDE
    decimal_places: int = Field(default=4, ge=0, le=10)


class ProportionSpec(BaseModel):
    kind: Literal["proportion"] = "proportion"
    numerator_predicate: ScopePredicate
    denominator_includes_null: bool = False
    decimal_places: int = Field(default=4, ge=0, le=10)


class ComparisonSpec(BaseModel):
    kind: Literal["comparison"] = "comparison"
    measure: Literal["count", "mean", "sum"] = "mean"
    field: Optional[str] = None
    group_a_predicate: ScopePredicate
    group_b_predicate: ScopePredicate
    output: Literal["difference", "ratio", "both"] = "difference"


class TopKSpec(BaseModel):
    kind: Literal["top_k"] = "top_k"
    group_by_field: str
    measure: Literal["count", "mean", "sum"] = "count"
    value_field: Optional[str] = None
    k: int = Field(..., ge=1)
    ascending: bool = False
    tie_policy: TiePolicy = TiePolicy.FIRST

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v > 100:
            raise ValueError("Top-k values larger than 100 are not supported")
        return v


class TrendSpec(BaseModel):
    kind: Literal["trend"] = "trend"
    time_field: str = "event_time"
    bucket: Literal["month", "quarter", "year"] = "month"
    measure: Literal["count", "mean", "sum"] = "count"
    value_field: Optional[str] = None
    null_policy: NullPolicy = NullPolicy.EXCLUDE


AggregationSpec = Union[
    CountSpec, SumSpec, MeanSpec, ProportionSpec, ComparisonSpec, TopKSpec, TrendSpec
]


# ---------------------------------------------------------------------------
# QuerySpec
# ---------------------------------------------------------------------------


class QuerySpec(BaseModel):
    """A complete, executable query specification."""

    query_id: str = Field(default_factory=lambda: str(uuid4()))
    family: QueryFamily
    natural_language_question: str = Field(..., min_length=10)
    scope_predicate: ScopePredicate = Field(..., description="Record filter (safe AST)")
    fact_spec: FactSpec
    aggregation_spec: AggregationSpec
    tolerance: float = Field(default=1e-6, ge=0.0, description="Tolerance for answer comparison")
    expected_evidence_requirement: str = Field(
        default="all_matching_records",
        description="Human-readable evidence requirement description",
    )
    world_id: str
    template_id: str = Field(default="manual")
    version: str = Field(default="1.0")
    created_at: datetime = Field(default_factory=_utcnow)

    # Prompt 2 additions (all optional, backward-compatible)
    dataset_snapshot_id: Optional[str] = Field(
        None, description="Snapshot ID if query was generated from an ingested dataset"
    )
    difficulty: Optional[str] = Field(
        None, description="Difficulty dimension: easy, medium, adversarial"
    )
    split: Optional[str] = Field(
        None, description="Dataset split: dev, val, test"
    )
    selectivity: Optional[str] = Field(
        None, description="Selectivity dimension: broad, narrow, selective"
    )

    def spec_hash(self) -> str:
        """Stable hash of the query specification (excluding query_id and timestamps)."""
        data = {
            "family": self.family.value,
            "scope_predicate": self.scope_predicate.model_dump(mode="json"),
            "fact_spec": self.fact_spec.model_dump(mode="json"),
            "aggregation_spec": self.aggregation_spec.model_dump(mode="json"),
            "world_id": self.world_id,
            "template_id": self.template_id,
        }
        return _stable_hash(data)


# ---------------------------------------------------------------------------
# GoldAnswer
# ---------------------------------------------------------------------------


class GoldAnswer(BaseModel):
    """Certified gold answer from dual evaluation."""

    answer_id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str
    world_id: str

    # Answer values
    answer_value: Any = Field(..., description="The primary answer value")
    answer_typed: Any = Field(None, description="Type-coerced representation")
    denominator: Optional[int] = Field(None, description="Denominator for proportions")
    numerator: Optional[int] = Field(None, description="Numerator for proportions")

    # Evidence
    eligible_record_count: int = Field(..., ge=0)
    contributing_record_ids: list[str] = Field(default_factory=list)
    evidence_hash: str = Field(..., description="Hash of contributing record IDs")

    # Dual engine results
    pandas_result: Any = Field(None)
    duckdb_result: Any = Field(None)
    agreement_status: AgreementStatus = AgreementStatus.NOT_COMPUTED
    tolerance: float = 1e-6

    # Metadata
    derivation_metadata: dict[str, Any] = Field(default_factory=dict)
    computed_at: datetime = Field(default_factory=_utcnow)
    schema_version: str = Field(default=SCHEMA_VERSION)

    # Prompt 2 additions (all optional, backward-compatible)
    eligibility_trace: Optional[dict[str, Any]] = Field(
        default=None,
        description="Trace of eligibility filtering for denominator computation"
    )
    denominator_trace: Optional[dict[str, Any]] = Field(
        default=None,
        description="Explicit denominator predicate and count for proportion/ranking queries"
    )
    tie_resolution_record: Optional[dict[str, Any]] = Field(
        default=None,
        description="Tie-resolution details for top-k and comparison queries"
    )
    disagreement_diagnostic: Optional[dict[str, Any]] = Field(
        default=None,
        description="Intermediate table hashes and engine states when Pandas/DuckDB disagree"
    )

    def answer_hash(self) -> str:
        return _stable_hash(
            {
                "query_id": self.query_id,
                "answer_value": str(self.answer_value),
                "evidence_hash": self.evidence_hash,
            }
        )


# ---------------------------------------------------------------------------
# PipelineRun
# ---------------------------------------------------------------------------


class PipelineRun(BaseModel):
    """An immutable record of a pipeline execution."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str
    pipeline_id: str
    provider_id: str = Field(default="deterministic")

    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = None

    status: RunStatus = RunStatus.PENDING
    answer: Any = Field(None, description="Pipeline's final answer")
    gold_answer_value: Any = Field(None, description="Gold answer for comparison")

    # Quality metrics
    is_correct: Optional[bool] = None
    is_within_tolerance: Optional[bool] = None
    loss: Optional[float] = None
    latency_ms: Optional[float] = None

    # Cost estimates
    token_estimate_input: int = 0
    token_estimate_output: int = 0

    # Error state
    error_message: Optional[str] = None
    error_stage: Optional[str] = None

    # Artifacts
    artifact_references: dict[str, str] = Field(
        default_factory=dict, description="Artifact name -> path mapping"
    )

    # Immutable configuration hash (computed at run creation time)
    config_hash: str = Field(default="", description="Hash of pipeline + query + world config")

    # Certification (Prompt 6)
    raw_answer: Any = Field(None, description="Pipeline's uncertified raw answer")
    policy_decision: Optional[str] = Field(None, description="CERTIFIED, ABSTAIN, PARTIAL, UNCERTIFIED")
    final_presented_answer: Any = Field(None, description="The answer actually presented to the user under policy")
    abstention_reason: Optional[str] = None
    certificate_id: Optional[str] = None
    certificate_hash: Optional[str] = None

    schema_version: str = Field(default=SCHEMA_VERSION)

    def compute_config_hash(self, query_spec: QuerySpec) -> str:
        data = {
            "pipeline_id": self.pipeline_id,
            "provider_id": self.provider_id,
            "query_spec_hash": query_spec.spec_hash(),
            "query_id": self.query_id,
        }
        return _stable_hash(data)


# ---------------------------------------------------------------------------
# P4/P5 Map Planner & Repair Models
# ---------------------------------------------------------------------------


class ExtractionUnit(BaseModel):
    """A single deterministic batch of records for extraction."""
    unit_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    record_ids: list[str] = Field(..., description="IDs to extract in this batch")
    batch_index: int
    token_estimate: int = 0
    status: RunStatus = RunStatus.PENDING
    cached: bool = False
    cache_key: Optional[str] = None


class MapPlan(BaseModel):
    """The complete extraction plan for a query."""
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    world_id: str
    total_eligible_records: int
    batch_size: int
    units: list[ExtractionUnit] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class RepairAttempt(BaseModel):
    attempt_number: int
    reason: RepairReason
    error_context: str
    prompt_hash: str
    output_hash: str
    is_successful: bool
    latency_ms: float
    token_cost: int


class RepairState(BaseModel):
    """The strict state machine tracking for a single extraction unit."""
    unit_id: str
    original_failure_reason: Optional[RepairReason] = None
    attempts: list[RepairAttempt] = Field(default_factory=list)
    final_status: Literal["success", "failed_permanent"] = "success"
    max_attempts: int = 2


# ---------------------------------------------------------------------------
# TraceEvent
# ---------------------------------------------------------------------------


class TraceEvent(BaseModel):
    """A single stage event in a pipeline trace."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    parent_event_id: Optional[str] = None

    timestamp: datetime = Field(default_factory=_utcnow)
    stage: str
    event_type: TraceEventType

    input_artifact_hash: Optional[str] = None
    output_artifact_hash: Optional[str] = None
    record_count_in: Optional[int] = None
    record_count_out: Optional[int] = None

    message: str = ""
    structured_payload: dict[str, Any] = Field(default_factory=dict)

    duration_ms: Optional[float] = None
    schema_version: str = Field(default=SCHEMA_VERSION)


# ---------------------------------------------------------------------------
# ComponentOutput
# ---------------------------------------------------------------------------


class ComponentOutput(BaseModel):
    """Output of a pipeline component stage."""

    component: Literal["retrieval", "extraction", "aggregation", "validation"]
    run_id: str
    stage_index: int

    # Retrieval / scope output
    scope_record_ids: Optional[list[str]] = None
    scope_record_count: Optional[int] = None
    scope_artifact_hash: Optional[str] = None

    # Extraction output
    extraction_rows: Optional[list[dict[str, Any]]] = None
    extraction_row_count: Optional[int] = None
    extraction_artifact_hash: Optional[str] = None

    # Aggregation output
    aggregation_plan: Optional[dict[str, Any]] = None
    aggregation_result: Optional[Any] = None
    aggregation_artifact_hash: Optional[str] = None

    # Validation
    validation_passed: Optional[bool] = None
    validation_message: Optional[str] = None
    validation_artifact_hash: Optional[str] = None

    created_at: datetime = Field(default_factory=_utcnow)


class EvidenceRequirement(BaseModel):
    """Operator-specific evidence completeness requirements."""
    requirement_type: Literal["exact", "lower_bound", "sample", "unknown"] = "exact"
    requires_full_scope: bool = True
    required_scope_count: Optional[int] = None
    required_denominator: Optional[int] = None
    required_tie_resolution: bool = False
    required_time_buckets: Optional[list[str]] = None
    
    @classmethod
    def from_query(cls, query: QuerySpec) -> "EvidenceRequirement":
        """Generate conservative requirements from the query spec."""
        req = cls()
        if isinstance(query.aggregation_spec, (CountSpec, SumSpec)):
            req.requires_full_scope = True
        elif isinstance(query.aggregation_spec, ProportionSpec):
            req.requires_full_scope = True
        elif isinstance(query.aggregation_spec, TopKSpec):
            req.requires_full_scope = True
            req.required_tie_resolution = True
        elif isinstance(query.aggregation_spec, TrendSpec):
            req.requires_full_scope = True
        return req


class CoverageObservation(BaseModel):
    """Pipeline observations measuring evidence completeness."""
    known_world_size: Optional[int] = None
    eligible_set_size_known: bool = False
    eligible_set_size: Optional[int] = None
    retrieved_units: int = 0
    unique_represented_record_ids: int = 0
    extracted_valid_rows: int = 0
    ambiguous_rows: int = 0
    failed_rows: int = 0
    missing_required_fields: int = 0
    denominator_evaluable: bool = False
    numerator_evaluable: bool = False
    time_bucket_completeness: Optional[float] = None
    ranking_candidate_completeness: Optional[float] = None
    tie_boundary_completeness: bool = False
    truncation_count: int = 0
    dropped_context_count: int = 0


class AnswerPolicyConfig(BaseModel):
    """Thresholds and configurations for answer policies."""
    policy_id: str
    version: str = "1.0"
    min_known_scope_coverage: float = 1.0
    min_extraction_completeness: float = 1.0
    max_ambiguous_tolerance: float = 0.0
    min_required_field_completeness: float = 1.0
    require_ranking_boundary_confidence: bool = True
    max_repair_failures: int = 0
    allow_partial: bool = False


class CoverageCertificate(BaseModel):
    """Immutable evidence coverage certificate for a pipeline run."""
    certificate_id: str = Field(default_factory=lambda: str(uuid4()))
    schema_version: str = Field(default=SCHEMA_VERSION)
    
    run_id: str
    query_id: str
    world_id: str
    pipeline_id: str
    config_hash: str
    
    evidence_requirement: EvidenceRequirement
    observations: CoverageObservation
    
    coverage_ratios: dict[str, float] = Field(default_factory=dict)
    unknown_dimensions: list[str] = Field(default_factory=list)
    
    decision: CoverageDecision = CoverageDecision.UNCERTIFIED
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    human_readable_explanation: str = ""
    
    policy_id: str = ""
    policy_version: str = ""
    
    artifact_lineage: dict[str, str] = Field(default_factory=dict)
    certificate_hash: str = ""

    @model_validator(mode="after")
    def compute_certificate_hash(self) -> "CoverageCertificate":
        if not self.certificate_hash:
            data = {
                "run_id": self.run_id,
                "config_hash": self.config_hash,
                "decision": self.decision.value,
                "ratios": self.coverage_ratios,
                "policy_id": self.policy_id,
            }
            object.__setattr__(self, "certificate_hash", _stable_hash(data))
        return self

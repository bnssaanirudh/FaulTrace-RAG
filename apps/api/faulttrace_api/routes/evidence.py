"""
Evidence API routes for FaultTrace-RAG.

Additive module — does not modify any existing routes.
Mounts at /api/v1/evidence.

Endpoints:
  POST /api/v1/evidence/extract      — extract from a single document
  POST /api/v1/evidence/validate     — validate citation integrity
  GET  /api/v1/evidence/graph/{id}   — return provenance graph for a dataset/run
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from faulttrace_core.evidence import (
    SupportStatus,
)
from faulttrace_core.extraction_providers import (
    BoundedRepairExtractor,
    DeterministicFixtureExtractor,
)
from faulttrace_core.knowledge_graph import ProvenanceGraph
from pydantic import BaseModel, Field

router = APIRouter(tags=["evidence"])

GRAPH_ARTIFACTS_DIR = Path("artifacts/graphs")


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    doc_id: str = Field(..., description="Source document ID")
    doc_text: str = Field(..., min_length=1, description="Document text content")
    query: str = Field(..., min_length=1, description="Query or claim to assess support for")
    dataset_id: str = Field(default="unknown")
    split: str = Field(default="test")
    max_repair_retries: int = Field(default=3, ge=1, le=10)


class ExtractResponse(BaseModel):
    doc_id: str
    dataset_id: str
    provider: str
    model_version: str | None
    support_status: str
    cited_doc_ids: list[str]
    supporting_quote: str | None
    abstention_reason: str | None
    total_facts: int
    repair_log: list[dict[str, Any]]
    schema_version: str


class ValidateCitationRequest(BaseModel):
    query_id: str
    dataset_id: str
    split: str
    cited_doc_ids: list[str] = Field(..., description="IDs claimed as evidence")
    known_doc_ids: list[str] = Field(
        ..., description="The actual document IDs in the retrieval set"
    )
    support_status: str
    answer_text: str | None = None
    abstention_reason: str | None = None


class ValidateCitationResponse(BaseModel):
    is_valid: bool
    violations: list[str]
    invented_doc_ids: list[str]
    citation_integrity_status: str  # "valid" | "violated" | "empty"


class GraphNode(BaseModel):
    node_id: str
    node_type: str
    label: str
    metadata: dict[str, Any]


class GraphEdge(BaseModel):
    edge_id: str
    edge_type: str
    source_id: str
    target_id: str
    confidence: float
    evidence_span: str | None


class GraphResponse(BaseModel):
    graph_id: str
    dataset_id: str
    description: str
    total_nodes: int
    total_edges: int
    node_types: dict[str, int]
    edge_types: dict[str, int]
    nodes: list[GraphNode]
    edges: list[GraphEdge]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/extract", response_model=ExtractResponse, summary="Extract evidence from a document")
def extract_evidence(req: ExtractRequest) -> ExtractResponse:
    """
    Run deterministic extraction on a single document and return structured evidence.

    Uses DeterministicFixtureExtractor wrapped in BoundedRepairExtractor.
    If OPENAI_API_KEY is set in the environment, uses SchemaConstrainedLLMExtractor instead.
    """
    base = DeterministicFixtureExtractor()
    extractor = BoundedRepairExtractor(base, max_retries=req.max_repair_retries)

    try:
        record = extractor.extract(
            doc_id=req.doc_id,
            doc_text=req.doc_text,
            query=req.query,
            dataset_id=req.dataset_id,
            split=req.split,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    cited_answer = record.cited_answer
    if cited_answer is None:
        support_status = SupportStatus.INSUFFICIENT_EVIDENCE.value
        cited_doc_ids: list[str] = []
        supporting_quote = None
        abstention_reason = "No cited answer produced"
    else:
        support_status = cited_answer.support_status.value
        cited_doc_ids = cited_answer.cited_doc_ids
        supporting_quote = cited_answer.supporting_quote
        abstention_reason = cited_answer.abstention_reason

    repair_log = [a.to_dict() for a in extractor.repair_log]

    return ExtractResponse(
        doc_id=record.doc_id,
        dataset_id=record.dataset_id,
        provider=record.provider,
        model_version=record.model_version,
        support_status=support_status,
        cited_doc_ids=cited_doc_ids,
        supporting_quote=supporting_quote,
        abstention_reason=abstention_reason,
        total_facts=len(record.facts),
        repair_log=repair_log,
        schema_version=record.schema_version,
    )


@router.post(
    "/validate",
    response_model=ValidateCitationResponse,
    summary="Validate citation integrity of a CitedAnswer",
)
def validate_citations(req: ValidateCitationRequest) -> ValidateCitationResponse:
    """
    Validate that all cited_doc_ids exist in the known_doc_ids set.

    Returns a list of violations with the specific problem for each invented ID.
    This endpoint is stateless — it does not require stored extraction records.
    """
    known_set = set(req.known_doc_ids)
    violations: list[str] = []
    invented_ids: list[str] = []

    for cited_id in req.cited_doc_ids:
        if cited_id not in known_set:
            invented_ids.append(cited_id)
            violations.append(
                f"INVENTED_DOCUMENT_ID: '{cited_id}' was cited but is not in the retrieval set"
            )

    if not req.cited_doc_ids:
        integrity_status = "empty"
    elif violations:
        integrity_status = "violated"
    else:
        integrity_status = "valid"

    return ValidateCitationResponse(
        is_valid=len(violations) == 0,
        violations=violations,
        invented_doc_ids=invented_ids,
        citation_integrity_status=integrity_status,
    )


@router.get(
    "/graph/{graph_id}", response_model=GraphResponse, summary="Return a saved provenance graph"
)
def get_provenance_graph(graph_id: str) -> GraphResponse:
    """
    Load and return a saved provenance graph by ID.

    Graphs are saved by the text extraction pipelines in artifacts/graphs/.
    Returns nodes, edges, and summary statistics.
    """
    graph_path = GRAPH_ARTIFACTS_DIR / f"{graph_id}.json"

    if not graph_path.exists():
        # Try subdirectories (benchmark_runs, text_runs)
        for subdir in ["benchmark_runs", "text_runs"]:
            alt = Path("artifacts") / subdir / f"graph_{graph_id}.json"
            if alt.exists():
                graph_path = alt
                break
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Provenance graph '{graph_id}' not found. "
                f"Run a text pipeline first to generate graphs.",
            )

    try:
        graph = ProvenanceGraph.load(graph_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load graph: {str(e)}")

    stats = graph.stats()

    return GraphResponse(
        graph_id=graph.graph_id,
        dataset_id=graph.dataset_id,
        description=graph.description,
        total_nodes=stats["total_nodes"],
        total_edges=stats["total_edges"],
        node_types=stats["node_types"],
        edge_types=stats["edge_types"],
        nodes=[
            GraphNode(
                node_id=n.node_id,
                node_type=n.node_type,
                label=n.label,
                metadata=n.metadata,
            )
            for n in graph.nodes.values()
        ],
        edges=[
            GraphEdge(
                edge_id=e.edge_id,
                edge_type=e.edge_type,
                source_id=e.source_id,
                target_id=e.target_id,
                confidence=e.confidence,
                evidence_span=e.evidence_span,
            )
            for e in graph.edges
        ],
    )

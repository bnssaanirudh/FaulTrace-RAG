"""
Provenance Knowledge Graph for FaultTrace-RAG.

This module implements a deterministic, file-persisted provenance graph that links:
  Dataset → Document → Chunk → ExtractedFact → Entity/Claim → CitedAnswer

IMPORTANT: This is a PROVENANCE TRACEABILITY GRAPH, not a Graph Neural Network (GNN).
There are no learned embeddings, no message passing, and no trained weights.
The graph is built deterministically from ExtractionRecords and persisted as JSON.

Node types: dataset, document, chunk, entity, claim, topic, extracted_fact
Edge types: contains, mentions, supports, refutes, related_to, semantic_similar_to
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from faulttrace_core.evidence import (
    AggregatedExtractionResult,
    ExtractionRecord,
    SupportStatus,
)

# ---------------------------------------------------------------------------
# Node and Edge Models
# ---------------------------------------------------------------------------


class KGNode(BaseModel):
    """A single node in the provenance graph."""

    node_id: str = Field(..., description="Unique node identifier")
    node_type: str = Field(
        ..., description="One of: dataset, document, chunk, entity, claim, topic, extracted_fact"
    )
    label: str = Field(..., description="Human-readable label")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class KGEdge(BaseModel):
    """A directed edge in the provenance graph."""

    edge_id: str = Field(..., description="Unique edge identifier")
    edge_type: str = Field(
        ...,
        description="One of: contains, mentions, supports, refutes, related_to, semantic_similar_to",
    )
    source_id: str = Field(..., description="Source node_id")
    target_id: str = Field(..., description="Target node_id")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_span: str | None = Field(None, description="Supporting quote for this edge")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ProvenanceGraph
# ---------------------------------------------------------------------------


class ProvenanceGraph(BaseModel):
    """
    Deterministic provenance graph persisted as JSON.

    This graph is built from ExtractionRecords and provides:
    - Fast node lookup by ID or type
    - Edge traversal (neighbors, subgraph)
    - JSON serialization/deserialization

    It is NOT a neural network. It does NOT learn. It does NOT generalize.
    It records provenance relationships that are explicitly built from
    structured extraction output.
    """

    graph_id: str = Field(..., description="Unique identifier for this graph instance")
    dataset_id: str = Field(..., description="Primary dataset this graph represents")
    description: str = Field(default="Provenance traceability graph (deterministic, not a GNN)")
    nodes: dict[str, KGNode] = Field(default_factory=dict)
    edges: list[KGEdge] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = "1.0.0"

    def add_node(self, node: KGNode) -> ProvenanceGraph:
        """Add a node (idempotent — same node_id updates in place)."""
        self.nodes[node.node_id] = node
        return self

    def add_edge(self, edge: KGEdge) -> ProvenanceGraph:
        """Add an edge. Source and target must exist as nodes."""
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node '{edge.source_id}' not in graph")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node '{edge.target_id}' not in graph")
        self.edges.append(edge)
        return self

    def get_node(self, node_id: str) -> KGNode | None:
        return self.nodes.get(node_id)

    def nodes_of_type(self, node_type: str) -> list[KGNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def edges_from(self, source_id: str) -> list[KGEdge]:
        return [e for e in self.edges if e.source_id == source_id]

    def edges_to(self, target_id: str) -> list[KGEdge]:
        return [e for e in self.edges if e.target_id == target_id]

    def neighbors(self, node_id: str, edge_type: str | None = None) -> list[KGNode]:
        """Return all nodes reachable from node_id in one hop."""
        result = []
        for edge in self.edges_from(node_id):
            if edge_type is None or edge.edge_type == edge_type:
                neighbor = self.nodes.get(edge.target_id)
                if neighbor:
                    result.append(neighbor)
        return result

    def subgraph(
        self,
        node_types: list[str] | None = None,
        edge_types: list[str] | None = None,
        root_node_id: str | None = None,
        max_hops: int = 2,
    ) -> ProvenanceGraph:
        """Return a filtered subgraph."""
        if root_node_id is not None:
            included_ids = self._bfs(root_node_id, max_hops)
        elif node_types is not None:
            included_ids = {n.node_id for n in self.nodes.values() if n.node_type in node_types}
        else:
            included_ids = set(self.nodes.keys())

        new_nodes = {nid: node for nid, node in self.nodes.items() if nid in included_ids}
        new_edges = [
            e
            for e in self.edges
            if e.source_id in included_ids
            and e.target_id in included_ids
            and (edge_types is None or e.edge_type in edge_types)
        ]

        return ProvenanceGraph(
            graph_id=f"{self.graph_id}_sub",
            dataset_id=self.dataset_id,
            description=f"Subgraph of {self.graph_id}",
            nodes=new_nodes,
            edges=new_edges,
        )

    def _bfs(self, start_id: str, max_hops: int) -> set[str]:
        """BFS to collect all node IDs within max_hops of start_id."""
        visited: set[str] = {start_id}
        frontier: set[str] = {start_id}
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for nid in frontier:
                for edge in self.edges_from(nid):
                    if edge.target_id not in visited:
                        visited.add(edge.target_id)
                        next_frontier.add(edge.target_id)
            frontier = next_frontier
        return visited

    def stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        node_type_counts: dict[str, int] = {}
        for node in self.nodes.values():
            node_type_counts[node.node_type] = node_type_counts.get(node.node_type, 0) + 1

        edge_type_counts: dict[str, int] = {}
        for edge in self.edges:
            edge_type_counts[edge.edge_type] = edge_type_counts.get(edge.edge_type, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": node_type_counts,
            "edge_types": edge_type_counts,
        }

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ProvenanceGraph:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# ProvenanceGraphBuilder
# ---------------------------------------------------------------------------


class ProvenanceGraphBuilder:
    """
    Builds a ProvenanceGraph from ExtractionRecords and dataset metadata.

    This is purely deterministic — no ML, no embeddings.
    The graph structure is derived entirely from the structured extraction output.
    """

    def __init__(self, graph_id: str, dataset_id: str):
        self.graph = ProvenanceGraph(
            graph_id=graph_id,
            dataset_id=dataset_id,
        )
        self._edge_counter = 0

    def _next_edge_id(self) -> str:
        self._edge_counter += 1
        return f"edge_{self._edge_counter:06d}"

    def add_dataset_node(
        self, dataset_id: str, label: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Add the root dataset node."""
        node_id = f"dataset:{dataset_id}"
        self.graph.add_node(
            KGNode(
                node_id=node_id,
                node_type="dataset",
                label=label,
                metadata=metadata or {},
            )
        )
        return node_id

    def add_document_node(
        self, doc_id: str, title: str, dataset_id: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Add a document node and link it to the dataset."""
        node_id = f"doc:{doc_id}"
        dataset_node_id = f"dataset:{dataset_id}"

        self.graph.add_node(
            KGNode(
                node_id=node_id,
                node_type="document",
                label=title,
                metadata={"doc_id": doc_id, **(metadata or {})},
            )
        )

        # Link to dataset if it exists
        if dataset_node_id in self.graph.nodes:
            self.graph.add_edge(
                KGEdge(
                    edge_id=self._next_edge_id(),
                    edge_type="contains",
                    source_id=dataset_node_id,
                    target_id=node_id,
                )
            )

        return node_id

    def add_chunk_node(
        self, chunk_id: str, doc_id: str, text_snippet: str, chunk_index: int
    ) -> str:
        """Add a chunk node and link it to its document."""
        node_id = f"chunk:{chunk_id}"
        doc_node_id = f"doc:{doc_id}"

        self.graph.add_node(
            KGNode(
                node_id=node_id,
                node_type="chunk",
                label=f"Chunk {chunk_index} of {doc_id}",
                metadata={
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "snippet": text_snippet[:100],
                },
            )
        )

        if doc_node_id in self.graph.nodes:
            self.graph.add_edge(
                KGEdge(
                    edge_id=self._next_edge_id(),
                    edge_type="contains",
                    source_id=doc_node_id,
                    target_id=node_id,
                )
            )

        return node_id

    def add_entity_node(
        self, entity_id: str, entity_type: str, label: str, aliases: list[str] | None = None
    ) -> str:
        """Add an entity node."""
        node_id = f"entity:{entity_id}"
        self.graph.add_node(
            KGNode(
                node_id=node_id,
                node_type="entity",
                label=label,
                metadata={"entity_type": entity_type, "aliases": aliases or []},
            )
        )
        return node_id

    def add_claim_node(self, claim_id: str, label: str, support_status: str) -> str:
        """Add a claim node (represents a CitedAnswer)."""
        node_id = f"claim:{claim_id}"
        self.graph.add_node(
            KGNode(
                node_id=node_id,
                node_type="claim",
                label=label[:200],
                metadata={"support_status": support_status},
            )
        )
        return node_id

    def ingest_extraction_record(self, record: ExtractionRecord) -> None:
        """
        Ingest a single ExtractionRecord into the graph.
        Adds document, chunk, fact, and entity nodes with appropriate edges.
        """
        doc_node_id = f"doc:{record.doc_id}"

        # Ensure document node exists
        if doc_node_id not in self.graph.nodes:
            self.graph.add_node(
                KGNode(
                    node_id=doc_node_id,
                    node_type="document",
                    label=record.doc_id,
                    metadata={"dataset_id": record.dataset_id, "split": record.split},
                )
            )
            dataset_node_id = f"dataset:{record.dataset_id}"
            if dataset_node_id in self.graph.nodes:
                self.graph.add_edge(
                    KGEdge(
                        edge_id=self._next_edge_id(),
                        edge_type="contains",
                        source_id=dataset_node_id,
                        target_id=doc_node_id,
                    )
                )

        # Add chunk node if applicable
        if record.chunk_id:
            chunk_node_id = f"chunk:{record.chunk_id}"
            if chunk_node_id not in self.graph.nodes:
                self.graph.add_node(
                    KGNode(
                        node_id=chunk_node_id,
                        node_type="chunk",
                        label=f"Chunk {record.chunk_id}",
                        metadata={"doc_id": record.doc_id},
                    )
                )
                self.graph.add_edge(
                    KGEdge(
                        edge_id=self._next_edge_id(),
                        edge_type="contains",
                        source_id=doc_node_id,
                        target_id=chunk_node_id,
                    )
                )

        # Add fact nodes
        for fact in record.facts:
            fact_node_id = f"fact:{fact.fact_id}"
            source_node_id = f"chunk:{record.chunk_id}" if record.chunk_id else doc_node_id

            if fact_node_id not in self.graph.nodes:
                self.graph.add_node(
                    KGNode(
                        node_id=fact_node_id,
                        node_type="extracted_fact",
                        label=f"{fact.fact_type.value}: {fact.normalized_value[:80]}",
                        metadata={
                            "fact_type": fact.fact_type.value,
                            "entity_type": fact.entity_type.value if fact.entity_type else None,
                            "normalized_value": fact.normalized_value,
                            "confidence": fact.confidence,
                            "validation_status": fact.validation_status.value,
                        },
                    )
                )

            self.graph.add_edge(
                KGEdge(
                    edge_id=self._next_edge_id(),
                    edge_type="mentions",
                    source_id=source_node_id,
                    target_id=fact_node_id,
                    confidence=fact.confidence,
                    evidence_span=fact.span.source_quote if fact.span else None,
                )
            )

        # Add claim node if there's a cited answer
        if record.cited_answer and record.query_id:
            claim_node_id = f"claim:{record.query_id}:{record.doc_id}"
            self.graph.add_node(
                KGNode(
                    node_id=claim_node_id,
                    node_type="claim",
                    label=(record.cited_answer.answer_text or "")[:200],
                    metadata={
                        "support_status": record.cited_answer.support_status.value,
                        "query_id": record.query_id,
                        "supporting_quote": (record.cited_answer.supporting_quote or "")[:300],
                        "abstention_reason": record.cited_answer.abstention_reason,
                    },
                )
            )

            edge_type = self._support_status_to_edge_type(record.cited_answer.support_status)
            self.graph.add_edge(
                KGEdge(
                    edge_id=self._next_edge_id(),
                    edge_type=edge_type,
                    source_id=doc_node_id,
                    target_id=claim_node_id,
                    confidence=1.0,
                    evidence_span=record.cited_answer.supporting_quote,
                )
            )

    def _support_status_to_edge_type(self, status: SupportStatus) -> str:
        mapping = {
            SupportStatus.SUPPORTED: "supports",
            SupportStatus.PARTIALLY_SUPPORTED: "supports",
            SupportStatus.UNSUPPORTED: "refutes",
            SupportStatus.CONFLICTING: "refutes",
            SupportStatus.INSUFFICIENT_EVIDENCE: "mentions",
        }
        return mapping.get(status, "related_to")

    def ingest_aggregated_result(self, result: AggregatedExtractionResult) -> None:
        """Ingest all records from a multi-document aggregated result."""
        for record in result.records:
            self.ingest_extraction_record(record)

        # Add a top-level claim node for the final answer
        if result.final_answer and result.final_answer.answer_text:
            claim_node_id = f"claim:{result.query_id}:final"
            self.graph.add_node(
                KGNode(
                    node_id=claim_node_id,
                    node_type="claim",
                    label=result.final_answer.answer_text[:200],
                    metadata={
                        "support_status": result.final_answer.support_status.value,
                        "query_id": result.query_id,
                        "cited_doc_ids": result.final_answer.cited_doc_ids,
                        "supporting_quote": (result.final_answer.supporting_quote or "")[:300],
                    },
                )
            )

            for doc_id in result.final_answer.cited_doc_ids:
                doc_node_id = f"doc:{doc_id}"
                if doc_node_id in self.graph.nodes:
                    edge_type = self._support_status_to_edge_type(
                        result.final_answer.support_status
                    )
                    self.graph.add_edge(
                        KGEdge(
                            edge_id=self._next_edge_id(),
                            edge_type=edge_type,
                            source_id=doc_node_id,
                            target_id=claim_node_id,
                            confidence=1.0,
                            evidence_span=result.final_answer.supporting_quote,
                        )
                    )

    def build(self) -> ProvenanceGraph:
        return self.graph

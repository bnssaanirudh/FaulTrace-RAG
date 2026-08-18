from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from faulttrace_core.models import (
    ComponentOutput,
    GoldAnswer,
    QuerySpec,
    TraceEvent,
    TraceEventType,
)
from faulttrace_gold.pandas_engine import PandasEvaluator

from faulttrace_pipelines.base import AbstractPipeline

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    torch = None
    Data = None


class GNNExtractorPipeline(AbstractPipeline):
    """
    [DEMO — No Trained Weights]

    This pipeline is a DEMONSTRATION of how a GNN-based extractor could be
    integrated into FaultTrace-RAG. It does NOT perform real graph learning:
    - Node features are random tensors (torch.randn)
    - Edges are sequential (i → i+1), not semantic
    - No message passing, no trained model, no inference

    For actual provenance graph construction, see:
        faulttrace_core.knowledge_graph.ProvenanceGraphBuilder

    The torch_geometric integration is included as a structural placeholder
    to show how the PyG Data object maps to a real GNN pipeline.
    If torch_geometric is not installed, it falls back to a mock dict.
    """

    pipeline_id = "gnn-extractor"
    provider_id = "pytorch-geometric"

    def __init__(self, artifacts_dir: Path = Path("artifacts/runs")):
        super().__init__(artifacts_dir)
        self.evaluator = PandasEvaluator()

    def _build_knowledge_graph(self, df: pd.DataFrame) -> Any:
        """
        Parses Springer ToC dataframe into a PyTorch Geometric Knowledge Graph.
        Returns a mock Data object if torch is not installed or just for demonstration.
        """
        if torch is None or Data is None:
            # Fallback/Mock behavior if environment lacks torch_geometric
            return {"nodes": len(df), "edges": len(df) * 2, "type": "mock_kg"}

        # Simplified node construction for demonstration
        # In a real GNN, this would map text embeddings to node features
        # and parse structural relations (e.g. ToC hierarchies) to edge indices
        num_nodes = len(df)
        x = torch.randn((num_nodes, 16))  # Mock 16-dim embeddings

        # Mock edge_index connecting consecutive nodes
        src = torch.arange(0, num_nodes - 1)
        dst = torch.arange(1, num_nodes)
        edge_index = torch.stack([src, dst], dim=0)

        data = Data(x=x, edge_index=edge_index)
        return data

    def _execute(
        self,
        run_id: str,
        query: QuerySpec,
        df: pd.DataFrame,
        parquet_path: Path | None,
        gold_answer: GoldAnswer | None,
    ) -> tuple[Any, list[TraceEvent], list[ComponentOutput], int, int]:

        trace_events = []
        component_outputs = []

        t0 = time.perf_counter()

        # 1. Ingestion Phase: Build GNN Knowledge Graph
        kg = self._build_knowledge_graph(df)

        trace_events.append(
            TraceEvent(
                run_id=run_id,
                event_type=TraceEventType.FACT_EXTRACT,
                stage="gnn_graph_construction",
                timestamp=datetime.now(UTC),
                structured_payload={"nodes": len(df), "kg_type": type(kg).__name__},
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        # Mocking the semantic extraction step that uses the GNN
        # In reality, this would run GNN inference over the graph
        extracted_rows = df.to_dict(orient="records")

        # Log extraction phase
        component_outputs.append(
            ComponentOutput(
                component="extraction",
                run_id=run_id,
                stage_index=1,
                extraction_rows=extracted_rows,
                extraction_row_count=len(extracted_rows),
            )
        )

        # 2. Aggregation Phase: Deterministic Evaluator
        t1 = time.perf_counter()
        try:
            # Convert extracted rows back to DF for aggregation
            extraction_df = pd.DataFrame(extracted_rows)
            # Evaluate using PandasEvaluator
            final_answer = self.evaluator.evaluate(query, extraction_df)
        except Exception as e:
            final_answer = None
            trace_events.append(
                TraceEvent(
                    run_id=run_id,
                    event_type=TraceEventType.ERROR,
                    stage="deterministic_aggregation",
                    timestamp=datetime.now(UTC),
                    message=str(e),
                    duration_ms=(time.perf_counter() - t1) * 1000,
                )
            )

        component_outputs.append(
            ComponentOutput(
                component="aggregation",
                run_id=run_id,
                stage_index=2,
                aggregation_result=final_answer,
            )
        )

        token_in = 0
        token_out = 0

        return final_answer, trace_events, component_outputs, token_in, token_out

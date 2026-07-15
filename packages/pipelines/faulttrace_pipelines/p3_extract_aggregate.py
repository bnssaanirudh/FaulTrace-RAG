from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import pandas as pd

from faulttrace_core.models import (
    ComponentOutput,
    GoldAnswer,
    QuerySpec,
    TraceEvent,
    TraceEventType,
)
from faulttrace_core.retrieval import DocumentRenderer
from faulttrace_core.llm import ProviderConfig, get_provider
from faulttrace_core.extraction import QueryExtractionResult
from faulttrace_core.models import FactSpec
from faulttrace_pipelines.base import AbstractPipeline
from faulttrace_pipelines.retrieval.bm25_retriever import BM25Retriever
from faulttrace_pipelines.retrieval.context_builder import ContextBuilder
from faulttrace_pipelines.prompts import build_extract_prompt

from faulttrace_gold.pandas_engine import PandasEvaluator

class P3ExtractAggregatePipeline(AbstractPipeline):
    """
    P3: Schema Extract + Deterministic Aggregate.
    Retrieves context, uses LLM to extract JSON row facts, 
    and then aggregates using deterministic Pandas reducer.
    """
    
    pipeline_id = "P3-extract-aggregate"

    def __init__(
        self, 
        provider_id: str = "deterministic",
        model_id: str = "deterministic-valid",
        top_k: int = 50,
        artifacts_dir: Path = Path("artifacts/runs")
    ):
        super().__init__(artifacts_dir)
        self.provider_id = provider_id
        self.model_id = model_id
        self.top_k = top_k
        self.provider_cls = get_provider(self.provider_id)
        self.provider = self.provider_cls()
        self.renderer = DocumentRenderer(chunk_size=None)
        self.context_builder = ContextBuilder(max_tokens=6000)
        self.evaluator = PandasEvaluator()

    def _execute(
        self,
        run_id: str,
        query: QuerySpec,
        df: pd.DataFrame,
        parquet_path: Optional[Path],
        gold_answer: Optional[GoldAnswer],
    ) -> tuple[Any, list[TraceEvent], list[ComponentOutput], int, int]:
        
        trace_events = []
        component_outputs = []
        
        # 1. Retrieval Phase
        t0 = time.perf_counter()
        data_dir = parquet_path.parent.parent.parent if parquet_path else Path("data")
        retriever = BM25Retriever(
            world_id=query.world_id,
            data_dir=data_dir,
            renderer=self.renderer
        )
        
        results = retriever.retrieve(query.natural_language, top_k=self.top_k)
        candidates = [res["unit"] for res in results]
        
        trace_events.append(TraceEvent(
            run_id=run_id,
            event_type=TraceEventType.SCOPE_NARROWED,
            timestamp=datetime.now(timezone.utc),
            details={"candidates_found": len(candidates), "top_k": self.top_k},
            latency_ms=(time.perf_counter() - t0) * 1000
        ))
        
        # 2. Context Building Phase
        t1 = time.perf_counter()
        context_str, manifest = self.context_builder.build_context(candidates)
        
        trace_events.append(TraceEvent(
            run_id=run_id,
            event_type=TraceEventType.COMPLETION_GENERATED,
            timestamp=datetime.now(timezone.utc),
            details=manifest.model_dump(),
            latency_ms=(time.perf_counter() - t1) * 1000
        ))
        
        # 3. Extraction Phase
        t2 = time.perf_counter()
        prompt = build_extract_prompt(query.natural_language, context_str)
        
        # Build strict JSON schema dynamically (we will just reuse QueryExtractionResult for simplicity)
        schema_dict = QueryExtractionResult.get_json_schema()
        
        config = ProviderConfig(
            model_id=self.model_id,
            temperature=0.0,
            max_tokens=2048,
            structured_schema=schema_dict,
            structured_schema_name="QueryExtractionResult"
        )
        
        output = self.provider.generate(prompt, config)
        
        trace_events.append(TraceEvent(
            run_id=run_id,
            event_type=TraceEventType.COMPLETION_GENERATED,
            timestamp=datetime.now(timezone.utc),
            details={
                "model_id": self.model_id,
                "provider": self.provider_id,
                "prompt_tokens": output.prompt_tokens,
                "completion_tokens": output.completion_tokens,
                "rejected_reason": output.rejected_reason
            },
            latency_ms=(time.perf_counter() - t2) * 1000
        ))

        # 4. Validation and Aggregation Phase
        t3 = time.perf_counter()
        
        extracted_df = pd.DataFrame()
        
        if output.is_successful_structured() and output.parsed_json:
            # We got valid JSON. Let's parse it and convert it to a DataFrame for aggregation.
            records = []
            extracted_data = output.parsed_json.get("record_extractions", [])
            for rec in extracted_data:
                if rec.get("is_relevant"):
                    row = {"record_id": rec.get("record_id")}
                    for field in rec.get("extracted_fields", []):
                        if not field.get("is_missing"):
                            row[field.get("field_name")] = field.get("value")
                    records.append(row)
            
            if records:
                extracted_df = pd.DataFrame(records)
                
                # We need to map string types to proper types if aggregation requires it.
                # If doing a count, len(extracted_df) is enough.
                # In P0, we relied on query.metric. We will do the same:
                if query.metric == "count":
                    answer_value = len(extracted_df)
                else:
                    # Let PandasEvaluator aggregate it
                    # But PandasEvaluator needs exactly the types it expects.
                    # Since we extracted strings, let's just attempt standard aggregation.
                    answer_value = self.evaluator._aggregate(extracted_df, query.metric, query.target_field)
            else:
                answer_value = 0 if query.metric == "count" else None
        else:
            answer_value = None
            
        trace_events.append(TraceEvent(
            run_id=run_id,
            event_type=TraceEventType.AGGREGATION_COMPUTED,
            timestamp=datetime.now(timezone.utc),
            details={"rows_extracted": len(extracted_df), "answer": answer_value},
            latency_ms=(time.perf_counter() - t3) * 1000
        ))
        
        return answer_value, trace_events, component_outputs, output.prompt_tokens, output.completion_tokens

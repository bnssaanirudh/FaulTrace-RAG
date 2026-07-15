from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import json

from faulttrace_core.models import (
    ComponentOutput,
    GoldAnswer,
    QuerySpec,
    TraceEvent,
    TraceEventType,
)
from faulttrace_core.retrieval import DocumentRenderer
from faulttrace_core.llm import ProviderConfig, get_provider
from faulttrace_pipelines.base import AbstractPipeline
from faulttrace_pipelines.retrieval.dense_retriever import DenseRetriever, MockEmbeddingProvider, SentenceTransformerProvider
from faulttrace_pipelines.retrieval.context_builder import ContextBuilder
from faulttrace_pipelines.prompts import build_direct_prompt

class P2DirectDensePipeline(AbstractPipeline):
    """
    P2: Direct Answer from Dense context.
    Retrieves top-k records using FAISS Dense embeddings and answers via LLM.
    """
    
    pipeline_id = "P2-direct-dense"

    def __init__(
        self, 
        provider_id: str = "deterministic",
        model_id: str = "deterministic-valid",
        embedding_mode: str = "mock",
        top_k: int = 20,
        artifacts_dir: Path = Path("artifacts/runs")
    ):
        super().__init__(artifacts_dir)
        self.provider_id = provider_id
        self.model_id = model_id
        self.top_k = top_k
        self.embedding_mode = embedding_mode
        self.provider_cls = get_provider(self.provider_id)
        self.provider = self.provider_cls()
        self.renderer = DocumentRenderer(chunk_size=None)
        self.context_builder = ContextBuilder(max_tokens=4000)
        
        if self.embedding_mode == "mock":
            self.embedding_provider = MockEmbeddingProvider()
        else:
            self.embedding_provider = SentenceTransformerProvider()

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
        retriever = DenseRetriever(
            world_id=query.world_id,
            data_dir=data_dir,
            renderer=self.renderer,
            provider=self.embedding_provider
        )
        
        results = retriever.retrieve(query.natural_language, top_k=self.top_k)
        
        candidates = [res["unit"] for res in results]
        retrieval_ms = (time.perf_counter() - t0) * 1000
        
        trace_events.append(TraceEvent(
            run_id=run_id,
            event_type=TraceEventType.SCOPE_NARROWED,
            timestamp=datetime.now(timezone.utc),
            details={"candidates_found": len(candidates), "top_k": self.top_k},
            latency_ms=retrieval_ms
        ))
        
        # 2. Context Building Phase
        t1 = time.perf_counter()
        context_str, manifest = self.context_builder.build_context(candidates)
        context_ms = (time.perf_counter() - t1) * 1000
        
        trace_events.append(TraceEvent(
            run_id=run_id,
            event_type=TraceEventType.COMPLETION_GENERATED,
            timestamp=datetime.now(timezone.utc),
            details=manifest.model_dump(),
            latency_ms=context_ms
        ))
        
        # 3. Generation Phase
        t2 = time.perf_counter()
        prompt = build_direct_prompt(query.natural_language, context_str)
        
        config = ProviderConfig(
            model_id=self.model_id,
            temperature=0.0,
            max_tokens=100
        )
        
        output = self.provider.generate(prompt, config)
        gen_ms = (time.perf_counter() - t2) * 1000
        
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
            latency_ms=gen_ms
        ))
        
        answer_value: Any = output.raw_response.strip()
        try:
            if answer_value.isdigit():
                answer_value = int(answer_value)
            else:
                answer_value = float(answer_value)
        except ValueError:
            pass
            
        return answer_value, trace_events, component_outputs, output.prompt_tokens, output.completion_tokens

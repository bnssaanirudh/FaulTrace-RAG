import time
import pandas as pd
from typing import Any, Optional, Dict, List
from pathlib import Path
import json

from faulttrace_core.models import (
    ComponentOutput, GoldAnswer, QuerySpec, TraceEventType, TraceEvent
)
from faulttrace_gold.pandas_engine import PandasEvaluator
from faulttrace_pipelines.base import AbstractPipeline
from faulttrace_pipelines.scope_service import ScopeService
from faulttrace_pipelines.planner import MapPlanner
from faulttrace_pipelines.schema_generator import SchemaGenerator
from faulttrace_pipelines.cache import ExtractionCache
from faulttrace_core.llm import get_provider, ProviderConfig

PIPELINE_ID = "P4-full-scope-mer"
PROVIDER_ID = "deterministic" # overrideable

class P4FullScopeMERPipeline(AbstractPipeline):
    """
    P4 — Full-Scope Map-Extract-Reduce
    Implements auditable enumeration, chunked map plan, JSON extraction, and deterministic reduce.
    """
    
    pipeline_id = PIPELINE_ID
    provider_id = PROVIDER_ID
    
    def __init__(self, artifacts_dir: Path = Path("artifacts/runs")):
        super().__init__(artifacts_dir)
        self.cache = ExtractionCache()
        self.evaluator = PandasEvaluator()
        
    def _execute(
        self,
        run_id: str,
        query: QuerySpec,
        df: pd.DataFrame,
        parquet_path: Optional[Path],
        gold_answer: Optional[GoldAnswer],
    ) -> tuple[Any, list[TraceEvent], list[ComponentOutput], int, int]:
        
        events: list[TraceEvent] = []
        components: list[ComponentOutput] = []
        token_in = 0
        token_out = 0
        
        provider_cls = get_provider(self.provider_id)
        provider = provider_cls()
        
        # ── Stage 1: Enumerate Scope
        t1 = time.perf_counter()
        scope_res = ScopeService.enumerate_scope(query, df)
        scope_duration = (time.perf_counter() - t1) * 1000
        
        events.append(self._make_event(
            run_id, "scope_enumerate", TraceEventType.SCOPE_ENUMERATE,
            scope_res.query_specific_explanation,
            record_count_in=len(df),
            record_count_out=scope_res.eligible_count,
            duration_ms=scope_duration,
            payload={
                "evidence_set_hash": scope_res.evidence_set_hash,
                "is_success": scope_res.is_success
            }
        ))
        
        if not scope_res.is_success:
            raise Exception(f"Scope compilation failed: {scope_res.failure_code}")
            
        # ── Stage 2: Map Planner
        t2 = time.perf_counter()
        # Ensure batch size configuration is readable. Using default of 10.
        plan = MapPlanner.create_plan(run_id, query.world_id, scope_res, batch_size=10)
        
        # Generate schema
        schema = SchemaGenerator.generate_extraction_schema(query.fact_spec, query.aggregation_spec, scope_res.eligible_record_ids)
        schema_hash = json.dumps(schema, sort_keys=True)
        schema_duration = (time.perf_counter() - t2) * 1000
        
        # ── Stage 3: Map Execute (Extraction)
        t3 = time.perf_counter()
        extracted_rows: list[dict[str, Any]] = []
        missing_record_ids: set[str] = set()
        
        for unit in plan.units:
            # Build records hash
            records_hash = "".join(unit.record_ids) # simplified hash for now
            # Prompt hash
            prompt_hash = "p4_prompt_v1"
            
            cache_key = self.cache.generate_key(
                self.provider_id, "default", prompt_hash, records_hash, schema_hash, 0
            )
            unit.cache_key = cache_key
            
            cached = self.cache.get(cache_key)
            if cached and cached.is_success:
                unit.cached = True
                parsed = cached.parsed_json or {}
                extracted_rows.extend(parsed.get("extracted_records", []))
                token_in += cached.prompt_tokens
                token_out += cached.completion_tokens
            else:
                unit.cached = False
                
                # Render records to JSON
                batch_df = df[df["record_id"].isin(unit.record_ids)]
                batch_json = batch_df.to_json(orient="records")
                prompt = f"Extract facts for the following records:\n{batch_json}\nReturn JSON."
                
                config = ProviderConfig(
                    model_id="default",
                    structured_schema=schema,
                    structured_schema_name="ExtractionOutput"
                )
                
                output = provider.generate(prompt, config)
                self.cache.store_output(cache_key, output, "1.0.0", 0.0)
                
                token_in += output.prompt_tokens
                token_out += output.completion_tokens
                
                if output.parsed_json:
                    extracted_rows.extend(output.parsed_json.get("extracted_records", []))
                else:
                    # Capture missing or failed
                    missing_record_ids.update(unit.record_ids)
                    
        extract_duration = (time.perf_counter() - t3) * 1000
        
        # Log event
        events.append(self._make_event(
            run_id, "fact_extract", TraceEventType.FACT_EXTRACT,
            f"Extracted {len(extracted_rows)} records from {len(plan.units)} batches.",
            record_count_in=scope_res.eligible_count,
            record_count_out=len(extracted_rows),
            duration_ms=extract_duration,
            payload={
                "cached_units": sum(1 for u in plan.units if u.cached),
                "total_units": len(plan.units),
                "missing_records": list(missing_record_ids)
            }
        ))
        
        # Save extraction artifact
        extract_path = self.artifacts_dir / run_id / "extraction.parquet"
        extract_path.parent.mkdir(parents=True, exist_ok=True)
        if extracted_rows:
            pd.DataFrame(extracted_rows).to_parquet(extract_path, index=False)
            
        # ── Stage 4: Reduce
        t4 = time.perf_counter()
        
        # Re-build DataFrame from extracted rows
        if not extracted_rows:
            answer_value = None
        else:
            ext_df = pd.DataFrame(extracted_rows)
            # Remove any rows marked missing_evidence or ambiguous
            if "scope_decision" in ext_df.columns:
                ext_df = ext_df[ext_df["scope_decision"] == "in_scope"]
            
            # Create a query copy without scope predicate since it's already scoped
            eval_query = query.model_copy(deep=True)
            from faulttrace_core.models import IsNotNullPredicate
            eval_query.scope_predicate = IsNotNullPredicate(field="record_id")
            
            # The evaluator needs the query to process
            answer_value = self.evaluator.evaluate(eval_query, ext_df).get("result")
            
        agg_duration = (time.perf_counter() - t4) * 1000
        events.append(self._make_event(
            run_id, "aggregate", TraceEventType.AGGREGATE,
            f"Reduced to answer: {answer_value}",
            duration_ms=agg_duration
        ))
        
        # ── Stage 5: Validate
        if gold_answer:
            events.append(self._make_event(
                run_id, "validate", TraceEventType.VALIDATE,
                f"Gold: {gold_answer.answer_value} | Model: {answer_value}"
            ))

        return answer_value, events, components, token_in, token_out


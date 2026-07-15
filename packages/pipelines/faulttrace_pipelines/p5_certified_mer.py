import time
import pandas as pd
from typing import Any, Optional, Dict, List
from pathlib import Path
import json

from faulttrace_core.models import (
    ComponentOutput, GoldAnswer, QuerySpec, TraceEventType, TraceEvent, RepairReason
)
from faulttrace_core.llm import get_provider, ProviderConfig
from faulttrace_pipelines.p4_full_scope_mer import P4FullScopeMERPipeline
from faulttrace_pipelines.scope_service import ScopeService
from faulttrace_pipelines.planner import MapPlanner
from faulttrace_pipelines.schema_generator import SchemaGenerator

PIPELINE_ID = "P5-certified-mer-repair"
PROVIDER_ID = "deterministic"

class P5CertifiedMERPipeline(P4FullScopeMERPipeline):
    """
    P5 — Certified Map-Extract-Reduce with Bounded Repair
    Executes P4 extraction, and performs strictly bounded schema-targeted repair
    for failed or ambiguous extraction units.
    """
    
    pipeline_id = PIPELINE_ID
    provider_id = PROVIDER_ID
    
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
        
        t1 = time.perf_counter()
        scope_res = ScopeService.enumerate_scope(query, df)
        events.append(self._make_event(run_id, "scope_enumerate", TraceEventType.SCOPE_ENUMERATE, scope_res.query_specific_explanation, duration_ms=(time.perf_counter()-t1)*1000))
        if not scope_res.is_success:
            raise Exception("Scope compilation failed")
            
        plan = MapPlanner.create_plan(run_id, query.world_id, scope_res, batch_size=10)
        schema = SchemaGenerator.generate_extraction_schema(query.fact_spec, query.aggregation_spec, scope_res.eligible_record_ids)
        schema_hash = json.dumps(schema, sort_keys=True)
        
        t3 = time.perf_counter()
        extracted_rows: list[dict[str, Any]] = []
        repaired_count = 0
        
        for unit in plan.units:
            records_hash = "".join(unit.record_ids)
            prompt_hash = "p5_prompt_v1"
            
            # Helper to execute a single attempt
            def run_attempt(attempt_num: int, reason: RepairReason = RepairReason.NONE, error_ctx: str = "") -> tuple[bool, Optional[dict]]:
                nonlocal token_in, token_out
                cache_key = self.cache.generate_key(self.provider_id, "default", prompt_hash, records_hash, schema_hash, attempt_num)
                cached = self.cache.get(cache_key)
                if cached and cached.is_success:
                    token_in += cached.prompt_tokens
                    token_out += cached.completion_tokens
                    return True, cached.parsed_json
                
                batch_df = df[df["record_id"].isin(unit.record_ids)]
                batch_json = batch_df.to_json(orient="records")
                
                prompt = f"Extract facts for the following records:\n{batch_json}\nReturn JSON."
                if attempt_num > 0:
                    prompt += f"\nPrevious attempt failed due to {reason.value}: {error_ctx}. Fix it without inventing records."
                
                config = ProviderConfig(model_id="default", structured_schema=schema, structured_schema_name="ExtractionOutput")
                output = provider.generate(prompt, config)
                self.cache.store_output(cache_key, output, "1.0.0", 0.0)
                
                token_in += output.prompt_tokens
                token_out += output.completion_tokens
                return output.parsed_json is not None, output.parsed_json

            # Attempt 0
            success, parsed = run_attempt(0)
            if success and parsed:
                extracted_rows.extend(parsed.get("extracted_records", []))
            else:
                # Attempt 1 (Repair)
                success, parsed = run_attempt(1, RepairReason.INVALID_JSON, "Failed to parse valid extraction array")
                repaired_count += 1
                if success and parsed:
                    extracted_rows.extend(parsed.get("extracted_records", []))
                else:
                    # Attempt 2 (Hard cap)
                    success, parsed = run_attempt(2, RepairReason.INVALID_JSON, "Second failure. Hard cap repair.")
                    repaired_count += 1
                    if success and parsed:
                        extracted_rows.extend(parsed.get("extracted_records", []))
                        
        extract_duration = (time.perf_counter() - t3) * 1000
        events.append(self._make_event(
            run_id, "fact_extract", TraceEventType.FACT_EXTRACT,
            f"Extracted {len(extracted_rows)} records with {repaired_count} repair attempts.",
            duration_ms=extract_duration
        ))
        
        # Aggregate
        answer_value = None
        if extracted_rows:
            ext_df = pd.DataFrame(extracted_rows)
            if "scope_decision" in ext_df.columns:
                ext_df = ext_df[ext_df["scope_decision"] == "in_scope"]
            
            # Create a query copy without scope predicate since it's already scoped
            eval_query = query.model_copy(deep=True)
            from faulttrace_core.models import IsNotNullPredicate
            eval_query.scope_predicate = IsNotNullPredicate(field="record_id")
                
            answer_value = self.evaluator.evaluate(eval_query, ext_df).get("result")
            
        events.append(self._make_event(run_id, "aggregate", TraceEventType.AGGREGATE, f"Answer: {answer_value}"))
        
        return answer_value, events, components, token_in, token_out

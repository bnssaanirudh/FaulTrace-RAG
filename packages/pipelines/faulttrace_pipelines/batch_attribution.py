"""
Batch Attribution runner for FaultTrace-RAG.

Executes CounterfactualAttributor over batches of runs and exports results
to Parquet and CSV.
"""

from typing import Any, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
import json

from faulttrace_core.models import PipelineRun, QuerySpec, GoldAnswer
from faulttrace_pipelines.attribution import CounterfactualAttributor


class BatchAttributionRunner:
    """Executes attribution over a batch of runs."""
    
    def __init__(self, artifacts_dir: Path = Path("artifacts/batch_diagnostics")):
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.attributor = CounterfactualAttributor()

    def run_batch(
        self,
        batch_id: str,
        runs_to_process: list[tuple[PipelineRun, QuerySpec, GoldAnswer, pd.DataFrame]]
    ) -> dict[str, Any]:
        """
        Executes counterfactual attribution over a set of runs.
        Returns a summary dictionary and saves detailed Parquet and CSV files.
        """
        results = []
        
        success_count = 0
        failure_count = 0
        
        for parent_run, query, gold_answer, corpus_df in runs_to_process:
            try:
                attr_res = self.attributor.attribute(
                    parent_run=parent_run,
                    query=query,
                    gold_answer_obj=gold_answer,
                    oracle_df=corpus_df
                )
                
                phi_dict = {c.component: c.shapley_value for c in attr_res.components}
                
                results.append({
                    "run_id": attr_res.run_id,
                    "query_id": attr_res.query_id,
                    "pipeline_id": attr_res.pipeline_id,
                    "is_correct": attr_res.is_correct,
                    "total_error": attr_res.total_error,
                    "interaction_term": attr_res.interaction_term,
                    "dominant_fault": attr_res.dominant_fault,
                    "dominant_fault_confidence": attr_res.dominant_fault_confidence,
                    "phi_scope": phi_dict.get("scope", 0.0),
                    "phi_facts": phi_dict.get("facts", 0.0),
                    "phi_aggregation": phi_dict.get("aggregation", 0.0),
                })
                success_count += 1
            except Exception as e:
                # Store failed/incomplete lattice
                results.append({
                    "run_id": parent_run.run_id,
                    "query_id": str(query.query_id),
                    "pipeline_id": parent_run.pipeline_id,
                    "is_correct": False,
                    "total_error": 1.0,
                    "interaction_term": 0.0,
                    "dominant_fault": "lattice_failure",
                    "dominant_fault_confidence": 1.0,
                    "phi_scope": 0.0,
                    "phi_facts": 0.0,
                    "phi_aggregation": 0.0,
                    "error_detail": str(e)
                })
                failure_count += 1

        df_results = pd.DataFrame(results)
        
        # Export Parquet
        parquet_path = self.artifacts_dir / f"{batch_id}.parquet"
        if not df_results.empty:
            df_results.to_parquet(parquet_path, index=False)
            
        # Export CSV
        csv_path = self.artifacts_dir / f"{batch_id}.csv"
        if not df_results.empty:
            df_results.to_csv(csv_path, index=False)
            
        # Generate Summary
        if not df_results.empty:
            valid_df = df_results[df_results["dominant_fault"] != "lattice_failure"]
            summary = {
                "batch_id": batch_id,
                "total_runs": len(results),
                "successful_lattices": success_count,
                "failed_lattices": failure_count,
                "mean_error": float(valid_df["total_error"].mean()) if not valid_df.empty else 0.0,
                "dominant_fault_counts": valid_df["dominant_fault"].value_counts().to_dict() if not valid_df.empty else {},
                "mean_phi_scope": float(valid_df["phi_scope"].mean()) if not valid_df.empty else 0.0,
                "mean_phi_facts": float(valid_df["phi_facts"].mean()) if not valid_df.empty else 0.0,
                "mean_phi_aggregation": float(valid_df["phi_aggregation"].mean()) if not valid_df.empty else 0.0,
                "parquet_export": str(parquet_path),
                "csv_export": str(csv_path),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            summary = {
                "batch_id": batch_id,
                "total_runs": 0,
                "successful_lattices": 0,
                "failed_lattices": 0
            }
            
        summary_path = self.artifacts_dir / f"{batch_id}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        
        return summary

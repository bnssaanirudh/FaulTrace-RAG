"""
Gold answer validator: runs both Pandas and DuckDB engines and certifies agreement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from faulttrace_core.models import (
    AgreementStatus,
    GoldAnswer,
    QuerySpec,
    TopKSpec,
    TrendSpec,
)
from faulttrace_gold.pandas_engine import PandasEvaluator
from faulttrace_gold.duckdb_engine import DuckDBEvaluator

pandas_eval = PandasEvaluator()
duckdb_eval = DuckDBEvaluator()


@dataclass
class GoldAgreementResult:
    """Result of agreement check between Pandas and DuckDB engines."""
    query_id: str
    agreed: bool
    pandas_result: Any
    duckdb_result: Any
    tolerance: float
    agreement_status: AgreementStatus
    diagnostic: str = ""
    gold_answer: Optional[GoldAnswer] = None


def _results_agree(
    a: Any,
    b: Any,
    tolerance: float = 1e-6,
) -> bool:
    """Check if two results agree within tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tolerance
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_results_agree(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_results_agree(a[k], b[k], tolerance) for k in a)
    # String/categorical comparison
    return str(a) == str(b)


class GoldValidator:
    """Validates query answers using dual Pandas + DuckDB engines."""

    def validate(
        self,
        query: QuerySpec,
        df: pd.DataFrame,
        parquet_path: Optional[Path] = None,
    ) -> GoldAgreementResult:
        """
        Run both evaluators and check agreement.
        
        If parquet_path is provided, DuckDB reads from Parquet.
        Otherwise, DuckDB uses the DataFrame directly.
        """
        # Run Pandas evaluator
        try:
            pd_result = pandas_eval.evaluate(query, df)
            pd_value = pd_result["result"]
            pd_ids = pd_result["contributing_ids"]
            pd_eligible = pd_result["eligible_count"]
        except Exception as e:
            pd_value = None
            pd_ids = []
            pd_eligible = 0
            pd_error = str(e)
        else:
            pd_error = None

        # Run DuckDB evaluator
        try:
            if parquet_path and parquet_path.exists():
                dk_result = duckdb_eval.evaluate(query, parquet_path)
            else:
                dk_result = duckdb_eval.evaluate_from_df(query, df)
            dk_value = dk_result["result"]
            dk_ids = dk_result["contributing_ids"]
            dk_eligible = dk_result["eligible_count"]
        except Exception as e:
            dk_value = None
            dk_ids = []
            dk_eligible = 0
            dk_error = str(e)
        else:
            dk_error = None

        # Check agreement
        agreed = _results_agree(pd_value, dk_value, query.tolerance)

        if pd_error or dk_error:
            status = AgreementStatus.SINGLE_ENGINE
            diagnostic = f"pandas_error={pd_error}, duckdb_error={dk_error}"
            agreed = False
        elif agreed:
            status = AgreementStatus.AGREED
            diagnostic = ""
        else:
            status = AgreementStatus.DISAGREED
            diagnostic = (
                f"pandas={pd_value!r} duckdb={dk_value!r} tolerance={query.tolerance}"
            )

        # Build GoldAnswer if agreed
        gold = None
        if agreed and status == AgreementStatus.AGREED:
            ids_for_gold = pd_ids or dk_ids
            evidence_hash = hashlib.sha256(
                "|".join(sorted(ids_for_gold)).encode()
            ).hexdigest()[:32]

            # Extract numerator/denominator for proportion queries
            from faulttrace_core.models import ProportionSpec
            numerator = None
            denominator = None
            if isinstance(query.aggregation_spec, ProportionSpec):
                numerator = pd_result.get("metadata", {}).get("numerator")
                denominator = pd_result.get("metadata", {}).get("denominator")

            gold = GoldAnswer(
                query_id=query.query_id,
                world_id=query.world_id,
                answer_value=pd_value,
                answer_typed=pd_value,
                denominator=denominator,
                numerator=numerator,
                eligible_record_count=pd_eligible,
                contributing_record_ids=ids_for_gold[:500],  # cap for storage
                evidence_hash=evidence_hash,
                pandas_result=pd_value,
                duckdb_result=dk_value,
                agreement_status=status,
                tolerance=query.tolerance,
                derivation_metadata={
                    "pandas_metadata": pd_result.get("metadata", {}),
                    "duckdb_metadata": dk_result.get("metadata", {}) if dk_error is None else {},
                },
            )

        return GoldAgreementResult(
            query_id=query.query_id,
            agreed=agreed,
            pandas_result=pd_value,
            duckdb_result=dk_value,
            tolerance=query.tolerance,
            agreement_status=status,
            diagnostic=diagnostic,
            gold_answer=gold,
        )


def validate_world_queries(
    world_id: str,
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Validate all generated queries for a world.
    
    Called by the CLI and used in smoke tests.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the world parquet
    world_dir = data_dir / "worlds" / world_id
    parquet_path = world_dir / "records.parquet"
    
    if not parquet_path.exists():
        return {
            "total": 0,
            "agreed": 0,
            "disagreed": 0,
            "error": f"Parquet not found: {parquet_path}",
        }
    
    # Load queries
    queries_path = output_dir.parent / "queries" / f"queries_{world_id}.jsonl"
    if not queries_path.exists():
        return {
            "total": 0,
            "agreed": 0,
            "disagreed": 0,
            "error": f"Queries not found: {queries_path}",
        }
    
    import pandas as pd_mod
    df = pd_mod.read_parquet(parquet_path)
    
    validator = GoldValidator()
    agreed = 0
    disagreed = 0
    results = []
    
    with open(queries_path) as f:
        for line in f:
            if not line.strip():
                continue
            q_data = json.loads(line)
            try:
                query = QuerySpec.model_validate(q_data)
                result = validator.validate(query, df, parquet_path)
                results.append({
                    "query_id": result.query_id,
                    "agreed": result.agreed,
                    "status": result.agreement_status.value,
                    "diagnostic": result.diagnostic,
                })
                if result.agreed:
                    agreed += 1
                else:
                    disagreed += 1
            except Exception as e:
                disagreed += 1
                results.append({"query_id": q_data.get("query_id"), "error": str(e)})
    
    report = {
        "total": agreed + disagreed,
        "agreed": agreed,
        "disagreed": disagreed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "world_id": world_id,
        "results": results,
    }
    
    report_path = output_dir / f"gold_report_{world_id}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    report["report_path"] = str(report_path)
    
    return report

"""
Explicit versioned oracle interfaces for the FaultTrace-RAG counterfactual engine.

Provides deterministic baseline truths for the three pipeline stages:
- R (ScopeOracle): The exact set of eligible records for a query.
- E (ExtractionOracle): The exact fact rows for a given set of records.
- A (AggregationOracle): The exact aggregated answer for a given set of fact rows.
"""

from typing import Any, Optional
import pandas as pd
from pydantic import BaseModel

from faulttrace_core.models import QuerySpec, FactSpec, AggregationSpec
from faulttrace_gold.pandas_engine import PandasEvaluator


class ScopeOracleResult(BaseModel):
    record_ids: list[str]
    metadata: dict[str, Any]


class ExtractionOracleResult(BaseModel):
    fact_rows: list[dict[str, Any]]
    metadata: dict[str, Any]


class AggregationOracleResult(BaseModel):
    answer_value: Any
    contributing_ids: list[str]
    metadata: dict[str, Any]


class ScopeOracle:
    """R*: Returns the gold eligible record set for a QuerySpec."""

    def __init__(self, evaluator: Optional[PandasEvaluator] = None):
        self.evaluator = evaluator or PandasEvaluator()

    def evaluate(self, query: QuerySpec, corpus_df: pd.DataFrame) -> ScopeOracleResult:
        scoped_df = self.evaluator._apply_scope(query.scope_predicate, corpus_df)
        record_ids = scoped_df["record_id"].tolist() if "record_id" in scoped_df.columns else []
        return ScopeOracleResult(
            record_ids=record_ids,
            metadata={"eligible_count": len(scoped_df)}
        )


class ExtractionOracle:
    """E*: Returns gold fact rows for a supplied record set."""

    def __init__(self, evaluator: Optional[PandasEvaluator] = None):
        self.evaluator = evaluator or PandasEvaluator()

    def evaluate(self, fact_spec: FactSpec, supplied_df: pd.DataFrame) -> ExtractionOracleResult:
        # Note: supplied_df is assumed to be filtered to the record set already.
        # The extraction oracle just applies the fact_spec.
        fact_df = self.evaluator._apply_fact_spec(fact_spec, supplied_df)
        fact_rows = fact_df.to_dict(orient="records")
        
        # Ensure all types are python natives, not pandas objects like Timestamp
        # Standardize types for comparison
        import json
        fact_rows = json.loads(json.dumps(fact_rows, default=str))
        
        return ExtractionOracleResult(
            fact_rows=fact_rows,
            metadata={"extracted_count": len(fact_rows), "shape": list(fact_df.shape)}
        )


class AggregationOracle:
    """A*: Applies gold deterministic AggregationSpec to supplied extracted rows."""

    def __init__(self, evaluator: Optional[PandasEvaluator] = None):
        self.evaluator = evaluator or PandasEvaluator()

    def evaluate(
        self, 
        agg_spec: AggregationSpec, 
        supplied_rows: list[dict[str, Any]], 
        query: QuerySpec
    ) -> AggregationOracleResult:
        # Convert supplied rows to a dataframe for the evaluator
        fact_df = pd.DataFrame(supplied_rows)
        if fact_df.empty:
            # Need to create empty DF with expected columns
            # But we can try just executing it
            pass
            
        # Ensure proper aggregation
        result, contributing_ids, metadata = self.evaluator._aggregate(agg_spec, fact_df, query)
        
        return AggregationOracleResult(
            answer_value=result,
            contributing_ids=contributing_ids,
            metadata=metadata
        )

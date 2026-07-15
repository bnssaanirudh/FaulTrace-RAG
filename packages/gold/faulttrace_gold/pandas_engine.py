"""
Pandas-based gold evaluator for FaultTrace-RAG.

Evaluates QuerySpec against a Pandas DataFrame of corpus records.
Uses independent code paths from the DuckDB evaluator.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any, Optional

import pandas as pd

from faulttrace_core.models import (
    AggregationSpec,
    AggregationKind,
    ComparisonSpec,
    CountSpec,
    FactSpec,
    GoldAnswer,
    MeanSpec,
    NullPolicy,
    ProportionSpec,
    QuerySpec,
    ScopePredicate,
    SumSpec,
    TopKSpec,
    TiePolicy,
    TrendSpec,
    AgreementStatus,
)
from faulttrace_core.predicates import PredicateCompiler

compiler = PredicateCompiler()


class PandasEvaluator:
    """
    Evaluates QuerySpec against a Pandas DataFrame.
    
    Independent from DuckDBEvaluator. Produces identical results when
    both engines are correct.
    """

    def evaluate(
        self,
        query: QuerySpec,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Evaluate query against df.
        
        Returns a dict with:
            result: the computed answer value
            eligible_count: records in scope
            contributing_ids: record IDs used in computation
            metadata: additional derivation info
        """
        # Step 1: Apply scope predicate
        scoped_df = self._apply_scope(query.scope_predicate, df)
        eligible_count = len(scoped_df)
        
        # Step 2: Apply fact spec (field selection + derived fields)
        fact_df = self._apply_fact_spec(query.fact_spec, scoped_df)
        
        # Step 3: Apply aggregation
        agg_spec = query.aggregation_spec
        result, contributing_ids, metadata = self._aggregate(agg_spec, fact_df, query)
        
        return {
            "result": result,
            "eligible_count": eligible_count,
            "contributing_ids": contributing_ids,
            "metadata": metadata,
            "fact_df_shape": fact_df.shape,
        }

    def _apply_scope(self, predicate: ScopePredicate, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scope predicate to DataFrame."""
        if len(df) == 0:
            return df
        mask = compiler.to_pandas_mask(predicate, df)
        return df[mask].copy()

    def _apply_fact_spec(self, fact_spec: FactSpec, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fact spec: select fields and compute derived fields."""
        # Select only specified fields (plus record_id for tracking)
        available = set(df.columns)
        select_fields = list(set(fact_spec.fields) & available)
        if "record_id" not in select_fields and "record_id" in available:
            select_fields = ["record_id"] + select_fields
        
        result_df = df[select_fields].copy()
        
        # Compute derived fields
        for derived in fact_spec.derived_fields:
            if derived.source_field not in result_df.columns:
                continue
            col = result_df[derived.source_field]
            if derived.expression_kind == "identity":
                result_df[derived.name] = col
            elif derived.expression_kind == "year":
                result_df[derived.name] = pd.to_datetime(col).dt.year
            elif derived.expression_kind == "month":
                result_df[derived.name] = pd.to_datetime(col).dt.month
            elif derived.expression_kind == "quarter":
                result_df[derived.name] = pd.to_datetime(col).dt.quarter
            elif derived.expression_kind == "log1p":
                import numpy as np
                result_df[derived.name] = np.log1p(pd.to_numeric(col, errors="coerce"))
        
        return result_df

    def _aggregate(
        self,
        agg_spec: AggregationSpec,
        fact_df: pd.DataFrame,
        query: QuerySpec,
    ) -> tuple[Any, list[str], dict[str, Any]]:
        """Execute aggregation. Returns (result, contributing_ids, metadata)."""
        ids = fact_df["record_id"].tolist() if "record_id" in fact_df.columns else []
        
        if isinstance(agg_spec, CountSpec):
            return self._count(agg_spec, fact_df, ids)
        elif isinstance(agg_spec, SumSpec):
            return self._sum(agg_spec, fact_df, ids)
        elif isinstance(agg_spec, MeanSpec):
            return self._mean(agg_spec, fact_df, ids)
        elif isinstance(agg_spec, ProportionSpec):
            return self._proportion(agg_spec, fact_df, query, ids)
        elif isinstance(agg_spec, ComparisonSpec):
            return self._comparison(agg_spec, fact_df, query, ids)
        elif isinstance(agg_spec, TopKSpec):
            return self._top_k(agg_spec, fact_df, ids)
        elif isinstance(agg_spec, TrendSpec):
            return self._trend(agg_spec, fact_df, ids)
        else:
            raise ValueError(f"Unsupported aggregation spec: {type(agg_spec)}")

    def _count(self, spec: CountSpec, df: pd.DataFrame, ids: list[str]) -> tuple[int, list[str], dict]:
        if spec.distinct and spec.field and spec.field in df.columns:
            col = self._apply_null_policy(df[spec.field], NullPolicy.EXCLUDE)
            result = int(col.nunique())
        else:
            result = len(df)
        return result, ids, {"distinct": spec.distinct, "field": spec.field}

    def _sum(self, spec: SumSpec, df: pd.DataFrame, ids: list[str]) -> tuple[float, list[str], dict]:
        if spec.field not in df.columns:
            return 0.0, [], {"error": f"Field {spec.field} not found"}
        col = self._apply_null_policy(df[spec.field], spec.null_policy)
        col = pd.to_numeric(col, errors="coerce")
        if spec.null_policy == NullPolicy.EXCLUDE:
            col = col.dropna()
        result = float(col.sum())
        return result, ids, {"field": spec.field, "null_policy": spec.null_policy.value}

    def _mean(self, spec: MeanSpec, df: pd.DataFrame, ids: list[str]) -> tuple[Optional[float], list[str], dict]:
        if spec.field not in df.columns:
            return None, [], {"error": f"Field {spec.field} not found"}
        col = pd.to_numeric(df[spec.field], errors="coerce")
        if spec.null_policy == NullPolicy.EXCLUDE:
            col = col.dropna()
        elif spec.null_policy == NullPolicy.INCLUDE_AS_ZERO:
            col = col.fillna(0.0)
        if len(col) == 0:
            return None, [], {"note": "empty_after_null_policy"}
        result = round(float(col.mean()), spec.decimal_places)
        return result, ids, {
            "field": spec.field,
            "n": len(col),
            "null_policy": spec.null_policy.value,
        }

    def _proportion(
        self, spec: ProportionSpec, df: pd.DataFrame, query: QuerySpec, ids: list[str]
    ) -> tuple[Optional[float], list[str], dict]:
        denominator = len(df)
        if denominator == 0:
            return None, [], {"note": "empty_denominator"}
        
        # Apply numerator predicate to scoped df
        # Need access to original full df for proportion - use fact_df as already scoped
        num_mask = compiler.to_pandas_mask(spec.numerator_predicate, df)
        numerator = int(num_mask.sum())
        
        proportion = round(numerator / denominator, spec.decimal_places)
        num_ids = df[num_mask]["record_id"].tolist() if "record_id" in df.columns else []
        
        return proportion, num_ids, {
            "numerator": numerator,
            "denominator": denominator,
        }

    def _comparison(
        self, spec: ComparisonSpec, df: pd.DataFrame, query: QuerySpec, ids: list[str]
    ) -> tuple[Any, list[str], dict]:
        mask_a = compiler.to_pandas_mask(spec.group_a_predicate, df)
        mask_b = compiler.to_pandas_mask(spec.group_b_predicate, df)
        df_a = df[mask_a]
        df_b = df[mask_b]
        
        val_a = self._group_measure(spec, df_a)
        val_b = self._group_measure(spec, df_b)
        
        if val_a is None or val_b is None:
            return None, [], {"note": "empty_group"}
        
        diff = round(val_a - val_b, 6)
        ratio = round(val_a / val_b, 6) if val_b != 0 else None
        
        if spec.output == "difference":
            result = diff
        elif spec.output == "ratio":
            result = ratio
        else:
            result = {"difference": diff, "ratio": ratio}
        
        return result, list(df_a["record_id"].tolist() if "record_id" in df_a.columns else []) + \
               list(df_b["record_id"].tolist() if "record_id" in df_b.columns else []), {
            "group_a_value": val_a,
            "group_b_value": val_b,
            "group_a_count": len(df_a),
            "group_b_count": len(df_b),
        }

    def _group_measure(self, spec: ComparisonSpec, df: pd.DataFrame) -> Optional[float]:
        if len(df) == 0:
            return None
        if spec.measure == "count":
            return float(len(df))
        elif spec.measure == "mean" and spec.field:
            col = pd.to_numeric(df[spec.field], errors="coerce").dropna()
            return float(col.mean()) if len(col) > 0 else None
        elif spec.measure == "sum" and spec.field:
            col = pd.to_numeric(df[spec.field], errors="coerce").fillna(0.0)
            return float(col.sum())
        return None

    def _top_k(self, spec: TopKSpec, df: pd.DataFrame, ids: list[str]) -> tuple[list[dict], list[str], dict]:
        if spec.group_by_field not in df.columns:
            return [], [], {"error": f"Field {spec.group_by_field} not found"}
        
        # Compute measure per group
        if spec.measure == "count":
            grouped = df.groupby(spec.group_by_field).size().reset_index(name="_measure")
        elif spec.measure == "mean" and spec.value_field and spec.value_field in df.columns:
            col = pd.to_numeric(df[spec.value_field], errors="coerce")
            df = df.copy()
            df["_val"] = col
            grouped = df.groupby(spec.group_by_field)["_val"].mean().reset_index(name="_measure")
        elif spec.measure == "sum" and spec.value_field and spec.value_field in df.columns:
            col = pd.to_numeric(df[spec.value_field], errors="coerce").fillna(0.0)
            df = df.copy()
            df["_val"] = col
            grouped = df.groupby(spec.group_by_field)["_val"].sum().reset_index(name="_measure")
        else:
            grouped = df.groupby(spec.group_by_field).size().reset_index(name="_measure")
        
        # Sort
        grouped = grouped.sort_values(
            ["_measure", spec.group_by_field],
            ascending=[spec.ascending, True],
        )
        
        # Apply tie policy
        k_val = min(spec.k, len(grouped))
        if spec.tie_policy == TiePolicy.FIRST:
            top = grouped.head(k_val)
        elif spec.tie_policy == TiePolicy.ALL:
            # Include all rows tied with the k-th value
            if k_val < len(grouped):
                kth_val = grouped.iloc[k_val - 1]["_measure"]
                top = grouped[grouped["_measure"] >= kth_val] if not spec.ascending else grouped[grouped["_measure"] <= kth_val]
            else:
                top = grouped
        else:
            top = grouped.head(k_val)
        
        result = [
            {spec.group_by_field: row[spec.group_by_field], "value": float(row["_measure"])}
            for _, row in top.iterrows()
        ]
        return result, ids, {
            "k": spec.k,
            "measure": spec.measure,
            "group_by": spec.group_by_field,
            "tie_policy": spec.tie_policy.value,
        }

    def _trend(self, spec: TrendSpec, df: pd.DataFrame, ids: list[str]) -> tuple[list[dict], list[str], dict]:
        if spec.time_field not in df.columns:
            return [], [], {"error": f"Field {spec.time_field} not found"}
        
        df = df.copy()
        time_col = pd.to_datetime(df[spec.time_field], errors="coerce", utc=True)
        
        if spec.bucket == "month":
            df["_bucket"] = time_col.dt.strftime("%Y-%m")
        elif spec.bucket == "quarter":
            # Normalize to YYYY-QN format matching DuckDB output
            df["_bucket"] = time_col.dt.year.astype(str) + "-Q" + time_col.dt.quarter.astype(str)
        elif spec.bucket == "year":
            df["_bucket"] = time_col.dt.strftime("%Y")
        else:
            df["_bucket"] = time_col.dt.strftime("%Y-%m")
        
        if spec.measure == "count":
            grouped = df.groupby("_bucket").size().reset_index(name="_measure")
        elif spec.measure == "mean" and spec.value_field and spec.value_field in df.columns:
            col = pd.to_numeric(df[spec.value_field], errors="coerce")
            if spec.null_policy == NullPolicy.EXCLUDE:
                df = df.copy()
                df["_val"] = col
                grouped = df.groupby("_bucket")["_val"].mean().reset_index(name="_measure")
            else:
                df = df.copy()
                df["_val"] = col.fillna(0.0)
                grouped = df.groupby("_bucket")["_val"].mean().reset_index(name="_measure")
        else:
            grouped = df.groupby("_bucket").size().reset_index(name="_measure")
        
        grouped = grouped.sort_values("_bucket")
        result = [
            {"bucket": row["_bucket"], "value": float(row["_measure"])}
            for _, row in grouped.iterrows()
        ]
        return result, ids, {"bucket": spec.bucket, "measure": spec.measure}

    def _apply_null_policy(self, col: pd.Series, policy: NullPolicy) -> pd.Series:
        if policy == NullPolicy.EXCLUDE:
            return col.dropna()
        elif policy == NullPolicy.INCLUDE_AS_ZERO:
            return col.fillna(0.0)
        return col

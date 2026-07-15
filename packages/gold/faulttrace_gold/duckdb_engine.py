"""
DuckDB-based gold evaluator for FaultTrace-RAG.

Evaluates QuerySpec against a Parquet file using DuckDB SQL.
Uses independent code paths from the Pandas evaluator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd

from faulttrace_core.models import (
    AggregationSpec,
    ComparisonSpec,
    CountSpec,
    FactSpec,
    MeanSpec,
    NullPolicy,
    ProportionSpec,
    QuerySpec,
    ScopePredicate,
    SumSpec,
    TopKSpec,
    TiePolicy,
    TrendSpec,
)
from faulttrace_core.predicates import PredicateCompiler, _validate_field_name

compiler = PredicateCompiler()


class DuckDBEvaluator:
    """
    Evaluates QuerySpec against a Parquet file using DuckDB SQL.
    
    Independent from PandasEvaluator. Uses SQL-based aggregation paths.
    """

    def evaluate(
        self,
        query: QuerySpec,
        parquet_path: Path,
    ) -> dict[str, Any]:
        """
        Evaluate query against a Parquet file.
        
        Returns a dict with:
            result: the computed answer value
            eligible_count: records in scope
            contributing_ids: record IDs used
            metadata: additional derivation info
            sql: the SQL used (for debugging)
        """
        with duckdb.connect(":memory:") as con:
            # Register the Parquet file
            con.execute(f"CREATE VIEW records AS SELECT * FROM read_parquet('{parquet_path}')")
            
            return self._evaluate_with_con(query, con)

    def evaluate_from_df(
        self,
        query: QuerySpec,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Evaluate against a Pandas DataFrame (registers as DuckDB table)."""
        with duckdb.connect(":memory:") as con:
            con.register("records", df)
            return self._evaluate_with_con(query, con)

    def _evaluate_with_con(self, query: QuerySpec, con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
        """Core evaluation logic given an open connection with 'records' view."""
        scope_sql = compiler.to_duckdb_sql(query.scope_predicate)
        
        # Step 1: Count eligible records
        eligible_sql = f"SELECT COUNT(*) FROM records WHERE {scope_sql}"
        eligible_count = int(con.execute(eligible_sql).fetchone()[0])
        
        # Step 2: Execute aggregation
        agg_spec = query.aggregation_spec
        result, contributing_ids, metadata, sql_used = self._aggregate(
            agg_spec, query, scope_sql, con
        )
        
        return {
            "result": result,
            "eligible_count": eligible_count,
            "contributing_ids": contributing_ids,
            "metadata": metadata,
            "sql": sql_used,
        }

    def _aggregate(
        self,
        agg_spec: AggregationSpec,
        query: QuerySpec,
        scope_sql: str,
        con: duckdb.DuckDBPyConnection,
    ) -> tuple[Any, list[str], dict[str, Any], str]:
        """Execute aggregation. Returns (result, contributing_ids, metadata, sql)."""
        if isinstance(agg_spec, CountSpec):
            return self._count(agg_spec, scope_sql, con)
        elif isinstance(agg_spec, SumSpec):
            return self._sum(agg_spec, scope_sql, con)
        elif isinstance(agg_spec, MeanSpec):
            return self._mean(agg_spec, scope_sql, con)
        elif isinstance(agg_spec, ProportionSpec):
            return self._proportion(agg_spec, scope_sql, con)
        elif isinstance(agg_spec, ComparisonSpec):
            return self._comparison(agg_spec, scope_sql, con)
        elif isinstance(agg_spec, TopKSpec):
            return self._top_k(agg_spec, scope_sql, con)
        elif isinstance(agg_spec, TrendSpec):
            return self._trend(agg_spec, scope_sql, con)
        else:
            raise ValueError(f"Unsupported aggregation spec: {type(agg_spec)}")

    def _count(self, spec: CountSpec, scope_sql: str, con: duckdb.DuckDBPyConnection) -> tuple:
        if spec.distinct and spec.field:
            field = _validate_field_name(spec.field)
            sql = f"SELECT COUNT(DISTINCT {field}) FROM records WHERE {scope_sql}"
        else:
            sql = f"SELECT COUNT(*) FROM records WHERE {scope_sql}"
        result = int(con.execute(sql).fetchone()[0])
        ids_sql = f"SELECT record_id FROM records WHERE {scope_sql}"
        ids = [row[0] for row in con.execute(ids_sql).fetchall()]
        return result, ids, {"distinct": spec.distinct}, sql

    def _sum(self, spec: SumSpec, scope_sql: str, con: duckdb.DuckDBPyConnection) -> tuple:
        field = _validate_field_name(spec.field)
        null_clause = "" if spec.null_policy == NullPolicy.EXCLUDE else f"COALESCE({field}, 0)"
        if spec.null_policy == NullPolicy.EXCLUDE:
            sql = f"SELECT SUM({field}) FROM records WHERE {scope_sql} AND {field} IS NOT NULL"
        else:
            sql = f"SELECT SUM(COALESCE({field}, 0)) FROM records WHERE {scope_sql}"
        val = con.execute(sql).fetchone()[0]
        result = float(val) if val is not None else 0.0
        ids_sql = f"SELECT record_id FROM records WHERE {scope_sql}"
        ids = [row[0] for row in con.execute(ids_sql).fetchall()]
        return result, ids, {"field": spec.field}, sql

    def _mean(self, spec: MeanSpec, scope_sql: str, con: duckdb.DuckDBPyConnection) -> tuple:
        field = _validate_field_name(spec.field)
        if spec.null_policy == NullPolicy.EXCLUDE:
            sql = f"SELECT ROUND(AVG({field}), {spec.decimal_places}), COUNT({field}) FROM records WHERE {scope_sql} AND {field} IS NOT NULL"
        elif spec.null_policy == NullPolicy.INCLUDE_AS_ZERO:
            sql = f"SELECT ROUND(AVG(COALESCE({field}, 0)), {spec.decimal_places}), COUNT(*) FROM records WHERE {scope_sql}"
        else:
            sql = f"SELECT ROUND(AVG({field}), {spec.decimal_places}), COUNT({field}) FROM records WHERE {scope_sql}"
        
        row = con.execute(sql).fetchone()
        val, n = row[0], row[1]
        result = float(val) if val is not None else None
        ids_sql = f"SELECT record_id FROM records WHERE {scope_sql}"
        ids = [row[0] for row in con.execute(ids_sql).fetchall()]
        return result, ids, {"field": spec.field, "n": n}, sql

    def _proportion(self, spec: ProportionSpec, scope_sql: str, con: duckdb.DuckDBPyConnection) -> tuple:
        num_sql = compiler.to_duckdb_sql(spec.numerator_predicate)
        
        denom_sql = f"SELECT COUNT(*) FROM records WHERE {scope_sql}"
        denominator = int(con.execute(denom_sql).fetchone()[0])
        
        numer_sql = f"SELECT COUNT(*) FROM records WHERE ({scope_sql}) AND ({num_sql})"
        numerator = int(con.execute(numer_sql).fetchone()[0])
        
        if denominator == 0:
            return None, [], {"note": "empty_denominator"}, denom_sql
        
        proportion = round(numerator / denominator, spec.decimal_places)
        ids_sql = f"SELECT record_id FROM records WHERE ({scope_sql}) AND ({num_sql})"
        ids = [row[0] for row in con.execute(ids_sql).fetchall()]
        
        sql = f"numerator: {numer_sql}\ndenominator: {denom_sql}"
        return proportion, ids, {"numerator": numerator, "denominator": denominator}, sql

    def _comparison(self, spec: ComparisonSpec, scope_sql: str, con: duckdb.DuckDBPyConnection) -> tuple:
        sql_a = compiler.to_duckdb_sql(spec.group_a_predicate)
        sql_b = compiler.to_duckdb_sql(spec.group_b_predicate)
        
        val_a = self._group_sql_measure(spec, f"({scope_sql}) AND ({sql_a})", con)
        val_b = self._group_sql_measure(spec, f"({scope_sql}) AND ({sql_b})", con)
        
        if val_a is None or val_b is None:
            return None, [], {"note": "empty_group"}, ""
        
        diff = round(val_a - val_b, 6)
        ratio = round(val_a / val_b, 6) if val_b != 0 else None
        
        if spec.output == "difference":
            result = diff
        elif spec.output == "ratio":
            result = ratio
        else:
            result = {"difference": diff, "ratio": ratio}
        
        sql = f"group_a: {sql_a}\ngroup_b: {sql_b}"
        ids_a_sql = f"SELECT record_id FROM records WHERE ({scope_sql}) AND ({sql_a})"
        ids_b_sql = f"SELECT record_id FROM records WHERE ({scope_sql}) AND ({sql_b})"
        ids = [r[0] for r in con.execute(ids_a_sql).fetchall()] + \
              [r[0] for r in con.execute(ids_b_sql).fetchall()]
        
        return result, ids, {"group_a": val_a, "group_b": val_b}, sql

    def _group_sql_measure(self, spec: ComparisonSpec, where_clause: str, con: duckdb.DuckDBPyConnection) -> Optional[float]:
        if spec.measure == "count":
            sql = f"SELECT COUNT(*) FROM records WHERE {where_clause}"
            val = con.execute(sql).fetchone()[0]
            return float(val)
        elif spec.measure == "mean" and spec.field:
            field = _validate_field_name(spec.field)
            sql = f"SELECT AVG({field}) FROM records WHERE {where_clause} AND {field} IS NOT NULL"
            val = con.execute(sql).fetchone()[0]
            return float(val) if val is not None else None
        elif spec.measure == "sum" and spec.field:
            field = _validate_field_name(spec.field)
            sql = f"SELECT SUM({field}) FROM records WHERE {where_clause}"
            val = con.execute(sql).fetchone()[0]
            return float(val) if val is not None else 0.0
        return None

    def _top_k(self, spec: TopKSpec, scope_sql: str, con: duckdb.DuckDBPyConnection) -> tuple:
        group_field = _validate_field_name(spec.group_by_field)
        
        if spec.measure == "count":
            inner_sql = f"SELECT {group_field}, COUNT(*) AS _measure FROM records WHERE {scope_sql} GROUP BY {group_field}"
        elif spec.measure == "mean" and spec.value_field:
            val_field = _validate_field_name(spec.value_field)
            inner_sql = f"SELECT {group_field}, AVG({val_field}) AS _measure FROM records WHERE {scope_sql} AND {val_field} IS NOT NULL GROUP BY {group_field}"
        elif spec.measure == "sum" and spec.value_field:
            val_field = _validate_field_name(spec.value_field)
            inner_sql = f"SELECT {group_field}, SUM({val_field}) AS _measure FROM records WHERE {scope_sql} GROUP BY {group_field}"
        else:
            inner_sql = f"SELECT {group_field}, COUNT(*) AS _measure FROM records WHERE {scope_sql} GROUP BY {group_field}"
        
        asc_desc = "ASC" if spec.ascending else "DESC"
        # Deterministic tie breaking: sort by measure desc/asc, then by field name asc
        sql = f"SELECT {group_field}, _measure FROM ({inner_sql}) t ORDER BY _measure {asc_desc}, {group_field} ASC LIMIT {spec.k}"
        
        rows = con.execute(sql).fetchall()
        result = [
            {spec.group_by_field: row[0], "value": float(row[1])}
            for row in rows
        ]
        ids_sql = f"SELECT record_id FROM records WHERE {scope_sql}"
        ids = [row[0] for row in con.execute(ids_sql).fetchall()]
        return result, ids, {"k": spec.k, "measure": spec.measure}, sql

    def _trend(self, spec: TrendSpec, scope_sql: str, con: duckdb.DuckDBPyConnection) -> tuple:
        time_field = _validate_field_name(spec.time_field)
        
        if spec.bucket == "month":
            bucket_expr = f"STRFTIME({time_field}::TIMESTAMP, '%Y-%m')"
        elif spec.bucket == "quarter":
            bucket_expr = f"CAST(YEAR({time_field}::TIMESTAMP) AS VARCHAR) || '-Q' || CAST(QUARTER({time_field}::TIMESTAMP) AS VARCHAR)"
        elif spec.bucket == "year":
            bucket_expr = f"STRFTIME({time_field}::TIMESTAMP, '%Y')"
        else:
            bucket_expr = f"STRFTIME({time_field}::TIMESTAMP, '%Y-%m')"
        
        if spec.measure == "count":
            inner = f"{bucket_expr} AS _bucket, COUNT(*) AS _measure"
        elif spec.measure == "mean" and spec.value_field:
            val_field = _validate_field_name(spec.value_field)
            null_clause = f" AND {val_field} IS NOT NULL" if spec.null_policy == NullPolicy.EXCLUDE else ""
            inner = f"{bucket_expr} AS _bucket, AVG({val_field}) AS _measure"
            scope_sql = scope_sql + null_clause
        else:
            inner = f"{bucket_expr} AS _bucket, COUNT(*) AS _measure"
        
        sql = f"SELECT {inner} FROM records WHERE {scope_sql} GROUP BY _bucket ORDER BY _bucket"
        
        rows = con.execute(sql).fetchall()
        result = [{"bucket": row[0], "value": float(row[1])} for row in rows]
        ids_sql = f"SELECT record_id FROM records WHERE {scope_sql}"
        ids = [row[0] for row in con.execute(ids_sql).fetchall()]
        return result, ids, {"bucket": spec.bucket, "measure": spec.measure}, sql

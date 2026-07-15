"""
Predicate compiler: transforms ScopePredicate AST into Pandas masks and DuckDB SQL.

The compiler never uses eval() or exec() on any user/corpus text.
All transformations are explicit case-by-case implementations.
"""

from __future__ import annotations

import re
from datetime import datetime, date
from typing import Any, TYPE_CHECKING

import pandas as pd

from faulttrace_core.models import (
    AndPredicate,
    EqPredicate,
    InPredicate,
    IsNotNullPredicate,
    IsNullPredicate,
    NeqPredicate,
    NotInPredicate,
    OrPredicate,
    RangePredicate,
    ScopePredicate,
)

# Safe field name pattern: alphanumeric + underscore only
_SAFE_FIELD_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

# Fields allowed in predicates
ALLOWED_FIELDS: frozenset[str] = frozenset({
    "record_id",
    "source",
    "world_id",
    "product_id",
    "parent_id",
    "category",
    "title",
    "brand",
    "rating",
    "helpful_votes",
    "verified_purchase",
    "event_time",
    "price",
    "text",
    "schema_version",
})


def _validate_field_name(field: str) -> str:
    """Ensure field name is safe for SQL generation."""
    if not _SAFE_FIELD_RE.match(field):
        raise ValueError(f"Unsafe field name: '{field}'")
    if field not in ALLOWED_FIELDS:
        raise ValueError(f"Field '{field}' not in allowed fields: {sorted(ALLOWED_FIELDS)}")
    return field


def _sql_literal(value: Any) -> str:
    """Convert a Python value to a SQL literal string (safe, no injection)."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (datetime, date)):
        return f"TIMESTAMP '{value.isoformat()}'"
    if isinstance(value, str):
        # Escape single quotes
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    raise ValueError(f"Unsupported literal type: {type(value)}")


def _sql_literals(values: list[Any]) -> str:
    return "(" + ", ".join(_sql_literal(v) for v in values) + ")"


class PredicateCompiler:
    """
    Compiles ScopePredicate AST to Pandas masks and DuckDB SQL WHERE clauses.
    
    Usage:
        compiler = PredicateCompiler()
        mask = compiler.to_pandas_mask(predicate, df)
        sql = compiler.to_duckdb_sql(predicate)
    """

    def to_pandas_mask(self, predicate: ScopePredicate, df: pd.DataFrame) -> pd.Series:
        """Compile predicate to a boolean Pandas Series mask."""
        return self._pandas(predicate, df)

    def to_duckdb_sql(self, predicate: ScopePredicate) -> str:
        """Compile predicate to a DuckDB-compatible SQL WHERE clause expression."""
        return self._sql(predicate)

    # --- Pandas compilation ---

    def _pandas(self, pred: ScopePredicate, df: pd.DataFrame) -> pd.Series:
        if isinstance(pred, EqPredicate):
            return self._pandas_eq(pred, df)
        elif isinstance(pred, NeqPredicate):
            return self._pandas_neq(pred, df)
        elif isinstance(pred, InPredicate):
            return self._pandas_in(pred, df)
        elif isinstance(pred, NotInPredicate):
            return ~self._pandas_in(InPredicate(field=pred.field, values=pred.values), df)
        elif isinstance(pred, RangePredicate):
            return self._pandas_range(pred, df)
        elif isinstance(pred, IsNullPredicate):
            return df[_validate_field_name(pred.field)].isna()
        elif isinstance(pred, IsNotNullPredicate):
            return df[_validate_field_name(pred.field)].notna()
        elif isinstance(pred, AndPredicate):
            result = self._pandas(pred.operands[0], df)
            for op in pred.operands[1:]:
                result = result & self._pandas(op, df)
            return result
        elif isinstance(pred, OrPredicate):
            result = self._pandas(pred.operands[0], df)
            for op in pred.operands[1:]:
                result = result | self._pandas(op, df)
            return result
        else:
            raise ValueError(f"Unknown predicate type: {type(pred)}")

    def _pandas_eq(self, pred: EqPredicate, df: pd.DataFrame) -> pd.Series:
        field = _validate_field_name(pred.field)
        col = df[field]
        if pred.value is None:
            return col.isna()
        # Handle enum values
        val = pred.value
        if hasattr(val, "value"):
            val = val.value
        return col == val

    def _pandas_neq(self, pred: NeqPredicate, df: pd.DataFrame) -> pd.Series:
        field = _validate_field_name(pred.field)
        col = df[field]
        if pred.value is None:
            return col.notna()
        val = pred.value
        if hasattr(val, "value"):
            val = val.value
        return col != val

    def _pandas_in(self, pred: InPredicate, df: pd.DataFrame) -> pd.Series:
        field = _validate_field_name(pred.field)
        values = [v.value if hasattr(v, "value") else v for v in pred.values]
        return df[field].isin(values)

    def _pandas_range(self, pred: RangePredicate, df: pd.DataFrame) -> pd.Series:
        field = _validate_field_name(pred.field)
        col = df[field]
        mask = pd.Series([True] * len(df), index=df.index)
        if pred.low is not None:
            if pred.low_inclusive:
                mask = mask & (col >= pred.low)
            else:
                mask = mask & (col > pred.low)
        if pred.high is not None:
            if pred.high_inclusive:
                mask = mask & (col <= pred.high)
            else:
                mask = mask & (col < pred.high)
        return mask

    # --- SQL compilation ---

    def _sql(self, pred: ScopePredicate) -> str:
        if isinstance(pred, EqPredicate):
            field = _validate_field_name(pred.field)
            if pred.value is None:
                return f"{field} IS NULL"
            val = pred.value.value if hasattr(pred.value, "value") else pred.value
            return f"{field} = {_sql_literal(val)}"
        elif isinstance(pred, NeqPredicate):
            field = _validate_field_name(pred.field)
            if pred.value is None:
                return f"{field} IS NOT NULL"
            val = pred.value.value if hasattr(pred.value, "value") else pred.value
            return f"{field} != {_sql_literal(val)}"
        elif isinstance(pred, InPredicate):
            field = _validate_field_name(pred.field)
            vals = [v.value if hasattr(v, "value") else v for v in pred.values]
            return f"{field} IN {_sql_literals(vals)}"
        elif isinstance(pred, NotInPredicate):
            field = _validate_field_name(pred.field)
            vals = [v.value if hasattr(v, "value") else v for v in pred.values]
            return f"{field} NOT IN {_sql_literals(vals)}"
        elif isinstance(pred, RangePredicate):
            return self._sql_range(pred)
        elif isinstance(pred, IsNullPredicate):
            field = _validate_field_name(pred.field)
            return f"{field} IS NULL"
        elif isinstance(pred, IsNotNullPredicate):
            field = _validate_field_name(pred.field)
            return f"{field} IS NOT NULL"
        elif isinstance(pred, AndPredicate):
            parts = [f"({self._sql(op)})" for op in pred.operands]
            return " AND ".join(parts)
        elif isinstance(pred, OrPredicate):
            parts = [f"({self._sql(op)})" for op in pred.operands]
            return " OR ".join(parts)
        else:
            raise ValueError(f"Unknown predicate type: {type(pred)}")

    def _sql_range(self, pred: RangePredicate) -> str:
        field = _validate_field_name(pred.field)
        parts = []
        if pred.low is not None:
            op = ">=" if pred.low_inclusive else ">"
            parts.append(f"{field} {op} {_sql_literal(pred.low)}")
        if pred.high is not None:
            op = "<=" if pred.high_inclusive else "<"
            parts.append(f"{field} {op} {_sql_literal(pred.high)}")
        return " AND ".join(parts)


# Singleton compiler instance
compiler = PredicateCompiler()

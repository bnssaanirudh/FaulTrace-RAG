"""faulttrace_gold: Dual gold engine for FaultTrace-RAG."""

from faulttrace_gold.pandas_engine import PandasEvaluator
from faulttrace_gold.duckdb_engine import DuckDBEvaluator
from faulttrace_gold.validator import GoldValidator, GoldAgreementResult

__all__ = ["PandasEvaluator", "DuckDBEvaluator", "GoldValidator", "GoldAgreementResult"]
__version__ = "0.1.0"

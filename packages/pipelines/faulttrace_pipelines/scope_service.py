import pandas as pd
from typing import Optional, List
from pydantic import BaseModel, Field

from faulttrace_core.models import QuerySpec, _stable_hash, FailureCode
from faulttrace_core.predicates import compiler

class ScopeResult(BaseModel):
    is_success: bool
    failure_code: Optional[FailureCode] = None
    eligible_record_ids: list[str] = Field(default_factory=list)
    total_world_count: int = 0
    eligible_count: int = 0
    excluded_count: int = 0
    evidence_set_hash: str = ""
    query_specific_explanation: str = ""

class ScopeService:
    """
    WP1: Full-Scope Enumerator.
    Compiles ScopePredicate to the canonical data engine and produces an auditable
    evidence requirement (eligible record IDs).
    """

    @classmethod
    def enumerate_scope(cls, query: QuerySpec, df: pd.DataFrame) -> ScopeResult:
        try:
            # Check for semantic track T stub (if any future predicates are semantic)
            # Currently all AST predicates are deterministic filter clauses.
            
            # Compile to pandas mask
            mask = compiler.to_pandas_mask(query.scope_predicate, df)
            eligible_df = df[mask]
            
            # Stable sort record IDs
            eligible_ids = sorted(eligible_df["record_id"].tolist())
            
            total = len(df)
            eligible = len(eligible_ids)
            excluded = total - eligible
            
            evidence_hash = _stable_hash(eligible_ids)
            
            return ScopeResult(
                is_success=True,
                eligible_record_ids=eligible_ids,
                total_world_count=total,
                eligible_count=eligible,
                excluded_count=excluded,
                evidence_set_hash=evidence_hash,
                query_specific_explanation=(
                    f"Full enumeration required to guarantee completeness. "
                    f"Found {eligible} eligible records from {total} total records."
                )
            )
            
        except NotImplementedError as e:
            return ScopeResult(
                is_success=False,
                failure_code=FailureCode.UNSUPPORTED_SEMANTIC_PREDICATE,
                query_specific_explanation=str(e)
            )
        except Exception as e:
            return ScopeResult(
                is_success=False,
                failure_code=FailureCode.SCOPE_COMPILATION_FAILURE,
                query_specific_explanation=str(e)
            )

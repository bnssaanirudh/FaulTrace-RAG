from fastapi import APIRouter
from faulttrace_core.models import AnswerPolicyConfig

router = APIRouter(tags=["policies"])

# In a real implementation this would come from a database or YAML
_POLICIES = {
    "strict_exact_v1": AnswerPolicyConfig(
        policy_id="strict_exact_v1",
        version="1.0",
        min_known_scope_coverage=1.0,
        min_extraction_completeness=1.0,
        max_ambiguous_tolerance=0.0,
        min_required_field_completeness=1.0,
        require_ranking_boundary_confidence=True,
        allow_partial=False
    ),
    "warn_partial_v1": AnswerPolicyConfig(
        policy_id="warn_partial_v1",
        version="1.0",
        min_known_scope_coverage=0.5,
        min_extraction_completeness=0.5,
        max_ambiguous_tolerance=0.1,
        min_required_field_completeness=0.8,
        require_ranking_boundary_confidence=False,
        allow_partial=True
    ),
    "benchmark_raw_v1": AnswerPolicyConfig(
        policy_id="benchmark_raw_v1",
        version="1.0",
        min_known_scope_coverage=0.0,
        min_extraction_completeness=0.0,
        max_ambiguous_tolerance=1.0,
        min_required_field_completeness=0.0,
        require_ranking_boundary_confidence=False,
        allow_partial=True
    )
}

@router.get("/policies", response_model=list[AnswerPolicyConfig], summary="List available answer policies")
async def list_policies():
    return list(_POLICIES.values())

@router.get("/policies/{policy_id}", response_model=AnswerPolicyConfig, summary="Get policy details")
async def get_policy(policy_id: str):
    if policy_id not in _POLICIES:
        return _POLICIES["strict_exact_v1"] # Default fallback
    return _POLICIES[policy_id]

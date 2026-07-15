"""
Typed loss functions for counterfactual attribution.

Defines exact metrics based on the answer family (Count, Mean, Proportion, TopK, Trend).
"""

from typing import Any, Optional
from pydantic import BaseModel

from faulttrace_core.models import AggregationSpec, CountSpec, MeanSpec, SumSpec, ProportionSpec, ComparisonSpec, TopKSpec, TrendSpec
from faulttrace_gold.validator import _results_agree


class LossDiagnostic(BaseModel):
    normalized_loss: float        # Unified loss in [0.0, 1.0] for Shapley attribution
    raw_error: Optional[float] = None     # Absolute difference for continuous values
    jaccard_distance: Optional[float] = None # For sets/TopK
    status: str = "valid"         # "valid", "invalid", "abstained", "partial"


def compute_loss(pipeline_answer: Any, gold_answer: Any, agg_spec: AggregationSpec, tolerance: float = 1e-4) -> LossDiagnostic:
    """Computes family-specific typed loss between pipeline output and gold truth."""
    # Handle invalid or missing pipeline answers
    if pipeline_answer is None and gold_answer is None:
        return LossDiagnostic(normalized_loss=0.0, status="valid")
    if pipeline_answer is None:
        return LossDiagnostic(normalized_loss=1.0, status="abstained")
        
    if isinstance(agg_spec, (CountSpec, SumSpec, MeanSpec, ProportionSpec, ComparisonSpec)):
        return _compute_scalar_loss(pipeline_answer, gold_answer, tolerance)
    elif isinstance(agg_spec, TopKSpec):
        return _compute_topk_loss(pipeline_answer, gold_answer)
    elif isinstance(agg_spec, TrendSpec):
        return _compute_trend_loss(pipeline_answer, gold_answer)
    else:
        # Fallback to exact match
        agrees = _results_agree(pipeline_answer, gold_answer, tolerance)
        return LossDiagnostic(normalized_loss=0.0 if agrees else 1.0, status="valid")


def _compute_scalar_loss(p_ans: Any, g_ans: Any, tol: float) -> LossDiagnostic:
    try:
        p_val = float(p_ans)
        g_val = float(g_ans)
    except (TypeError, ValueError):
        # Invalid format
        return LossDiagnostic(normalized_loss=1.0, status="invalid")
        
    raw_err = abs(p_val - g_val)
    if raw_err <= tol:
        return LossDiagnostic(normalized_loss=0.0, raw_error=raw_err, status="valid")
        
    normalizer = max(abs(g_val), 1.0)
    norm_loss = min(raw_err / normalizer, 1.0)
    
    return LossDiagnostic(normalized_loss=norm_loss, raw_error=raw_err, status="valid")


def _compute_topk_loss(p_ans: Any, g_ans: Any) -> LossDiagnostic:
    if not isinstance(p_ans, list) or not isinstance(g_ans, list):
        return LossDiagnostic(normalized_loss=1.0, status="invalid")
        
    def get_key(x):
        if isinstance(x, dict) and len(x) > 0:
            return str(list(x.values())[0]) # usually group_by field is the first or key
        return str(x)
        
    set_p = {get_key(x) for x in p_ans}
    set_g = {get_key(x) for x in g_ans}
    
    if not set_p and not set_g:
        return LossDiagnostic(normalized_loss=0.0, jaccard_distance=0.0, status="valid")
        
    jaccard_sim = len(set_p & set_g) / len(set_p | set_g)
    jaccard_dist = 1.0 - jaccard_sim
    
    return LossDiagnostic(normalized_loss=jaccard_dist, jaccard_distance=jaccard_dist, status="valid")


def _compute_trend_loss(p_ans: Any, g_ans: Any) -> LossDiagnostic:
    if not isinstance(p_ans, list) or not isinstance(g_ans, list):
        return LossDiagnostic(normalized_loss=1.0, status="invalid")
        
    # We treat trend as a set of (bucket, value) mappings
    dict_g = {str(x.get("bucket", "")): float(x.get("value", 0)) for x in g_ans if isinstance(x, dict)}
    
    # If gold is empty, and prediction is empty, 0 loss
    if not dict_g and not p_ans:
        return LossDiagnostic(normalized_loss=0.0, status="valid")
        
    total_g = sum(abs(v) for v in dict_g.values())
    normalizer = max(total_g, 1.0)
    
    dict_p = {}
    try:
        for x in p_ans:
            if isinstance(x, dict):
                dict_p[str(x.get("bucket", ""))] = float(x.get("value", 0))
    except (TypeError, ValueError):
        return LossDiagnostic(normalized_loss=1.0, status="invalid")
        
    # L1 distance over all buckets in the union of keys
    all_keys = set(dict_g.keys()) | set(dict_p.keys())
    l1_err = 0.0
    for k in all_keys:
        l1_err += abs(dict_g.get(k, 0.0) - dict_p.get(k, 0.0))
        
    norm_loss = min(l1_err / normalizer, 1.0)
    return LossDiagnostic(normalized_loss=norm_loss, raw_error=l1_err, status="valid")

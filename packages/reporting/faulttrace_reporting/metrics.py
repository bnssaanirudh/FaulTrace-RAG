import pandas as pd
from typing import List, Dict, Any
from faulttrace_core.models import PipelineRun

class MetricsComputer:
    """Computes aggregate metrics from pipeline runs."""
    
    @staticmethod
    def compute_metrics(runs: List[PipelineRun]) -> Dict[str, Any]:
        if not runs:
            return {}
            
        total_runs = len(runs)
        completed = [r for r in runs if r.status == "completed"]
        correct = [r for r in completed if r.is_correct]
        within_tol = [r for r in completed if r.is_within_tolerance]
        
        total_latency = sum((r.latency_ms or 0) for r in completed)
        total_input_tokens = sum(r.token_estimate_input for r in completed)
        total_output_tokens = sum(r.token_estimate_output for r in completed)
        
        return {
            "total_runs": total_runs,
            "completed": len(completed),
            "correct": len(correct),
            "correctness_rate": len(correct) / len(completed) if completed else 0.0,
            "within_tolerance_rate": len(within_tol) / len(completed) if completed else 0.0,
            "avg_latency_ms": total_latency / len(completed) if completed else 0.0,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens
        }

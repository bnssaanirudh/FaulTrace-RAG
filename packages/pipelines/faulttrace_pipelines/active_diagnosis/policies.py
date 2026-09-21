import math
from typing import Dict, List, Optional
import numpy as np
from .models import DiagnosticState, DiagnosticResult
from .base import DiagnosisPolicy

def _calculate_posterior(obs: Dict[int, float], hypotheses: List[str], template_mean: Dict[str, List[float]], template_std: Dict[str, List[float]]) -> List[float]:
    logp = np.zeros(len(hypotheses), dtype=float)
    
    for hi, h in enumerate(hypotheses):
        mu = template_mean[h]
        sd = template_std[h]
        
        for j, y in obs.items():
            z = (y - mu[j]) / sd[j]
            logp[hi] += (-0.5 * z * z) - math.log(sd[j])
            
    logp -= np.max(logp)
    p = np.exp(logp)
    return (p / p.sum()).tolist()

def run_active_diagnosis(
    row: Dict[str, float], 
    budget: int, 
    policy: DiagnosisPolicy, 
    initial_state: DiagnosticState, 
    template_mean: Dict[str, List[float]], 
    template_std: Dict[str, List[float]], 
    confidence_threshold: float = 0.95
) -> DiagnosticResult:
    """Runs active diagnosis loop using a specified policy."""
    state = initial_state
    
    for step in range(budget):
        j = policy.choose_next_intervention(state)
        if j is None:
            break
            
        # Observe the new intervention
        intervention_name = state.intervention_names[j]
        y = float(row.get(intervention_name, row.get(f"norm__{intervention_name}", 0.0)))
        state.observed[j] = y
        
        # Update posterior
        state.posterior = _calculate_posterior(state.observed, state.hypotheses, template_mean, template_std)
        
        if len(state.observed) >= 3 and max(state.posterior) >= confidence_threshold:
            break
            
    pred_idx = int(np.argmax(state.posterior))
    pred = state.hypotheses[pred_idx]
    
    return DiagnosticResult(
        pred=pred,
        n_probes=len(state.observed),
        confidence=float(max(state.posterior)),
        observed_names=[state.intervention_names[j] for j in state.observed]
    )

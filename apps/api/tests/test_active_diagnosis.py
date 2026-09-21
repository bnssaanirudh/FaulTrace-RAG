import pytest
from faulttrace_pipelines.active_diagnosis.models import DiagnosticState
from faulttrace_pipelines.active_diagnosis.costs import default_intervention_cost
from faulttrace_pipelines.active_diagnosis.greedy import GreedyPolicy
from faulttrace_pipelines.active_diagnosis.bayesian import BayesianPolicy
from faulttrace_pipelines.active_diagnosis.policies import run_active_diagnosis

def test_greedy_policy():
    state = DiagnosticState(
        candidate_indices=[1, 2, 3],
        hypotheses=["R", "E", "A"],
        posterior=[0.33, 0.33, 0.34],
        observed={1: 0.1},
        intervention_sets=[frozenset([]), frozenset(["R"]), frozenset(["E"]), frozenset(["R", "E"])],
        intervention_names=["none", "R", "E", "R_E"]
    )
    
    policy = GreedyPolicy()
    
    # Next intervention should be 2 because 1 is observed
    assert policy.choose_next_intervention(state) == 2
    
    # If all observed, returns None
    state.observed = {1: 0.1, 2: 0.2, 3: 0.3}
    assert policy.choose_next_intervention(state) is None

def test_bayesian_policy():
    template_mean = {
        "R": [0.5, 0.0, 0.5, 0.0],
        "E": [0.5, 0.5, 0.0, 0.0]
    }
    
    cost_fn = lambda x: 1.0
    
    policy = BayesianPolicy(template_mean, cost_fn)
    
    state = DiagnosticState(
        candidate_indices=[1, 2, 3],
        hypotheses=["R", "E"],
        posterior=[0.5, 0.5],
        observed={},
        intervention_sets=[frozenset([]), frozenset(["R"]), frozenset(["E"]), frozenset(["R", "E"])],
        intervention_names=["none", "R", "E", "R_E"]
    )
    
    # For R (idx 1): R is 0.0 for H='R' and 0.5 for H='E'. Disagreement = 0.5 * (0.0 - 0.25)^2 + 0.5 * (0.5 - 0.25)^2 = 0.5 * 0.0625 + 0.5 * 0.0625 = 0.0625.
    # For E (idx 2): R is 0.5 for H='R' and 0.0 for H='E'. Disagreement = 0.0625.
    # For R_E (idx 3): 0.0 for both. Disagreement = 0.0.
    # Both 1 and 2 give the same disagreement (and cost). Max should pick 2.
    assert policy.choose_next_intervention(state) == 2
    
    state.observed = {2: 0.0}
    state.posterior = [1.0, 0.0]
    # No disagreement anymore (posterior is peaked)
    assert policy.choose_next_intervention(state) == 3 # will pick last one since all disagreements are 0.0

def test_run_active_diagnosis():
    template_mean = {
        "R": [0.5, 0.0, 0.5, 0.0],
        "E": [0.5, 0.5, 0.0, 0.0]
    }
    template_std = {
        "R": [0.1, 0.1, 0.1, 0.1],
        "E": [0.1, 0.1, 0.1, 0.1]
    }
    
    state = DiagnosticState(
        candidate_indices=[1, 2, 3],
        hypotheses=["R", "E"],
        posterior=[0.5, 0.5],
        observed={},
        intervention_sets=[frozenset([]), frozenset(["R"]), frozenset(["E"]), frozenset(["R", "E"])],
        intervention_names=["none", "R", "E", "R_E"]
    )
    
    policy = GreedyPolicy()
    
    # Simulate an actual observation row
    row = {"norm__none": 0.5, "norm__R": 0.0, "norm__E": 0.5, "norm__R_E": 0.0}
    
    result = run_active_diagnosis(
        row=row,
        budget=3,
        policy=policy,
        initial_state=state,
        template_mean=template_mean,
        template_std=template_std,
        confidence_threshold=0.99
    )
    
    assert result.pred == "R"
    assert result.n_probes > 0
    assert result.confidence > 0.99

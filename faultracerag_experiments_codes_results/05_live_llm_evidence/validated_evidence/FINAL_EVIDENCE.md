# FaulTrace-RAG Final Experimental Evidence

Repository commit: `433f3d580e0fe526113e94af89398d137ffc84a0`
Generated: 2026-09-14T13:07:45.426710+00:00

## Cross-domain
- Identifiable cases: 343,197
- Mean Shapley F1: 0.9280
- Mean singleton-delta F1: 0.8935
- Mean random F1: 0.5906
- Mean MCR exact recovery: 0.7581

## Bayesian Active Diagnosis
- Best tested budget: 10
- BACD F1: 0.9369
- Mean probes used: 4.55
- Exhaustive Shapley F1: 0.9401
- Exhaustive worlds: 32

## Cost-aware MCR
- See `FINAL_cost_aware_MCR.csv` for ambiguity and savings under all pre-specified cost profiles.

## Certification
- Tested certification settings: 16
- Non-zero held-out coverage settings: 0
- Zero-coverage settings are retained and must not be hidden.

## Manuscript rule
Only PAPER_MODE results should be used in the final headline tables.
Do not select a model, budget, alpha, tolerance, or cost profile using the held-out test set.
# FaulTrace-RAG Live-LLM External Validation

FaulTrace source commit: `433f3d580e0fe526113e94af89398d137ffc84a0`
Generated UTC: 2026-09-14T17:35:38.386260+00:00

## Models
- Qwen2.5-3B-Instruct: `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Phi-4-mini-instruct: `microsoft/Phi-4-mini-instruct` @ `cfbefacb99257ffa30c83adab238a50856ac3083`
- Mistral-7B-Instruct-v0.3: `mistralai/Mistral-7B-Instruct-v0.3` @ `c170c708c41dac9275d15a8fff4eca08d52bab71`

## Experimental rule
- Retrieval and extraction can be repaired counterfactually.
- The final answer model is never replaced with the gold answer.
- Therefore residual loss after R+E is reported as generation/reasoning residual.

## Mistral-7B-Instruct-v0.3
- n: 100
- baseline EM: 0.2000
- baseline F1: 0.3434
- R-repaired F1: 0.4436
- E-repaired F1: 0.3531
- R+E-repaired F1: 0.5451
- natural failure rate: 0.7600
- upstream-repairable rate (all cases): 0.2800
- residual generation-failure rate (all cases): 0.4800
- mean phi_R: 0.1461
- mean phi_E: 0.0556

## Phi-4-mini-instruct
- n: 100
- baseline EM: 0.0100
- baseline F1: 0.1075
- R-repaired F1: 0.1633
- E-repaired F1: 0.1201
- R+E-repaired F1: 0.1449
- natural failure rate: 0.9800
- upstream-repairable rate (all cases): 0.0100
- residual generation-failure rate (all cases): 0.9700
- mean phi_R: 0.0403
- mean phi_E: -0.0029

## Qwen2.5-3B-Instruct
- n: 100
- baseline EM: 0.1700
- baseline F1: 0.1952
- R-repaired F1: 0.2859
- E-repaired F1: 0.3067
- R+E-repaired F1: 0.4097
- natural failure rate: 0.8200
- upstream-repairable rate (all cases): 0.2200
- residual generation-failure rate (all cases): 0.6000
- mean phi_R: 0.0968
- mean phi_E: 0.1176

## Interpretation guardrail
Natural failures do not provide independent component labels automatically.
Use the exported human-audit sheet if the manuscript claims stage-label accuracy on natural failures.
Without human labels, report counterfactual repairability, attribution, residual loss, and efficiency only.
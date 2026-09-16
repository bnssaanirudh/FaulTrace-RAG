# FaulTrace-RAG Live-LLM Natural-Failure Validation

FaulTrace source commit: `433f3d580e0fe526113e94af89398d137ffc84a0`

This experiment evaluates naturally occurring errors from real instruction-tuned
language models on HotpotQA and 2WikiMultihopQA.

Counterfactual repairs are restricted to retrieval (R) and extraction (E).
The answer model is never replaced by a gold-answer oracle, so residual loss
after R+E represents model-side answer synthesis / reasoning failure.

Use LIVE_LLM_FINAL_EVIDENCE.md for the headline experiment summary.
Use the human-audit CSV only after a human has actually filled the label columns.

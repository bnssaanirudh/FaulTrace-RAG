# FaulTrace-RAG Live-LLM External Validation - Audited Set

The manuscript uses **200 validated natural-error cases**: 100 Qwen2.5-3B-Instruct and 100 Mistral-7B-Instruct-v0.3 cases, split evenly across HotpotQA and 2WikiMultiHopQA.

A preliminary Phi-4-mini-instruct run is **excluded from every manuscript statistic** because a post-run audit found a generation-termination configuration mismatch. The preliminary files are retained only for provenance and are clearly labeled `PRELIMINARY_WITH_PHI`.

## Validated results

- Qwen2.5-3B-Instruct: baseline F1 0.1952; joint R+E repair F1 0.4097; paired Wilcoxon p = 1.22e-4; 26.83% of baseline natural failures reach F1 >= 0.80 after joint upstream repair.
- Mistral-7B-Instruct-v0.3: baseline F1 0.3434; joint R+E repair F1 0.5451; paired Wilcoxon p = 8.73e-4; 36.84% of baseline natural failures reach F1 >= 0.80 after joint upstream repair.
- On natural failures, mean (phi_R, phi_E) is (0.142, 0.174) for Qwen and (0.259, 0.111) for Mistral.

## Guardrails

Gold answers are used only for scoring. The same answer model remains active in all four worlds (empty, R, E, RE). Natural failures do not provide independent human component labels, so the manuscript reports repairability and counterfactual response rather than natural-failure localization accuracy.

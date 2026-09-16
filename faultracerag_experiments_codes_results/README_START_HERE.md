# FaulTrace-RAG — Complete Experiments, Code and Results Archive

This package is the reproducibility archive for the final manuscripts.

## Folder map

- `01_notebooks/` — every experiment notebook revision from Notebook A/B through
  the final closure, live-LLM validation fixes, and corrected Phi rerun.
- `02_exported_python_code/` — code-cell-only `.py` exports corresponding to
  each notebook.
- `03_raw_result_archives/` — original result ZIPs produced during the
  experimental progression.
- `04_final_results/` — expanded final closure and master outputs.
- `05_live_llm_evidence/` — validated live-model evidence and original live
  export where available.
- `06_audits_and_manifests/` — independent result/citation checks and Phi
  exclusion rationale.
- `07_final_paper_sources/` — the final manuscript/Overleaf package.
- `08_repository_reference/` — canonical GitHub repository and recorded source
  commit.

## Which results are manuscript-grade?

Use the FINAL_CLOSURE PAPER_MODE results and the validated Qwen/Mistral live
evidence for headline claims. Earlier FAST_MODE results are retained only for
provenance and debugging history.

The preliminary Phi-4-mini run must not be quoted as a final model result
because the initial generation code overrode Phi's multi-EOS stopping
configuration. `FaulTrace_Phi4_Corrected_Rerun.ipynb` is included for a clean
future rerun. The current final manuscripts do not depend on Phi.

## Recomputed headline values

- PAPER_MODE injected cases: 488,250
- Identifiable cases: 343,197
- Shapley fault-set F1: 0.9280
- Singleton/delta F1: 0.8935
- Random F1: 0.5906
- MCR exact recovery: 0.7581
- BACD budget-8 F1: ~0.9314 using ~4.47 mean probes
- Exhaustive active-benchmark Shapley F1: ~0.9401 using 32 worlds
- Validated live evidence: Qwen2.5-3B-Instruct + Mistral-7B-Instruct-v0.3
- Statistical certification sweep: zero non-zero-coverage certified settings

See the included audit files for exact unrounded values.

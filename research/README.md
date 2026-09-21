# Evidence Registry

This directory (`research/`) is the canonical source of truth for all experimental evidence associated with the FaultTrace-RAG manuscript.

## Registry Authority

`EVIDENCE_REGISTRY.json` is the machine-readable authority determining which evidence can appear in the manuscript. The paper-generation script (`scripts/build_manuscript_evidence.py`) consults this registry to select eligible experiments.

## Directory Structure

- `canonical/` — Final paper-mode tables, audited live evidence, manifests, and recomputation audits.
- `exploratory/` — FAST_MODE results, early ablations, and debugging experiments. Not for manuscript use.
- `excluded/` — Preliminary Phi-4 run and known-invalid experiments. Preserved for provenance.
- `archived/` — Historical development iterations.

## Eligibility Criteria

An experiment is manuscript-eligible (`manuscript_eligibility: true`) if:
1. It ran in `PAPER_MODE` or `LIVE_LLM` mode.
2. It was produced from commit `433f3d58`.
3. It passed attribution audit verification.
4. It has not been superseded by a more recent canonical run.

> [!CAUTION]
> Do not manually add experiments to `EVIDENCE_REGISTRY.json`. All modifications must go through a verified audit process to preserve provenance integrity.

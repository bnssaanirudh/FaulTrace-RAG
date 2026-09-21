# FaultTrace-RAG — Research Release Notes

**Version**: 0.1.0  
**Canonical Commit**: `433f3d58`  
**Release Date**: 2026-09-20  
**DOI**: `10.5281/zenodo.1234567`

---

## Summary

This release is the research artifact associated with the FaultTrace-RAG manuscript. It contains the full source code, experiment configurations, canonical paper-mode tables, validated live evidence, and documentation required for reproduction and peer review.

## Included in This Release

- **Core Framework**: Source code for the complete FaultTrace-RAG pipeline (retrieval, extraction, aggregation, Shapley attribution, active diagnosis, repair, and certification).
- **Paper-Mode Tables**: 488,250 controlled fault-injection cases reproduced in `paper/generated/`.
- **Live LLM Validation**: 200 audited Qwen/Mistral natural-failure cases with attributed results.
- **RAGTruth Sweep**: Statistical risk-control sweep results with an honest negative certification finding (max certified coverage = 0.0%).
- **Documentation**: Architecture, threat model, certification scope, and reproducibility guides.
- **CI/CD**: Mandatory GitHub Actions workflows covering Python typing, testing, coverage, and frontend build.

## Excluded from This Release

- `.venv`, `node_modules`, caches, and development artifacts
- All FAST_MODE exploratory outputs (quarantined in `03_raw_result_archives/fast_mode/`)
- Preliminary Phi-4 results (excluded per `paper/PHI_AUDIT_EXCLUSION.md`)
- Raw upstream datasets (see `REPRODUCIBILITY.md` for acquisition instructions)

## Known Limitations

See `docs/PROJECT_STATUS.md` for a detailed component maturity matrix and `paper/limitations.md` for research scope limitations.

## Reproducing Results

See `REPRODUCIBILITY.md` for the step-by-step guide.

## Citing This Work

See `CITATION.cff` for the complete citation metadata.

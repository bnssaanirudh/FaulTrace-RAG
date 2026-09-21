# Research Release Provenance

## Canonical Evaluated Commit
`433f3d580e0fe526113e94af89398d137ffc84a0`

## Canonical Research Tag
`v1.0.0-paper-evaluated`

## Archive / Integration Commit
`cb6d8fd6b528074ed2781d3212f83fb2805cc6ab`

## Provenance Relationship
The canonical evaluated commit (`433f3d580e0fe526113e94af89398d137ffc84a0`) represents the exact state of the source code against which the final PAPER_MODE experiments (September 2026 archive) were executed. 

Later repository and archive integration commits (such as `cb6d8fd6b528074ed2781d3212f83fb2805cc6ab` and subsequent work) may contain documentation, archival packaging, tooling updates, presentation updates, and subsequent theoretical formulations. These later commits **must not be described as having generated the manuscript headline measurements** unless the experiments are fully rerun and recorded. 

Experimental results were generated strictly against commit `433f3d580e0fe526113e94af89398d137ffc84a0`.

## Immutable Experiment Locations
All primary metrics for the manuscript are sourced from the immutable evidence archives rather than recalculated on the fly. 
- Hash manifests are stored within `faultracerag_experiments_codes_results/` and the subsequent DOI archive releases.
- Live-model evidence is located within the validated test sets under the `live-llm/` registry. 

## Model Inclusions and Exclusions
- The manuscript-grade results validate findings across two audited models: **Qwen2.5-3B-Instruct** and **Mistral-7B-Instruct-v0.3**.
- **Phi-4-mini** experiments are classified as preliminary and are **excluded** from the manuscript statistics. 

## Reproducibility
For reproducing or validating the exact results from the manuscript:
1. Check out the canonical tag: `git checkout v1.0.0-paper-evaluated`
2. Install the reproducible environment constraints from `requirements.lock.txt`.
3. Load the corresponding configuration parameters found in the experiment manifests.

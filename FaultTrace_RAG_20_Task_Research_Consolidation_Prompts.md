# FaultTrace-RAG — 20-Task Research Consolidation Prompt Pack

## Global instructions for every task

You are working on the repository:

`bnssaanirudh/FaulTrace-RAG`

Project name:

**FaultTrace-RAG**

The repository contains two important generations of work:

- the August 2026 three-stage R/E/A counterfactual framework and verified deterministic/external experiments;
- the September 2026 expanded experimental program including PAPER_MODE experiments, multi-stage diagnosis, BACD, MCR, RAGTruth analysis, and audited live-LLM experiments.

Important provenance already present in the repository includes:

- evaluated source commit referenced by the September final experiment archive:
  `433f3d580e0fe526113e94af89398d137ffc84a0`
- later repository/archive integration commit:
  `cb6d8fd6b528074ed2781d3212f83fb2805cc6ab`
- PAPER_MODE injected rows: `488,250`
- identifiable cases: `343,197`
- identifiable fraction: approximately `70.29%`
- exact Shapley fault-set F1: approximately `0.9280`
- singleton/delta F1: approximately `0.8935`
- random F1: approximately `0.5906`
- MCR exact recovery: approximately `0.7581`
- BACD budget-8 F1: approximately `0.9314`
- mean BACD probes at budget 8: approximately `4.47`
- exhaustive active-benchmark Shapley F1: approximately `0.9401`
- exhaustive five-stage worlds: `32`
- live-model validated cases: `200`
- live models:
  - Qwen2.5-3B-Instruct: 100
  - Mistral-7B-Instruct-v0.3: 100
- preliminary Phi-4-mini experiment is excluded from manuscript statistics.
- statistical certification sweep contains 16 settings and found zero non-zero-coverage held-out settings.

### General rules

1. Inspect the real code and evidence before changing anything.
2. Never invent experimental numbers.
3. Never silently recompute or reinterpret headline metrics.
4. Never overwrite immutable/raw research artifacts.
5. Preserve negative experimental results.
6. Clearly distinguish:
   - measured results,
   - exploratory results,
   - proposed methods,
   - future work.
7. Do not claim unrestricted causal identification.
8. Do not call source consistency a truth guarantee.
9. Do not claim production readiness where authentication/security remain absent.
10. Add tests for every substantial code change.
11. Run appropriate lint/type/test/build checks after modifications.
12. Update documentation whenever code behavior changes.
13. Preserve provenance hashes, model revisions, dataset identifiers, experiment modes, seeds, and source-commit references.
14. Prefer reusable package modules over notebook-only implementations.
15. Do not remove useful historical research material unless it is safely archived.
16. Maintain backwards compatibility where practical.
17. Keep publication-facing outputs clean and conservative.
18. Every task should end with:
    - files changed,
    - tests executed,
    - tests passed/failed,
    - unresolved issues,
    - scientific implications,
    - next recommended task.

---

# PROMPT 1 — Freeze a canonical evaluated commit and research tag

## Objective

Establish an unambiguous canonical evaluated source version for FaultTrace-RAG so the paper, experiment archive, release package, and repository all identify exactly which source revision produced the manuscript-grade results.

## Tasks

Audit all references to Git commits in:

- `README.md`
- `paper/`
- `docs/`
- `faultracerag_experiments_codes_results/`
- experiment manifests
- final evidence reports
- model manifests
- release scripts
- reproducibility documentation.

Determine which commit corresponds to the code against which the final PAPER_MODE experiments were actually executed.

Current evidence indicates:

`433f3d580e0fe526113e94af89398d137ffc84a0`

was the source commit used by the final September experiment archive.

Do not blindly assume this. Verify it from repository evidence.

Create a canonical research tag proposal such as:

`v1.0.0-paper-evaluated`

or:

`faulttrace-paper-2026-09`

If the environment permits safe tag creation, create an annotated tag pointing to the verified evaluated commit.

The tag annotation should contain:

- project name,
- evaluated commit,
- date,
- experiment families covered,
- paper-mode status,
- statement that later commits may contain documentation/archive integration not used to generate headline numbers.

Add:

`docs/RESEARCH_RELEASE_PROVENANCE.md`

containing:

- canonical evaluated commit;
- canonical research tag;
- archive/integration commit;
- relationship between the two;
- immutable experiment locations;
- hashes/manifests;
- live-model evidence locations;
- Phi exclusion explanation;
- instructions for reproducing or validating the results.

Update relevant documentation so wording follows this convention:

> Experimental results were generated against commit `<evaluated-commit>`. Later repository commits may contain documentation, archival packaging, or presentation updates and must not be described as having generated those measurements unless experiments are rerun.

## Acceptance criteria

- One canonical evaluated commit is documented.
- One canonical research tag exists or is ready to create.
- No document ambiguously calls the current HEAD the source of older results.
- Provenance relationships are clear.
- Existing hashes and evidence remain untouched.
- Relevant tests/build checks still pass.

---

# PROMPT 2 — Unify the August and September research narratives

## Objective

Turn the repository from two partially disconnected research stories into one coherent FaultTrace-RAG research program.

## Current problem

The top-level paper and README predominantly describe the August three-stage:

`R → E → A`

framework.

The September archive contains a substantially expanded program including:

- PAPER_MODE cross-domain studies,
- five-stage/multi-stage fault diagnosis,
- 488,250 injected cases,
- BACD,
- MCR,
- cost-aware repair,
- RAGTruth grounding experiments,
- live Qwen/Mistral natural-failure experiments,
- stricter risk-controlled certification experiments.

## Tasks

Audit:

- `README.md`
- `paper/manuscript.md`
- `paper/results.md`
- `paper/limitations.md`
- `paper/hypothesis_mapping.md`
- `paper/novelty_matrix.md`
- experiment archive READMEs.

Create a single chronological and conceptual research narrative:

### Phase A — deterministic analytical framework
Explain the original R/E/A formulation.

### Phase B — controlled compound-fault localization
Explain synthetic Track-M results and artifact-discrepancy diagnosis.

### Phase C — generalized multi-stage counterfactual diagnosis
Introduce a general component graph rather than presenting R/E/A as universal.

### Phase D — active diagnosis and repair
Introduce BACD and MCR.

### Phase E — external/live-model validation
Introduce retrieval benchmarks, RAGTruth, and audited Qwen/Mistral natural failures.

### Phase F — certification limitations
Preserve both earlier coverage/FCR experiments and the stricter zero-coverage statistical result.

Make sure the new story does not make later exploratory studies look preregistered if they were not.

Add a clearly labeled section:

`Evolution of the research framework`

and explain which analyses are:

- confirmatory;
- exploratory;
- post-hoc;
- external validation;
- engineering validation.

Remove contradictions.

## Acceptance criteria

A reader starting from `README.md` and proceeding into the paper should encounter one coherent research story, not two separate projects.

---

# PROMPT 3 — Generalize R/E/A to an n-stage causal pipeline theory

## Objective

Refactor the theoretical formulation from a hard-coded three-stage system into a general counterfactual diagnosis framework over `n` pipeline components.

## Required theory

Represent a pipeline as:

`C1 → C2 → ... → Cn`

or more generally a DAG:

`G = (V, E)`

where each node is a replaceable pipeline component.

The existing R/E/A formulation should become the simple special case:

`n = 3`.

The newer multihop formulation:

`S → R → E → A → G`

should become an example with:

`n = 5`.

For a subset `S ⊆ V`, define an oracle intervention:

`do(C_i = C_i*)`

for every selected component.

Define:

`L(S)`

as the downstream loss after replacing components in subset `S` and replaying their descendants.

Define:

`v(S) = L(∅) - L(S)`.

Then define exact Shapley attribution for component `i`:

`φ_i = Σ [ |S|!(n-|S|-1)! / n! ] [v(S∪{i}) - v(S)]`

over subsets not containing `i`.

Discuss complexity:

`2^n`

counterfactual worlds for exhaustive evaluation.

Introduce why this motivates BACD.

## Engineering tasks

Refactor appropriate core abstractions so the system can represent arbitrary ordered/DAG stages without breaking the current R/E/A API.

Add objects resembling:

- `PipelineStage`
- `PipelineGraph`
- `InterventionSet`
- `CounterfactualWorld`
- `StageOracle`
- `StageReplayBoundary`

Do not prematurely rewrite every existing pipeline.

Build compatibility adapters that map:

- R → retrieval/scope
- E → extraction
- A → aggregation.

Add unit tests for:

- n=1
- n=2
- n=3
- n=5
- invalid DAG
- missing oracle
- intervention on upstream stage with downstream replay
- complete-lattice enumeration.

Document exponential complexity.

## Acceptance criteria

Theoretical documentation and code abstractions support arbitrary stage counts while the original three-stage experiments remain reproducible.

---

# PROMPT 4 — Integrate the 488,250-case PAPER_MODE results

## Objective

Promote the September PAPER_MODE cross-domain evidence into the canonical research results documentation.

## Verified numbers to cross-check

Expected evidence includes approximately:

- total PAPER_MODE cases: `488,250`
- identifiable cases: `343,197`
- identifiable fraction: `0.702912...`
- exact Shapley fault-set F1: `0.928007...`
- singleton F1: `0.893530...`
- random F1: `0.590619...`
- MCR exact recovery: `0.758095...`
- MCR residual: approximately `1.762e-05`.

Do not type these values from this prompt directly into the paper.

Read them from the immutable audit/evidence files and use those as the source of truth.

## Tasks

Update:

`paper/results.md`

Add a dedicated section:

`Large-scale PAPER_MODE counterfactual validation`

Report:

- number of cases;
- number and fraction identifiable;
- datasets/domains;
- number of seeds;
- stage graph;
- fault families;
- fault severities;
- attribution methods;
- matched baselines;
- confidence intervals/statistical tests where available;
- MCR outcomes.

Clearly define what `identifiable` means.

Do not compute headline F1 only on a favorable subset without explicitly saying so.

Explain both:

`488,250 total`

and:

`343,197 identifiable`.

Add a table comparing:

- exact Shapley;
- singleton/delta;
- random;
- active diagnosis where appropriate.

Add links to immutable evidence and recomputation audit.

Update the manuscript but preserve the earlier August Track-M results as controlled engineering validation rather than deleting them.

## Acceptance criteria

The paper's headline results accurately reflect the strongest manuscript-grade PAPER_MODE experiment while preserving provenance and scope.

---

# PROMPT 5 — Integrate the 200 audited Qwen/Mistral natural-failure cases

## Objective

Promote the audited live-model study into the primary external-validity section of the paper.

## Validated set

The manuscript-grade dataset consists of:

- 100 Qwen2.5-3B-Instruct cases;
- 100 Mistral-7B-Instruct-v0.3 cases;
- balanced across HotpotQA and 2WikiMultiHopQA.

Phi-4 preliminary results are excluded.

## Tasks

Use only audited live-model files.

Create a paper section:

`Live-LLM natural-failure validation`

Explain the four evaluated worlds:

- baseline;
- R repair;
- E repair;
- joint R+E repair.

Make clear that:

- the same answer model is used across all worlds;
- gold answers are used only for evaluation;
- natural failures do not provide independent human component-fault labels;
- therefore the experiment evaluates repairability and intervention response, not supervised localization accuracy.

Report model-specific:

- baseline EM/F1;
- repaired EM/F1;
- paired significance tests;
- natural failure count;
- repairability fraction;
- average counterfactual contributions;
- latency/tokens if validated.

Explicitly state the Phi exclusion and reason.

Add a pointer to:

`PHI_AUDIT_EXCLUSION.md`

Do not mix preliminary Phi outputs into averages.

## Acceptance criteria

The live-LLM study is clearly presented as external validation without overstating fault-localization accuracy.

---

# PROMPT 6 — Promote BACD and MCR into first-class library modules

## Objective

Move Bayesian Active Counterfactual Diagnosis and Minimum Counterfactual Repair out of notebook-only research code into maintainable package modules.

## Target location

Create production-quality modules under:

`packages/pipelines/faulttrace_pipelines/`

Suggested structure:

`active_diagnosis/`

containing:

- `base.py`
- `greedy.py`
- `bayesian.py`
- `policies.py`
- `costs.py`
- `models.py`

and:

`repair/`

containing:

- `mcr.py`
- `cost_aware.py`
- `models.py`
- `objectives.py`.

## BACD requirements

Represent:

- candidate stages;
- posterior/belief state;
- available interventions;
- intervention costs;
- probe budget;
- stopping criteria;
- observed loss;
- predicted informativeness;
- chosen next probe.

Support:

- deterministic greedy policy;
- Bayesian policy;
- explicit budget;
- audit trace;
- reproducible seed.

## MCR requirements

Implement:

`S* = argmin_S C(S)`

subject to acceptable residual loss, or the equivalent weighted objective:

`L(S) + λC(S)`.

Support:

- intervention count cost;
- stage-specific costs;
- latency cost;
- model-token cost;
- custom user cost vectors.

## Testing

Create deterministic fixtures reproducing known notebook examples.

Test:

- one faulty stage;
- multiple faults;
- equal-cost ties;
- infeasible repairs;
- residual loss;
- budget exhaustion;
- deterministic reproducibility;
- no-fault baseline.

Do not make package tests depend on heavyweight model downloads.

## Acceptance criteria

The notebooks import BACD/MCR from package code rather than containing separate algorithm implementations.

---

# PROMPT 7 — Preserve zero-coverage statistical certification honestly

## Objective

Ensure the repository does not misrepresent selective certification after the stricter September experiment found no nonzero-coverage setting satisfying the requested held-out constraints.

## Tasks

Audit every usage of terms including:

- certified;
- certification;
- guarantee;
- safe;
- reliable;
- risk-controlled;
- zero false certification;
- semantic certificate.

Make clear there are at least two different experiments:

### Earlier empirical structured-semantic certificate
Observed zero false certification in specific controlled experiments, often at substantial coverage cost.

### Later statistical risk-control sweep
Sixteen tested settings resulted in:

`maximum held-out certified coverage = 0`.

The latter means the stronger requested statistical constraint was not met at useful coverage.

Rewrite relevant claims so that:

- empirical zero observed errors are not mathematical guarantees;
- controlled structured certificates remain valid as engineering measurements;
- statistical certification is reported as a negative result;
- no nonzero-coverage guarantee is claimed.

Add:

`docs/CERTIFICATION_SCOPE.md`

with a matrix explaining:

| Mechanism | Guarantee | Coverage | Dataset | Limitation |
|---|---|---|---|---|

Include:

- structural certificate;
- structured-semantic v2;
- RAGBench lexical/numeric certificate;
- RAGTruth risk-controlled experiment.

## Acceptance criteria

No publication-facing document can be interpreted as claiming a statistically guaranteed nonzero-coverage correctness certificate.

---

# PROMPT 8 — Archive FAST_MODE outputs

## Objective

Ensure FAST_MODE experiments remain available for provenance/debugging but cannot be accidentally cited as manuscript-grade evidence.

## Tasks

Audit all experiment directories.

Classify artifacts into:

- `paper_mode`
- `fast_mode`
- `preliminary`
- `excluded`
- `exploratory`
- `canonical`.

Create:

`faultracerag_experiments_codes_results/PROVENANCE_INDEX.md`

Move or logically reorganize FAST_MODE outputs into a clearly marked archive such as:

`03_raw_result_archives/fast_mode/`

Do not destroy files.

Add banners/README warnings:

> FAST_MODE results are exploratory/debugging outputs and must not be used for manuscript headline claims.

Add machine-readable metadata:

`artifact_classification.json`

containing:

- path;
- experiment mode;
- manuscript eligibility;
- source commit;
- reason;
- superseded by.

Modify table-generation scripts so FAST_MODE outputs are rejected unless an explicit override flag is supplied.

## Acceptance criteria

A manuscript-generation script cannot accidentally consume FAST_MODE evidence.

---

# PROMPT 9 — Remove stale Prompt-N documents from publication-facing structure

## Objective

Remove visible development artifacts that describe implementation as “Prompt 1”, “Prompt 3”, “Prompt 10”, etc., while retaining history for provenance.

## Tasks

Locate:

- `prompt-*`
- `Prompt N`
- completion reports
- preflight files
- implementation-stage notes.

Classify each file:

1. valuable technical design;
2. redundant development log;
3. obsolete;
4. useful historical provenance.

Move useful history into:

`archive/development_history/`

or equivalent.

Rewrite valuable content into domain-based documentation:

- architecture;
- methodology;
- experiments;
- APIs;
- reproducibility;
- testing;
- security.

Remove references such as:

> implemented in Prompt 3

and replace them with meaningful descriptions such as:

> implemented in the provider/retrieval subsystem.

Update hyperlinks.

Do not rewrite Git history.

## Acceptance criteria

A reviewer or recruiter browsing the repository should see a coherent engineering/research project rather than prompt-by-prompt development logs.

---

# PROMPT 10 — Rewrite `docs/ARCHITECTURE.md`

## Objective

Replace the stale architecture document with an accurate representation of the current system.

## Required sections

### 1. System purpose

Explain FaultTrace-RAG as a counterfactual diagnosis and repair framework.

### 2. Logical layers

Document:

- data;
- query specification;
- gold engine;
- pipeline runtime;
- retrieval;
- extraction;
- aggregation/generation;
- counterfactual engine;
- active diagnosis;
- repair;
- certification;
- reporting;
- API;
- frontend;
- artifact storage.

### 3. Current pipeline graph

Explain both:

`R → E → A`

and generalized:

`C1 → ... → Cn`

plus:

`S → R → E → A → G`

as an experimental five-stage example.

### 4. Counterfactual execution

Explain descendant replay.

### 5. Artifact lineage

Explain run artifacts, hashes, manifests, immutable bundles.

### 6. Certification

Explain what is and is not certified.

### 7. Experiment architecture

Separate:

- controlled synthetic;
- external retrieval;
- deterministic mechanics;
- PAPER_MODE;
- live-LLM;
- RAGTruth.

### 8. Service architecture

FastAPI + Next.js + SQLite/Parquet.

### 9. Security boundary

Local-only by default.

### 10. Extension points

Document how to add:

- new stage;
- new retriever;
- new oracle;
- new provider;
- new dataset;
- new active diagnosis policy;
- new repair objective.

Use Mermaid diagrams where helpful.

## Acceptance criteria

Nothing in the architecture document calls implemented functionality “future work.”

---

# PROMPT 11 — Replace the false `100% COMPLETE` statement

## Objective

Make project status scientifically honest.

## Tasks

Replace:

`OVERALL STATUS: 100% COMPLETE`

with a maturity matrix.

Suggested categories:

- core engine;
- reproducibility;
- controlled validation;
- large-scale injected validation;
- external retrieval;
- natural LLM validation;
- compound-fault diagnosis;
- active diagnosis;
- repair;
- certification;
- production security;
- documentation;
- release engineering.

Use statuses such as:

- Complete for current research scope
- Validated
- Partial
- Experimental
- Planned
- Out of scope

Do not use fake precision percentages unless directly measured.

Create:

`docs/PROJECT_STATUS.md`

Differentiate:

### Research prototype completion

from:

### Production deployment readiness

and:

### Manuscript evidence maturity.

Update `docs/BUILD_STATE.md` to link to the new matrix.

## Acceptance criteria

The repository no longer communicates that every research or production problem has been solved.

---

# PROMPT 12 — Standardize FaultTrace-RAG naming

## Objective

Eliminate inconsistent naming such as:

- FaulTrace-RAG
- FaultTrace-RAG
- FaulTrace
- FaultTrace.

## Tasks

Use the canonical human-facing name:

`FaultTrace-RAG`.

Use package/module prefix:

`faulttrace_`

where already established.

Audit:

- README;
- docs;
- paper;
- UI;
- notebook titles;
- generated figures;
- manifests;
- Docker labels;
- API metadata;
- CLI help;
- package metadata.

Do not rename immutable historical artifacts when doing so would break hashes or provenance.

Instead maintain a compatibility note:

> Historical experiment artifacts may retain the legacy “FaulTrace” spelling. Their filenames are preserved to maintain checksum and provenance integrity.

If renaming the GitHub repository itself is desired, document migration instructions but do not break external links unnecessarily.

Add one canonical branding statement.

## Acceptance criteria

All active/current publication-facing content uses `FaultTrace-RAG`.

---

# PROMPT 13 — Make CI typing, lint and coverage real gates

## Objective

Convert informational checks into enforceable release-quality gates.

## Tasks

Inspect:

`.github/workflows/ci.yml`

Remove inappropriate:

`continue-on-error: true`

from checks that are supposed to define a successful release.

Targets:

### Python

- Ruff must pass.
- Mypy must pass for the enforced typed surface.
- Pytest must pass.
- Coverage threshold must fail CI if below threshold.

Do not suddenly require repository-wide strict mypy if the code does not pass.

Instead define explicit staged targets.

For example:

Phase 1 mandatory:

- core
- gold
- counterfactual engine
- active diagnosis
- repair.

Then expand coverage over time.

### Frontend

- TypeScript check mandatory.
- ESLint mandatory.
- production build mandatory.

### Release gate

Require all upstream jobs.

Add:

- dependency caching;
- Python 3.11 and 3.12;
- Node 20;
- artifact upload for coverage reports if useful.

Add badge/status documentation to README only if workflow actually passes.

## Acceptance criteria

A broken lint/type/test/coverage/build condition causes CI failure.

---

# PROMPT 14 — Fix `pytest-timeout` and dependency hygiene

## Objective

Ensure a clean environment can run the test command exactly as CI invokes it.

## Tasks

Inspect CI for:

`--timeout=120`.

Verify whether `pytest-timeout` is explicitly installed.

If not, add it to development dependencies.

Search for other implicit tools/plugins used but not declared.

Examples to inspect:

- pytest-timeout;
- pytest-cov;
- pytest-asyncio;
- Playwright;
- Ruff;
- mypy;
- Alembic;
- Node scripts.

Add a dependency-verification CI step using a fresh environment.

Run:

`pytest --help`

and verify timeout options exist.

Ensure no required tool is present only because of transitive dependencies.

## Acceptance criteria

Fresh installation reproduces CI without unknown pytest flags or undeclared development tools.

---

# PROMPT 15 — Consolidate dependency and package installation

## Objective

Replace fragmented setup procedures with one canonical installation path.

## Target experience

A developer should be able to run:

`pip install -e ".[dev]"`

or equivalent.

## Tasks

Use root `pyproject.toml` as the canonical Python package metadata source.

Define extras such as:

- `dev`
- `research`
- `api`
- `retrieval`
- optional `llm`
- optional `gpu` if appropriate.

Avoid pinning huge GPU stacks unnecessarily.

Align:

- root editable install;
- package discovery;
- CLI entry points;
- FastAPI app dependencies;
- experiment dependencies.

Retire or auto-generate redundant requirements files if practical.

If requirements files remain for deployment reasons, explain which one is authoritative.

Update:

- README;
- Docker;
- CI;
- Makefile/task scripts.

Add a clean installation smoke test.

## Acceptance criteria

One documented installation path installs all code needed for development and testing.

---

# PROMPT 16 — Add one command to regenerate all manuscript tables

## Objective

Create a deterministic manuscript-evidence build command.

## Desired interface

Something similar to:

`python scripts/build_manuscript_evidence.py`

or:

`make paper`

or:

`faulttrace research build-paper-evidence`

## Requirements

The command must:

1. locate canonical immutable experiment artifacts;
2. verify hashes;
3. verify source commit metadata;
4. reject FAST_MODE;
5. reject preliminary Phi results;
6. load PAPER_MODE cross-domain evidence;
7. load BACD results;
8. load MCR results;
9. load live Qwen/Mistral validation;
10. load RAGTruth results;
11. load certification sweep;
12. recompute reported summary metrics;
13. compare recomputed values against recorded audit values;
14. fail on mismatch beyond defined tolerance;
15. generate publication tables;
16. generate machine-readable JSON;
17. optionally generate figures;
18. write a provenance manifest.

Suggested output:

`paper/generated/`

containing:

- `table_main_results.csv`
- `table_active_diagnosis.csv`
- `table_mcr.csv`
- `table_live_llm.csv`
- `table_ragtruth.csv`
- `table_certification.csv`
- `metrics.json`
- `provenance.json`.

Generated tables should not be manually edited.

## Acceptance criteria

The entire manuscript evidence set can be regenerated using one deterministic command.

---

# PROMPT 17 — Create a clean research release and DOI-ready archive

## Objective

Build a publication-grade release artifact suitable for Zenodo/Figshare/OSF DOI archiving.

## Tasks

Create a release builder that includes:

- source code;
- canonical evaluated commit metadata;
- environment specification;
- experiment configurations;
- paper-mode tables;
- validated live evidence;
- generated manuscript evidence;
- documentation;
- model revision manifests;
- dataset acquisition instructions;
- licenses;
- checksums.

Exclude:

- secrets;
- `.env`;
- caches;
- node_modules;
- `.venv`;
- huge upstream datasets where redistribution is not permitted;
- excluded preliminary results from headline folders;
- temporary artifacts.

Add:

- `CITATION.cff`
- `codemeta.json` if useful;
- `REPRODUCIBILITY.md`;
- `RELEASE_NOTES.md`;
- `LICENSES/` or third-party notices;
- dataset-license notes.

Generate:

`dist/faulttrace-rag-research-release-<version>.zip`

and:

`dist/SHA256SUMS`.

Document exact steps for uploading to a DOI archive.

Do not fabricate a DOI.

Provide a placeholder field to insert the DOI after archival service assignment.

## Acceptance criteria

The produced archive can be handed to a reviewer and independently inspected without needing the development working tree.

---

# PROMPT 18 — Separate publication evidence from exploratory notebooks

## Objective

Make it impossible to confuse experimental development notebooks with manuscript evidence.

## Proposed structure

Create:

`research/`

with:

- `canonical/`
- `exploratory/`
- `excluded/`
- `archived/`.

Or adapt the existing archive while preserving hashes.

Canonical should contain only:

- final paper-mode tables;
- final evidence summaries;
- audited live evidence;
- manifests;
- recomputation audits.

Exploratory should contain:

- notebook iterations;
- FAST_MODE;
- early ablations;
- debugging experiments.

Excluded should contain:

- preliminary Phi run;
- known-invalid experiments;
- superseded evidence.

Create:

`research/EVIDENCE_REGISTRY.json`

Each experiment should include:

- experiment ID;
- title;
- status;
- source commit;
- datasets;
- model revisions;
- mode;
- manuscript eligibility;
- replacement/supersession relation;
- reason for inclusion/exclusion.

Update paper-generation scripts to consume the registry rather than hard-coded random paths.

## Acceptance criteria

There is one machine-readable authority indicating which evidence can appear in the manuscript.

---

# PROMPT 19 — Add independent human stage-level annotations

## Objective

Create a human-annotated natural-failure benchmark that allows real localization accuracy to be evaluated rather than only repairability.

## Scope

Sample a sufficiently diverse subset from the 200 audited Qwen/Mistral live cases.

Target at least:

`50–100 cases`

if practical.

Stratify by:

- model;
- dataset;
- baseline severity;
- repair success;
- retrieval behavior;
- multihop complexity.

## Annotation taxonomy

Use independent labels for:

- retrieval/scope failure;
- extraction/evidence failure;
- aggregation/reasoning failure;
- generation failure;
- mixed/compound;
- insufficient information;
- annotation uncertain.

If the generalized framework uses:

`S → R → E → A → G`

define each stage precisely.

## Annotation protocol

Create:

`docs/HUMAN_ANNOTATION_PROTOCOL.md`

containing:

- annotation instructions;
- examples;
- edge cases;
- allowed multilabel combinations;
- confidence rating;
- evidence fields;
- adjudication policy.

Use at least two independent annotators where possible.

Calculate:

- raw agreement;
- Cohen's kappa for appropriate categorical decisions;
- multilabel agreement metric where relevant;
- adjudicated gold labels.

Never silently treat an LLM judge as a human annotation.

Create a clean CSV/JSONL annotation schema.

Then evaluate FaultTrace localization against the adjudicated labels.

Separate:

- intervention response;
- localization accuracy;
- annotator agreement.

## Acceptance criteria

The project has at least one natural-failure dataset with independent component-level labels suitable for evaluating fault localization.

---

# PROMPT 20 — Add a strong matched contemporary diagnostic baseline

## Objective

Compare FaultTrace against at least one strong recent RAG diagnostic/evaluation baseline under matched conditions.

## Baseline selection

Start by reviewing technically appropriate baselines already identified in the repository's novelty matrix, such as:

- RAGChecker;
- ARES;
- RAGAS component metrics;
- suitable judge-based component diagnosis;
- a contemporary attribution/diagnosis method compatible with the task.

Do not choose a weak baseline just because it is easier to beat.

## Matching requirements

The comparison must use the same:

- dataset;
- queries;
- answer outputs;
- evidence;
- model outputs;
- fault conditions;
- number of cases;
- evaluation labels;
- token budget where applicable;
- intervention/probe budget where applicable.

If the baseline does not naturally produce stage labels, define a transparent conversion rule before evaluation.

Do not tune the baseline on test labels.

## Metrics

Report:

- exact fault-set accuracy;
- macro F1;
- micro F1;
- per-stage precision/recall/F1;
- compound-fault performance;
- latency;
- token usage;
- number of probes/interventions;
- monetary cost where measurable.

For active diagnosis include budget-matched comparison:

`FaultTrace BACD vs baseline at equal probe/cost budget`.

For exhaustive diagnosis include:

`FaultTrace Shapley vs baseline with access to the same cases`.

Statistical analysis should use paired tests where possible.

Add bootstrap confidence intervals.

## Scientific requirements

Discuss where the baseline wins.

Do not claim superiority if differences are insignificant.

If FaultTrace's advantage comes from oracle access unavailable to the baseline, state this explicitly.

## Deliverables

Create:

- reproducible baseline adapter;
- test suite;
- experiment configuration;
- comparison table;
- statistical analysis;
- failure cases;
- methodology documentation;
- manuscript subsection.

## Acceptance criteria

The final paper includes at least one serious contemporary matched baseline comparison that a reviewer would consider technically fair.

---

# Final consolidation prompt

After completing Prompts 1–20, perform a full repository audit.

Verify that:

- one canonical evaluated release exists;
- all headline numbers come from immutable evidence;
- the generalized theory matches the implemented experiments;
- BACD and MCR are library components;
- live-model validation is integrated;
- Phi preliminary results remain excluded;
- FAST_MODE cannot contaminate manuscript generation;
- risk-controlled certification is honestly described as currently yielding zero useful held-out coverage under the tested settings;
- all publication-facing documentation uses FaultTrace-RAG;
- stale Prompt-N language is archived;
- architecture is current;
- CI is actually enforceable;
- dependencies install cleanly;
- one command regenerates manuscript evidence;
- research evidence is separated from exploratory work;
- human annotations are available for a natural-failure subset;
- at least one strong matched contemporary baseline has been evaluated;
- all tests pass;
- README, manuscript, results, limitations, architecture, provenance, and release notes agree with one another.

Finally produce:

`FINAL_RESEARCH_READINESS_AUDIT.md`

with:

1. implemented contributions;
2. experimental evidence;
3. externally validated contributions;
4. negative findings;
5. remaining weaknesses;
6. reproducibility status;
7. release status;
8. paper claim-to-evidence matrix;
9. unresolved reviewer risks;
10. exact recommended claims for abstract/conclusion;
11. claims that must not be made;
12. remaining tasks before submission.

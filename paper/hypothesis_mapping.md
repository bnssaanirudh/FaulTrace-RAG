# Hypothesis status after the 2026-08-28 controlled run

The preregistered confirmatory protocol remains in [preregistration.md](preregistration.md).
The statuses below apply only to the deterministic synthetic Track-M audit.

- **H1 — Scale interaction: not tested confirmatorily.** Scale-stratified runs exist, but
  the planned matched model and multiplicity-corrected interaction analysis has not been
  completed. No scale-effect claim is made.
- **H2 — Matched repair benefit: not tested.** There is no matched repair/non-repair design
  with measured model cost. Existing P4/P5 labels are fault conditions, not a repair trial.
- **H3 — Single-fault localization: provisionally supported in the controlled audit.**
  Dominant-component attribution identified 40/60 single injected faults (0.667), with a
  query-clustered 95% bootstrap interval of [0.534, 0.789] and macro-F1 of 0.786. External
  baselines and real-model replication are still required. An exploratory stage-matched
  artifact diagnostic reached 0.817 exact-set accuracy [0.714, 0.915] and 0.901 macro-F1,
  but it was added after the original audit and requires oracle access.
- **H4 — Compound-fault localization: exploratory partial evidence, not confirmatory
  support.** Cross-fitted Shapley multilabeling improved overall exact-set accuracy from
  0.500 to 0.600 and macro-F1 from 0.646 to 0.746. Stage-matched artifact diagnosis reached
  0.550 and 0.350 exact-set accuracy on the two- and three-fault groups. These analyses were
  developed after inspecting the earlier controlled run, so H4 requires a sealed external
  evaluation before its status can change to supported.
- **H5 — Semantic certification: provisionally supported in the controlled audit.** The
  observed false-certification rate fell from 0.697 to 0.000 for P2 and from 0.852 to 0.000
  for P3, while coverage fell from 0.908 to 0.150 and from 0.958 to 0.142. This demonstrates
  a risk/coverage trade-off, not a correctness guarantee.

Only immutable bundle results may change these statuses. External datasets, real language
models, strong baselines, and a sealed confirmatory evaluation remain mandatory.

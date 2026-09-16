# Certification semantics

FaultTrace exposes two distinct policies. Their names and claims must remain separate in
the API, artifacts, figures, and manuscript.

## Structural coverage policy

The structural policy checks evidence membership, expected-scope recall and precision,
extraction completeness, required-field presence, ambiguity, truncation, and
operator-specific conditions such as denominator, ranking-boundary, and time-bucket
completeness. It makes no claim about the truth of extracted values or the correctness of
the final arithmetic.

## Structured-semantic policy

For datasets with immutable source records and stable `record_id` lineage, the v2 policy
adds three gold-independent guards:

1. **Provenance coverage:** every extracted row maps to exactly one immutable source row.
2. **Source fact fidelity:** required extracted fields recursively agree with their source
   values under the query tolerance.
3. **Aggregation replay:** deterministic execution of the declared aggregation over the
   persisted extraction reproduces the pipeline answer.

For an answer to be certified under the strict v2 policy,

\[
C_{v2}=C_{structural}\land C_{provenance}\land C_{facts}\land C_{aggregation}.
\]

The certificate never reads the gold answer. Gold is used only after execution to measure
false certification and selective risk. If source lineage or deterministic replay is not
available, the semantic dimensions are unknown and strict v2 withholds certification.

## Scope of the claim

Structured-semantic certification detects disagreement with the supplied immutable source
and declared aggregation. It does not prove that the source is true, current, unbiased, or
safe; that the query specification represents user intent; or that unstructured textual
entailment is correct. Therefore the valid claim is **source-and-program consistency for
structured analytical runs**, not answer truth or general RAG safety.

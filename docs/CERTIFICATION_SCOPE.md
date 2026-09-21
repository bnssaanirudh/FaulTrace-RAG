# Certification Scope and Limitations

This document explicitly defines the scope, guarantees, and limitations of the certification mechanisms implemented and evaluated in the FaultTrace-RAG repository.

> [!WARNING]
> No mechanism in this repository provides a mathematically guaranteed, nonzero-coverage "truth certificate." All zero-error observations are empirical operating points bounded to specific experimental configurations.

## Certification Matrix

| Mechanism | Guarantee | Coverage | Dataset | Limitation |
|---|---|---|---|---|
| **Structural certificate** | Formatting, syntax, and presence of mandatory fields. | High | Track-M (synthetic) | Fails to detect semantic errors; frequently falsely certifies factually incorrect aggregations. |
| **Structured-semantic v2** | Source-and-program consistency with zero *observed* false certifications. | Very Low (e.g., 14.2%) | Track-M (synthetic) | This is an **empirical operating point**, not a mathematical guarantee. Coverage drops severely as the constraints tighten. |
| **RAGBench lexical/numeric certificate** | Minimum token overlap or exact numeric match against source. | Modest (e.g., ~30.5%) | RAGBench / COVID-QA | Narrow source-consistency policy. Validation-calibrated thresholds do not transfer perfectly (e.g., 9.3% false-certification on test data). Not a truth guarantee. |
| **RAGTruth risk-controlled experiment** | Statistical risk control. | **Zero** | RAGTruth | **Negative result**: Across 16 tested settings, the maximum held-out certified coverage was exactly zero. The stricter statistical risk-control constraint could not be met at any useful coverage. |

## Interpretation

The statistical risk-controlled sweep demonstrates a negative result: stricter mathematical bounds on error rates lead directly to zero coverage. As such, the repository's mechanisms act as **engineering and structural constraints** rather than absolute, risk-controlled mathematical guarantees. 

Any reference to "zero false certification" in the paper or documentation refers exclusively to an empirical finding within the `Structured-semantic v2` controlled evaluation, and should never be extrapolated as a generalized safety or truth guarantee.

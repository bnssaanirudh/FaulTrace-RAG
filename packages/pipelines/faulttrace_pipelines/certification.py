"""
Certification Engine.

Applies Answer Policies to Coverage Observations to generate Coverage Certificates.
"""

from faulttrace_core.models import (
    AnswerPolicyConfig,
    CoverageCertificate,
    CoverageDecision,
    CoverageObservation,
    EvidenceRequirement,
    PipelineRun,
    ProportionSpec,
    QuerySpec,
    ReasonCode,
    TopKSpec,
    TrendSpec,
)


class CertificationEngine:
    """Evaluates observations against requirements according to a policy."""

    def __init__(self, policy: AnswerPolicyConfig):
        self.policy = policy

    def certify(
        self, run: PipelineRun, query: QuerySpec, obs: CoverageObservation
    ) -> CoverageCertificate:
        req = EvidenceRequirement.from_query(query)

        ratios = {}
        unknowns = []
        codes = []

        # 1. Scope Coverage
        if req.requires_full_scope:
            if (
                obs.scope_membership_known
                and obs.eligible_set_size_known
                and obs.eligible_set_size is not None
                and obs.eligible_set_size > 0
            ):
                scope_coverage = obs.eligible_record_ids_covered / obs.eligible_set_size
                ratios["scope_coverage"] = scope_coverage
                scope_precision = (
                    obs.eligible_record_ids_covered
                    / (obs.eligible_record_ids_covered + obs.unexpected_record_ids)
                    if obs.eligible_record_ids_covered + obs.unexpected_record_ids > 0
                    else 0.0
                )
                ratios["scope_precision"] = scope_precision
                if (
                    scope_coverage < self.policy.min_known_scope_coverage
                    or scope_precision < self.policy.min_known_scope_coverage
                ):
                    codes.append(ReasonCode.SCOPE_COVERAGE_BELOW_REQUIRED)
            elif obs.scope_membership_known and obs.eligible_set_size == 0:
                # Legitimate empty scope
                empty_scope_valid = obs.unexpected_record_ids == 0
                ratios["scope_coverage"] = 1.0 if empty_scope_valid else 0.0
                ratios["scope_precision"] = 1.0 if empty_scope_valid else 0.0
                if not empty_scope_valid:
                    codes.append(ReasonCode.SCOPE_COVERAGE_BELOW_REQUIRED)
            else:
                unknowns.append("scope_coverage")
                codes.append(ReasonCode.SCOPE_COVERAGE_UNKNOWN)

        # 2. Extraction Completeness
        if obs.retrieved_units > 0:
            extraction_completeness = obs.extracted_valid_rows / obs.retrieved_units
            ratios["extraction_completeness"] = extraction_completeness
            if extraction_completeness < self.policy.min_extraction_completeness:
                codes.append(ReasonCode.EXTRACTION_ROWS_MISSING)
        elif obs.eligible_set_size == 0:
            ratios["extraction_completeness"] = 1.0
        else:
            unknowns.append("extraction_completeness")

        # 3. Required Fields
        if obs.extracted_valid_rows > 0:
            field_completeness = (
                obs.extracted_valid_rows - obs.missing_required_fields
            ) / obs.extracted_valid_rows
            ratios["field_completeness"] = field_completeness
            if field_completeness < self.policy.min_required_field_completeness:
                codes.append(ReasonCode.REQUIRED_FIELD_MISSING)

        # 4. Ambiguity and failed extraction rows
        ambiguity_denominator = max(obs.retrieved_units, 1)
        ambiguity_ratio = obs.ambiguous_rows / ambiguity_denominator
        ratios["ambiguity_ratio"] = ambiguity_ratio
        if ambiguity_ratio > self.policy.max_ambiguous_tolerance:
            codes.append(ReasonCode.EXTRACTION_AMBIGUOUS)
        if obs.failed_rows > self.policy.max_repair_failures:
            codes.append(ReasonCode.EXTRACTION_ROWS_MISSING)

        # 5. Context completeness
        if obs.truncation_count > 0 or obs.dropped_context_count > 0:
            codes.append(ReasonCode.CONTEXT_TRUNCATED)

        # 6. Operator-specific evidence requirements
        if isinstance(query.aggregation_spec, ProportionSpec):
            if not obs.denominator_evaluable or not obs.numerator_evaluable:
                codes.append(ReasonCode.DENOMINATOR_INCOMPLETE)

        if isinstance(query.aggregation_spec, TopKSpec):
            ranking = obs.ranking_candidate_completeness
            if ranking is None:
                unknowns.append("ranking_candidate_completeness")
                codes.append(ReasonCode.RANKING_DOMAIN_INCOMPLETE)
            else:
                ratios["ranking_candidate_completeness"] = ranking
                if ranking < 1.0:
                    codes.append(ReasonCode.RANKING_DOMAIN_INCOMPLETE)
            if self.policy.require_ranking_boundary_confidence and not obs.tie_boundary_completeness:
                codes.append(ReasonCode.TIE_BOUNDARY_UNRESOLVED)

        if isinstance(query.aggregation_spec, TrendSpec):
            time_coverage = obs.time_bucket_completeness
            if time_coverage is None:
                unknowns.append("time_bucket_completeness")
                codes.append(ReasonCode.TIME_BUCKET_INCOMPLETE)
            else:
                ratios["time_bucket_completeness"] = time_coverage
                if time_coverage < 1.0:
                    codes.append(ReasonCode.TIME_BUCKET_INCOMPLETE)

        # 7. Optional source-grounded semantic integrity checks. These checks
        # never use the gold answer: they inspect immutable source records and
        # replay the declared aggregation from the extraction artifact.
        if self.policy.require_provenance_verification:
            if not obs.provenance_verifiable or obs.provenance_coverage is None:
                unknowns.append("provenance_coverage")
                codes.append(ReasonCode.PROVENANCE_UNVERIFIABLE)
            else:
                ratios["provenance_coverage"] = obs.provenance_coverage
                if obs.provenance_coverage < 1.0:
                    codes.append(ReasonCode.PROVENANCE_MISMATCH)

        if self.policy.min_source_fact_fidelity is not None:
            if obs.source_fact_fidelity is None:
                unknowns.append("source_fact_fidelity")
                codes.append(ReasonCode.FACT_FIDELITY_UNKNOWN)
            else:
                ratios["source_fact_fidelity"] = obs.source_fact_fidelity
                if obs.source_fact_fidelity < self.policy.min_source_fact_fidelity:
                    codes.append(ReasonCode.FACT_FIDELITY_BELOW_REQUIRED)

        if self.policy.min_numeric_fidelity is not None:
            if obs.numeric_fidelity is None:
                unknowns.append("numeric_fidelity")
                codes.append(ReasonCode.NUMERIC_FIDELITY_UNKNOWN)
            else:
                ratios["numeric_fidelity"] = obs.numeric_fidelity
                if obs.numeric_fidelity < self.policy.min_numeric_fidelity:
                    codes.append(ReasonCode.NUMERIC_FIDELITY_BELOW_REQUIRED)

        if self.policy.require_aggregation_replay:
            if not obs.aggregation_replay_evaluable or obs.aggregation_replay_consistent is None:
                unknowns.append("aggregation_replay")
                codes.append(ReasonCode.AGGREGATION_REPLAY_UNKNOWN)
            elif not obs.aggregation_replay_consistent:
                codes.append(ReasonCode.AGGREGATION_REPLAY_MISMATCH)
            else:
                ratios["aggregation_replay_consistency"] = 1.0

        # Preserve deterministic ordering without duplicate reason codes.
        codes = list(dict.fromkeys(codes))
        unknowns = list(dict.fromkeys(unknowns))

        # Determine Decision
        if ReasonCode.SCOPE_COVERAGE_UNKNOWN in codes:
            decision = CoverageDecision.UNCERTIFIED
        elif codes:
            decision = (
                CoverageDecision.PARTIAL if self.policy.allow_partial else CoverageDecision.ABSTAIN
            )
        else:
            decision = CoverageDecision.CERTIFIED
            codes.append(ReasonCode.CERTIFIED)

        if run.answer is None and obs.eligible_set_size != 0:
            # If the pipeline errored out completely before generating an answer
            decision = CoverageDecision.ABSTAIN
            if ReasonCode.CERTIFIED in codes:
                codes.remove(ReasonCode.CERTIFIED)
            if ReasonCode.AGGREGATION_INVALID not in codes:
                codes.append(ReasonCode.AGGREGATION_INVALID)

        explanation = (
            "All configured evidence coverage requirements were satisfied."
            if decision == CoverageDecision.CERTIFIED
            else "Certification withheld: " + ", ".join(code.value for code in codes)
        )

        return CoverageCertificate(
            run_id=run.run_id,
            query_id=query.query_id,
            world_id=query.world_id,
            pipeline_id=run.pipeline_id,
            config_hash=run.config_hash,
            evidence_requirement=req,
            observations=obs,
            coverage_ratios=ratios,
            unknown_dimensions=unknowns,
            decision=decision,
            reason_codes=codes,
            human_readable_explanation=explanation,
            policy_id=self.policy.policy_id,
            policy_version=self.policy.version,
            assurance_scope=(
                "structured_semantic"
                if any(
                    (
                        self.policy.require_provenance_verification,
                        self.policy.min_source_fact_fidelity is not None,
                        self.policy.min_numeric_fidelity is not None,
                        self.policy.require_aggregation_replay,
                    )
                )
                else "structural_coverage"
            ),
        )

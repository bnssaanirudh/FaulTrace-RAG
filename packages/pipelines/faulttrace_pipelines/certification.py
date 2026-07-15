"""
Certification Engine.

Applies Answer Policies to Coverage Observations to generate Coverage Certificates.
"""

from typing import Any, Optional
from faulttrace_core.models import (
    PipelineRun, QuerySpec, CoverageObservation, EvidenceRequirement,
    CoverageCertificate, CoverageDecision, ReasonCode, AnswerPolicyConfig
)


class CertificationEngine:
    """Evaluates observations against requirements according to a policy."""
    
    def __init__(self, policy: AnswerPolicyConfig):
        self.policy = policy
        
    def certify(self, run: PipelineRun, query: QuerySpec, obs: CoverageObservation) -> CoverageCertificate:
        req = EvidenceRequirement.from_query(query)
        
        ratios = {}
        unknowns = []
        codes = []
        
        # 1. Scope Coverage
        if req.requires_full_scope:
            if obs.eligible_set_size_known and obs.eligible_set_size is not None and obs.eligible_set_size > 0:
                scope_coverage = obs.unique_represented_record_ids / obs.eligible_set_size
                ratios["scope_coverage"] = scope_coverage
                if scope_coverage < self.policy.min_known_scope_coverage:
                    codes.append(ReasonCode.SCOPE_COVERAGE_BELOW_REQUIRED)
            elif obs.eligible_set_size == 0:
                # Legitimate empty scope
                ratios["scope_coverage"] = 1.0
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
            field_completeness = (obs.extracted_valid_rows - obs.missing_required_fields) / obs.extracted_valid_rows
            ratios["field_completeness"] = field_completeness
            if field_completeness < self.policy.min_required_field_completeness:
                codes.append(ReasonCode.REQUIRED_FIELD_MISSING)
                
        # Determine Decision
        if ReasonCode.SCOPE_COVERAGE_UNKNOWN in codes:
            decision = CoverageDecision.UNCERTIFIED
        elif codes:
            decision = CoverageDecision.PARTIAL if self.policy.allow_partial else CoverageDecision.ABSTAIN
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
            policy_id=self.policy.policy_id,
            policy_version=self.policy.version
        )

"""
Selective Prediction Evaluation Metrics (Prompt 6).

Computes risk-coverage curves and evaluates abstention policies over a set of runs.
"""

from typing import Any

from faulttrace_core.models import CoverageDecision, PipelineRun
from pydantic import BaseModel


class PolicyEvaluationResult(BaseModel):
    total_queries: int
    answered_queries: int
    abstained_queries: int

    answer_coverage_rate: float

    risk: float  # Mean error on answered queries
    selective_accuracy: float  # Accuracy on answered queries

    abstention_precision: float  # Ratio of abstentions that were actually wrong
    false_certification_rate: float  # Ratio of answered queries that are wrong
    unnecessary_abstention_rate: float  # Ratio of abstentions that were actually correct


def evaluate_certification_policy(runs: list[PipelineRun]) -> PolicyEvaluationResult:
    """Evaluates the currently applied AnswerPolicy across a list of runs."""
    total = len(runs)
    if total == 0:
        return PolicyEvaluationResult(
            total_queries=0,
            answered_queries=0,
            abstained_queries=0,
            answer_coverage_rate=0,
            risk=0,
            selective_accuracy=0,
            abstention_precision=0,
            false_certification_rate=0,
            unnecessary_abstention_rate=0,
        )

    answered = [r for r in runs if r.policy_decision == CoverageDecision.CERTIFIED.value]
    abstained = [
        r
        for r in runs
        if r.policy_decision
        in (
            CoverageDecision.ABSTAIN.value,
            CoverageDecision.UNCERTIFIED.value,
            CoverageDecision.PARTIAL.value,
        )
    ]

    ans_count = len(answered)
    abs_count = len(abstained)

    # Calculate risks and accuracies
    risk = 0.0
    sel_acc = 0.0
    if ans_count > 0:
        risk = sum(r.loss for r in answered if r.loss is not None) / ans_count
        sel_acc = sum(1 for r in answered if r.is_correct) / ans_count

    # Evaluate abstentions against gold (offline evaluation)
    # A "correct" abstention is one where the raw answer was actually wrong.
    # An "unnecessary" abstention is one where the raw answer was actually right.
    abs_correct = sum(1 for r in abstained if r.is_correct is False)
    abs_wrong = sum(1 for r in abstained if r.is_correct is True)

    abs_precision = abs_correct / abs_count if abs_count > 0 else 0.0
    unnecessary = abs_wrong / abs_count if abs_count > 0 else 0.0
    false_cert = 1.0 - sel_acc if ans_count > 0 else 0.0

    return PolicyEvaluationResult(
        total_queries=total,
        answered_queries=ans_count,
        abstained_queries=abs_count,
        answer_coverage_rate=ans_count / total,
        risk=risk,
        selective_accuracy=sel_acc,
        abstention_precision=abs_precision,
        false_certification_rate=false_cert,
        unnecessary_abstention_rate=unnecessary,
    )


def generate_risk_coverage_curve(runs: list[PipelineRun]) -> list[dict[str, Any]]:
    """Return measured operating points; no unobserved threshold sweep is fabricated."""
    if not runs:
        return []

    evaluable = [run for run in runs if run.is_correct is not None]
    answered = [
        run
        for run in evaluable
        if run.policy_decision == CoverageDecision.CERTIFIED.value
    ]

    def point(label: str, selected: list[PipelineRun]) -> dict[str, Any]:
        losses = [float(run.loss) for run in selected if run.loss is not None]
        return {
            "label": label,
            "coverage_rate": len(selected) / len(evaluable) if evaluable else 0.0,
            "risk": sum(losses) / len(losses) if losses else None,
            "selective_accuracy": (
                sum(1 for run in selected if run.is_correct is True) / len(selected)
                if selected
                else None
            ),
            "measurement_status": "measured",
        }

    return [point("raw_evaluable", evaluable), point("applied_policy", answered)]

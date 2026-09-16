"""
Metrics engine to calculate analytical RAG performance, selective prediction risk, and Shapley attributions.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AggregateMetrics(BaseModel):
    sample_count: int
    missing_count: int
    failure_count: int

    # Core Accuracy & Loss
    accuracy: float
    mean_loss: float

    # Retrieval & Scope
    mean_scope_coverage: float | None
    retrieval_recall: float | None
    retrieval_precision: float | None

    # Extraction
    extraction_field_accuracy: float | None
    extraction_macro_f1: float | None

    # Certificates & Selective Prediction
    certified_rate: float
    selective_risk: float
    false_certification_rate: float

    # Performance & Footprint
    mean_latency_ms: float
    total_cost_usd: float | None
    cache_hit_rate: float | None

    # Attribution averages
    avg_phi_r: float
    avg_phi_e: float
    avg_phi_a: float
    measurement_status: dict[str, str] = Field(default_factory=dict)


class MetricsComputer:
    """Computes publication-ready performance and diagnostics metrics."""

    @staticmethod
    def compute_all(
        runs: list[dict[str, Any]], attributions: list[dict[str, Any]] | None = None
    ) -> AggregateMetrics:
        total = len(runs)
        if total == 0:
            return AggregateMetrics(
                sample_count=0,
                missing_count=0,
                failure_count=0,
                accuracy=0.0,
                mean_loss=0.0,
                mean_scope_coverage=None,
                retrieval_recall=None,
                retrieval_precision=None,
                extraction_field_accuracy=None,
                extraction_macro_f1=None,
                certified_rate=0.0,
                selective_risk=0.0,
                false_certification_rate=0.0,
                mean_latency_ms=0.0,
                total_cost_usd=None,
                cache_hit_rate=None,
                avg_phi_r=0.0,
                avg_phi_e=0.0,
                avg_phi_a=0.0,
            )

        completed_runs = [r for r in runs if r.get("status") == "completed"]
        completed_count = len(completed_runs)
        failure_count = total - completed_count

        if completed_count == 0:
            return AggregateMetrics(
                sample_count=total,
                missing_count=total,
                failure_count=failure_count,
                accuracy=0.0,
                mean_loss=0.0,
                mean_scope_coverage=None,
                retrieval_recall=None,
                retrieval_precision=None,
                extraction_field_accuracy=None,
                extraction_macro_f1=None,
                certified_rate=0.0,
                selective_risk=0.0,
                false_certification_rate=0.0,
                mean_latency_ms=0.0,
                total_cost_usd=None,
                cache_hit_rate=None,
                avg_phi_r=0.0,
                avg_phi_e=0.0,
                avg_phi_a=0.0,
            )

        # Basic Correctness & Loss
        scored_runs = [r for r in completed_runs if r.get("is_correct") is not None]
        loss_runs = [r for r in completed_runs if r.get("loss") is not None]
        correct_count = sum(1 for r in scored_runs if r.get("is_correct") is True)
        accuracy = correct_count / len(scored_runs) if scored_runs else 0.0
        mean_loss = (
            sum(float(r["loss"]) for r in loss_runs) / len(loss_runs) if loss_runs else 0.0
        )

        def measured_mean(field: str) -> float | None:
            values = [float(r[field]) for r in completed_runs if r.get(field) is not None]
            return sum(values) / len(values) if values else None

        mean_scope_cov = measured_mean("scope_coverage")
        retrieval_recall = measured_mean("retrieval_recall")
        retrieval_precision = measured_mean("retrieval_precision")
        extraction_field_accuracy = measured_mean("extraction_field_accuracy")
        extraction_macro_f1 = measured_mean("extraction_macro_f1")

        # Selective Prediction / Certification Rates
        certified_runs = [r for r in completed_runs if r.get("policy_decision") == "certified"]
        certified_count = len(certified_runs)
        certified_rate = certified_count / completed_count

        selective_risk = 0.0
        false_cert_rate = 0.0
        if certified_count > 0:
            certified_correct = sum(1 for r in certified_runs if r.get("is_correct") is True)
            certified_incorrect = certified_count - certified_correct
            selective_risk = (
                sum(float(r.get("loss") or 0) for r in certified_runs) / certified_count
            )
            false_cert_rate = certified_incorrect / certified_count

        # Latency & Cost
        mean_latency = (
            sum(float(r.get("latency_ms") or 0) for r in completed_runs) / completed_count
        )
        measured_costs = [float(r["cost_usd"]) for r in completed_runs if r.get("cost_usd") is not None]
        total_cost = sum(measured_costs) if measured_costs else None
        cache_values = [bool(r["cache_hit"]) for r in completed_runs if r.get("cache_hit") is not None]
        cache_hit_rate = (
            sum(1 for value in cache_values if value) / len(cache_values) if cache_values else None
        )

        # Average Attributions if provided
        phi_r_vals = []
        phi_e_vals = []
        phi_a_vals = []
        if attributions:
            for attr in attributions:
                comps = attr.get("components", [])
                for c in comps:
                    if c.get("component") == "scope":
                        phi_r_vals.append(c.get("shapley_value", 0.0))
                    elif c.get("component") in ("facts", "extraction"):
                        phi_e_vals.append(c.get("shapley_value", 0.0))
                    elif c.get("component") == "aggregation":
                        phi_a_vals.append(c.get("shapley_value", 0.0))

        avg_phi_r = sum(phi_r_vals) / len(phi_r_vals) if phi_r_vals else 0.0
        avg_phi_e = sum(phi_e_vals) / len(phi_e_vals) if phi_e_vals else 0.0
        avg_phi_a = sum(phi_a_vals) / len(phi_a_vals) if phi_a_vals else 0.0

        # Incomplete / missing values mapping
        missing_count = sum(1 for r in completed_runs if r.get("answer") is None)

        return AggregateMetrics(
            sample_count=total,
            missing_count=missing_count,
            failure_count=failure_count,
            accuracy=accuracy,
            mean_loss=mean_loss,
            mean_scope_coverage=mean_scope_cov,
            retrieval_recall=retrieval_recall,
            retrieval_precision=retrieval_precision,
            extraction_field_accuracy=extraction_field_accuracy,
            extraction_macro_f1=extraction_macro_f1,
            certified_rate=certified_rate,
            selective_risk=selective_risk,
            false_certification_rate=false_cert_rate,
            mean_latency_ms=mean_latency,
            total_cost_usd=total_cost,
            cache_hit_rate=cache_hit_rate,
            avg_phi_r=avg_phi_r,
            avg_phi_e=avg_phi_e,
            avg_phi_a=avg_phi_a,
            measurement_status={
                name: ("measured" if value is not None else "unavailable_not_simulated")
                for name, value in {
                    "mean_scope_coverage": mean_scope_cov,
                    "retrieval_recall": retrieval_recall,
                    "retrieval_precision": retrieval_precision,
                    "extraction_field_accuracy": extraction_field_accuracy,
                    "extraction_macro_f1": extraction_macro_f1,
                    "total_cost_usd": total_cost,
                    "cache_hit_rate": cache_hit_rate,
                }.items()
            },
        )

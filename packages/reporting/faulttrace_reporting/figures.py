"""
Figure Generator: generates publication-grade vector SVG/PNG graphics and CSV backing values.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("faulttrace.figures")

# Styling constants - Colorblind-safe palette
COLORS = ["#ea580c", "#0f172a", "#10b981", "#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444"]


class FigureGenerator:
    def __init__(self, run_records: list[dict[str, Any]], output_dir: Path):
        self.runs = run_records
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Matplotlib global styling configuration
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Inter", "Arial"]
        plt.rcParams["text.color"] = "#333333"
        plt.rcParams["axes.labelcolor"] = "#333333"
        plt.rcParams["xtick.color"] = "#333333"
        plt.rcParams["ytick.color"] = "#333333"

    def _add_watermark(self, fig: plt.Figure):
        fig.text(
            0.5,
            0.5,
            "MEASURED RUN DATA",
            fontsize=40,
            color="gray",
            ha="center",
            va="center",
            alpha=0.1,
            rotation=45,
        )

    def generate_all(self) -> list[str]:
        """Generate only figures supported by measured input columns."""
        generated_paths = []
        df = pd.DataFrame(self.runs)
        if df.empty:
            logger.warning("No run records provided to generate figures. Writing empty stub plots.")
            return []

        def has(*columns: str) -> bool:
            return all(column in df.columns for column in columns)

        if has("scale_n", "pipeline_id", "is_correct"):
            generated_paths.append(self._plot_accuracy_vs_scale(df))
        if has("scale_n", "loss"):
            generated_paths.append(self._plot_loss_vs_scale(df))
        if has("policy_decision", "loss"):
            generated_paths.append(self._plot_coverage_vs_error(df))
            generated_paths.append(self._plot_risk_coverage_curves(df))
        if has("top_k", "retrieval_recall"):
            generated_paths.append(self._plot_topk_sensitivity(df))
        if has("query_family", "extraction_macro_f1"):
            generated_paths.append(self._plot_extraction_f1(df))
        if has("phi_scope", "phi_facts", "phi_aggregation"):
            generated_paths.append(self._plot_attribution_distributions(df))
        if has("scale_n", "dominant_fault"):
            generated_paths.append(self._plot_dominant_fault_by_scale(df))
        if has("pipeline_id", "latency_ms", "is_correct"):
            generated_paths.append(self._plot_cost_latency_vs_accuracy(df))
            if df["pipeline_id"].astype(str).str.startswith(("P4", "P5")).any():
                generated_paths.append(self._plot_p4_p5_repair_benefit(df))

        return generated_paths

    def _write_csv(self, name: str, headers: list[str], rows: list[list[Any]]):
        csv_path = self.output_dir / f"{name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
            writer.writerow(["# provenance: measured run records supplied to FigureGenerator"])

    def _plot_accuracy_vs_scale(self, df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(6, 4))
        scales = sorted(df["scale_n"].unique())
        pipelines = df["pipeline_id"].unique()

        csv_rows = []
        for p in pipelines:
            accuracies = []
            for s in scales:
                subset = df[(df["pipeline_id"] == p) & (df["scale_n"] == s)]
                acc = subset["is_correct"].mean() if not subset.empty else 0.0
                accuracies.append(acc)
                csv_rows.append([p, s, acc])

            ax.plot(
                scales,
                accuracies,
                marker="o",
                label=p.split("-")[0],
                color=COLORS[len(csv_rows) % len(COLORS)],
            )

        ax.set_xscale("log")
        ax.set_xlabel("Corpus Scale N (log)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs. Corpus Scale")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        self._add_watermark(fig)
        svg_path = self.output_dir / "accuracy_vs_scale.svg"
        png_path = self.output_dir / "accuracy_vs_scale.png"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        self._write_csv("accuracy_vs_scale", ["pipeline", "scale_n", "accuracy"], csv_rows)
        return str(svg_path)

    def _plot_loss_vs_scale(self, df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(6, 4))
        scales = sorted(df["scale_n"].unique())

        loss_means = []
        csv_rows = []
        for s in scales:
            subset = df[df["scale_n"] == s]
            mean_loss = subset["loss"].mean() if not subset.empty else 0.0
            loss_means.append(mean_loss)
            csv_rows.append([s, mean_loss])

        ax.bar([str(s) for s in scales], loss_means, color="#ea580c", width=0.4)
        ax.set_xlabel("Corpus Scale N")
        ax.set_ylabel("Mean Loss")
        ax.set_title("Normalized Error Loss across Scales")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        self._add_watermark(fig)
        svg_path = self.output_dir / "loss_vs_scale.svg"
        png_path = self.output_dir / "loss_vs_scale.png"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        self._write_csv("loss_vs_scale", ["scale_n", "mean_loss"], csv_rows)
        return str(svg_path)

    def _plot_coverage_vs_error(self, df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(6, 4))
        evaluable = df[df["loss"].notna()]
        certified = evaluable[evaluable["policy_decision"] == "certified"]
        selected = [("raw", evaluable), ("certified", certified)]
        coverages = [len(part) / len(evaluable) if len(evaluable) else 0.0 for _, part in selected]
        errors = [part["loss"].mean() if len(part) else np.nan for _, part in selected]

        ax.plot(coverages, errors, marker="s", color="#ea580c", linewidth=2)
        ax.set_xlabel("Answer Coverage Rate")
        ax.set_ylabel("Empirical Risk (Mean Loss)")
        ax.set_title("Risk-Coverage Trade-off Curve")
        ax.grid(True, linestyle="--", alpha=0.5)

        self._add_watermark(fig)
        svg_path = self.output_dir / "coverage_vs_error.svg"
        png_path = self.output_dir / "coverage_vs_error.png"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        rows = [[label, cov, err] for (label, _), cov, err in zip(selected, coverages, errors, strict=False)]
        self._write_csv("coverage_vs_error", ["operating_point", "coverage", "error_loss"], rows)
        return str(svg_path)

    def _plot_topk_sensitivity(self, df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(6, 4))
        measured = df.dropna(subset=["top_k", "retrieval_recall"])
        grouped = measured.groupby("top_k")["retrieval_recall"].mean().sort_index()
        top_k_values = grouped.index.tolist()
        recalls = grouped.tolist()

        ax.plot(top_k_values, recalls, marker="^", color="#3b82f6", linewidth=2)
        ax.set_xlabel("Retrieval Top-K limit")
        ax.set_ylabel("Evidence Recall")
        ax.set_title("Retrieval Omission Sensitivity (Top-K)")
        ax.grid(True, linestyle="--", alpha=0.5)

        self._add_watermark(fig)
        svg_path = self.output_dir / "topk_sensitivity.svg"
        png_path = self.output_dir / "topk_sensitivity.png"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        rows = [[tk, rec] for tk, rec in zip(top_k_values, recalls, strict=False)]
        self._write_csv("topk_sensitivity", ["top_k", "recall"], rows)
        return str(svg_path)

    def _plot_extraction_f1(self, df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(6, 4))
        grouped = df.dropna(subset=["extraction_macro_f1"]).groupby("query_family")[
            "extraction_macro_f1"
        ].mean()
        families = grouped.index.tolist()
        f1_scores = grouped.tolist()

        ax.bar(families, f1_scores, color="#10b981", width=0.4)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Query Family")
        ax.set_ylabel("Extraction Macro F1")
        ax.set_title("Extraction Attribute Correctness by Family")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        self._add_watermark(fig)
        svg_path = self.output_dir / "extraction_f1.svg"
        png_path = self.output_dir / "extraction_f1.png"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        rows = [[fam, f1] for fam, f1 in zip(families, f1_scores, strict=False)]
        self._write_csv("extraction_f1", ["query_family", "macro_f1"], rows)
        return str(svg_path)

    def _plot_attribution_distributions(self, df: pd.DataFrame) -> str:
        svg_path = self.output_dir / "attribution_dist.svg"
        fig, ax = plt.subplots(figsize=(6, 4))
        data = [
            df["phi_scope"].dropna().tolist(),
            df["phi_facts"].dropna().tolist(),
            df["phi_aggregation"].dropna().tolist(),
        ]

        ax.boxplot(data, tick_labels=["Scope R", "Extraction E", "Aggregation A"])
        ax.set_ylabel("Shapley Value Contribution")
        ax.set_title("REA Component Error Distribution")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        self._add_watermark(fig)
        svg_path = self.output_dir / "attribution_dist.svg"
        png_path = self.output_dir / "attribution_dist.png"
        fig.tight_layout()
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        rows = [
            [name, float(np.mean(values)) if values else None]
            for name, values in zip(("scope", "facts", "aggregation"), data, strict=False)
        ]
        self._write_csv("attribution_distributions", ["component", "mean_contribution"], rows)
        return str(svg_path)

    def _plot_dominant_fault_by_scale(self, df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(6, 4))
        table = pd.crosstab(df["scale_n"], df["dominant_fault"], normalize="index")
        scales = [str(value) for value in table.index]
        scope_faults = table.get("scope", pd.Series(0.0, index=table.index)).tolist()
        extract_faults = table.get("facts", pd.Series(0.0, index=table.index)).tolist()
        agg_faults = table.get("aggregation", pd.Series(0.0, index=table.index)).tolist()

        ax.bar(scales, scope_faults, label="Scope (R)", color="#ea580c", width=0.4)
        ax.bar(
            scales,
            extract_faults,
            bottom=scope_faults,
            label="Extraction (E)",
            color="#3b82f6",
            width=0.4,
        )
        bottoms = np.array(scope_faults) + np.array(extract_faults)
        ax.bar(
            scales, agg_faults, bottom=bottoms, label="Aggregation (A)", color="#8b5cf6", width=0.4
        )

        ax.set_xlabel("Corpus Scale N")
        ax.set_ylabel("Fault Share")
        ax.set_title("Dominant Fault Origin by Scale")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        self._add_watermark(fig)
        svg_path = self.output_dir / "dominant_fault_by_scale.svg"
        png_path = self.output_dir / "dominant_fault_by_scale.png"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        rows = [list(row) for row in zip(scales, scope_faults, extract_faults, agg_faults, strict=False)]
        self._write_csv(
            "dominant_fault_by_scale",
            ["scale_n", "scope_fault", "extraction_fault", "aggregation_fault"],
            rows,
        )
        return str(svg_path)

    def _plot_cost_latency_vs_accuracy(self, df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(6, 4))
        grouped = df.groupby("pipeline_id").agg(
            latency_ms=("latency_ms", "mean"), accuracy=("is_correct", "mean")
        ).dropna()
        latencies = grouped["latency_ms"].tolist()
        accuracies = grouped["accuracy"].tolist()

        ax.scatter(latencies, accuracies, color="#ea580c", s=100)
        for i, txt in enumerate(grouped.index.tolist()):
            ax.annotate(txt, (latencies[i] + 15, accuracies[i] - 0.01), fontsize=8)

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Latency-Accuracy Trade-off Profile")
        ax.grid(True, linestyle="--", alpha=0.5)

        self._add_watermark(fig)
        svg_path = self.output_dir / "cost_latency_vs_accuracy.svg"
        png_path = self.output_dir / "cost_latency_vs_accuracy.png"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        rows = [[pipeline, lat, acc] for pipeline, lat, acc in zip(grouped.index, latencies, accuracies, strict=False)]
        self._write_csv("cost_latency_vs_accuracy", ["pipeline", "latency_ms", "accuracy"], rows)
        return str(svg_path)

    def _plot_risk_coverage_curves(self, df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(6, 4))
        evaluable = df[df["loss"].notna()]
        certified = evaluable[evaluable["policy_decision"] == "certified"]
        coverages = [1.0 if len(evaluable) else 0.0, len(certified) / len(evaluable) if len(evaluable) else 0.0]
        risks = [evaluable["loss"].mean() if len(evaluable) else np.nan, certified["loss"].mean() if len(certified) else np.nan]

        ax.plot(coverages, risks, marker="o", color="#ea580c", label="strict_exact_v1")
        ax.set_xlabel("Answer Coverage Rate")
        ax.set_ylabel("Empirical Selective Prediction Risk")
        ax.set_title("Selective Prediction Calibration Curves")
        ax.grid(True, linestyle="--", alpha=0.5)

        self._add_watermark(fig)
        svg_path = self.output_dir / "risk_coverage_curves.svg"
        png_path = self.output_dir / "risk_coverage_curves.png"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        rows = [[label, cov, risk] for label, cov, risk in zip(("raw", "applied_policy"), coverages, risks, strict=False)]
        self._write_csv("risk_coverage_curves", ["operating_point", "coverage", "risk"], rows)
        return str(svg_path)

    def _plot_p4_p5_repair_benefit(self, df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(6, 4))
        subset = df[df["pipeline_id"].astype(str).str.startswith(("P4", "P5"))]
        grouped = subset.groupby("pipeline_id")["is_correct"].mean().dropna()
        pipelines = grouped.index.tolist()
        accuracies = grouped.tolist()

        ax.bar(pipelines, accuracies, color=["#ef4444", "#10b981"], width=0.3)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Pipeline Accuracy")
        ax.set_title("Auto-Repair Mitigation Benefit (P4 vs. P5)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        self._add_watermark(fig)
        svg_path = self.output_dir / "p4_p5_repair_benefit.svg"
        png_path = self.output_dir / "p4_p5_repair_benefit.png"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        rows = [[pipe, acc] for pipe, acc in zip(pipelines, accuracies, strict=False)]
        self._write_csv("p4_p5_repair_benefit", ["pipeline", "accuracy"], rows)
        return str(svg_path)

"""Summarize measured, provenance-complete experiment performance records."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Measured experiment CSV not found: {input_path}")
    frame = pd.read_csv(input_path)
    required = {"experiment_config_hash", "dataset_id", "pipeline_id", "scale_n", "latency_ms"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is not a measured provenance-complete export; missing {sorted(missing)}")

    aggregations = {
        "runs": ("latency_ms", "count"),
        "mean_latency_ms": ("latency_ms", "mean"),
        "p50_latency_ms": ("latency_ms", "median"),
        "p95_latency_ms": ("latency_ms", lambda values: values.quantile(0.95)),
    }
    if "peak_memory_mb" in frame.columns:
        aggregations["mean_peak_memory_mb"] = ("peak_memory_mb", "mean")
    summary = frame.groupby(["dataset_id", "pipeline_id", "scale_n"], dropna=False).agg(
        **aggregations
    ).reset_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path.with_suffix(".csv"), index=False)
    output_path.write_text(
        "# FaultTrace-RAG Measured Performance Summary\n\n"
        + f"Source: `{input_path}`\n\n"
        + summary.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Provenance-complete measured experiment CSV")
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark_results.md"))
    args = parser.parse_args()
    summarize(args.input, args.output)

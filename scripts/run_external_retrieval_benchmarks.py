"""Run traceable external retrieval benchmarks over immutable local snapshots.

SciFact is evaluated as full-corpus retrieval. HotpotQA distractor and RAGBench
COVID-QA are evaluated over the candidate sets supplied with each question; they
must not be described as global-corpus retrieval results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faulttrace_core.evaluation import evaluate_retrieval
from faulttrace_core.retrieval import RetrievalUnit, TextDocument
from faulttrace_core.retrieval_bm25 import BM25Retriever
from faulttrace_core.retrieval_dense import DenseRetriever
from faulttrace_core.retrieval_hybrid import HybridRetriever
from faulttrace_data.benchmarks.covidqa import CovidQAAdapter
from faulttrace_data.benchmarks.hotpotqa import HotpotQAAdapter
from faulttrace_data.benchmarks.provenance import build_benchmark_manifest
from faulttrace_data.benchmarks.scifact import SciFactAdapter

SEED = 20260828


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def to_units(documents: list[TextDocument]) -> list[RetrievalUnit]:
    return [
        RetrievalUnit(
            unit_id=document.doc_id,
            record_id=document.doc_id,
            text=f"{document.title} {document.text}".strip(),
            metadata=document.metadata,
        )
        for document in documents
    ]


def ranked(engine: Any, query: str, top_k: int) -> tuple[list[str], list[float]]:
    hits = engine.search(query, top_k=top_k)
    return (
        [str(hit["unit"].record_id) for hit in hits],
        [float(hit["score"]) for hit in hits],
    )


def confidence_interval(
    results: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    metric: str,
    k: int,
) -> list[float]:
    values = np.asarray(
        [
            evaluate_retrieval(
                {query_id: results[query_id]}, {query_id: qrels[query_id]}, [k]
            )[metric]
            for query_id in sorted(results)
        ],
        dtype=float,
    )
    generator = np.random.default_rng(SEED)
    means = np.empty(10_000, dtype=float)
    for start in range(0, 10_000, 100):
        indices = generator.integers(0, len(values), size=(100, len(values)))
        means[start : start + 100] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def validate_inputs(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    candidate_sets: dict[str, list[TextDocument]] | None = None,
) -> list[str]:
    query_ids = sorted(set(queries) & set(qrels))
    if not query_ids:
        raise ValueError("No scored queries remain after joining queries and qrels")
    for query_id in query_ids:
        if not queries[query_id].strip():
            raise ValueError(f"Empty scored query: {query_id}")
        if not any(score > 0 for score in qrels[query_id].values()):
            raise ValueError(f"Scored query has no positive qrel: {query_id}")
        if candidate_sets is not None:
            candidate_ids = {document.doc_id for document in candidate_sets[query_id]}
            missing = set(qrels[query_id]) - candidate_ids
            if missing:
                raise ValueError(f"Gold documents absent from candidates for {query_id}: {missing}")
    return query_ids


def run_candidate_benchmark(
    *,
    dataset_id: str,
    split: str,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    candidate_sets: dict[str, list[TextDocument]],
    k_values: list[int],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    query_ids = validate_inputs(queries, qrels, candidate_sets)
    all_rankings: dict[str, list[str]] = {}
    latencies = []
    candidate_counts = []
    top_k = max(k_values)
    started = time.perf_counter()
    for query_id in query_ids:
        units = to_units(candidate_sets[query_id])
        engine = BM25Retriever()
        engine.build_index(units)
        query_started = time.perf_counter()
        doc_ids, scores = ranked(engine, queries[query_id], top_k)
        latency_ms = (time.perf_counter() - query_started) * 1000
        all_rankings[query_id] = doc_ids
        latencies.append(latency_ms)
        candidate_counts.append(len(units))
        records.append(
            {
                "dataset_id": dataset_id,
                "split": split,
                "evaluation_semantics": "per_query_supplied_candidate_retrieval",
                "retriever": "bm25",
                "query_id": query_id,
                "candidate_count": len(units),
                "ranked_doc_ids": doc_ids,
                "ranked_scores": scores,
                "latency_ms": latency_ms,
            }
        )
    metrics = evaluate_retrieval(all_rankings, qrels, k_values)
    primary_k = max(k_values)
    primary_metric = f"ndcg@{primary_k}"
    return {
        "dataset_id": dataset_id,
        "split": split,
        "evaluation_semantics": "per_query_supplied_candidate_retrieval",
        "retriever": "bm25",
        "queries_evaluated": len(query_ids),
        "queries_excluded_without_positive_qrels": len(queries) - len(query_ids),
        "candidate_count_min": min(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "metrics": metrics,
        "primary_metric": primary_metric,
        "primary_metric_95_ci": confidence_interval(
            all_rankings, qrels, primary_metric, primary_k
        ),
        "mean_query_latency_ms": float(np.mean(latencies)),
        "p95_query_latency_ms": float(np.quantile(latencies, 0.95)),
        "wall_time_seconds": time.perf_counter() - started,
    }


def run_scifact(
    adapter: SciFactAdapter,
    records: list[dict[str, Any]],
    include_dense: bool,
) -> list[dict[str, Any]]:
    documents = adapter.load_corpus()
    queries = adapter.load_queries()
    qrels = adapter.load_qrels("test")
    query_ids = validate_inputs(queries, qrels)
    units = to_units(documents)
    engines: dict[str, Any] = {"bm25": BM25Retriever()}
    if include_dense:
        engines["dense"] = DenseRetriever()
    build_seconds: dict[str, float] = {}
    for name, engine in engines.items():
        started = time.perf_counter()
        engine.build_index(units)
        build_seconds[name] = time.perf_counter() - started
    if include_dense:
        hybrid = HybridRetriever(engines["bm25"], engines["dense"])
        hybrid.units = units
        engines["hybrid"] = hybrid
        build_seconds["hybrid"] = 0.0

    summaries = []
    for name, engine in engines.items():
        rankings: dict[str, list[str]] = {}
        latencies = []
        started = time.perf_counter()
        for query_id in query_ids:
            query_started = time.perf_counter()
            doc_ids, scores = ranked(engine, queries[query_id], 10)
            latency_ms = (time.perf_counter() - query_started) * 1000
            rankings[query_id] = doc_ids
            latencies.append(latency_ms)
            records.append(
                {
                    "dataset_id": "scifact-beir",
                    "split": "test",
                    "evaluation_semantics": "full_corpus_retrieval",
                    "retriever": name,
                    "query_id": query_id,
                    "candidate_count": len(units),
                    "ranked_doc_ids": doc_ids,
                    "ranked_scores": scores,
                    "latency_ms": latency_ms,
                }
            )
        metrics = evaluate_retrieval(rankings, qrels, [1, 5, 10])
        summaries.append(
            {
                "dataset_id": "scifact-beir",
                "split": "test",
                "evaluation_semantics": "full_corpus_retrieval",
                "retriever": name,
                "queries_evaluated": len(query_ids),
                "corpus_documents": len(units),
                "metrics": metrics,
                "primary_metric": "ndcg@10",
                "primary_metric_95_ci": confidence_interval(
                    rankings, qrels, "ndcg@10", 10
                ),
                "index_build_seconds": build_seconds[name],
                "mean_query_latency_ms": float(np.mean(latencies)),
                "p95_query_latency_ms": float(np.quantile(latencies, 0.95)),
                "wall_time_seconds": time.perf_counter() - started,
            }
        )
    return summaries


def split_ids(path: Path) -> set[str]:
    return set(pd.read_parquet(path, columns=["id"])["id"].astype(str))


def create_manifests(data_root: Path) -> dict[str, dict[str, Any]]:
    scifact = data_root / "scifact" / "beir" / "scifact"
    hotpot = data_root / "hotpotqa" / "distractor"
    covid = data_root / "ragbench" / "covidqa"
    manifests = {
        "scifact-beir": build_benchmark_manifest(
            dataset_id="scifact-beir",
            root=data_root,
            split_files={
                "corpus": [scifact / "corpus.jsonl"],
                "queries": [scifact / "queries.jsonl"],
                "train_qrels": [scifact / "qrels" / "train.tsv"],
                "test_qrels": [scifact / "qrels" / "test.tsv"],
            },
            source_url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
            license_name="CC BY-NC 2.0",
            license_url="https://github.com/allenai/scifact/blob/master/LICENSE.md",
            acquisition_method="historical acquisition record absent; local files hashed at run time",
            query_ids_by_split={
                "train": set(adapter_qrels(scifact / "qrels" / "train.tsv")),
                "test": set(adapter_qrels(scifact / "qrels" / "test.tsv")),
            },
        ),
        "hotpotqa-distractor": build_benchmark_manifest(
            dataset_id="hotpotqa-distractor",
            root=data_root,
            split_files={
                "train": sorted(hotpot.glob("train-*.parquet")),
                "validation": [hotpot / "validation-00000-of-00001.parquet"],
            },
            source_url="https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/main/distractor",
            license_name="CC BY-SA 4.0",
            license_url="https://hotpotqa.github.io/",
            acquisition_method="historical acquisition record absent; Hugging Face schema metadata present and local files hashed at run time",
            query_ids_by_split={
                "train_0": split_ids(hotpot / "train-00000-of-00002.parquet"),
                "train_1": split_ids(hotpot / "train-00001-of-00002.parquet"),
                "validation": split_ids(hotpot / "validation-00000-of-00001.parquet"),
            },
        ),
        "ragbench-covidqa": build_benchmark_manifest(
            dataset_id="ragbench-covidqa",
            root=data_root,
            split_files={
                "train": [covid / "train-00000-of-00001.parquet"],
                "validation": [covid / "validation-00000-of-00001.parquet"],
                "test": [covid / "test-00000-of-00001.parquet"],
            },
            source_url="https://huggingface.co/datasets/galileo-ai/ragbench/tree/main/covidqa",
            license_name="CC BY 4.0",
            license_url="https://huggingface.co/datasets/galileo-ai/ragbench",
            acquisition_method="historical acquisition record absent; Hugging Face schema metadata present and local files hashed at run time",
            query_ids_by_split={
                "train": split_ids(covid / "train-00000-of-00001.parquet"),
                "validation": split_ids(covid / "validation-00000-of-00001.parquet"),
                "test": split_ids(covid / "test-00000-of-00001.parquet"),
            },
        ),
    }
    return {key: value.model_dump(mode="json") for key, value in manifests.items()}


def adapter_qrels(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t")
    return frame["query-id"].astype(str).tolist()


def write_report(output: Path, summary: dict[str, Any]) -> None:
    rows = []
    for result in summary["results"]:
        metrics = result["metrics"]
        primary = result["primary_metric"]
        low, high = result["primary_metric_95_ci"]
        rows.append(
            f"| {result['dataset_id']} | {result['evaluation_semantics']} | "
            f"{result['retriever']} | {result['queries_evaluated']} | "
            f"{metrics[primary]:.4f} [{low:.4f}, {high:.4f}] | "
            f"{metrics.get('recall@10', metrics.get('recall@4', 0.0)):.4f} |"
        )
    content = """# External retrieval benchmark run

This directory contains measured results from the complete local external files, not
the hand-authored test fixtures. SciFact is a full-corpus retrieval task. HotpotQA
distractor and RAGBench COVID-QA are per-question supplied-candidate tasks; their scores
are not comparable to open-domain retrieval scores.

| Dataset | Evaluation semantics | Retriever | Queries | Primary nDCG [95% bootstrap CI] | Recall at largest reported k |
|---|---|---:|---:|---:|---:|
""" + "\n".join(rows) + "\n\nSee `summary.json`, `per_query_results.parquet`, and `manifests/` for traceability.\n"
    (output / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    manifests = create_manifests(args.data_root)
    input_hashes_before = {
        item["relative_path"]: item["sha256"]
        for manifest in manifests.values()
        for item in manifest["files"]
    }
    manifest_dir = args.output / "manifests"
    manifest_dir.mkdir()
    for dataset_id, manifest in manifests.items():
        (manifest_dir / f"{dataset_id}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    records: list[dict[str, Any]] = []
    results = run_scifact(SciFactAdapter(args.data_root), records, not args.bm25_only)
    hotpot = HotpotQAAdapter(args.data_root)
    results.append(
        run_candidate_benchmark(
            dataset_id="hotpotqa-distractor",
            split="validation",
            queries=hotpot.load_queries(),
            qrels=hotpot.load_qrels(),
            candidate_sets=hotpot.load_candidate_sets(),
            k_values=[1, 2, 5, 10],
            records=records,
        )
    )
    covid = CovidQAAdapter(args.data_root)
    results.append(
        run_candidate_benchmark(
            dataset_id="ragbench-covidqa",
            split="test",
            queries=covid.load_queries(),
            qrels=covid.load_qrels(),
            candidate_sets=covid.load_candidate_sets(),
            k_values=[1, 2, 4],
            records=records,
        )
    )

    current_hashes = {
        relative: sha256_file(args.data_root / relative)
        for relative in input_hashes_before
    }
    if current_hashes != input_hashes_before:
        raise RuntimeError("A benchmark source file changed during evaluation")

    frame = pd.DataFrame(records)
    result_path = args.output / "per_query_results.parquet"
    frame.to_parquet(result_path, index=False)
    implementation_files = [
        Path(__file__),
        Path("packages/core/faulttrace_core/evaluation.py"),
        Path("packages/core/faulttrace_core/retrieval_bm25.py"),
        Path("packages/core/faulttrace_core/retrieval_dense.py"),
        Path("packages/core/faulttrace_core/retrieval_hybrid.py"),
        Path("packages/data/faulttrace_data/benchmarks/scifact.py"),
        Path("packages/data/faulttrace_data/benchmarks/hotpotqa.py"),
        Path("packages/data/faulttrace_data/benchmarks/covidqa.py"),
        Path("packages/data/faulttrace_data/benchmarks/provenance.py"),
    ]
    summary = {
        "schema_version": "1.0.0",
        "measurement_status": "measured_from_complete_local_external_snapshots",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_sha256": hashlib.sha256(
            (git_value("status", "--porcelain=v1") or "").encode("utf-8")
        ).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "implementation_files": {
            str(path.as_posix()): sha256_file(path) for path in implementation_files
        },
        "dataset_snapshot_sha256": {
            key: manifest["snapshot_sha256"] for key, manifest in manifests.items()
        },
        "input_files_unchanged_during_run": True,
        "per_query_result_sha256": sha256_file(result_path),
        "results": results,
        "claim_boundary": (
            "Retrieval-only external validation. No extraction, answer generation, "
            "R/E/A attribution, or semantic certification was evaluated."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(args.output, summary)
    checksummed = sorted(
        path for path in args.output.rglob("*") if path.is_file()
    )
    (args.output / "checksums.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(args.output).as_posix()}"
            for path in checksummed
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

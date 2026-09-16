"""Run leakage-aware external text pipeline and certification audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faulttrace_core.extraction_providers import DeterministicFixtureExtractor
from faulttrace_core.retrieval import RetrievalUnit, TextDocument
from faulttrace_core.retrieval_bm25 import BM25Retriever
from faulttrace_data.benchmarks.covidqa import CovidQAAdapter
from faulttrace_data.benchmarks.hotpotqa import HotpotQAAdapter
from faulttrace_data.benchmarks.provenance import build_benchmark_manifest
from faulttrace_data.benchmarks.scifact import SciFactAdapter, SciFactClaimCase
from faulttrace_pipelines.external_text_evaluation import (
    aggregate_support_status,
    answer_exact_match,
    answer_token_f1,
    certificate_metrics,
    flip_support_status,
    lexical_grounding_score,
    select_certificate_threshold,
)
from faulttrace_pipelines.text_attribution import TextAttributor
from sklearn.metrics import accuracy_score, f1_score

SEED = 20260828
SUBSETS = ("none", "R", "E", "A", "RE", "RA", "EA", "REA")
COMPONENTS = ("R", "E", "A")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def to_unit(document: TextDocument) -> RetrievalUnit:
    return RetrievalUnit(
        unit_id=document.doc_id,
        record_id=document.doc_id,
        text=f"{document.title} {document.text}".strip(),
        metadata=document.metadata,
    )


def classification_summary(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    labels = sorted(set(gold) | set(predicted))
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "accuracy_95_ci": bootstrap_mean(
            [float(expected == actual) for expected, actual in zip(gold, predicted, strict=True)]
        ),
        "macro_f1": float(f1_score(gold, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_f1_labels": labels,
        "label_support": dict(Counter(gold)),
        "prediction_counts": dict(Counter(predicted)),
    }


def bootstrap_mean(values: list[float], draws: int = 10_000) -> list[float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return [float("nan"), float("nan")]
    generator = np.random.default_rng(SEED)
    means = np.empty(draws, dtype=float)
    batch = 100
    for start in range(0, draws, batch):
        size = min(batch, draws - start)
        indices = generator.integers(0, len(array), size=(size, len(array)))
        means[start : start + size] = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def multilabel_metrics(gold_sets: list[set[str]], predicted_sets: list[set[str]]) -> dict[str, float]:
    exact = np.mean([gold == predicted for gold, predicted in zip(gold_sets, predicted_sets, strict=True)])
    per_label_f1 = []
    for label in COMPONENTS:
        tp = sum(label in gold and label in predicted for gold, predicted in zip(gold_sets, predicted_sets, strict=True))
        fp = sum(label not in gold and label in predicted for gold, predicted in zip(gold_sets, predicted_sets, strict=True))
        fn = sum(label in gold and label not in predicted for gold, predicted in zip(gold_sets, predicted_sets, strict=True))
        denominator = 2 * tp + fp + fn
        per_label_f1.append(2 * tp / denominator if denominator else 1.0)
    return {"exact_set_accuracy": float(exact), "macro_f1": float(np.mean(per_label_f1))}


def oracle_scope(case: SciFactClaimCase) -> list[str]:
    return sorted(case.evidence_doc_statuses)


def oracle_extract(case: SciFactClaimCase, scope: list[str]) -> dict[str, str]:
    return {
        doc_id: case.evidence_doc_statuses.get(doc_id, "insufficient_evidence")
        for doc_id in scope
    }


def corrupt_extraction(statuses: dict[str, str]) -> dict[str, str]:
    return {doc_id: flip_support_status(status) for doc_id, status in statuses.items()}


def run_injected_subset(
    case: SciFactClaimCase,
    subset: str,
    faults: set[str],
    bad_scope: list[str],
) -> str:
    scope = oracle_scope(case) if "R" in subset or "R" not in faults else bad_scope
    statuses = oracle_extract(case, scope)
    if "E" not in subset and "E" in faults:
        statuses = corrupt_extraction(statuses)
    answer = aggregate_support_status(statuses.values())
    if "A" not in subset and "A" in faults:
        answer = flip_support_status(answer)
    return answer


def run_scifact(
    data_root: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    adapter = SciFactAdapter(data_root)
    documents = adapter.load_official_corpus()
    cases = adapter.load_official_claims("dev")
    document_map = {document.doc_id: document for document in documents}
    units = [to_unit(document) for document in documents]
    unit_map = {unit.record_id: unit for unit in units}
    retriever = BM25Retriever()
    build_started = time.perf_counter()
    retriever.build_index(units)
    build_seconds = time.perf_counter() - build_started
    extractor = DeterministicFixtureExtractor()
    attributor = TextAttributor()

    extraction_cache: dict[tuple[str, str], tuple[str, str | None]] = {}

    def evaluated_extract(case: SciFactClaimCase, scope: list[str]) -> dict[str, str]:
        statuses = {}
        for doc_id in scope:
            key = (case.query_id, doc_id)
            if key not in extraction_cache:
                unit = unit_map[doc_id]
                record = extractor.extract(
                    doc_id=doc_id,
                    doc_text=unit.text,
                    query=case.claim,
                    dataset_id="scifact-official",
                    split="dev",
                )
                answer = record.cited_answer
                extraction_cache[key] = (
                    answer.support_status.value,
                    answer.supporting_quote,
                )
            statuses[doc_id] = extraction_cache[key][0]
        return statuses

    predictions = []
    gold = []
    retrieval_recalls = []
    structural_certified = []
    natural_attributions = []
    baseline_scopes: dict[str, list[str]] = {}
    for case in cases:
        hits = retriever.search(case.claim, top_k=5)
        scope = [str(hit["unit"].record_id) for hit in hits]
        baseline_scopes[case.query_id] = scope
        statuses = evaluated_extract(case, scope)
        prediction = aggregate_support_status(statuses.values())
        quotes = [
            extraction_cache[(case.query_id, doc_id)][1]
            for doc_id in scope
            if extraction_cache[(case.query_id, doc_id)][0] == "supported"
            and extraction_cache[(case.query_id, doc_id)][1]
        ]
        answer_text = quotes[0] if quotes else None
        prediction_correct = prediction == case.gold_support_status
        citation_integrity = all(doc_id in document_map for doc_id in scope)
        quote_grounded = answer_text is None or any(
            answer_text in unit_map[doc_id].text for doc_id in scope
        )
        certified = citation_integrity and quote_grounded and prediction != "insufficient_evidence"
        structural_certified.append((certified, prediction_correct))
        gold.append(case.gold_support_status)
        predictions.append(prediction)
        evidence_ids = set(case.evidence_doc_statuses)
        if evidence_ids:
            retrieval_recalls.append(len(evidence_ids & set(scope)) / len(evidence_ids))

        oracle_results = {}
        for subset in SUBSETS:
            current_scope = oracle_scope(case) if "R" in subset else scope
            current_statuses = (
                oracle_extract(case, current_scope)
                if "E" in subset
                else evaluated_extract(case, current_scope)
            )
            oracle_results[subset] = aggregate_support_status(current_statuses.values())
        attribution = attributor.attribute(
            query_id=case.query_id,
            dataset_id="scifact-official",
            pipeline_id="bm25-deterministic-extract",
            pipeline_answer=answer_text,
            pipeline_support_status=prediction,
            gold_support_status=case.gold_support_status,
            oracle_results=oracle_results,
        )
        natural_attributions.append(attribution)
        rows.append(
            {
                "task": "scifact_end_to_end",
                "dataset_id": "scifact-official",
                "split": "dev",
                "query_id": case.query_id,
                "gold": case.gold_support_status,
                "prediction": prediction,
                "correct": prediction_correct,
                "certified": certified,
                "score": None,
                "details_json": json.dumps(
                    {
                        "retrieved_doc_ids": scope,
                        "evidence_doc_ids": sorted(evidence_ids),
                        "answer_text": answer_text,
                        "attribution": attribution.to_dict(),
                    },
                    sort_keys=True,
                ),
            }
        )

    certified_pairs = [pair for pair in structural_certified if pair[0]]
    natural_summary = {
        "wrong_predictions": sum(not value for value in [g == p for g, p in zip(gold, predictions, strict=True)]),
        "mean_phi_retrieval": float(np.mean([value.phi_retrieval for value in natural_attributions])),
        "mean_phi_extraction": float(np.mean([value.phi_extraction for value in natural_attributions])),
        "mean_phi_aggregation": float(np.mean([value.phi_aggregation for value in natural_attributions])),
        "full_oracle_correct_rate": float(np.mean([
            aggregate_support_status(oracle_extract(case, oracle_scope(case)).values()) == case.gold_support_status
            for case in cases
        ])),
    }

    audit_cases = [
        case
        for status in ("supported", "unsupported")
        for case in sorted(
            [item for item in cases if item.gold_support_status == status],
            key=lambda item: item.query_id,
        )[:60]
    ]
    fault_conditions = {
        "P0": set(),
        "R": {"R"},
        "E": {"E"},
        "A": {"A"},
        "RE": {"R", "E"},
        "RA": {"R", "A"},
        "EA": {"E", "A"},
        "REA": {"R", "E", "A"},
    }
    all_doc_ids = sorted(document_map)
    injected_gold_sets = []
    injected_predicted_sets = []
    injected_results = []
    for condition, faults in fault_conditions.items():
        for case in audit_cases:
            bad_scope = next(
                [doc_id]
                for doc_id in all_doc_ids
                if doc_id not in case.evidence_doc_statuses
                and doc_id not in case.cited_doc_ids
            )
            oracle_results = {
                subset: run_injected_subset(case, subset, faults, bad_scope)
                for subset in SUBSETS
            }
            baseline = oracle_results["none"]
            attribution = attributor.attribute(
                query_id=case.query_id,
                dataset_id="scifact-official-injected",
                pipeline_id=condition,
                pipeline_answer=None,
                pipeline_support_status=baseline,
                gold_support_status=case.gold_support_status,
                oracle_results=oracle_results,
            )
            predicted_faults = {
                component
                for component, value in {
                    "R": attribution.phi_retrieval,
                    "E": attribution.phi_extraction,
                    "A": attribution.phi_aggregation,
                }.items()
                if value > 0.05
            }
            efficiency_residual = abs(
                attribution.total_recoverable_error
                - (
                    attribution.phi_retrieval
                    + attribution.phi_extraction
                    + attribution.phi_aggregation
                    + attribution.interaction_term
                )
            )
            injected_gold_sets.append(faults)
            injected_predicted_sets.append(predicted_faults)
            injected_results.append(
                {
                    "condition": condition,
                    "query_id": case.query_id,
                    "gold_faults": sorted(faults),
                    "predicted_faults": sorted(predicted_faults),
                    "full_oracle_correct": oracle_results["REA"] == case.gold_support_status,
                    "efficiency_residual": efficiency_residual,
                }
            )
            rows.append(
                {
                    "task": "scifact_injected_attribution",
                    "dataset_id": "scifact-official",
                    "split": "dev-injected-audit",
                    "query_id": f"{condition}:{case.query_id}",
                    "gold": json.dumps(sorted(faults)),
                    "prediction": json.dumps(sorted(predicted_faults)),
                    "correct": predicted_faults == faults,
                    "certified": None,
                    "score": attribution.total_recoverable_error,
                    "details_json": json.dumps(
                        {
                            "source_query_id": case.query_id,
                            "condition": condition,
                            "oracle_results": oracle_results,
                            "attribution": attribution.to_dict(),
                            "full_oracle_correct": oracle_results["REA"]
                            == case.gold_support_status,
                            "efficiency_residual": efficiency_residual,
                        },
                        sort_keys=True,
                    ),
                }
            )
    injected_metrics = multilabel_metrics(injected_gold_sets, injected_predicted_sets)
    injected_metrics["lattices"] = len(injected_results)
    injected_metrics["exact_set_accuracy_95_ci"] = bootstrap_mean(
        [
            float(gold_set == predicted_set)
            for gold_set, predicted_set in zip(
                injected_gold_sets, injected_predicted_sets, strict=True
            )
        ]
    )
    injected_metrics["full_oracle_correct_rate"] = float(
        np.mean([item["full_oracle_correct"] for item in injected_results])
    )
    injected_metrics["max_abs_efficiency_residual"] = max(
        item["efficiency_residual"] for item in injected_results
    )
    injected_metrics["condition_exact_set_accuracy"] = {
        condition: float(
            np.mean(
                [
                    set(item["gold_faults"]) == set(item["predicted_faults"])
                    for item in injected_results
                    if item["condition"] == condition
                ]
            )
        )
        for condition in fault_conditions
    }

    return {
        "dataset_id": "scifact-official",
        "split": "dev",
        "provider": "deterministic_fixture",
        "corpus_documents": len(documents),
        "queries": len(cases),
        "retriever": "bm25",
        "top_k": 5,
        "index_build_seconds": build_seconds,
        "classification": classification_summary(gold, predictions),
        "evidence_recall_at_5": float(np.mean(retrieval_recalls)),
        "structural_certificate": {
            "coverage": len(certified_pairs) / len(cases),
            "false_certification_rate": (
                sum(not correct for _, correct in certified_pairs) / len(certified_pairs)
                if certified_pairs
                else None
            ),
            "false_certification_rate_95_ci": bootstrap_mean(
                [float(not correct) for _, correct in certified_pairs]
            ),
        },
        "natural_counterfactual_attribution": natural_summary,
        "injected_fault_attribution": injected_metrics,
        "claim_boundary": "Deterministic rule-based claim-status baseline; not an LLM or state-of-the-art comparison.",
    }


def run_hotpot(data_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    adapter = HotpotQAAdapter(data_root)
    split_file = adapter.hotpot_dir / "validation-00000-of-00001.parquet"
    frame = pd.read_parquet(split_file, columns=["id", "question", "answer"])
    answers = {str(row.id): str(row.answer) for row in frame.itertuples()}
    queries = adapter.load_queries()
    qrels = adapter.load_qrels()
    candidates = adapter.load_candidate_sets()
    extractor = DeterministicFixtureExtractor()
    exact_matches = []
    token_f1s = []
    citation_recalls = []
    grounded = []
    certified_pairs = []
    for query_id in sorted(queries):
        units = [to_unit(document) for document in candidates[query_id]]
        unit_map = {unit.record_id: unit for unit in units}
        retriever = BM25Retriever()
        retriever.build_index(units)
        hits = retriever.search(queries[query_id], top_k=2)
        records = [
            extractor.extract(
                doc_id=hit["unit"].record_id,
                doc_text=hit["unit"].text,
                query=queries[query_id],
                dataset_id="hotpotqa-distractor",
                split="validation",
            )
            for hit in hits
        ]
        statuses = [record.cited_answer.support_status.value for record in records]
        support_status = aggregate_support_status(statuses)
        quotes = [
            record.cited_answer.supporting_quote
            for record in records
            if record.cited_answer.support_status.value == "supported"
            and record.cited_answer.supporting_quote
        ]
        prediction = quotes[0] if quotes else None
        cited_doc_ids = list(dict.fromkeys(
            doc_id
            for record in records
            for doc_id in record.cited_answer.cited_doc_ids
        ))
        exact = answer_exact_match(prediction, answers[query_id])
        f1 = answer_token_f1(prediction, answers[query_id])
        gold_docs = set(qrels[query_id])
        citation_recall = len(gold_docs & set(cited_doc_ids)) / len(gold_docs)
        quote_grounded = prediction is None or any(
            prediction in unit_map[doc_id].text for doc_id in cited_doc_ids
        )
        certificate = bool(cited_doc_ids) and quote_grounded and support_status != "insufficient_evidence"
        exact_matches.append(exact)
        token_f1s.append(f1)
        citation_recalls.append(citation_recall)
        grounded.append(quote_grounded)
        certified_pairs.append((certificate, bool(exact)))
        rows.append(
            {
                "task": "hotpotqa_cited_answer",
                "dataset_id": "hotpotqa-distractor",
                "split": "validation",
                "query_id": query_id,
                "gold": answers[query_id],
                "prediction": prediction,
                "correct": bool(exact),
                "certified": certificate,
                "score": f1,
                "details_json": json.dumps(
                    {
                        "cited_doc_ids": cited_doc_ids,
                        "gold_supporting_doc_ids": sorted(gold_docs),
                        "citation_recall": citation_recall,
                        "support_status": support_status,
                        "quote_grounded": quote_grounded,
                    },
                    sort_keys=True,
                ),
            }
        )
    certified = [item for item in certified_pairs if item[0]]
    return {
        "dataset_id": "hotpotqa-distractor",
        "split": "validation",
        "provider": "deterministic_fixture_evidence_sentence",
        "queries": len(queries),
        "top_k": 2,
        "answer_exact_match": float(np.mean(exact_matches)),
        "answer_exact_match_95_ci": bootstrap_mean(exact_matches),
        "answer_token_f1": float(np.mean(token_f1s)),
        "answer_token_f1_95_ci": bootstrap_mean(token_f1s),
        "supporting_document_recall_at_2": float(np.mean(citation_recalls)),
        "supporting_document_recall_at_2_95_ci": bootstrap_mean(citation_recalls),
        "quote_grounding_rate": float(np.mean(grounded)),
        "structural_certificate": {
            "coverage": len(certified) / len(queries),
            "false_certification_rate_against_exact_match": (
                sum(not correct for _, correct in certified) / len(certified)
                if certified
                else None
            ),
            "false_certification_rate_95_ci": bootstrap_mean(
                [float(not correct) for _, correct in certified]
            ),
        },
        "claim_boundary": "Evidence-sentence output from a deterministic extractor; not a trained QA generator.",
    }


def nested_sentence_texts(value: Any) -> list[str]:
    texts = []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list | tuple):
        if len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], str):
            return [value[1]]
        for item in value:
            texts.extend(nested_sentence_texts(item))
    return texts


def score_ragbench(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[float], list[bool]]:
    records = []
    scores = []
    labels = []
    for row in frame.itertuples():
        response_sentences = nested_sentence_texts(row.response_sentences)
        document_sentences = nested_sentence_texts(row.documents_sentences)
        grounding = lexical_grounding_score(response_sentences, document_sentences)
        scores.append(grounding.score)
        labels.append(bool(row.adherence_score))
        records.append(
            {
                "query_id": str(row.id),
                "score": grounding.score,
                "gold_adherent": bool(row.adherence_score),
                "numeric_fidelity": grounding.numeric_fidelity,
                "minimum_sentence_coverage": grounding.minimum_sentence_coverage,
                "mean_sentence_coverage": grounding.mean_sentence_coverage,
                "response_sentence_count": grounding.response_sentence_count,
            }
        )
    return records, scores, labels


def run_ragbench(data_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    root = CovidQAAdapter(data_root).covidqa_dir
    validation = pd.read_parquet(root / "validation-00000-of-00001.parquet")
    test = pd.read_parquet(root / "test-00000-of-00001.parquet")
    validation_records, validation_scores, validation_labels = score_ragbench(validation)
    calibration = select_certificate_threshold(
        validation_scores, validation_labels, target_fcr=0.05
    )
    test_records, test_scores, test_labels = score_ragbench(test)
    threshold = float(calibration["threshold"])
    test_metrics = certificate_metrics(test_scores, test_labels, threshold)
    structural_metrics = certificate_metrics(test_scores, test_labels, 0.0)
    certified_test_indices = [
        index for index, score in enumerate(test_scores) if score >= threshold
    ]
    test_metrics["false_certification_rate_95_ci"] = bootstrap_mean(
        [float(not test_labels[index]) for index in certified_test_indices]
    )
    test_metrics["coverage_95_ci"] = bootstrap_mean(
        [float(score >= threshold) for score in test_scores]
    )
    for record in test_records:
        certified = record["score"] >= threshold
        rows.append(
            {
                "task": "ragbench_grounding_certificate",
                "dataset_id": "ragbench-covidqa",
                "split": "test",
                "query_id": record["query_id"],
                "gold": str(record["gold_adherent"]).lower(),
                "prediction": str(certified).lower(),
                "correct": certified == record["gold_adherent"],
                "certified": certified,
                "score": record["score"],
                "details_json": json.dumps(record, sort_keys=True),
            }
        )
    return {
        "dataset_id": "ragbench-covidqa",
        "calibration_split": "validation",
        "calibration_examples": len(validation_records),
        "target_validation_fcr": 0.05,
        "calibration": calibration,
        "test_split": "test",
        "test": test_metrics,
        "test_structural_nonempty_baseline": structural_metrics,
        "policy": "minimum per-response-sentence source-token coverage with exact numeric fidelity",
        "claim_boundary": "Lexical/numeric source-consistency certificate; not semantic entailment or truth certification.",
    }


def create_manifests(data_root: Path) -> dict[str, dict[str, Any]]:
    scifact = data_root / "scifact" / "official" / "data"
    hotpot = data_root / "hotpotqa" / "distractor"
    covid = data_root / "ragbench" / "covidqa"
    manifests = {
        "scifact-official-dev": build_benchmark_manifest(
            dataset_id="scifact-official-dev",
            root=data_root,
            split_files={
                "corpus": [scifact / "corpus.jsonl"],
                "train": [scifact / "claims_train.jsonl"],
                "dev": [scifact / "claims_dev.jsonl"],
                "test": [scifact / "claims_test.jsonl"],
            },
            source_url="https://github.com/allenai/scifact",
            license_name="CC BY-NC 2.0",
            license_url="https://github.com/allenai/scifact/blob/master/LICENSE.md",
            acquisition_method="historical acquisition record absent; local files hashed at run time",
            query_ids_by_split={
                "train": {
                    str(json.loads(line)["id"])
                    for line in (scifact / "claims_train.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                },
                "dev": {
                    str(json.loads(line)["id"])
                    for line in (scifact / "claims_dev.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                },
                "test": {
                    str(json.loads(line)["id"])
                    for line in (scifact / "claims_test.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                },
            },
        ),
        "hotpotqa-distractor-validation": build_benchmark_manifest(
            dataset_id="hotpotqa-distractor-validation",
            root=data_root,
            split_files={"validation": [hotpot / "validation-00000-of-00001.parquet"]},
            source_url="https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/main/distractor",
            license_name="CC BY-SA 4.0",
            license_url="https://hotpotqa.github.io/",
            acquisition_method="historical acquisition record absent; local file hashed at run time",
        ),
        "ragbench-covidqa-validation-test": build_benchmark_manifest(
            dataset_id="ragbench-covidqa-validation-test",
            root=data_root,
            split_files={
                "validation": [covid / "validation-00000-of-00001.parquet"],
                "test": [covid / "test-00000-of-00001.parquet"],
            },
            source_url="https://huggingface.co/datasets/galileo-ai/ragbench/tree/main/covidqa",
            license_name="CC BY 4.0",
            license_url="https://huggingface.co/datasets/galileo-ai/ragbench",
            acquisition_method="historical acquisition record absent; local files hashed at run time",
            query_ids_by_split={
                "validation": set(
                    pd.read_parquet(
                        covid / "validation-00000-of-00001.parquet", columns=["id"]
                    )["id"].astype(str)
                ),
                "test": set(
                    pd.read_parquet(
                        covid / "test-00000-of-00001.parquet", columns=["id"]
                    )["id"].astype(str)
                ),
            },
        ),
    }
    return {key: value.model_dump(mode="json") for key, value in manifests.items()}


def write_report(output: Path, summary: dict[str, Any]) -> None:
    scifact = summary["scifact"]
    hotpot = summary["hotpotqa"]
    ragbench = summary["ragbench"]
    test_certificate = ragbench["test"]
    content = f"""# External end-to-end and certification audit

This measured audit uses complete local external snapshots. Providers and task semantics
are reported explicitly; no deterministic result is labeled as an LLM result.

| Evaluation | Main result |
|---|---:|
| SciFact deterministic claim-status accuracy / macro-F1 | {scifact['classification']['accuracy']:.4f} / {scifact['classification']['macro_f1']:.4f} |
| SciFact evidence Recall@5 | {scifact['evidence_recall_at_5']:.4f} |
| SciFact injected-fault attribution exact set / macro-F1 | {scifact['injected_fault_attribution']['exact_set_accuracy']:.4f} / {scifact['injected_fault_attribution']['macro_f1']:.4f} |
| HotpotQA evidence-sentence answer EM / token F1 | {hotpot['answer_exact_match']:.4f} / {hotpot['answer_token_f1']:.4f} |
| HotpotQA supporting-document Recall@2 | {hotpot['supporting_document_recall_at_2']:.4f} |
| RAGBench test certificate coverage / false-certification rate | {test_certificate['coverage']:.4f} / {test_certificate['false_certification_rate'] if test_certificate['false_certification_rate'] is not None else 'n/a'} |

SciFact and HotpotQA use the deterministic fixture extractor because no external-provider
credential was available. RAGBench evaluates a validation-calibrated lexical/numeric
source-consistency certificate over dataset-provided responses. These results do not
establish production-LLM quality, semantic entailment, or truth certification.
"""
    (output / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    manifests = create_manifests(args.data_root)
    manifest_dir = args.output / "manifests"
    manifest_dir.mkdir()
    for dataset_id, manifest in manifests.items():
        (manifest_dir / f"{dataset_id}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    before_hashes = {
        item["relative_path"]: item["sha256"]
        for manifest in manifests.values()
        for item in manifest["files"]
    }

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    scifact = run_scifact(args.data_root, rows)
    hotpot = run_hotpot(args.data_root, rows)
    ragbench = run_ragbench(args.data_root, rows)
    current_hashes = {
        relative: sha256_file(args.data_root / relative)
        for relative in before_hashes
    }
    if current_hashes != before_hashes:
        raise RuntimeError("An external source file changed during evaluation")

    results_path = args.output / "per_case_results.parquet"
    pd.DataFrame(rows).to_parquet(results_path, index=False)
    implementation_files = [
        Path(__file__),
        Path("packages/pipelines/faulttrace_pipelines/external_text_evaluation.py"),
        Path("packages/pipelines/faulttrace_pipelines/text_attribution.py"),
        Path("packages/core/faulttrace_core/extraction_providers.py"),
        Path("packages/core/faulttrace_core/retrieval_bm25.py"),
        Path("packages/data/faulttrace_data/benchmarks/scifact.py"),
        Path("packages/data/faulttrace_data/benchmarks/hotpotqa.py"),
        Path("packages/data/faulttrace_data/benchmarks/covidqa.py"),
    ]
    summary = {
        "schema_version": "1.0.0",
        "measurement_status": "measured_external_text_pipeline_audit",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_sha256": hashlib.sha256(
            (git_value("status", "--porcelain=v1") or "").encode("utf-8")
        ).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "wall_time_seconds": time.perf_counter() - started,
        "dataset_snapshot_sha256": {
            key: manifest["snapshot_sha256"] for key, manifest in manifests.items()
        },
        "input_files_unchanged_during_run": True,
        "implementation_files": {
            path.as_posix(): sha256_file(path) for path in implementation_files
        },
        "per_case_result_sha256": sha256_file(results_path),
        "scifact": scifact,
        "hotpotqa": hotpot,
        "ragbench": ragbench,
        "global_claim_boundary": (
            "External pipeline mechanics and narrow source-consistency policies only; "
            "no production LLM or general semantic correctness claim."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(args.output, summary)
    files = sorted(path for path in args.output.rglob("*") if path.is_file())
    (args.output / "checksums.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(args.output).as_posix()}"
            for path in files
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

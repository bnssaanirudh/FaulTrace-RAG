"""Leakage-aware utilities for external text benchmark evaluation."""

from __future__ import annotations

import re
import string
from collections.abc import Iterable
from dataclasses import dataclass

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def normalize_answer(text: str | None) -> str:
    value = (text or "").lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return " ".join(token for token in value.split() if token not in {"a", "an", "the"})


def answer_exact_match(prediction: str | None, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def answer_token_f1(prediction: str | None, gold: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    predicted_counts = {token: predicted.count(token) for token in set(predicted)}
    expected_counts = {token: expected.count(token) for token in set(expected)}
    overlap = sum(
        min(count, expected_counts.get(token, 0))
        for token, count in predicted_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def aggregate_support_status(statuses: Iterable[str]) -> str:
    values = list(statuses)
    if not values:
        return "insufficient_evidence"
    status_set = set(values)
    if "conflicting" in status_set or {"supported", "unsupported"} <= status_set:
        return "conflicting"
    if "supported" in status_set:
        return "supported"
    if "partially_supported" in status_set:
        return "partially_supported"
    if "unsupported" in status_set:
        return "unsupported"
    return "insufficient_evidence"


def flip_support_status(status: str) -> str:
    return {
        "supported": "unsupported",
        "unsupported": "supported",
        "partially_supported": "unsupported",
        "conflicting": "supported",
        "insufficient_evidence": "supported",
    }.get(status, "supported")


def aggregate_nli_predictions(predictions: Iterable[tuple[str, float]]) -> str:
    """Aggregate document-level NLI labels using the strongest non-neutral decision."""
    non_neutral = [item for item in predictions if item[0] != "neutral"]
    if not non_neutral:
        return "insufficient_evidence"
    label, _confidence = max(non_neutral, key=lambda item: item[1])
    return {
        "entailment": "supported",
        "contradiction": "unsupported",
    }[label]


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\b\w+\b", text.lower())
        if token not in STOPWORDS
    }


@dataclass(frozen=True)
class GroundingScore:
    score: float
    minimum_sentence_coverage: float
    mean_sentence_coverage: float
    numeric_fidelity: bool
    response_sentence_count: int


def lexical_grounding_score(
    response_sentences: Iterable[str], document_sentences: Iterable[str]
) -> GroundingScore:
    """Score source-token coverage without using support annotations or gold labels."""
    responses = [sentence for sentence in response_sentences if sentence.strip()]
    sources = [sentence for sentence in document_sentences if sentence.strip()]
    source_token_sets = [content_tokens(sentence) for sentence in sources]
    coverages = []
    for sentence in responses:
        response_tokens = content_tokens(sentence)
        if not response_tokens or not source_token_sets:
            coverages.append(0.0)
            continue
        coverages.append(
            max(len(response_tokens & source_tokens) / len(response_tokens) for source_tokens in source_token_sets)
        )
    all_source_text = " ".join(sources).lower()
    response_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", " ".join(responses)))
    source_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", all_source_text))
    numeric_fidelity = response_numbers <= source_numbers
    minimum = min(coverages) if coverages else 0.0
    mean = sum(coverages) / len(coverages) if coverages else 0.0
    score = minimum if numeric_fidelity else 0.0
    return GroundingScore(
        score=score,
        minimum_sentence_coverage=minimum,
        mean_sentence_coverage=mean,
        numeric_fidelity=numeric_fidelity,
        response_sentence_count=len(responses),
    )


def select_certificate_threshold(
    scores: list[float], labels: list[bool], target_fcr: float = 0.05
) -> dict[str, float | int]:
    """Choose maximum validation coverage subject to an empirical FCR constraint."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and aligned")
    candidates = sorted(set(scores), reverse=True) + [1.000001]
    best: dict[str, float | int] | None = None
    for threshold in candidates:
        selected = [index for index, score in enumerate(scores) if score >= threshold]
        fcr = (
            0.0
            if not selected
            else sum(not labels[index] for index in selected) / len(selected)
        )
        coverage = len(selected) / len(scores)
        if fcr <= target_fcr and (
            best is None
            or coverage > float(best["coverage"])
            or (coverage == float(best["coverage"]) and threshold < float(best["threshold"]))
        ):
            best = {
                "threshold": float(threshold),
                "coverage": float(coverage),
                "false_certification_rate": float(fcr),
                "certified": len(selected),
            }
    assert best is not None
    return best


def certificate_metrics(
    scores: list[float], labels: list[bool], threshold: float
) -> dict[str, float | int | None]:
    selected = [index for index, score in enumerate(scores) if score >= threshold]
    true_positives = sum(labels[index] for index in selected)
    false_positives = len(selected) - true_positives
    positives = sum(labels)
    return {
        "examples": len(scores),
        "certified": len(selected),
        "coverage": len(selected) / len(scores) if scores else 0.0,
        "false_certification_rate": (
            false_positives / len(selected) if selected else None
        ),
        "precision": true_positives / len(selected) if selected else None,
        "recall": true_positives / positives if positives else None,
    }

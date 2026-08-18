import math
from typing import Any


def mean_reciprocal_rank(results: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""
    if not results or not qrels:
        return 0.0

    mrr_sum = 0.0
    queries_evaluated = 0

    for q_id, retrieved_ids in results.items():
        if q_id not in qrels:
            continue

        queries_evaluated += 1
        relevant_docs = {doc_id for doc_id, score in qrels[q_id].items() if score > 0}

        for rank, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_docs:
                mrr_sum += 1.0 / (rank + 1)
                break

    return mrr_sum / queries_evaluated if queries_evaluated > 0 else 0.0


def recall_at_k(
    results: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int = 10
) -> float:
    """Calculate Recall@k."""
    if not results or not qrels:
        return 0.0

    recall_sum = 0.0
    queries_evaluated = 0

    for q_id, retrieved_ids in results.items():
        if q_id not in qrels:
            continue

        queries_evaluated += 1
        relevant_docs = {doc_id for doc_id, score in qrels[q_id].items() if score > 0}
        if not relevant_docs:
            continue

        retrieved_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_docs)
        recall_sum += hits / len(relevant_docs)

    return recall_sum / queries_evaluated if queries_evaluated > 0 else 0.0


def precision_at_k(
    results: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int = 10
) -> float:
    """Calculate Precision@k."""
    if not results or not qrels:
        return 0.0

    precision_sum = 0.0
    queries_evaluated = 0

    for q_id, retrieved_ids in results.items():
        if q_id not in qrels:
            continue

        queries_evaluated += 1
        relevant_docs = {doc_id for doc_id, score in qrels[q_id].items() if score > 0}

        retrieved_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_docs)
        precision_sum += hits / k if k > 0 else 0.0

    return precision_sum / queries_evaluated if queries_evaluated > 0 else 0.0


def ndcg_at_k(
    results: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int = 10
) -> float:
    """Calculate nDCG@k."""
    if not results or not qrels:
        return 0.0

    ndcg_sum = 0.0
    queries_evaluated = 0

    for q_id, retrieved_ids in results.items():
        if q_id not in qrels:
            continue

        queries_evaluated += 1
        query_qrels = qrels[q_id]

        # Calculate DCG
        dcg = 0.0
        retrieved_k = retrieved_ids[:k]
        for rank, doc_id in enumerate(retrieved_k):
            rel = query_qrels.get(doc_id, 0)
            if rel > 0:
                dcg += (2**rel - 1) / math.log2(rank + 2)

        # Calculate IDCG
        ideal_rels = sorted([rel for rel in query_qrels.values() if rel > 0], reverse=True)
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels[:k]):
            idcg += (2**rel - 1) / math.log2(rank + 2)

        if idcg > 0:
            ndcg_sum += dcg / idcg

    return ndcg_sum / queries_evaluated if queries_evaluated > 0 else 0.0


def evaluate_retrieval(
    results: dict[str, list[str]], qrels: dict[str, dict[str, int]], k_values: list[int] = None
) -> dict[str, Any]:
    """Calculate a standard suite of retrieval metrics."""
    if k_values is None:
        k_values = [1, 5, 10]
    metrics = {"mrr": mean_reciprocal_rank(results, qrels)}

    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(results, qrels, k)
        metrics[f"precision@{k}"] = precision_at_k(results, qrels, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(results, qrels, k)

    return metrics

# SciFact Retrieval Benchmark Results
**Timestamp:** 2026-08-18T17:22:40.984138+00:00
**Git Commit:** df59c3759dbd28bd8d79205246edbc947650ea7b

## Environment Summary
- **Device:** cpu
- **Dense Model:** `all-MiniLM-L6-v2`

## Dataset Details
- **Path:** `data/benchmarks/scifact/beir/scifact/`
- **Corpus Documents:** 5183
- **Total Queries:** 1109
- **Test Qrels:** 300

## Comparison Table (Full Test)
| Metric | BM25 | Dense | Hybrid |
|---|---|---|---|
| NDCG@10 | 0.6408 | 0.6451 | 0.6873 |
| RECALL@10 | 0.7667 | 0.7833 | 0.8186 |
| PRECISION@10 | 0.0847 | 0.0883 | 0.0913 |
| MRR | 0.6057 | 0.6047 | 0.6530 |
| RECALL@1 | 0.4969 | 0.4823 | 0.5423 |
| RECALL@5 | 0.7111 | 0.7379 | 0.7344 |
| PRECISION@1 | 0.5167 | 0.5033 | 0.5667 |
| PRECISION@5 | 0.1540 | 0.1640 | 0.1620 |
| NDCG@1 | 0.5167 | 0.5033 | 0.5667 |
| NDCG@5 | 0.6213 | 0.6293 | 0.6590 |
| **Queries Evaluated** | 300 | 300 | 300 |
| **Avg Latency (ms)** | 54.07 | 28.13 | 79.57 |

## Quick 50 vs Full Test Comparison (NDCG@10)
| Retriever | Quick 50 | Full Test |
|---|---|---|
| BM25 | 0.8060 | 0.6408 |
| Dense | 0.7384 | 0.6451 |
| Hybrid | 0.8529 | 0.6873 |

## Interpretation
BM25 typically relies on exact keyword matching, while Dense retrieval handles semantic similarity. Hybrid retrieval via reciprocal rank fusion combines both signals and usually achieves the highest overall accuracy on scientific domains.

## Limitations
- **Retrieval Only:** These results represent SciFact retrieval performance strictly. They do not indicate the end-to-end claim verification accuracy of the RAG pipeline.
- **Genuine Results:** These metrics were computed purely from local corpus, queries, and qrel sets. No simulated benchmark scripts or mocked fixtures were used.
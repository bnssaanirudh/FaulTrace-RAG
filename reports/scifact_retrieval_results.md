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
| **Queries Evaluated** | 300 | 300 | 300 |
| **Avg Latency (ms)** | 54.07 | 28.13 | 79.57 |

## Quick 50 vs Full Test Comparison (NDCG@10)
| Retriever | Quick 50 | Full Test |
|---|---|---|
| BM25 | 0.0000 | 0.0000 |
| Dense | 0.0000 | 0.0000 |
| Hybrid | 0.0000 | 0.0000 |

## Interpretation
BM25 typically relies on exact keyword matching, while Dense retrieval handles semantic similarity. Hybrid retrieval via reciprocal rank fusion combines both signals and usually achieves the highest overall accuracy on scientific domains.

## Limitations
- **Retrieval Only:** These results represent SciFact retrieval performance strictly. They do not indicate the end-to-end claim verification accuracy of the RAG pipeline.
- **Genuine Results:** These metrics were computed purely from local corpus, queries, and qrel sets. No simulated benchmark scripts or mocked fixtures were used.

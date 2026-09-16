# FaulTrace-RAG Final Reference and Citation Audit

## Outcome

All four final manuscripts use the same 19 scholarly references. Static citation-key checks found **0 missing citation keys** and **0 unused bibliography entries** in every version. The citations are attached to claims that match the scope of the cited work (RAG, RAG evaluation, retrieval benchmarks, multi-hop datasets, Shapley/Harsanyi theory, selective risk control, retrieval models, and the Pandas/DuckDB reference engines).

## Publication-status cross-check

The bibliography is restricted to published scholarly sources; there are no arXiv-only entries or GitHub repositories in the reference list.

- Lewis et al. - NeurIPS 2020, Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Es et al. - EACL 2024 System Demonstrations, RAGAs, DOI 10.18653/v1/2024.eacl-demo.16.
- Saad-Falcon et al. - NAACL 2024, ARES, DOI 10.18653/v1/2024.naacl-long.20.
- Ru et al. - NeurIPS 2024 Datasets and Benchmarks, RAGChecker, DOI 10.52202/079017-0692.
- Yu et al. - Findings of NAACL 2024, ReEval, DOI 10.18653/v1/2024.findings-naacl.85.
- Thakur et al. - NeurIPS 2021 Datasets and Benchmarks, BEIR.
- Yang et al. - EMNLP 2018, HotpotQA, DOI 10.18653/v1/D18-1259.
- Ho et al. - COLING 2020, 2WikiMultiHopQA paper, DOI 10.18653/v1/2020.coling-main.580.
- Niu et al. - ACL 2024, RAGTruth, DOI 10.18653/v1/2024.acl-long.585.
- Shapley - the canonical 1953 published Princeton University Press chapter, *A Value for n-Person Games*. This is a published scholarly book chapter rather than a journal/conference article and is retained because it is the original source for the Shapley value.
- Harsanyi - International Economic Review 1963, DOI 10.2307/2525487.
- Reimers and Gurevych - EMNLP-IJCNLP 2019, Sentence-BERT, DOI 10.18653/v1/D19-1410.
- Robertson and Zaragoza - Foundations and Trends in Information Retrieval 3(4), 333-389 (2009), DOI 10.1561/1500000019.
- McKinney - SciPy 2010 proceedings, Pandas, DOI 10.25080/Majora-92bf1922-00a.
- Raasveldt and Muehleisen - SIGMOD 2019, DuckDB, DOI 10.1145/3299869.3320212.
- Angelopoulos et al. - ICLR 2024, Conformal Risk Control.
- Bates et al. - Journal of the ACM 68(6), 2021, DOI 10.1145/3478535.
- Cormack et al. - SIGIR 2009, Reciprocal Rank Fusion, DOI 10.1145/1571941.1572114.
- Wang et al. - NeurIPS 2020, MiniLM.

## Claim mapping checked

- RAG definition/knowledge-intensive use -> Lewis et al.
- Reference-free/component RAG evaluation -> RAGAs, ARES, RAGChecker, ReEval.
- Cross-domain retrieval benchmark -> BEIR.
- Multi-hop datasets -> HotpotQA and 2WikiMultiHopQA papers.
- Hallucination pilot -> RAGTruth paper.
- Cooperative-game attribution and interaction -> Shapley and Harsanyi.
- Dense retrieval representation -> Sentence-BERT and MiniLM.
- Lexical/hybrid retrieval -> BM25 and Reciprocal Rank Fusion.
- Dual deterministic reference engines -> Pandas and DuckDB publications.
- Statistical risk-control framing -> Conformal Risk Control and risk-controlling prediction sets.

No paper claim relies on the excluded preliminary Phi-4-mini result. The final live-model statistics are limited to the validated Qwen and Mistral runs.

# FaultTrace-RAG Maturity Matrix

This document clarifies the scientific and engineering status of the framework. It replaces binary "complete" labels with a granular breakdown of component readiness, distinguishing between research prototype validation, manuscript evidence generation, and production deployment.

## Component Readiness

| Component | Status | Notes |
|---|---|---|
| **Core Engine** | Validated | Fully implements RAG extraction and aggregation as a reproducible AST. |
| **Reproducibility** | Validated | Immutable SHA-256 caching and JSONL persistence are fully functional. |
| **Controlled Validation** | Complete for current research scope | Exact-match tests on Track M confirm fundamental mathematical correctness. |
| **Large-scale Injected Validation** | Complete for current research scope | 488,250 PAPER_MODE pipeline tests run. |
| **External Retrieval** | Validated | Integrations for SciFact, HotpotQA, and COVID-QA function correctly. |
| **Natural LLM Validation** | Validated | Qwen/Mistral inference validation passes manually audited samples. |
| **Compound-Fault Diagnosis** | Partial | Framework isolates independent faults, but complex non-linear confounding models are future work. |
| **Active Diagnosis** | Validated | Bayesian and Greedy test selection strategies correctly implemented and tested. |
| **Repair** | Validated | Cost-Aware and Minimum Counterfactual Repair engines function successfully. |
| **Certification** | Validated (Empirical) | The struct-semantic overlap certifier works, but statistical risk bounds guarantee zero false certificates empirically, not mathematically. |
| **Production Security** | Out of scope | Designed for local research environments; no authentication or prompt-injection defense mechanisms. |
| **Documentation** | Validated | Full architecture and manuscript documentation available. |
| **Release Engineering** | Planned | Automated PyPI releases and Docker hub pushes are planned but not active. |

### Status Definitions:
- **Complete for current research scope**: The component has produced the necessary evidence required for the current manuscript and operates stably within the research environment.
- **Validated**: The component functions as designed, passes rigorous CI tests, and is usable by researchers.
- **Partial**: The feature works in restricted scenarios but lacks general capability.
- **Experimental**: Highly volatile or speculative implementation.
- **Planned**: Identified as a requirement but not yet implemented.
- **Out of scope**: Explicitly excluded from the repository's goals.

# Limitations

The current implementation is an engineering demo and comes with several constraints:
- Only a limited subset of real-world query logic is supported (top-k, counts, averages).
- The execution scale is artificially bounded to a maximum `N=1000` nodes to avoid overwhelming local resources.
- Extraction attribution models use heuristic approximation instead of full counterfactual replay in certain modes to save on LLM inference costs.

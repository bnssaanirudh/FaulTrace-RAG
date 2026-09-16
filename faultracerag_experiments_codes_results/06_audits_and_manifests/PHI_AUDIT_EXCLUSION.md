# Phi-4-mini audit exclusion

A preliminary 100-case Phi-4-mini-instruct live run was excluded from all manuscript statistics after a post-run audit identified a generation-termination configuration mismatch: the notebook overrode the model generation configuration with a single tokenizer EOS id, while the checkpoint uses multiple termination ids. This caused prompt-like continuation text in a large fraction of decoded Phi outputs.

No Phi result from that preliminary run is used in the revised manuscripts. The validated live-model analysis therefore contains 200 cases: 100 Qwen2.5-3B-Instruct and 100 Mistral-7B-Instruct-v0.3 cases. A corrected Phi rerun should allow the model generation configuration to supply its own EOS settings.

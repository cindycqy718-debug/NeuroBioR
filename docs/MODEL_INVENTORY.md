# Retained Model Inventory

Every listed row contains 15 retained runs: three datasets × request indices 41--45.
No missing result is silently replaced, and every retained directory contains the
generated predictor, probability matrix, generation transcript, configuration, score,
and summary.

| Display name | Directory tag | Reporting group | Runs |
|---|---|---|---:|
| GPT-4.1-mini | `gpt41_mini_fastfill_v1` | standard endpoint | 15 |
| GPT-4o | `gpt4o_nonthinking_retrymax5_v1` | standard endpoint | 15 |
| Qwen3-235B-A22B Thinking 2507 | `qwen3_235b_a22b_thinking_2507_v1` | thinking-designated | 15 |
| Gemini 3 Flash Thinking | `gemini3_flash_preview_thinking_explicit_v1` | thinking-designated | 15 |
| GLM-5 | `glm5_reasoning_v1` | thinking-designated | 15 |

The group label describes the endpoint configuration used in this study; it is not a
claim that internal chain-of-thought was available to the evaluator.

Failed calls were allowed to retry only when model, request index, prompt, data, and all
parameters were unchanged. The first valid result in chronological order was retained;
the maximum was five attempts per case. The retained 75-run release contains only valid,
complete rows. Generation transcripts remain available for auditing.


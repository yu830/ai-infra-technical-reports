# Reducing Backend Sampling Host Buffers in llama.cpp

[llama.cpp PR #29576](https://github.com/ggml-org/llama.cpp/pull/29576) · [Issue #29487](https://github.com/ggml-org/llama.cpp/issues/29487)

## Problem

Backend Top-K sampling can return only K candidates per output row, while the host probability and candidate buffers were allocated at vocabulary width V. For large vocabularies and small K, those two arrays reserve substantially more pinned host memory than their outputs require.

## Implementation

The patch probes the bound sampler chains and sizes the probability and candidate buffers using the maximum supported output width. The sampled logits and ordinary logits keep vocabulary-width layouts. Copy operations, getters and output reordering use the same candidate stride.

The sampler chain caches the width. Adding or removing a sampler invalidates it, with a full-vocabulary fallback until a safe width is available. Before a changed layout reuses or resizes the buffer, the context waits for outstanding asynchronous copies.

## Results and validation

For **Qwen3.5-0.8B-Q4_0**, vocabulary size **248,320**, **K=8** and **64 output rows**, the measured total host output buffer fell from **254,279,936 bytes (242.5 MiB)** to **127,144,192 bytes (121.3 MiB)**, approximately **50%**. The two logits buffers remain vocabulary-width. These 64 rows are output rows, not a claim of 64 concurrent requests.

The capacity regression failed on the unmodified baseline and passed with the patch. Recorded tests covered mixed K, sparse outputs, layout changes with pending copies and sampler removal; CPU/CUDA sampler tests passed 21 cases. Recorded CPU and CUDA perplexity matched at 14.6649 ± 1.41828 on the specified Wikitext2 test.

The result concerns pinned **host** output buffers. Runtime samples varied, so the report makes no inference speedup claim. GPU memcheck and other backends remain unverified; the PR contains the full configuration and timing records.

## Attribution

The upstream PR discloses that OpenAI Codex implemented the patch and tests, ran local validation and drafted the PR description; Gemini 3.8 Flash reviewed the diff, and Claude Opus 4.6 discussed cache invalidation. Codex also assisted with this report.

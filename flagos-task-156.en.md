# FlagOS Task 156: A Triton Implementation of Segmented LoRA K Projection

## Overview

Task 156's step_a_q operator computes the query-side LoRA correction for absorbed MLA. It selects an adapter for each request segment and multiplies each attention head's inputs by that adapter's K-weight slice. The implementation tiles the work by segment and head, groups multiple heads for selected short-decode shapes, and provides a layout-dependent Kunlunxin path.

The FlagOS task leaderboard recorded team YuJiacheng at **rank 3** with a platform-average speedup of **10.74×** on October 22, 2026. The leaderboard entry and the source package examined below are separate records; their relationship is described in the results section.

## Operator contract

The input q_nope has shape [S, H, K], where S is the number of token rows and H is the number of heads. B_buf has shape [number of adapters, H × full_K_per_head, R], with fixed LoRA rank R. The output has shape [S, H, R].

Each segment selects an adapter through weight_indices. An optional permutation maps segment positions back to original token rows. For an active segment, token row i and head h:

    out[i, h, :] = q_nope[i, h, :] @
                   B_buf[adapter, h × full_K_per_head : h × full_K_per_head + K, :]

Each head's K and V weights share a full_K_per_head span; this operator reads only the first K rows. Empty segments and zero-rank adapters are skipped, and uncovered output rows remain zero. The reference computes the product in FP32 before casting to the input dtype. The published BF16 tolerance is atol=rtol=0.015.

## Implementation

### Generic path

The implementation dispatches over segments, heads and output columns. Each program forms a token-by-rank output tile and reduces over K. In the inspected candidate, BM is 16 or 64 according to segment length, BN is 16, 32 or 64 according to rank, and BT is 64. Triton dot uses IEEE input precision with FP32 accumulation. Loads use the actual input and weight strides.

When the maximum segment length is at most 4, H ≥ 8, S×H ≥ 128, 16 < R ≤ 32 and K ≤ 256, a short-decode path processes one token and groups four heads per program. This reduces the number of separately scheduled head work units for those shapes; the benefit depends on the workload and backend.

A separate Triton kernel normally initializes the output. When permutation is absent and the guarded conditions are satisfied, the candidate combines part of the zero initialization with segmented projection. Routing metadata is read for the current batch on every call.

### Kunlunxin path

Without permutation, the candidate uses direct tiled projection and keeps the actual q_nope and B_buf strides. With permutation, it flattens [head, rank] output positions and traverses K using four FP32 accumulators. This avoids a matrix-gather path for reordered rows while changing the weight-reuse tradeoff.

## Platform results and version identity

The following values are from the task leaderboard recorded at 20:14 on October 22, 2026. Individual case shapes and timing samples were not published.

| Platform device label | Speedup |
|---|---:|
| Tianshu | 9.73× |
| MetaX | 12.11× |
| Hygon | 27.56× |
| Kunlunxin | 1.71× |
| Huawei | 2.71× |
| International chip A | 13.40× |
| International chip B | 7.96× |

The leaderboard reported an average speedup of 10.74142857×, a ranking score of 1.2 and task rank 3. Ranking uses per-device ranks, so the average alone does not determine placement.

The source package examined for this report was the 16:35 candidate on October 22, with source commit identifier 13721f12c6e9 and ZIP SHA-256 cfeabdae2429744f0bfcde72e6530934c3d61b653c0f7f91c410b76f48d2c1f3e. The user-provided platform receipt for that package displayed an average of **10.73×**. The later **10.74×** leaderboard entry is a separate record; the available evidence does not establish that this exact ZIP produced that leaderboard result or identify the source actually loaded by the evaluator.

## Development tradeoffs

An earlier, separate one-dimensional candidate met correctness in platform feedback but scored below 0.01× on Kunlunxin, under the task's 0.1× threshold. A different candidate using packed inputs and a matrix path then produced numerical mismatches and illegal memory access on the target evaluation. These were different versions. Their feedback motivated later layout-specific routes and should not be used to attribute the performance of the candidate examined here.

The published PyTorch reference includes host metadata reads and launch overhead. The competition's operator-level speedup is not an end-to-end inference speedup over a production SGLang path.

## Sources and attribution

- [Official FlagOS Task 156 statement and reference](https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/step_a_q)
- [FlagGems-sglang project](https://github.com/flagos-ai/FlagGems-sglang)
- The leaderboard record and candidate package are retained separately. Candidate source files carry FlagOS Contributors copyright and Apache-2.0 notices.
- OpenAI Codex assisted in preparing this report; existing source attribution and disclosures remain in place.

# FlagOS Task 156: Developing a Segmented LoRA K Projection

## Overview

Task 156's step_a_q operator computes a segmented matrix product for the query-side LoRA correction in absorbed MLA. Each segment selects an adapter, and each attention head reads only that adapter's K-weight slice. The engineering problem includes short segments, reordered token rows, non-contiguous layouts and compiler behavior across different backends.

The work moved from a correct but slow vector route to layout-dependent matrix tiling, then tested tile width, zero-initialization fusion and indexing as separate interventions. On October 22, 2026, the task leaderboard recorded team YuJiacheng at **rank 3** with a platform-average speedup of **10.74×**. This report retains the unsuccessful experiments and separates the leaderboard entry from the inspected source package.

## 1. Operator contract

q_nope has shape [S, H, K], B_buf has shape [number of adapters, H × full_K_per_head, R], and output has shape [S, H, R]. Segment boundaries come from seg_indptr; weight_indices selects an adapter. An optional permutation maps segment positions back to original token rows. For an active segment, row i and head h:

    out[i, h, :] = q_nope[i, h, :] @
                   B_buf[adapter, h × full_K_per_head : h × full_K_per_head + K, :]

Each head's weight span contains both K and V portions; this operator reads only the first K rows. Empty segments, zero-rank adapters and uncovered output rows must remain zero. The reference computes in FP32 before casting back to the input dtype. Actual strides, partial tiles, changing metadata and graph replay are part of the contract.

### Inspected candidate

The generic path in the 16:35 candidate dispatches output tiles by segment, head and rank columns. BM is 16 when the longest segment is shorter than 32 and 64 otherwise; BN is 16/32/64 according to rank, and the K tile BT is 64. For short decode satisfying length≤4, H≥8, S×H≥128, 16<R≤32 and K≤256, a separate route groups four heads per program. Input and weight loads use their actual strides, and routing metadata is read on each call.

The corresponding Kunlunxin file uses direct matrix tiles without permutation; with permutation it flattens [head, rank] and walks K with four FP32 accumulators. These are implementation facts from the inspected candidate. The individual contributions of those paths to the platform's aggregate result were not measured.

## 2. From a correct vector route to layout-dependent dispatch

### Why increase the unit of work?

An early scalar-output route passed the reported correctness evaluation but displayed **<0.01×** on Kunlunxin. A packed-vector route combined output columns yet also displayed **<0.01×**. For two public mirror dimensions, source-level grid accounting reduced projection programs from about 2,097,152 to 131,072. These counts describe the mirror inputs, not the platform's hidden cases. Both target results were censored, so their exact relative performance is unknown. [Grid record](evidence/task-156/10-routing-count-extract.json)

The source suggested limited reuse of weights between token rows and substantial scheduling work for small programs; without a target stage trace, these remained cost hypotheses. A 16-fold reduction in program count alone did not establish a competitive target result.

### Why use direct matrix tiles?

For inputs without permutation, a BM32/BN16/BK64 direct matrix route allows a tile to reuse adapter weights across token rows without global A/W/C packing buffers. The candidate retains the earlier correctness-supported one-dimensional route when permutation is present, rather than introducing an unverified matrix-gather path at the same time.

In one self-designed local CUDA BF16 case, S=96, H=16, K=128, R=32 and no permutation, CUDA Graph replay medians were **70.489 μs** for the earlier vector candidate and **6.314 μs** for the direct candidate, approximately **11.16×**. A different case with permutation was nearly unchanged. These measurements are local candidate comparisons, not Kunlunxin timings or confirmed hidden platform shapes. [Samples and source hashes](evidence/task-156/09-direct-cuda-extract.json)

A subsequent user-reported platform result for candidate 325f072 showed **0.84×** on Kunlunxin, above the then-current 0.1× threshold; the previous candidate had shown <0.01×. The submissions were not paired per case, so the result establishes that the threshold was crossed, not the isolated contribution of any one change. [Feedback record](evidence/task-156/01-direct-feedback.json)

## 3. BM64 helped; why was BN32 rejected?

External research suggested trying a larger token tile on the verified matrix route. The BM64 control retained BN16, BK64 and the other main paths, changing BM only. Longer local cases benefited, while the smallest direct case regressed. Two subsequent platform aggregates show Kunlunxin changing from **0.8365×** to **1.471×**. This supported retaining BM64/BN16 as the measured baseline at that point; the submissions lack paired per-case timings, so the difference is not a pure causal estimate of BM64. [BM64 feedback](evidence/task-156/02-bm64-feedback.json)

The next control changed BN16 to BN32, placing more rank columns in one tile and potentially increasing work and reuse per program. It passed local regressions and offline builds with the public XPU compiler. On the platform, however, Kunlunxin Case 1 failed during compilation: Linalg inferred an output width of 16 while the accumulator width was 32. The downloaded submission ZIP matched the local archive byte-for-byte; the source loaded after deployment and the precise transformation causing the mismatch remain unavailable.

The error was wrapped as OutOfResources with a uni_sram message, but that is not a measurement of SRAM consumption. Eight further public-compiler alignment controls all built successfully and did not reproduce the platform failure. Uniform BN32 was rejected and BM64/BN16 retained; the triggering compiler conditions were not established. [Compilation-failure record](evidence/task-156/03-bn32-failure.json)

## 4. Fusing zero initialization can also make a path slower

The original output must be initialized. Fusing zeroing with projection appeared to remove a kernel launch, but also changed control flow and live data within the program. In one skewed local CUDA case, a uniform-fusion control achieved only **0.644×** the baseline CUDA Graph speed; its compiled kernel reported 40 spills versus zero for the baseline. The records show these observations together, not that spills alone explain the entire regression. [Uniform-fusion samples](evidence/task-156/11-uniform-fusion-cuda-extract.json)

The later candidate restricted fusion to guarded paths without permutation, outside the short-decode route, and with BN at most 32. Other paths retained the original kernel. Against the same mid64 baseline, 14 self-designed CUDA Graph cases produced different outcomes:

| BF16 shape | Baseline A/A | Baseline / candidate | Speed ratio |
|---|---:|---:|---:|
| Eight segments of length 64, H=128, K=128, R=32 | 1.0025× | 21.967 / 18.624 μs | 1.1795× |
| One segment of length 32, H=8, K=1024, R=32 | 0.9919× | 5.575 / 5.772 μs | 0.9658× |

The run used 50 warmups, rotated/reversed order and nine 100 ms CUDA Graph samples, with synchronized full-wrapper wall samples recorded separately. GPU exclusivity was not verified. The long-K case is a real graph-replay regression in this run; graph timing also excludes host allocation and dispatch. [All 14-case samples, sanitized](evidence/task-156/07-mid64-fusion-cuda-extract.json)

## 5. BN16 for narrow rank, followed by counterexamples

On the then-measured platform baseline, a separate candidate changed only the generic R≤16 path from BN32 to BN16; the Kunlunxin file was unchanged. For a local BF16 long-segment case with H=128, K=64 and R=16, the graph-replay median changed from **18.703 μs** to **15.941 μs**, a **1.1733×** ratio with **1.0010×** A/A. This supported a target trial for narrow rank, not a uniform replacement for every rank. [BN16 samples](evidence/task-156/08-bn16-cuda-extract.json)

The user-provided feedback for filename 01-step_a_q-bn16-for-r16.zip at 16:35 on October 22 displayed a platform average of **10.73×**. The local same-name ZIP has SHA-256 cfeabdae2429744f0bfcde72e6530934c3d61b653c0f7f91c410b76f48d2c1f3e. The remote platform ZIP bytes and the evaluator's loaded source were not independently verified. [Feedback record](evidence/task-156/04-bn16-feedback.json)

Later experiments produced counterexamples:

- **Hygon index32:** The user-reported Hygon metric was **27.16× → 27.02×** relative to the 17:49 control. These are unpaired observations with unknown variance, so they establish neither a gain nor a reliable regression. The candidate was not adopted. [Record](evidence/task-156/05-hygon-index32-feedback.json)
- **Kunlunxin guarded-half:** The candidate had passed 372 local CUDA checks and 35 public-compiler offline builds, but the user-reported target result showed **319/320, 5606/5632 and 1281771/2097152** mismatched outputs across three cases. It was rejected. Final-output errors alone do not identify whether packing, flags, matrix lowering or runtime caused the failure. [Failure record](evidence/task-156/06-halfpack-failure.json)

These are separate submission packages. They do not inherit the BN16 package's result, and per-device maxima from different packages cannot be combined into a new official score.

## 6. Leaderboard result and attribution

The task leaderboard recorded team YuJiacheng at rank 3 and platform-average speedup 10.74142857× at 20:14 on October 22, 2026. Device-label results were Tianshu 9.73×, MetaX 12.11×, Hygon 27.56×, Kunlunxin 1.71×, Huawei 2.71×, international chip A 13.40× and chip B 7.96×.

That **10.74×** leaderboard row and the **10.73×** filename receipt above are separate records. The available evidence does not precisely bind the leaderboard value to the local ZIP. Ranking uses per-device ranks rather than average speedup alone, and the platform's operator result is not an end-to-end SGLang inference speedup.

## Conclusion and sources

The central design choice was to select work granularity by segment length, rank and routing layout while preserving a path the target compiler could use. Local wins did not automatically transfer across shapes or devices: uniform fusion and BN32 encountered performance and target-compilation counterexamples, while guarded-half passed local checks but failed target correctness.

- [Official FlagOS Task 156 statement and reference](https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/step_a_q)
- [FlagGems-sglang project and contribution guide](https://github.com/flagos-ai/FlagGems-sglang)
- Linked feedback files preserve the recorded feedback. Local CUDA benchmark files are sanitized extracts that retain the original artifact SHA-256, parameters and samples.
- The single-variable BM64 direction drew on external research advice. Candidate sources retain FlagOS Contributors and Apache-2.0 notices. OpenAI Codex assisted in preparing this report; existing code attribution and AI disclosures remain in place.

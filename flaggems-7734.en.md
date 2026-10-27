# Handling Non-Contiguous slice_backward Gradients in FlagGems

[FlagGems PR #7734](https://github.com/flagos-ai/FlagGems/pull/7734) · [Issue #7396](https://github.com/flagos-ai/FlagGems/issues/7396)

## Problem

The generic slice_backward kernel reads grad_output using linear offsets. A gradient can instead be an expanded zero-stride view, a transpose or a stepped view with a storage offset. In particular, an expanded tensor can represent many logical values using a single backing element. Linear access without respecting that layout does not implement the logical gradient correctly.

## Implementation

The wrapper materializes grad_output with contiguous() before the existing kernel reads it. For already-contiguous input, this is a no-op; for non-contiguous input, it introduces a copy and temporary storage proportional to the logical gradient size.

The approach follows the materialization proposal by **lvyufeng** in [Issue #7396](https://github.com/flagos-ai/FlagGems/issues/7396). It changes the generic implementation, not vendor-specific overrides.

## Validation

The new layout regressions exposed **36 failures and 36 passes** on the baseline. The candidate's targeted selection reported **150 passed, 468 deselected**; this selection includes more than the baseline's new layout cases. Twelve separate autograd dispatch checks across FP16, BF16 and FP32 reached the generic implementation and matched CPU gradients exactly.

Coverage included expansion, transpose, stepped and offset views, contiguous offset views, negative dimensions and empty gradients. GPU memcheck and vendor-specific overrides were not validated in the recorded run. The PR contains the test command and full environment; no speedup is claimed.

## Attribution

OpenAI Codex assisted with implementation, review, local checks and the PR description, as disclosed in the upstream PR. Codex also assisted with this report.

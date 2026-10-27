# Preserving Integer Equality Semantics in FlagGems

[FlagGems PR #7733](https://github.com/flagos-ai/FlagGems/pull/7733) · [Issue #7584](https://github.com/flagos-ai/FlagGems/issues/7584)

## Problem

The generic equality kernel converted operands to FP32 before comparison. FP32 cannot distinguish every adjacent integer above 2²⁴: for example, converting both 2²⁴ and 2²⁴ + 1 to FP32 can produce the same value. Integer equality must preserve the input values rather than compare rounded floating-point representations.

## Implementation

The generic Triton equality kernels now compare integral operands as integers. Narrow integer and Boolean values are widened to a supported integer width. For mixed integral tensors, the wrapper uses PyTorch result-type rules, including zero-dimensional tensors; integer scalar handling follows the input dtype's wrapping semantics. Boolean scalar arguments are normalized to an integer launch signature.

The Boolean output allocation does not itself determine input promotion. The floating-point path remains in place. Mixed integral types may require a conversion, so this change is a correctness fix rather than a claim of faster comparison.

## Validation

On the generic implementation with PyTorch 2.9.1 and Triton 3.5.1, the same regression selection produced **33 failures and 91 passes** on the baseline and **124 passes** with the candidate. Tests covered integer precision and range boundaries, mixed integer types, zero-dimensional tensors, scalar wrapping, strided inputs and storage offsets. Separate normal dispatch checks matched independent CPU references.

The PR provides the command and full environment. Vendor-specific overrides were outside the tested scope; no performance gain is claimed.

## Attribution

OpenAI Codex assisted with implementation, review, local checks and the PR description, as disclosed in the upstream PR. Codex also assisted with this report.

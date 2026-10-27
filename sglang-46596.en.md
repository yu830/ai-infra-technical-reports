# Measuring Scheduler-to-Detokenizer Router Delivery Latency in SGLang

[SGLang PR #46596](https://github.com/sgl-project/sglang/pull/46596) · [Issue #43304](https://github.com/sgl-project/sglang/issues/43304)

## Problem

With multiple detokenizer workers, a router receives messages from the scheduler and distributes them to workers. Existing router CPU and processing-time metrics do not show how long a message takes to reach the router. A backlog can therefore grow at this boundary without appearing as router processing time.

## Implementation

The patch adds the Prometheus histogram **sglang:detokenizer_router_delivery_seconds** for the metrics-enabled, multiple-detokenizer path. The sender records a monotonic timestamp immediately before serialization and sends it as an eight-byte frame alongside the payload in an atomic multipart message. The router records elapsed time after decoding and before grouping or fan-out. The interval includes serialization, send blocking, IPC waiting and decoding.

The receiver accepts legacy single-frame messages without recording a delivery observation. Metrics-off, single-worker and skip-tokenizer paths retain the existing single-frame communication path. Timestamps rely on a same-host monotonic clock and compatible endpoints.

## Validation

- Focused manager, sender and transport tests: **29 passed, 18 subtests passed**. The current test head is **7040d357**.
- Real ZMQ subprocess tests covered msgpack and pickle, receiver backlog, histogram count and sum, fan-out and legacy messages. Four timed incoming messages produced four observations even when fan-out created seven downstream messages.
- A Qwen2.5-1.5B-Instruct smoke test with two detokenizers ran ordinary, batched and streaming requests with metrics on and off. All four checked final texts matched. With metrics enabled, histogram count rose from **3 to 17**; with metrics disabled, the endpoint returned 404. This run used production commit **37a69e8**; the later **7040d357** commit changes tests only.

The smoke used eager torch_native attention with CUDA Graph disabled. The metric is an observability feature; the measurements do not establish a serving speedup. The upstream PR contains the full test environment, commands and current CI status.

## Attribution

OpenAI Codex assisted with implementation, review, test execution and the PR description, as disclosed in the upstream PR. Codex also assisted with this report.

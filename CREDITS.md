# Credits

This release exists because Khaled Bakeer (0xBakeer) first made
DeepSeek-V4.1-Flash practical on a single DGX Spark and published the engine,
profiling work, and measured optimization path openly.

## Upstream runtime: Khaled Bakeer / 0xBakeer

The runtime is derived from
[`0xBakeer/deepseek-v41-flash-spark`](https://github.com/0xBakeer/deepseek-v41-flash-spark)
at commit `8b68fdde188fea5e8e9cd030f0e1b299dc16ac85`.

Bakeer's work supplied the load-bearing single-Spark implementation: Engram
lookups from NVMe, the resident expert arena and transient slots, CB3 routed
experts, Triton kernels, DSpark speculative decoding, chunked prefill, memory
guards, and the benchmark-driven pruning approach. This release materially
relies on that work. His MIT copyright and license are retained in `LICENSE`.

The release-specific additions are the immutable 154-expert-per-layer
selection, deterministic CB3-v2 packed weights, dense-only retained source
shards, strict manifest and digest validation, trace-free warm start, and the
512K qualification profile.

Bakeer's related
[`qwen38-flash-next-spark`](https://github.com/0xBakeer/qwen38-flash-next-spark)
and [`ling3-flash-spark`](https://github.com/0xBakeer/ling3-flash-spark)
projects also established techniques used by the upstream runtime.

## Model and platform

- [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) created
  DeepSeek-V4.1-Flash, its weights, architecture, tokenizer, encoding, DSpark
  drafter, Engram design, and reference implementation. Those files retain
  DeepSeek's upstream license and notices.
- NVIDIA provides the DGX Spark / GB10 platform, CUDA toolchain, and container
  runtime used for this single-device profile.
- [OpenAI Triton](https://github.com/triton-lang/triton) provides the kernel
  language and JIT used by the GPU kernels.

Release packaging, immutable K154 artifact construction, verification, and
512K qualification by `sayyidfareed`.

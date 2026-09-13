---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash
library_name: pytorch
pipeline_tag: text-generation
tags:
- deepseek
- deepseek-v4.1
- dgx-spark
- gb10
- long-context
- 512k-context
- mixture-of-experts
- cb3
- speculative-decoding
---

# DeepSeek-V4.1-Flash-Next-DGX-Spark-512K

**A self-contained, 512K-qualified DeepSeek-V4.1-Flash deployment for one 128
GB NVIDIA DGX Spark.**

On an ASUS GX10, this fixed K154 release handled a 524,293-token prompt,
returned all five buried values, and generated 44 more tokens. Warm short coding
prompts decoded at a 27.79 tok/s median with 1.32 s median TTFT.

The released DeepSeek-V4.1-Flash checkpoint is about 510 GB—far beyond the
memory of one Spark. Khaled Bakeer
([0xBakeer](https://github.com/0xBakeer)) built and published the single-Spark
runtime that made this possible: Engram lookups from NVMe, a resident expert
arena, CB3 experts, Triton kernels, DSpark speculative decoding, and the
benchmark-driven pruning path.

This release takes the next deployment step. It permanently packages 154 routed
experts in each of the 40 main layers, removes the original main routed-expert
tensors, and verifies every retained shard before serving. There is no routing
trace to supply, no machine-local warm-start artifact, and no expert
re-quantization at startup.

> This repository contains the complete self-contained model payload: retained
> DeepSeek weights, the 40-layer K154 CB3-v2 expert pack, tokenizer, encoding,
> configuration, DSpark drafter, manifest, and digests. The matching runtime is
> [`sayyidfareed/deepseek-v4.1-flash-next-dgx-spark-512k`](https://github.com/sayyidfareed/deepseek-v4.1-flash-next-dgx-spark-512k),
> tag `v1.0.0`.

## Measured on the GX10

| Test | Result |
|---|---:|
| Engine capacity | 1,048,576 tokens |
| Validated request | 524,293 prompt + 44 output = **524,337 tokens** |
| Distributed needle retrieval | **5/5** at 5%, 25%, 50%, 75%, 95% |
| Long-context prefill | **147.69 tok/s** |
| Long-context decode | **5.69 tok/s** |
| Median decode, five coding languages | **27.794 tok/s** |
| Minimum warm decode | **25.235 tok/s** |
| Maximum warm decode | **33.438 tok/s** |
| Median warm TTFT | **1.324 s** |
| HumanEval | **150/164 (91.5%)** |
| HumanEval+ Mini | **142/164 (86.6%)** |
| Executable coding microbenchmark | **29/34** |
| Minimum available memory during 512K prefill | **7.09 GiB** |
| New swap used during 512K request | **1.17 GiB** |

Hardware: NVIDIA GX10 / GB10 with 121.63 GiB usable unified memory and local
NVMe. The model service was the only large workload on the box. Short-speed
numbers are the second identical five-language pass after graph warmup; the
first pass contained one first-shape compilation outlier and is retained in the
qualification evidence rather than silently mixed into the warm result.

## What K154 means

DeepSeek has 384 routed expert blocks in each of 40 main layers. For every
token, the router normally picks six. The complete checkpoint contains 15,360
of those expert blocks, which is too much to keep in a Spark's unified memory.

K154 means this release keeps a fixed set of **154 experts per layer**:
6,160 total. Those experts are stored in a compact three-bit-per-row CB3-v2
format and loaded into the 89.16 GB resident arena. The router is restricted to
that exact set, so serving does not fall back to missing source experts.

This is not a retrained model. Dense weights, Engram tables, tokenizer,
configuration, and DSpark drafter come from DeepSeek's checkpoint. The tradeoff
is deliberate: fixed-expert pruning makes a 748B-class model practical on one
128 GB machine, but it can change quality outside the coding and reasoning
workloads used to choose and validate the set.

## Why a custom package was necessary

The original checkpoint's two Engram tables account for about 203 GB and stay
on NVMe. The routed experts account for another 288.78 GB. Keeping the original
routed tensors alongside a resident packed copy would make a supposedly
publishable model hundreds of gigabytes larger than necessary.

This repository therefore contains:

- 221,508,192,600 bytes of retained non-main-routed tensors;
- 89,041,469,440 bytes of K154 CB3-v2 routed experts;
- 310,549,662,040 tensor bytes in total, about 289.22 GiB;
- no original main routed-expert tensors.

The 40 rewritten base shards and 40 CB3 layer shards are authenticated by the
manifest. Startup fails before the API port binds if a shard, tensor layout,
selection, or source identity differs.

## Quickstart

Requirements:

- DGX Spark, ASUS GX10, or compatible GB10 system with 128 GB unified memory;
- NVIDIA container runtime and Docker;
- at least 330 GB free on fast local NVMe;
- Hugging Face CLI for the roughly 311 GB model download;
- about 17 minutes for the tested cold integrity check and model load.

```bash
hf download sayyidfareed/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K \
  --local-dir /models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K

git clone --branch v1.0.0 \
  https://github.com/sayyidfareed/deepseek-v4.1-flash-next-dgx-spark-512k.git
cd deepseek-v4.1-flash-next-dgx-spark-512k

MODEL_PARENT=/models docker compose -f compose.k154.yaml up --build
```

The server is ready when the log says `serving
DeepSeek-V4.1-Flash-Next-DGX-Spark-512K`. It exposes an OpenAI-compatible API
on `127.0.0.1:8000` by default:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V4.1-Flash-Next-DGX-Spark-512K",
    "messages": [{"role":"user","content":"Write a lock-free ring buffer in Rust."}],
    "temperature": 0,
    "max_tokens": 1024,
    "stream": true,
    "tool_choice": "none"
  }'
```

The API has no built-in authentication. Keep it on loopback or place an
authenticated proxy and trusted-network boundary in front of it.

## The qualified serving profile

| Setting | Value |
|---|---:|
| Fixed experts | 154 per layer × 40 layers = 6,160 |
| Expert format | CB3-v2, three bits per row |
| Resident arena | 89.16 GB |
| Transient slots | 8 |
| Free-memory floor | 2.5 GiB |
| Maximum sequences | 1 |
| Engine capacity | 1,048,576 tokens |
| Prefill chunk | 256 tokens |
| Sparse-window replay | enabled |
| DSpark speculative decoding | enabled |
| Dense FP4 groups | attention and `wo_a` |
| Head format | FP8 |

The immutable launcher supplies these values and refuses routing-trace or
arbitrary extra flags. Changing the runtime, expert set, memory floor, or
kernels creates a new profile that needs fresh qualification.

The qualified container used Ubuntu 24.04, Python 3.12.3, CUDA 13.0,
PyTorch 2.13.0+cu130, Triton 3.7.1, Transformers 5.17.0, Tokenizers 0.23.2,
Safetensors 0.8.0, NumPy 2.3.5, and SymPy 1.14.0. The Dockerfile pins the CUDA
base by digest and the Python package versions used by the release.

## Short-context quality check

The static package was rerun through the complete 164-problem HumanEval+
Mini generation set and the frozen executable coding microbenchmark. Generated
programs were executed in network-disabled containers. The model produced:

- HumanEval: **150/164**;
- HumanEval+ Mini: **142/164**;
- executable microbenchmark: **29/34**.

The 29/34 microbenchmark result exactly matches the earlier accepted K154
runtime. This is a bounded coding qualification, not a general benchmark claim.

## Reproduce the 512K test

Start with a small smoke request:

```bash
python3 bench/long_context_probe.py \
  --base-url http://127.0.0.1:8000 \
  --target-tokens 5000 \
  --tokenizer-json /models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/tokenizer.json \
  --output smoke-5k.json
```

The exact accepted run takes about one hour on the tested machine:

```bash
python3 bench/long_context_probe.py \
  --base-url http://127.0.0.1:8000 \
  --target-tokens 522744 \
  --max-output-tokens 128 \
  --timeout 7200 \
  --tokenizer-json /models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/tokenizer.json \
  --expected-prompt-tokens 524293 \
  --prompt-token-tolerance 0 \
  --expected-prompt-sha256 af9b0c94b91caf0fd53f6bc324236d3d94d9070740b88b66e7f25cf2c74e2c54 \
  --expected-completion-tokens 44 \
  --min-prefill-tok-s 145 \
  --min-decode-tok-s 4 \
  --output reproduction-512k.json
```

The probe generates deterministic varied archive records and inserts five
unique values at roughly 5%, 25%, 50%, 75%, and 95%. It is a retrieval and
operational-stability test, not a comprehensive long-context reasoning test.

## Operational limits

- **512K is qualified; 1M is not.** The engine reserves a 1,048,576-token
  capacity, but the successful evidence stops at 524,337 total tokens.
- **This is a single-sequence profile.** It is tuned for one large request, not
  aggregate multi-user throughput.
- **Do not co-locate another large model.** CPU, page cache, GPU allocations,
  and KV cache share the same 128 GB pool.
- **Use fast local NVMe.** The 203 GB Engram lookup tables remain on storage and
  are accessed during inference.
- **Expect a long cold start.** Integrity hashing and deterministic loading are
  intentional; warm-request TTFT does not include them.
- **The fixed expert set is workload-sensitive.** Re-evaluate quality before
  using it as a specialist outside the validated coding/reasoning profile.
- Multimodal input, multi-user concurrency, and 1M-context operation were not
  part of this qualification.

## Reproducibility

| Component | Pinned identity |
|---|---|
| Source checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash` |
| Source revision | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Upstream runtime | `0xBakeer/deepseek-v41-flash-spark` |
| Upstream commit | `8b68fdde188fea5e8e9cd030f0e1b299dc16ac85` |
| K154 selection SHA-256 | `783ae482c5ffd5b8bdd154ec96462650e00452ca36e8f0cb74101c9ab703d811` |
| Release index SHA-256 | `09e45cbf8c2c370d4f7549af591c5c7375c9699abdfad49c8ea5bc615ed16561` |
| Exact 512K prompt SHA-256 | `af9b0c94b91caf0fd53f6bc324236d3d94d9070740b88b66e7f25cf2c74e2c54` |

## Credits and licenses

- DeepSeek created DeepSeek-V4.1-Flash and released the source checkpoint.
- **Khaled Bakeer (0xBakeer)** created the upstream single-DGX-Spark runtime and
  the core Engram, expert-residency, CB3, kernel, memory, and optimization work
  this release materially depends on. See
  [`0xBakeer/deepseek-v41-flash-spark`](https://github.com/0xBakeer/deepseek-v41-flash-spark)
  and `CREDITS.md`.
- NVIDIA provides the DGX Spark / GB10 and CUDA stack; OpenAI Triton provides
  the kernel language and JIT.

The runtime code is MIT licensed. DeepSeek-derived weights and reference files
retain their upstream license. See `NOTICE` before redistribution.

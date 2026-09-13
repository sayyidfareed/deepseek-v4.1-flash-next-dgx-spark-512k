# DeepSeek-V4.1-Flash-Next-DGX-Spark-512K

**A self-contained, 512K-qualified DeepSeek-V4.1-Flash deployment for one 128
GB NVIDIA DGX Spark.** On an ASUS GX10, the release handled a 524,293-token
prompt, recovered five values buried across the full prompt, and generated 44
more tokens. Warm short coding prompts decoded at a 27.79 tok/s median with
1.32 s median TTFT.

The original DeepSeek checkpoint is about 510 GB. Khaled Bakeer
([0xBakeer](https://github.com/0xBakeer)) built the open single-Spark runtime
that made the model practical: Engram lookup from NVMe, a resident expert
arena, CB3 experts, Triton kernels, DSpark speculative decoding, and a measured
expert-pruning path.

This release packages one fixed K154 deployment. It carries 154 routed experts
for each of the 40 main layers, removes the original main routed-expert tensors,
and validates every retained shard before the API binds. Startup does not need
a routing trace, a machine-local result directory, or runtime re-quantization.

The complete weights are public at
[`sayyidfareed/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K`](https://huggingface.co/sayyidfareed/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K).

## Measured on the GX10

| Test | Result |
|---|---:|
| Engine capacity | 1,048,576 tokens |
| Validated request | 524,293 prompt + 44 output = **524,337 tokens** |
| Distributed needle retrieval | **5/5** at 5%, 25%, 50%, 75%, 95% |
| Long-context prefill | **147.69 tok/s** |
| Long-context decode | **5.69 tok/s** |
| Median warm decode, five coding languages | **27.794 tok/s** |
| Minimum warm decode | **25.235 tok/s** |
| Maximum warm decode | **33.438 tok/s** |
| Median warm TTFT | **1.324 s** |
| HumanEval | **150/164 (91.5%)** |
| HumanEval+ Mini | **142/164 (86.6%)** |
| Executable coding microbenchmark | **29/34** |
| Minimum available memory during 512K prefill | **7.09 GiB** |
| New swap used during 512K request | **1.17 GiB** |

Hardware: NVIDIA GX10 / GB10 with 121.63 GiB usable unified memory and local
NVMe. The model service was the only large workload on the box. The short-speed
row is the second identical pass after graph warmup; the first pass is kept in
the evidence because one first-shape compilation sample fell below steady-state
speed.

## What K154 means

DeepSeek has 384 routed expert blocks in each of 40 main layers. For each token,
the router normally chooses six. The complete checkpoint contains 15,360 of
these blocks—far too many to keep in one Spark's unified memory.

K154 keeps a fixed set of **154 experts per layer**, or 6,160 total, in a compact
three-bit-per-row CB3-v2 representation. They occupy the 89.16 GB resident arena.
The router is restricted to that exact set, so the release cannot silently fall
back to an omitted source expert.

This is not a retrained model. Dense weights, Engram tables, tokenizer,
configuration, and the DSpark drafter derive from DeepSeek's checkpoint. The
fixed expert set makes the deployment fit, but it can change quality outside
the coding and reasoning workloads used to choose and validate it.

## Quickstart

Requirements:

- DGX Spark, ASUS GX10, or compatible GB10 with 128 GB unified memory;
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

The API has no built-in authentication. The launcher defaults to loopback. The
Compose profile binds inside the container but publishes only to host
`127.0.0.1`. Override that boundary only behind a trusted network or an
authenticated reverse proxy.

## The qualified profile

```text
experts per layer = 154
resident experts  = 6,160
expert format     = CB3-v2
arena             = 89.16 GB
transient slots   = 8
free-memory floor = 2.5 GiB
maximum sequences = 1
engine capacity   = 1,048,576 tokens
prefill chunk     = 256 tokens
DSpark            = enabled
sparse replay     = enabled
dense FP4 groups  = attention, wo_a
head format       = FP8
```

Start through `scripts/k154_entrypoint.sh` or `Dockerfile.k154`. The launcher
always supplies the immutable expert pack and rejects routing traces and
arbitrary extra flags. The engine hashes the release index, retained base
shards, selection, and all 40 CB3 shards before it binds the API.

### Runtime lock

| Component | Qualified version |
|---|---|
| OS | Ubuntu 24.04 |
| Python | 3.12.3 |
| CUDA | 13.0 |
| GPU target | NVIDIA GB10, `sm_121a` |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1, installed with PyTorch |
| Transformers | 5.17.0 |
| Tokenizers | 0.23.2 |
| Safetensors | 0.8.0 |
| NumPy | 2.3.5 |
| SymPy | 1.14.0 |

The Dockerfile pins the CUDA base by its ARM64 image digest and pins every
direct Python dependency to the version used by the qualified container.

## Reproduce the 512K test

First run a 5K smoke probe:

```bash
python3 bench/long_context_probe.py \
  --base-url http://127.0.0.1:8000 \
  --target-tokens 5000 \
  --tokenizer-json /models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/tokenizer.json \
  --output smoke-5k.json
```

Then run the exact accepted profile, which takes about one hour on the tested
GX10:

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

The prompt is deterministic and its accepted SHA-256 is
`af9b0c94b91caf0fd53f6bc324236d3d94d9070740b88b66e7f25cf2c74e2c54`.
The five values are placed at roughly 5%, 25%, 50%, 75%, and 95%. This tests
retrieval and operational stability, not every kind of long-context reasoning.

## Operational limits

- **512K is qualified; 1M is not.** The engine reserves a 1,048,576-token
  capacity, but the successful evidence stops at 524,337 total tokens.
- **This is a single-sequence profile.** It is tuned for one large request, not
  aggregate multi-user throughput.
- **Do not co-locate another large model.** CPU, page cache, GPU allocations,
  and KV cache share the same 128 GB pool.
- **Use fast local NVMe.** The 203 GB Engram tables remain on storage and are
  accessed during inference.
- **Expect a long cold start.** Integrity hashing and deterministic loading are
  intentional; warm-request TTFT does not include them.
- **The fixed expert set is workload-sensitive.** Re-evaluate quality for a
  different specialty before treating this as a general-purpose quant.
- Multimodal input, multi-user concurrency, and 1M-context operation were not
  part of this qualification.

## Reproducibility

| Component | Pinned identity |
|---|---|
| Model repository | `sayyidfareed/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K` |
| Source checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash` |
| Source revision | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Upstream runtime | `0xBakeer/deepseek-v41-flash-spark` |
| Upstream commit | `8b68fdde188fea5e8e9cd030f0e1b299dc16ac85` |
| K154 selection SHA-256 | `783ae482c5ffd5b8bdd154ec96462650e00452ca36e8f0cb74101c9ab703d811` |
| Manifest SHA-256 | `eb1d4b432609c9d61e82685de8a42dbf6959af4248563aa5070587967a07b17d` |
| Release index SHA-256 | `09e45cbf8c2c370d4f7549af591c5c7375c9699abdfad49c8ea5bc615ed16561` |

## Credits and licenses

This release materially relies on **Khaled Bakeer (0xBakeer)** and his
[`deepseek-v41-flash-spark`](https://github.com/0xBakeer/deepseek-v41-flash-spark)
engine at commit `8b68fdde188fea5e8e9cd030f0e1b299dc16ac85`. Bakeer's work supplied the
single-Spark architecture, Engram and expert offload, CB3 path, kernels, DSpark,
memory controls, and optimization methodology. His MIT copyright is retained
in [LICENSE](LICENSE), with fuller attribution in [CREDITS.md](CREDITS.md).

DeepSeek-V4.1-Flash weights and reference files retain the upstream DeepSeek
license. The MIT license for this runtime does not replace or extend the model
weight license; see [NOTICE](NOTICE).

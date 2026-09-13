#!/usr/bin/env python3
"""Bounded long-context retrieval probe for an OpenAI-compatible local server.

The probe creates deterministic, varied archive records, inserts five unique
needles through the prompt, requests all values, and records token counts and
timings as JSON. It intentionally generates only one request at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request


WORDS = (
    "amber birch cedar delta ember frost granite harbor indigo juniper kinetic "
    "lantern maple nickel orchid pebble quartz river saffron timber umber violet "
    "willow xenon yellow zephyr atlas beacon copper drift elm fjord galaxy helix "
    "island jasper kelp lunar meadow north opal prairie quiet ridge solar tundra"
).split()

NEEDLES = [
    ("ALPHA", "K7-VIOLET-314159"),
    ("BRAVO", "M2-CEDAR-271828"),
    ("CHARLIE", "R9-QUARTZ-161803"),
    ("DELTA", "T4-HARBOR-141421"),
    ("ECHO", "W8-JUNIPER-173205"),
]

RELEASE_MODEL = "DeepSeek-V4.1-Flash-Next-DGX-Spark-512K"
RELEASE_EXPERT_PACK_SUFFIX = "/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/k154-cb3"
RELEASE_ENGINE_CONFIG = {
    "engine": "v41",
    "arena_slots": 6168,
    "lru_slots": 6160,
    "transient_slots": 8,
    "max_seq": 1048576,
    "spec": True,
    "trace_stats": None,
    "kernel": "triton-cb3",
    "expert_format": "cb3",
    "dense_fp4": "attn,wo_a",
    "head_fmt": "fp8",
    "routed_topk": 6,
    "swa_replay": True,
    "prune_select": "manifest",
    "prune_rank": "manifest",
    "prefill_chunk": 256,
    "fast_reference_shapes": False,
}

# These probes target an explicitly supplied local or trusted-network server. Do not let
# workstation HTTP proxy variables redirect multi-megabyte prompts elsewhere.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post_json(base: str, path: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body, separators=(",", ":")).encode()
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with OPENER.open(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:4000]}") from exc


def get_json(base: str, path: str, timeout: float) -> dict:
    with OPENER.open(base.rstrip("/") + path, timeout=timeout) as response:
        return json.load(response)


def validate_release_health(health: dict, *, expect_idle: bool) -> None:
    """Reject a healthy-looking server that is not the qualified K154 profile."""
    if health.get("status") != "ok":
        raise RuntimeError(f"server is not healthy: {health.get('status')!r}")
    if health.get("model") != RELEASE_MODEL:
        raise RuntimeError(f"wrong served model: {health.get('model')!r}")
    if health.get("max_context") != 1048576:
        raise RuntimeError(f"wrong max_context: {health.get('max_context')!r}")
    if expect_idle and health.get("busy") is not False:
        raise RuntimeError("server must be idle before the acceptance request")
    config = health.get("engine_config")
    if not isinstance(config, dict):
        raise RuntimeError("health response lacks engine_config")
    for key, expected in RELEASE_ENGINE_CONFIG.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"wrong engine_config.{key}: {config.get(key)!r}; expected {expected!r}"
            )
    if abs(float(config.get("arena_gb", -1)) - 89.2) > 0.05:
        raise RuntimeError(f"wrong engine_config.arena_gb: {config.get('arena_gb')!r}")
    if not str(config.get("expert_pack", "")).endswith(RELEASE_EXPERT_PACK_SUFFIX):
        raise RuntimeError(f"wrong engine_config.expert_pack: {config.get('expert_pack')!r}")


def record(i: int) -> str:
    # Deterministic but sufficiently varied to avoid an unrealistically tiny PLE
    # hot set from one repeated sentence.
    x = (i * 1103515245 + 12345) & 0x7FFFFFFF
    chosen = [WORDS[(x >> (j % 19) ^ i * (j + 3)) % len(WORDS)] for j in range(18)]
    return f"Archive record {i:07d}: " + " ".join(chosen) + ".\n"


def make_prompt(target_tokens: int, chars_per_token: float) -> str:
    intro = (
        "Read the archive carefully. Five IMPORTANT records contain secret values. "
        "Ignore ordinary records and retain every labelled secret value.\n"
    )
    question = (
        "\nEND OF ARCHIVE. Return the five secret values for ALPHA, BRAVO, CHARLIE, "
        "DELTA, and ECHO in that order. Output only the values separated by commas.\n"
    )
    target_chars = max(10_000, int(target_tokens * chars_per_token))
    needle_at = [int(target_chars * fraction) for fraction in (0.05, 0.25, 0.50, 0.75, 0.95)]
    parts = [intro]
    chars = len(intro)
    inserted = 0
    i = 0
    while chars < target_chars - len(question):
        if inserted < len(NEEDLES) and chars >= needle_at[inserted]:
            label, value = NEEDLES[inserted]
            line = (
                f"IMPORTANT RECORD {label}: The secret value for {label} is {value}. "
                f"Remember {value} exactly.\n"
            )
            parts.append(line)
            chars += len(line)
            inserted += 1
        line = record(i)
        parts.append(line)
        chars += len(line)
        i += 1
    while inserted < len(NEEDLES):
        label, value = NEEDLES[inserted]
        parts.append(f"IMPORTANT RECORD {label}: The secret value for {label} is {value}.\n")
        inserted += 1
    parts.append(question)
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--model", default=RELEASE_MODEL
    )
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--output")
    parser.add_argument("--expected-prompt-tokens", type=int)
    parser.add_argument("--prompt-token-tolerance", type=int, default=32)
    parser.add_argument("--expected-prompt-sha256")
    parser.add_argument("--expected-completion-tokens", type=int)
    parser.add_argument("--min-prefill-tok-s", type=float)
    parser.add_argument("--min-decode-tok-s", type=float)
    args = parser.parse_args()
    if args.model != RELEASE_MODEL:
        raise SystemExit(f"ERROR: this acceptance probe requires model {RELEASE_MODEL}")
    health_before = get_json(args.base_url, "/health", min(args.timeout, 120))
    validate_release_health(health_before, expect_idle=True)

    # Calibrate text density on representative records without submitting a
    # second ultra-long request merely to count tokens.
    sample = "".join(record(i) for i in range(400))
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer_json)
    sample_tokens = len(tokenizer.encode(sample).ids)
    calibration = {"count": sample_tokens, "source": args.tokenizer_json}
    if not sample_tokens:
        raise RuntimeError(f"tokenizer response lacks a count: {calibration.keys()}")
    chars_per_token = len(sample) / sample_tokens
    prompt = make_prompt(args.target_tokens, chars_per_token)
    exact_prompt_tokens = len(tokenizer.encode(prompt).ids)
    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
    if (
        args.expected_prompt_sha256 is not None
        and prompt_sha256 != args.expected_prompt_sha256
    ):
        raise RuntimeError(
            f"prompt SHA-256 is {prompt_sha256}; expected {args.expected_prompt_sha256}"
        )
    debug_prompt = post_json(
        args.base_url,
        "/v1/debug/prompt",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        min(args.timeout, 300),
    )
    rendered_prompt_tokens = debug_prompt.get("prompt_tokens")
    preflight = {
        "model": args.model,
        "target_tokens": args.target_tokens,
        "prompt_characters": len(prompt),
        "locally_tokenized_prompt_tokens": exact_prompt_tokens,
        "rendered_prompt_tokens": rendered_prompt_tokens,
        "prompt_sha256": prompt_sha256,
        "calibrated_characters_per_token": chars_per_token,
    }
    if args.output:
        with open(args.output + ".preflight.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(preflight, indent=2) + "\n")
    print(json.dumps({"preflight": preflight}, indent=2), flush=True)
    if args.expected_prompt_tokens is not None:
        if rendered_prompt_tokens is None:
            raise RuntimeError("debug prompt response lacks prompt_tokens")
        delta = abs(rendered_prompt_tokens - args.expected_prompt_tokens)
        if delta > args.prompt_token_tolerance:
            raise RuntimeError(
                f"rendered prompt is {rendered_prompt_tokens} tokens; expected "
                f"{args.expected_prompt_tokens} +/- {args.prompt_token_tolerance}"
            )

    started = time.time()
    response = post_json(
        args.base_url,
        "/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": args.max_output_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        args.timeout,
    )
    elapsed = time.time() - started
    message = response["choices"][0]["message"]
    content = message.get("content") or ""
    found = {label: value in content for label, value in NEEDLES}
    usage = response.get("usage", {})
    engine_stats = response.get("x_engine_stats") or response.get("engine_stats")
    health_after = get_json(args.base_url, "/health", min(args.timeout, 120))
    validate_release_health(health_after, expect_idle=True)
    result = {
        "model": args.model,
        "health_before": health_before,
        "target_tokens": args.target_tokens,
        "prompt_characters": len(prompt),
        "locally_tokenized_prompt_tokens": exact_prompt_tokens,
        "rendered_prompt_tokens": rendered_prompt_tokens,
        "prompt_sha256": prompt_sha256,
        "calibrated_characters_per_token": chars_per_token,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_seconds": elapsed,
        "needles_found": found,
        "score": f"{sum(found.values())}/{len(found)}",
        "response": content,
        "finish_reason": response["choices"][0].get("finish_reason"),
        "usage": usage,
        "engine_stats": engine_stats,
        "health_after": health_after,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(rendered + "\n")
    failures = []
    if not all(found.values()):
        failures.append("not all five needles were returned")
    if usage.get("prompt_tokens") != rendered_prompt_tokens:
        failures.append(
            f"usage.prompt_tokens={usage.get('prompt_tokens')!r} does not match "
            f"rendered prompt tokens={rendered_prompt_tokens!r}"
        )
    if (
        args.expected_completion_tokens is not None
        and usage.get("completion_tokens") != args.expected_completion_tokens
    ):
        failures.append(
            f"usage.completion_tokens={usage.get('completion_tokens')!r}; expected "
            f"{args.expected_completion_tokens}"
        )
    if not isinstance(engine_stats, dict):
        failures.append("response lacks engine timing statistics")
    else:
        if engine_stats.get("expert_hit_rate") != 1.0:
            failures.append(
                f"engine expert_hit_rate={engine_stats.get('expert_hit_rate')!r}; expected 1.0"
            )
        if engine_stats.get("expert_misses") != 0:
            failures.append(
                f"engine expert_misses={engine_stats.get('expert_misses')!r}; expected 0"
            )
        if engine_stats.get("prefill_expert_misses") != 0:
            failures.append(
                "engine prefill_expert_misses="
                f"{engine_stats.get('prefill_expert_misses')!r}; expected 0"
            )
        for flag, field in (
            (args.min_prefill_tok_s, "prefill_tok_s"),
            (args.min_decode_tok_s, "decode_tok_s"),
        ):
            if flag is not None:
                measured = engine_stats.get(field)
                if not isinstance(measured, (int, float)) or measured < flag:
                    failures.append(f"engine {field}={measured!r}; minimum {flag}")
    if failures:
        raise SystemExit("ERROR: " + "; ".join(failures))


if __name__ == "__main__":
    main()

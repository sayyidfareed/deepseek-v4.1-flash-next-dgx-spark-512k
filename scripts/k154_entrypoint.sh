#!/usr/bin/env bash
# Immutable public launcher for DeepSeek-V4.1-Flash-Next-DGX-Spark-512K.
# It deliberately has no trace-selection or generic warm-start route.
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# This entrypoint is the qualified release profile, not a generic tuning
# surface. Force every model-shape and kernel choice that can change the
# published quality, speed, or long-context result. HOST and PORT remain
# configurable because they do not change inference.
readonly MODEL_DIR="/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K"
readonly EXPERT_PACK="$MODEL_DIR/k154-cb3"
readonly SERVED_MODEL_NAME="DeepSeek-V4.1-Flash-Next-DGX-Spark-512K"
readonly MAX_SEQ="1048576"
readonly ARENA_GB="89.16"
readonly KEEP_FREE_GB="2.5"
readonly SPEC="1"

export DSV41_DENSE_FP4="attn,wo_a"
export DSV41_HEAD_FMT="fp8"
export DSV41_HEAD_FP32="0"
export DSV41_PREFILL_CHUNK="256"
export DSV41_RING="4096"
export DSV41_SWA_REPLAY="1"
export DSV41_TOPK="6"
export DSV41_BLOCK="5"
export DSV41_FAST_REFERENCE_SHAPES="0"
export DSV41_MEM_FLOOR_GB="2.5"
export DSV41_DENSE_FP8="1"
export DSV41_WOA_FP8="1"
export DSV41_FP4_DENSE_PREFILL="kernel"
export DSV41_FP4_DENSE_F16="1"
export DSV41_CB3_PREFILL="fp4"
export DSV41_CB3_PREFILL_MIN_P="65"
export DSV41_FAST="1"
export DSV41_GRAPHS="1"
export DSV41_LUT="1"
export DSV41_HC_KERNEL="1"
export DSV41_FUSED_ATTN="0"
export DSV41_GRAPH_SEGMENTS="1"
export DSV41_LEAN_STEP="1"
export DSV41_CYCLE_BREAK="1"

: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"
: "${DEFAULT_THINKING:=off}"
: "${DEFAULT_EFFORT:=75}"

[[ -z "${TRACE_STATS:-}" ]] || die 'TRACE_STATS is forbidden for the immutable K154 release'
[[ -z "${EXTRA_FLAGS:-}" ]] || die 'EXTRA_FLAGS is disabled for the immutable K154 release'
[[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || die "missing release index in $MODEL_DIR"
[[ -f "$EXPERT_PACK/manifest.json" ]] || die "missing K154 manifest in $EXPERT_PACK"
[[ -f "$EXPERT_PACK/k154-selection.json" ]] || die "missing immutable K154 selection in $EXPERT_PACK"
case "$DEFAULT_THINKING" in on|off) ;; *) die 'DEFAULT_THINKING must be on or off' ;; esac
case "$SPEC" in 0|1) ;; *) die 'SPEC must be 0 or 1' ;; esac

flags=(--model-dir "$MODEL_DIR" --host "$HOST" --port "$PORT"
       --served-model-name "$SERVED_MODEL_NAME" --default-thinking "$DEFAULT_THINKING"
       --default-effort "$DEFAULT_EFFORT" --engine v41 --max-seq "$MAX_SEQ"
       --arena-gb "$ARENA_GB" --engine-kwargs
       "{\"expert_pack\":\"$EXPERT_PACK\",\"expert_format\":\"cb3\",\"transient_slots\":8,\"keep_free_gb\":$KEEP_FREE_GB}")
[[ "$SPEC" == 0 ]] && flags+=(--no-spec)
exec python3 server/app.py "${flags[@]}"

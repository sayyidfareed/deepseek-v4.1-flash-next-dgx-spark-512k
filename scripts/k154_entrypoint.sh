#!/usr/bin/env bash
# Immutable public launcher for DeepSeek-V4.1-Flash-Next-DGX-Spark-512K.
# It deliberately has no trace-selection or generic warm-start route.
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
: "${MODEL_DIR:=/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K}"
: "${EXPERT_PACK:=$MODEL_DIR/k154-cb3}"
: "${SERVED_MODEL_NAME:=DeepSeek-V4.1-Flash-Next-DGX-Spark-512K}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"
: "${MAX_SEQ:=1048576}"
: "${ARENA_GB:=89.16}"
: "${KEEP_FREE_GB:=2.5}"
: "${DEFAULT_THINKING:=off}"
: "${DEFAULT_EFFORT:=75}"
: "${SPEC:=1}"

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

#!/usr/bin/env bash
set -euo pipefail
f="${1:-scripts/k154_entrypoint.sh}"
bash -n "$f"
grep -Fq 'TRACE_STATS is forbidden' "$f"
grep -Fq 'server/app.py' "$f"
grep -Fq '\"expert_pack\"' "$f"
grep -Fq '\"expert_format\":\"cb3\"' "$f"
grep -Fq '\"transient_slots\":8' "$f"
! grep -Fq -- '--trace-stats' "$f"

fixed=(
  'readonly MODEL_DIR="/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K"'
  'readonly EXPERT_PACK="$MODEL_DIR/k154-cb3"'
  'readonly SERVED_MODEL_NAME="DeepSeek-V4.1-Flash-Next-DGX-Spark-512K"'
  'readonly MAX_SEQ="1048576"'
  'readonly ARENA_GB="89.16"'
  'readonly KEEP_FREE_GB="2.5"'
  'readonly SPEC="1"'
  'export DSV41_DENSE_FP4="attn,wo_a"'
  'export DSV41_HEAD_FMT="fp8"'
  'export DSV41_HEAD_FP32="0"'
  'export DSV41_PREFILL_CHUNK="256"'
  'export DSV41_RING="4096"'
  'export DSV41_SWA_REPLAY="1"'
  'export DSV41_TOPK="6"'
  'export DSV41_BLOCK="5"'
  'export DSV41_FAST_REFERENCE_SHAPES="0"'
  'export DSV41_MEM_FLOOR_GB="2.5"'
  'export DSV41_DENSE_FP8="1"'
  'export DSV41_WOA_FP8="1"'
  'export DSV41_FP4_DENSE_PREFILL="kernel"'
  'export DSV41_FP4_DENSE_F16="1"'
  'export DSV41_CB3_PREFILL="fp4"'
  'export DSV41_CB3_PREFILL_MIN_P="65"'
  'export DSV41_FAST="1"'
  'export DSV41_GRAPHS="1"'
  'export DSV41_LUT="1"'
  'export DSV41_HC_KERNEL="1"'
  'export DSV41_FUSED_ATTN="0"'
  'export DSV41_GRAPH_SEGMENTS="1"'
  'export DSV41_LEAN_STEP="1"'
  'export DSV41_CYCLE_BREAK="1"'
)
for contract in "${fixed[@]}"; do
  grep -Fqx "$contract" "$f"
done

#!/usr/bin/env bash
set -euo pipefail
f="${1:-scripts/k154_entrypoint.sh}"
grep -Fq 'TRACE_STATS is forbidden' "$f"
grep -Fq 'server/app.py' "$f"
grep -Fq '\"expert_pack\"' "$f"
grep -Fq '\"expert_format\":\"cb3\"' "$f"
grep -Fq '\"transient_slots\":8' "$f"
! grep -Fq -- '--trace-stats' "$f"

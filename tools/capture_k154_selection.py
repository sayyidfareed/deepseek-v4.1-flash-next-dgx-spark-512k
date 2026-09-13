#!/usr/bin/env python3
"""Capture the accepted K154 coding selection as an immutable public artifact.

Run once against the identified qualification inputs.  The resulting JSON is
the only selection input accepted by the exporter; serving never reads a trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.prepacked_cb3 import canonical_json

K154, LAYERS, EXPERTS = 154, 40, 384


def digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(coverage_path: str, qualification_path: str) -> dict:
    coverage = json.loads(Path(coverage_path).read_text(encoding="utf-8"))
    qualification = json.loads(Path(qualification_path).read_text(encoding="utf-8"))
    stats = qualification.get("engine_stats", {})
    if (qualification.get("prompt_tokens"), qualification.get("completion_tokens"), qualification.get("score"),
            stats.get("max_seq")) != (524293, 44, "5/5", 1048576):
        raise ValueError("qualification is not the accepted 524293+44, 5/5, max_seq=1048576 record")
    per_layer = coverage.get("per_layer")
    if not isinstance(per_layer, dict) or len(per_layer) != LAYERS:
        raise ValueError("coverage must contain exactly 40 per-layer records")
    rows = []
    for layer in range(LAYERS):
        values = per_layer.get(str(layer), {}).get("counts_coding")
        if not isinstance(values, list) or len(values) != EXPERTS:
            raise ValueError(f"layer {layer} has no 384-wide coding histogram")
        total = sum(values)
        if total <= 0:
            raise ValueError(f"layer {layer} has no coding routing observations")
        # Match the accepted policy exactly: normalized coding-only score and
        # descending expert-id tie break.  Persist the output, never this rule.
        ids = sorted(range(EXPERTS), key=lambda expert: (float(values[expert]) / total, expert), reverse=True)[:K154]
        if len(set(ids)) != K154:
            raise ValueError(f"layer {layer} selection is not unique")
        rows.append({"layer": layer, "expert_ids": ids})
    return {
        "format": "dsv41-k154-selection", "version": 1, "variant": "K154-CB3",
        "selection_policy": "coding-counts-uniform-0.40", "layers": rows,
        "provenance": {"coverage_sha256": digest(coverage_path), "qualification_sha256": digest(qualification_path),
                       "qualification_prompt_tokens": qualification["prompt_tokens"],
                       "qualification_completion_tokens": qualification["completion_tokens"],
                       "qualification_needles": qualification["score"], "qualification_max_seq": stats["max_seq"]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", required=True); ap.add_argument("--qualification", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = capture(a.coverage, a.qualification)
    target = Path(a.out)
    if target.exists():
        raise SystemExit(f"ERROR: refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json(result) + b"\n")
    print(json.dumps({"out": str(target), "sha256": digest(str(target)), "experts": LAYERS * K154}, sort_keys=True))


if __name__ == "__main__":
    main()

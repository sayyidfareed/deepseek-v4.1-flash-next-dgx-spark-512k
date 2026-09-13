#!/usr/bin/env python3
"""Static guard: the public image may copy only explicit allowlisted files."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOW = {line.strip() for line in (ROOT / "PUBLIC_SOURCE_ALLOWLIST.txt").read_text().splitlines()
         if line.strip() and not line.startswith("#")}
FORBIDDEN_PREFIXES = ("corpus/", "results/", "bench/", "docs/", "scripts/diag", "engine/diag", ".git", ".env")


def main() -> None:
    copied: set[str] = set()
    for raw in (ROOT / "Dockerfile.k154").read_text().splitlines():
        line = raw.strip()
        if not line.startswith("COPY "):
            continue
        fields = line.split()[1:]
        if len(fields) < 2:
            raise SystemExit(f"ERROR: malformed COPY: {line}")
        sources = fields[:-1]
        if any(source in (".", "./") or source.endswith("/") or "*" in source for source in sources):
            raise SystemExit(f"ERROR: non-explicit Docker COPY: {line}")
        copied.update(sources)
    if not copied:
        raise SystemExit("ERROR: Dockerfile has no explicit runtime copies")
    outside = copied - ALLOW
    if outside:
        raise SystemExit(f"ERROR: Docker COPY outside public allowlist: {sorted(outside)}")
    forbidden = [source for source in copied if source.startswith(FORBIDDEN_PREFIXES)]
    if forbidden:
        raise SystemExit(f"ERROR: excluded diagnostics/corpus/results reach image: {forbidden}")
    required = {"engine/v41_engine.py", "engine/prepacked_cb3.py", "server/app.py",
                "scripts/k154_entrypoint.sh", "k154-cb3/k154-selection.json"}
    if not required <= copied:
        raise SystemExit(f"ERROR: image misses runtime inputs: {sorted(required - copied)}")
    print(f"OK: Docker copies {len(copied)} explicit allowlisted runtime files only")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a clean public repository from the reviewed source allowlist."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=".", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"ERROR: output already exists: {output}")
    allowlist = source / "PUBLIC_SOURCE_ALLOWLIST.txt"
    names = [
        line.strip()
        for line in allowlist.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(names) != len(set(names)):
        raise SystemExit("ERROR: duplicate public allowlist entry")

    for name in names:
        origin = (source / name).resolve()
        if not origin.is_file() or source not in origin.parents:
            raise SystemExit(f"ERROR: missing or unsafe source: {name}")
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)

    # GitHub's canonical landing page is the reviewed public release document.
    shutil.copy2(source / "PUBLIC_RELEASE.md", output / "README.md")
    expected = set(names) | {"README.md"}
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise SystemExit(
            f"ERROR: assembled inventory mismatch; missing={sorted(expected - actual)} "
            f"unexpected={sorted(actual - expected)}"
        )
    print(f"OK: assembled {len(actual)} reviewed public files at {output}")


if __name__ == "__main__":
    main()

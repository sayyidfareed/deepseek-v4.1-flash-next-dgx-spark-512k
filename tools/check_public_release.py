#!/usr/bin/env python3
"""Fail closed when a proposed public runtime bundle contains private material."""
from __future__ import annotations

import argparse
from pathlib import Path

# Split literals so the scanner can audit its own source without reporting its
# deny-list as a leak.
FORBIDDEN = ("bed" + "rova", "phan" + "tom", "pack" + "wolf", "tail" + "scale",
             "/home/" + "fareed", "/users/" + "fareed", "100." + "86.",
             "100." + "89.", "10." + "168.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--allowlist", default="PUBLIC_SOURCE_ALLOWLIST.txt")
    ap.add_argument("--exact-inventory", action="store_true")
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--reject-placeholders", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    allow = root / args.allowlist
    names = [line.strip() for line in allow.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(names) != len(set(names)):
        raise SystemExit("ERROR: duplicate public allowlist entry")
    scan_names = names + args.extra
    if len(scan_names) != len(set(scan_names)):
        raise SystemExit("ERROR: duplicate file across allowlist and extras")
    for name in scan_names:
        path = (root / name).resolve()
        if not path.is_file() or root not in path.parents:
            raise SystemExit(f"ERROR: missing or unsafe allowlist entry: {name}")
        if path.suffix.lower() not in {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ""}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        bad = next((needle for needle in FORBIDDEN if needle in text), None)
        if bad:
            raise SystemExit(f"ERROR: private identifier {bad!r} in {name}")
        if args.reject_placeholders and path.suffix.lower() == ".md" and "__" in text:
            raise SystemExit(f"ERROR: unresolved placeholder in {name}")
    if args.exact_inventory:
        expected = set(names) | set(args.extra)
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and ".git" not in path.relative_to(root).parts
        }
        if actual != expected:
            raise SystemExit(
                f"ERROR: public inventory mismatch; missing={sorted(expected - actual)} "
                f"unexpected={sorted(actual - expected)}"
            )
    print(f"OK: {len(names)} public files passed allowlist and sanitization checks")


if __name__ == "__main__":
    main()

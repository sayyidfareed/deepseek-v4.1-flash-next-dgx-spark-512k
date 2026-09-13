#!/usr/bin/env python3
"""Streaming exporter for the public K154 CB3-v2 release.

This tool is intentionally *not* an automatic model builder.  It requires an
immutable 40-by-154 selection file and works on exactly one shard per command.
That lets a DGX Spark export, hash, upload, and discard each staging file without
ever requiring a second copy of the 222 GB retained base or the 89 GB CB3 sidecar.

No network operation is performed here.  Upload is a separate, reviewable step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from engine.prepacked_cb3 import (  # noqa: E402
    BYTES_PER_EXPERT, EXPERTS_PER_LAYER, FORMAT, K154, LAYERS, PLANES, VERSION,
    canonical_json, load_manifest, sha256_file, validate_manifest, verify_release_weights,
)

SOURCE_REPOSITORY = "deepseek-ai/DeepSeek-V4.1-Flash"
SOURCE_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
SELECTION_FORMAT = "dsv41-k154-selection"
RETAINED_BASE_TENSOR_BYTES = 221_508_192_600
ROUTED_SOURCE_TENSOR_BYTES = 288_777_830_400
CB3_TENSOR_BYTES = LAYERS * K154 * BYTES_PER_EXPERT
FULL_PACKAGE_TENSOR_BYTES = RETAINED_BASE_TENSOR_BYTES + CB3_TENSOR_BYTES
REUSABLE_SOURCE_SHARDS = {1, 2, 43, 44, 45, 46, 47, 48}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def load_selection(path: str) -> dict[int, list[int]]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read selection: {exc}")
    if doc.get("format") != SELECTION_FORMAT or doc.get("version") != 1 or doc.get("variant") != "K154-CB3":
        fail("selection must be a versioned K154-CB3 selection artifact")
    rows = doc.get("layers")
    if not isinstance(rows, list) or len(rows) != LAYERS:
        fail("selection must have exactly 40 layer records")
    out: dict[int, list[int]] = {}
    for row in rows:
        layer, ids = row.get("layer"), row.get("expert_ids")
        if not isinstance(layer, int) or layer in out or not 0 <= layer < LAYERS:
            fail("selection has a duplicate or invalid layer")
        if not isinstance(ids, list) or len(ids) != K154 or any(not isinstance(x, int) for x in ids):
            fail(f"layer {layer} does not contain 154 integer expert ids")
        if len(set(ids)) != K154 or any(x < 0 or x >= EXPERTS_PER_LAYER for x in ids):
            fail(f"layer {layer} has duplicate or out-of-range expert ids")
        out[layer] = ids
    if set(out) != set(range(LAYERS)):
        fail("selection does not cover layers 0 through 39")
    return out


def source_digests(model_dir: str) -> dict[str, str]:
    root = Path(model_dir)
    names = ("config.json", "inference/config.json", "inference/engram.py", "tokenizer.json",
             "tokenizer_config.json", "encoding/encoding.py")
    found = {name: sha256_file(root / name) for name in names if (root / name).is_file()}
    if "inference/config.json" not in found:
        fail("model directory is missing inference/config.json")
    return found


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(canonical_json(value) + b"\n")
    os.replace(tmp, path)


def export_cb3_layer(model_dir: str, selection_path: str, layer: int, out: str) -> dict[str, Any]:
    if not 0 <= layer < LAYERS:
        fail("--layer must be in 0..39")
    ids = load_selection(selection_path)[layer]
    try:
        import torch
        from safetensors.torch import save_file
        import cb3_moe as C3
        from engine import experts as EX
        from engine.codebook_sim import CodebookSim
    except ImportError as exc:
        fail(f"export requires the public CUDA runtime dependencies: {exc}")
    if not torch.cuda.is_available():
        fail("export requires CUDA because CB3-v2 packing is performed by the tested GPU implementation")
    # Nine slots satisfy ExpertStore's eight-slot transient invariant while retaining
    # only one active packing slot.  No full routed-expert arena is allocated.
    arena = C3.CB3ArenaV2(9, "cuda")
    arena.sim = CodebookSim(3, "cuda")
    index = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())
    store = EX.ExpertStore(model_dir, index, arena, LAYERS, transient_slots=8, io_threads=1, read_threads=1)
    planes: dict[str, list[Any]] = {name: [] for name in PLANES}
    try:
        for position, expert in enumerate(ids):
            w1, s1, w2, s2, w3, s3 = store.read_expert(layer, expert)
            arena.load_slot(0, w1.view(*EX.W13_SHAPE), s1.view(*EX.S13_SHAPE), w2.view(*EX.W2_SHAPE),
                            s2.view(*EX.S2_SHAPE), w3.view(*EX.W13_SHAPE), s3.view(*EX.S13_SHAPE))
            torch.cuda.synchronize()
            for name in PLANES:
                planes[name].append(getattr(arena, name)[0].detach().cpu().contiguous())
            print(f"layer {layer:02d}: packed {position + 1}/{K154} expert {expert}", flush=True)
        tensors = {name: torch.stack(rows, dim=0).contiguous() for name, rows in planes.items()}
        tensor_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
        if tensor_bytes != K154 * BYTES_PER_EXPERT:
            fail(f"CB3 tensor byte mismatch: {tensor_bytes} != {K154 * BYTES_PER_EXPERT}")
        dest = Path(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name + ".partial")
        if temp.exists() or dest.exists():
            fail(f"refusing to overwrite existing output {dest}")
        save_file(tensors, str(temp), metadata={"format": FORMAT, "version": str(VERSION), "layout": "cb3-v2",
                                                "layer": str(layer), "selection_sha256": sha256_file(selection_path)})
        os.replace(temp, dest)
    finally:
        store.pool.shutdown(wait=True)
        store.read_pool.shutdown(wait=True)
    receipt = {"layer": layer, "file": f"layers/layer-{layer:02d}.safetensors", "file_bytes": Path(out).stat().st_size,
               "tensor_bytes": K154 * BYTES_PER_EXPERT, "sha256": sha256_file(out),
               "expert_ids": ids, "slot_order": ids}
    return receipt


def write_receipt(receipt: dict[str, Any], path: str) -> None:
    atomic_json(Path(path), receipt)


def load_base_receipts(model_dir: str, receipt_dir: str) -> list[dict[str, Any]]:
    """Load exactly one immutable receipt for every final retained base shard."""
    plan = base_plan(model_dir)
    expected = {row["source_file"]: row for row in plan["shards"]}
    rows: list[dict[str, Any]] = []
    for source_file, planned in sorted(expected.items()):
        path = Path(receipt_dir) / f"{source_file}.json"
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"missing/invalid base receipt for {source_file}: {exc}")
        required = {"source_file": source_file, "file": source_file,
                    "mode": "copy-server-side" if planned["mode"] == "copy" else "rewrite",
                    "tensor_bytes": planned["kept_tensor_bytes"]}
        if any(row.get(key) != value for key, value in required.items()):
            fail(f"base receipt does not match immutable plan: {source_file}")
        if not isinstance(row.get("file_bytes"), int) or row["file_bytes"] <= 0:
            fail(f"base receipt has invalid file size: {source_file}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            fail(f"base receipt has invalid sha256: {source_file}")
        rows.append(row)
    return rows


def write_manifest(model_dir: str, selection_path: str, receipt_dir: str, base_receipt_dir: str,
                   release_index: str, out: str) -> None:
    selection = load_selection(selection_path)
    rows = []
    for layer in range(LAYERS):
        receipt_path = Path(receipt_dir) / f"layer-{layer:02d}.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"missing/invalid receipt for layer {layer}: {exc}")
        if receipt.get("layer") != layer or receipt.get("expert_ids") != selection[layer] or receipt.get("slot_order") != selection[layer]:
            fail(f"receipt selection mismatch for layer {layer}")
        rows.append(receipt)
    manifest = {
        "format": FORMAT, "version": VERSION, "variant": "K154-CB3", "layout": "cb3-v2",
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        "source_digests": source_digests(model_dir), "release_index_sha256": sha256_file(release_index), "layers": LAYERS, "experts_per_layer": EXPERTS_PER_LAYER,
        "experts_kept_per_layer": K154, "bytes_per_expert": BYTES_PER_EXPERT,
        "selection_sha256": sha256_file(selection_path), "base_shards": load_base_receipts(model_dir, base_receipt_dir),
        "retained_weight_map_sha256": hashlib.sha256(canonical_json(base_plan(model_dir)["retained_weight_map"])).hexdigest(),
        "shards": rows,
    }
    validate_manifest(manifest)
    atomic_json(Path(out), manifest)


def is_main_routed_expert(name: str) -> bool:
    # DSpark experts use mtp.* and are deliberately retained.  Only the 40 main
    # MoE layers move into the CB3 sidecar.
    return name.startswith("layers.") and ".ffn.experts." in name


def safetensors_tensor_sizes(path: Path) -> dict[str, int]:
    """Read only a safetensors header; never map or materialize a tensor."""
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            fail(f"truncated safetensors header: {path.name}")
        header_bytes = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(header_bytes))
    return {name: int(value["data_offsets"][1]) - int(value["data_offsets"][0])
            for name, value in header.items() if name != "__metadata__"}


def base_plan(model_dir: str) -> dict[str, Any]:
    root = Path(model_dir)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        fail("source index has no weight_map")
    grouped: dict[str, list[str]] = {}
    kept, dropped = [], []
    for name, shard in weight_map.items():
        (dropped if is_main_routed_expert(name) else kept).append(name)
        grouped.setdefault(shard, []).append(name)
    if len(dropped) != LAYERS * EXPERTS_PER_LAYER * 6:
        fail(f"unexpected routed expert tensor count {len(dropped)}")
    header_sizes: dict[str, int] = {}
    for shard in grouped:
        header_sizes.update(safetensors_tensor_sizes(root / shard))
    if set(header_sizes) != set(weight_map):
        fail("source index and safetensors headers disagree")
    retained_bytes = sum(header_sizes[name] for name in kept)
    routed_bytes = sum(header_sizes[name] for name in dropped)
    if retained_bytes != RETAINED_BASE_TENSOR_BYTES or routed_bytes != ROUTED_SOURCE_TENSOR_BYTES:
        fail(f"source tensor accounting changed: retained={retained_bytes}, routed={routed_bytes}")
    entries = []
    for shard in sorted(grouped):
        number = int(shard.split("-")[1])
        names = grouped[shard]
        entries.append({"source_file": shard, "output_file": shard, "mode": "copy" if number in REUSABLE_SOURCE_SHARDS else "rewrite",
                        "kept_tensor_count": sum(not is_main_routed_expert(n) for n in names),
                        "kept_tensor_bytes": sum(header_sizes[n] for n in names if not is_main_routed_expert(n)),
                        "dropped_routed_tensor_count": sum(is_main_routed_expert(n) for n in names)})
    return {"source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
            "retained_base_tensor_bytes": RETAINED_BASE_TENSOR_BYTES,
            "excluded_main_routed_tensor_bytes": ROUTED_SOURCE_TENSOR_BYTES,
            "cb3_tensor_bytes": CB3_TENSOR_BYTES,
            "full_package_tensor_bytes": FULL_PACKAGE_TENSOR_BYTES,
            "reusable_source_shards": sorted(REUSABLE_SOURCE_SHARDS), "shards": entries,
            "retained_weight_map": {k: v for k, v in weight_map.items() if not is_main_routed_expert(k)},
            "source_index_metadata": index.get("metadata", {})}


def export_base_shard(model_dir: str, source_shard: str, out: str) -> dict[str, Any]:
    """Write exactly one retained base shard.  Caller uploads then removes this staging file."""
    plan = base_plan(model_dir)
    row = next((x for x in plan["shards"] if x["source_file"] == source_shard), None)
    if row is None:
        fail(f"unknown source shard {source_shard}")
    source, dest = Path(model_dir) / source_shard, Path(out)
    if dest.exists():
        fail(f"refusing to overwrite existing output {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if row["mode"] == "copy":
        fail("unchanged shards must use server-side copy; use write-copy-receipt, never a local duplicate")
    else:
        try:
            from safetensors import safe_open
            from safetensors.torch import save_file
        except ImportError as exc:
            fail(f"safetensors is required to rewrite a base shard: {exc}")
        tensors = {}
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not is_main_routed_expert(key):
                    tensors[key] = handle.get_tensor(key).contiguous()
        temp = dest.with_name(dest.name + ".partial")
        save_file(tensors, str(temp), metadata={"source": source_shard, "variant": "K154-CB3"})
        os.replace(temp, dest)
    return {"source_file": source_shard, "file": source_shard, "mode": "rewrite",
            "tensor_bytes": row["kept_tensor_bytes"], "file_bytes": dest.stat().st_size,
            "sha256": sha256_file(dest)}


def write_copy_receipt(model_dir: str, source_shard: str, out: str) -> None:
    """Record a source hash for a Hugging Face server-side copied unchanged shard.

    The receipt is not proof of a completed upload; `preflight-release` verifies
    the final downloaded release object against this digest before serving.
    """
    plan = base_plan(model_dir)
    row = next((x for x in plan["shards"] if x["source_file"] == source_shard), None)
    if row is None or row["mode"] != "copy":
        fail("write-copy-receipt accepts only an unchanged reusable source shard")
    source = Path(model_dir) / source_shard
    receipt = {"source_file": source_shard, "file": source_shard, "mode": "copy-server-side",
               "tensor_bytes": row["kept_tensor_bytes"], "file_bytes": source.stat().st_size,
               "sha256": sha256_file(source)}
    target = Path(out)
    if target.exists():
        fail(f"refusing to overwrite existing receipt {target}")
    atomic_json(target, receipt)


def write_retained_index(model_dir: str, out: str) -> None:
    plan = base_plan(model_dir)
    metadata = dict(plan["source_index_metadata"])
    metadata["total_size"] = RETAINED_BASE_TENSOR_BYTES
    index = {"metadata": metadata, "weight_map": plan["retained_weight_map"]}
    atomic_json(Path(out), index)


def preflight_release(model_dir: str, pack_dir: str) -> dict[str, Any]:
    """Fail closed unless every final weight file matches the immutable manifest."""
    manifest = load_manifest(pack_dir)
    verify_release_weights(model_dir, pack_dir, manifest)
    return {"ok": True, "base_shards": len(manifest["base_shards"]),
            "cb3_layers": len(manifest["shards"]), "release_index_sha256": manifest["release_index_sha256"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    layer = sub.add_parser("export-layer", help="export one CB3 layer and print a receipt")
    layer.add_argument("--model-dir", required=True); layer.add_argument("--selection", required=True)
    layer.add_argument("--layer", required=True, type=int); layer.add_argument("--out", required=True); layer.add_argument("--receipt", required=True)
    manifest = sub.add_parser("write-manifest", help="assemble final manifest from the 40 verified receipts")
    manifest.add_argument("--model-dir", required=True); manifest.add_argument("--selection", required=True)
    manifest.add_argument("--receipt-dir", required=True); manifest.add_argument("--base-receipt-dir", required=True); manifest.add_argument("--release-index", required=True); manifest.add_argument("--out", required=True)
    base = sub.add_parser("base-plan", help="write the exact retained/rewrite base-weight plan")
    base.add_argument("--model-dir", required=True); base.add_argument("--out", required=True)
    shard = sub.add_parser("export-base-shard", help="stage one retained base shard")
    shard.add_argument("--model-dir", required=True); shard.add_argument("--source-shard", required=True)
    shard.add_argument("--out", required=True); shard.add_argument("--receipt", required=True)
    copy = sub.add_parser("write-copy-receipt", help="hash one unchanged source shard for server-side copy")
    copy.add_argument("--model-dir", required=True); copy.add_argument("--source-shard", required=True); copy.add_argument("--out", required=True)
    index = sub.add_parser("write-retained-index", help="write the index for retained base weights")
    index.add_argument("--model-dir", required=True); index.add_argument("--out", required=True)
    preflight = sub.add_parser("preflight-release", help="hash every final base and CB3 weight before serving")
    preflight.add_argument("--model-dir", required=True); preflight.add_argument("--pack-dir", required=True)
    args = ap.parse_args()
    if args.command == "export-layer":
        write_receipt(export_cb3_layer(args.model_dir, args.selection, args.layer, args.out), args.receipt)
    elif args.command == "write-manifest":
        write_manifest(args.model_dir, args.selection, args.receipt_dir, args.base_receipt_dir, args.release_index, args.out)
    elif args.command == "base-plan":
        atomic_json(Path(args.out), base_plan(args.model_dir))
    elif args.command == "export-base-shard":
        receipt = export_base_shard(args.model_dir, args.source_shard, args.out)
        write_receipt(receipt, args.receipt)
        print(json.dumps(receipt, sort_keys=True))
    elif args.command == "write-copy-receipt":
        write_copy_receipt(args.model_dir, args.source_shard, args.out)
    elif args.command == "write-retained-index":
        write_retained_index(args.model_dir, args.out)
    elif args.command == "preflight-release":
        print(json.dumps(preflight_release(args.model_dir, args.pack_dir), sort_keys=True))


if __name__ == "__main__":
    main()

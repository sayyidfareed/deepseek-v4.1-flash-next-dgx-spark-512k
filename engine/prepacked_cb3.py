"""Fail-closed reader for the K154 CB3 routed-expert sidecar.

The upstream DeepSeek checkpoint remains the source of every dense, Engram, router,
and draft tensor.  This sidecar contains only the selected routed experts after the
public CB3 transform.  It is deliberately self-describing and refuses to load when
the exact source checkpoint or any sidecar shard differs from the manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

FORMAT = "dsv41-cb3-expert-pack"
VERSION = 1
LAYERS = 40
EXPERTS_PER_LAYER = 384
K154 = 154
BASE_SHARDS = 48

# Tensor names in one per-layer safetensors shard.  Shapes omit the expert axis;
# the loader adds the selected-expert count from manifest.json.
PLANES: dict[str, tuple[int, ...]] = {
    "w1_lo": (2304, 1280), "w1_hi": (2304, 640), "w1_cb": (2304, 8), "s1": (2304, 160),
    "w3_lo": (2304, 1280), "w3_hi": (2304, 640), "w3_cb": (2304, 8), "s3": (2304, 160),
    "w2_lo": (5120, 576), "w2_hi": (5120, 288), "w2_cb": (5120, 8), "s2": (5120, 72),
}
BYTES_PER_EXPERT = sum(int(__import__("math").prod(shape)) for shape in PLANES.values())


class PackValidationError(RuntimeError):
    """The pack is absent, malformed, or does not match its declared source."""


def sha256_file(path: str | os.PathLike[str], block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PackValidationError(message)


def _digest_spec(spec: Any) -> dict[str, str]:
    _require(isinstance(spec, dict) and spec, "source_digests must be a non-empty object")
    out: dict[str, str] = {}
    for relative, expected in spec.items():
        _require(isinstance(relative, str) and relative and not os.path.isabs(relative) and ".." not in Path(relative).parts,
                 "unsafe source digest path")
        _require(isinstance(expected, str) and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected),
                 f"invalid sha256 for {relative}")
        out[relative] = expected
    return out


def _hex_digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and
             all(c in "0123456789abcdef" for c in value), f"invalid {label}")
    return value


def _base_shards(spec: Any) -> list[dict[str, Any]]:
    """Validate receipts for every final non-routed checkpoint shard.

    A copied upstream shard is a final release file just like a rewritten shard:
    the receipt makes that explicit and lets the release preflight authenticate it
    after Hugging Face's server-side copy operation.
    """
    _require(isinstance(spec, list) and len(spec) == BASE_SHARDS,
             "expected receipts for all 48 retained base shards")
    out, seen = [], set()
    for row in spec:
        _require(isinstance(row, dict), "invalid base-shard receipt")
        name = row.get("file")
        _require(isinstance(name, str) and name.startswith("model-") and name.endswith(".safetensors"),
                 "invalid retained base filename")
        _require(name not in seen, "duplicate retained base filename")
        seen.add(name)
        _require(row.get("source_file") == name, "base receipt source/output mismatch")
        _require(row.get("mode") in ("rewrite", "copy-server-side"), "invalid base receipt mode")
        _hex_digest(row.get("sha256"), f"base-shard sha256 for {name}")
        _require(isinstance(row.get("file_bytes"), int) and row["file_bytes"] > 0,
                 f"invalid base-shard size for {name}")
        _require(isinstance(row.get("tensor_bytes"), int) and 0 < row["tensor_bytes"] <= row["file_bytes"],
                 f"invalid base-shard tensor bytes for {name}")
        out.append(row)
    expected = {f"model-{number:05d}-of-00048.safetensors" for number in range(1, BASE_SHARDS + 1)}
    _require(seen == expected, "retained base receipt filenames are incomplete or noncanonical")
    return out


def _weight_map_digest(value: Any) -> str:
    _require(isinstance(value, str) and len(value) == 64 and
             all(c in "0123456789abcdef" for c in value), "missing retained_weight_map_sha256")
    return value


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate schema and return the same mapping.  No untrusted path is opened here."""
    _require(manifest.get("format") == FORMAT and manifest.get("version") == VERSION,
             "unsupported expert-pack format/version")
    _require(manifest.get("layout") == "cb3-v2", "unsupported CB3 layout")
    _require(manifest.get("variant") == "K154-CB3", "this loader only accepts K154-CB3")
    source = manifest.get("source")
    _require(isinstance(source, dict) and source.get("repository") == "deepseek-ai/DeepSeek-V4.1-Flash"
             and source.get("revision") == "dba1be0a40aa45a94ad051997016db3960a90277",
             "missing immutable DeepSeek source revision")
    _require(manifest.get("layers") == LAYERS and manifest.get("experts_per_layer") == EXPERTS_PER_LAYER,
             "unexpected model expert geometry")
    _require(manifest.get("experts_kept_per_layer") == K154, "expected exactly 154 experts per layer")
    _require(manifest.get("bytes_per_expert") == BYTES_PER_EXPERT, "incorrect CB3 byte geometry")
    _digest_spec(manifest.get("source_digests"))
    _hex_digest(manifest.get("selection_sha256"), "selection_sha256")
    _hex_digest(manifest.get("release_index_sha256"), "release_index_sha256")
    _weight_map_digest(manifest.get("retained_weight_map_sha256"))
    _base_shards(manifest.get("base_shards"))
    shards = manifest.get("shards")
    _require(isinstance(shards, list) and len(shards) == LAYERS, "expected one shard for each of 40 layers")
    seen_layers: set[int] = set()
    for entry in shards:
        _require(isinstance(entry, dict), "invalid shard entry")
        layer = entry.get("layer")
        _require(isinstance(layer, int) and 0 <= layer < LAYERS and layer not in seen_layers, "duplicate/invalid layer")
        seen_layers.add(layer)
        filename = entry.get("file")
        _require(isinstance(filename, str) and filename == f"layers/layer-{layer:02d}.safetensors", "unsafe/noncanonical shard filename")
        digest = entry.get("sha256")
        _require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                 f"invalid shard sha256 for layer {layer}")
        _require(entry.get("tensor_bytes") == K154 * BYTES_PER_EXPERT, f"unexpected tensor bytes for layer {layer}")
        _require(isinstance(entry.get("file_bytes"), int) and entry["file_bytes"] >= entry["tensor_bytes"],
                 f"invalid safetensors file bytes for layer {layer}")
        ids = entry.get("expert_ids")
        _require(isinstance(ids, list) and len(ids) == K154 and all(isinstance(x, int) for x in ids),
                 f"invalid expert selection for layer {layer}")
        _require(len(set(ids)) == K154 and all(0 <= x < EXPERTS_PER_LAYER for x in ids),
                 f"expert ids out of range/duplicate for layer {layer}")
        # Axis zero is the declared stable slot order.  Keeping the redundant field
        # makes the contract reviewable without having to infer meaning from tensor
        # storage order, and catches a mistaken manifest from another packer.
        _require(entry.get("slot_order") == ids, f"slot order must equal tensor-axis order for layer {layer}")
    _require(len(seen_layers) == LAYERS, "missing layer shard")
    return manifest


def load_manifest(pack_dir: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(pack_dir).resolve()
    path = root / "manifest.json"
    _require(path.is_file(), "manifest.json is missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackValidationError(f"cannot read manifest.json: {exc}") from exc
    return validate_manifest(manifest)


def verify_source(model_dir: str | os.PathLike[str], manifest: dict[str, Any]) -> None:
    """Hash every declared source metadata file before touching a prepacked expert."""
    root = Path(model_dir).resolve()
    for relative, expected in _digest_spec(manifest["source_digests"]).items():
        path = (root / relative).resolve()
        _require(path.is_file() and os.path.commonpath((str(root), str(path))) == str(root),
                 f"required source file is missing: {relative}")
        actual = sha256_file(path)
        _require(actual == expected, f"source digest mismatch: {relative}")
    release_index = root / "model.safetensors.index.json"
    _require(release_index.is_file() and sha256_file(release_index) == manifest["release_index_sha256"],
             "release model.safetensors.index.json digest mismatch")
    try:
        index = json.loads(release_index.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackValidationError(f"cannot parse release index: {exc}") from exc
    weight_map = index.get("weight_map")
    _require(isinstance(weight_map, dict) and all(isinstance(k, str) and isinstance(v, str)
             for k, v in weight_map.items()), "release index has invalid weight_map")
    _require(not any(name.startswith("layers.") and ".ffn.experts." in name for name in weight_map),
             "release index still contains original main routed experts")
    _require(hashlib.sha256(canonical_json(weight_map)).hexdigest() == manifest["retained_weight_map_sha256"],
             "release index weight_map differs from retained source-derived contract")
    receipts = {row["file"] for row in _base_shards(manifest["base_shards"])}
    mapped = set(weight_map.values())
    _require(mapped == receipts, "release index shards differ from canonical base receipts")


def verify_release_base(model_dir: str | os.PathLike[str], pack_dir: str | os.PathLike[str],
                        manifest: dict[str, Any]) -> None:
    """Authenticate every release weight before inference starts.

    This is intentionally an explicit, linear disk pass. It detects a truncated
    or substituted Hugging Face object, including files created by server-side
    copy, before a request can reach the model.
    """
    verify_source(model_dir, manifest)
    root = Path(model_dir).resolve()
    pack_root = Path(pack_dir).resolve()
    selection = pack_root / "k154-selection.json"
    _require(selection.is_file() and sha256_file(selection) == manifest["selection_sha256"],
             "k154-selection.json digest mismatch")
    for row in _base_shards(manifest["base_shards"]):
        path = (root / row["file"]).resolve()
        _require(path.is_file() and os.path.commonpath((str(root), str(path))) == str(root),
                 f"missing retained base shard: {row['file']}")
        _require(path.stat().st_size == row["file_bytes"], f"base-shard size mismatch: {row['file']}")
        _require(sha256_file(path) == row["sha256"], f"base-shard digest mismatch: {row['file']}")


def verify_release_weights(model_dir: str | os.PathLike[str], pack_dir: str | os.PathLike[str],
                           manifest: dict[str, Any]) -> None:
    """Offline full preflight, including CB3 files before any serve attempt."""
    verify_release_base(model_dir, pack_dir, manifest)
    for entry in manifest["shards"]:
        verified_layer_path(pack_dir, entry)


def verified_layer_path(pack_dir: str | os.PathLike[str], entry: dict[str, Any]) -> Path:
    root = Path(pack_dir).resolve()
    path = (root / entry["file"]).resolve()
    _require(path.is_file() and os.path.commonpath((str(root), str(path))) == str(root),
             f"missing sidecar shard for layer {entry['layer']}")
    _require(path.stat().st_size == entry["file_bytes"], f"size mismatch for layer {entry['layer']}")
    _require(sha256_file(path) == entry["sha256"], f"digest mismatch for layer {entry['layer']}")
    return path


def load_layer_cpu(pack_dir: str | os.PathLike[str], entry: dict[str, Any]) -> dict[str, Any]:
    """Verify one safetensors shard then load it on CPU with exact uint8 geometry."""
    path = verified_layer_path(pack_dir, entry)
    try:
        from safetensors.torch import load_file
        import torch
    except ImportError as exc:  # import late so schema tests do not need CUDA/PyTorch
        raise PackValidationError("safetensors and torch are required to load an expert pack") from exc
    tensors = load_file(str(path), device="cpu")
    _require(set(tensors) == set(PLANES), f"unexpected tensor names in layer {entry['layer']}")
    for name, suffix in PLANES.items():
        value = tensors[name]
        _require(value.dtype == torch.uint8, f"{name} is not uint8 in layer {entry['layer']}")
        _require(tuple(value.shape) == (K154, *suffix),
                 f"{name} has wrong shape in layer {entry['layer']}: {tuple(value.shape)}")
        _require(value.is_contiguous(), f"{name} is noncontiguous in layer {entry['layer']}")
    return tensors


def manifest_layer(manifest: dict[str, Any], layer: int) -> dict[str, Any]:
    for entry in manifest["shards"]:
        if entry["layer"] == layer:
            return entry
    raise PackValidationError(f"no sidecar entry for layer {layer}")

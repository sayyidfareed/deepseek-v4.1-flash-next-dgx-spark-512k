"""No-GPU contract tests for the public K154 CB3 sidecar schema."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from engine.prepacked_cb3 import (BYTES_PER_EXPERT, FORMAT, K154, LAYERS, PackValidationError,
                                  VERSION, canonical_json, validate_manifest, verify_source)


def manifest() -> dict:
    digest = "0" * 64
    base = [{"source_file": f"model-{i:05d}-of-00048.safetensors",
             "file": f"model-{i:05d}-of-00048.safetensors",
             "mode": "copy-server-side", "tensor_bytes": 1, "file_bytes": 1, "sha256": digest}
            for i in range(1, 49)]
    return {
        "format": FORMAT, "version": VERSION, "variant": "K154-CB3", "layout": "cb3-v2", "selection_sha256": digest, "release_index_sha256": digest,
        "source": {"repository": "deepseek-ai/DeepSeek-V4.1-Flash", "revision": "dba1be0a40aa45a94ad051997016db3960a90277"},
        "source_digests": {"inference/config.json": digest}, "layers": 40, "experts_per_layer": 384,
        "experts_kept_per_layer": K154, "bytes_per_expert": BYTES_PER_EXPERT,
        "base_shards": base, "retained_weight_map_sha256": digest,
        "shards": [{"layer": layer, "file": f"layers/layer-{layer:02d}.safetensors", "sha256": digest,
                    "tensor_bytes": K154 * BYTES_PER_EXPERT, "file_bytes": K154 * BYTES_PER_EXPERT,
                    "expert_ids": list(range(K154)), "slot_order": list(range(K154))} for layer in range(LAYERS)],
    }


class PrepackedCB3SchemaTests(unittest.TestCase):
    def test_valid_schema_has_exact_6160_residents(self):
        doc = validate_manifest(manifest())
        self.assertEqual(sum(len(x["expert_ids"]) for x in doc["shards"]), 6160)

    def test_duplicate_expert_is_rejected(self):
        doc = manifest()
        doc["shards"][0]["expert_ids"][-1] = 0
        with self.assertRaises(PackValidationError):
            validate_manifest(doc)

    def test_slot_order_is_a_contract(self):
        doc = manifest()
        doc["shards"][4]["slot_order"] = list(reversed(doc["shards"][4]["slot_order"]))
        with self.assertRaises(PackValidationError):
            validate_manifest(doc)

    def test_unexpected_file_path_is_rejected(self):
        doc = manifest()
        doc["shards"][0]["file"] = "../layer.safetensors"
        with self.assertRaises(PackValidationError):
            validate_manifest(doc)

    def test_wrong_layout_and_missing_selection_are_rejected(self):
        doc = manifest(); doc["layout"] = "cb3-v1"
        with self.assertRaises(PackValidationError):
            validate_manifest(doc)

    def test_missing_or_duplicate_base_receipt_is_rejected(self):
        doc = manifest(); doc["base_shards"].pop()
        with self.assertRaises(PackValidationError):
            validate_manifest(doc)
        doc = manifest(); doc["base_shards"][1]["file"] = doc["base_shards"][0]["file"]
        with self.assertRaises(PackValidationError):
            validate_manifest(doc)
        doc = manifest(); doc.pop("selection_sha256")
        with self.assertRaises(PackValidationError):
            validate_manifest(doc)

    def test_source_digest_is_verified(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "inference").mkdir()
            content = b"known-source-metadata"
            (root / "inference" / "config.json").write_bytes(content)
            doc = manifest()
            doc["source_digests"] = {"inference/config.json": hashlib.sha256(content).hexdigest()}
            weight_map = {f"dense.{i}": row["file"] for i, row in enumerate(doc["base_shards"])}
            doc["retained_weight_map_sha256"] = hashlib.sha256(canonical_json(weight_map)).hexdigest()
            index = canonical_json({"metadata": {"total_size": 221508192600}, "weight_map": weight_map})
            (root / "model.safetensors.index.json").write_bytes(index)
            doc["release_index_sha256"] = hashlib.sha256(index).hexdigest()
            verify_source(root, validate_manifest(doc))
            (root / "inference" / "config.json").write_bytes(b"different")
            with self.assertRaises(PackValidationError):
                verify_source(root, validate_manifest(doc))

    def test_index_contract_rejects_original_expert_tensor(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); (root / "inference").mkdir()
            content = b"known-source-metadata"; (root / "inference" / "config.json").write_bytes(content)
            doc = manifest(); doc["source_digests"] = {"inference/config.json": hashlib.sha256(content).hexdigest()}
            weight_map = {f"dense.{i}": row["file"] for i, row in enumerate(doc["base_shards"])}
            weight_map["layers.0.ffn.experts.0.down_proj.weight"] = doc["base_shards"][0]["file"]
            doc["retained_weight_map_sha256"] = hashlib.sha256(canonical_json(weight_map)).hexdigest()
            index = canonical_json({"metadata": {}, "weight_map": weight_map})
            (root / "model.safetensors.index.json").write_bytes(index)
            doc["release_index_sha256"] = hashlib.sha256(index).hexdigest()
            with self.assertRaises(PackValidationError):
                verify_source(root, validate_manifest(doc))


if __name__ == "__main__":
    unittest.main()

"""No-GPU structural contracts for the immutable public K154 serve path."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicK154ContractTests(unittest.TestCase):
    def test_public_identity_and_network_boundary_are_consistent(self):
        name = "DeepSeek-V4.1-Flash-Next-DGX-Spark-512K"
        launcher = (ROOT / "scripts" / "k154_entrypoint.sh").read_text()
        compose = (ROOT / "compose.k154.yaml").read_text()
        server = (ROOT / "server" / "app.py").read_text()
        self.assertIn(f'readonly SERVED_MODEL_NAME="{name}"', launcher)
        self.assertIn(f"MODEL_DIR: /models/{name}", compose)
        self.assertIn('HOST:=127.0.0.1', launcher)
        self.assertIn('ports: ["127.0.0.1:${PORT:-8000}:8000"]', compose)
        self.assertIn('${MODEL_PARENT:?set MODEL_PARENT}:/models:ro', compose)
        self.assertIn('HOST: 0.0.0.0', compose)
        self.assertIn('"owned_by": "sayyidfareed"', server)

    def test_launcher_forces_the_qualified_inference_profile(self):
        launcher = (ROOT / "scripts" / "k154_entrypoint.sh").read_text()
        fixed = {
            "MODEL_DIR": "/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K",
            "SERVED_MODEL_NAME": "DeepSeek-V4.1-Flash-Next-DGX-Spark-512K",
            "MAX_SEQ": "1048576",
            "ARENA_GB": "89.16",
            "KEEP_FREE_GB": "2.5",
            "SPEC": "1",
        }
        for name, value in fixed.items():
            self.assertIn(f'readonly {name}="{value}"', launcher)
            self.assertNotIn(f'${{{name}:=', launcher)

        env = {
            "DSV41_DENSE_FP4": "attn,wo_a",
            "DSV41_HEAD_FMT": "fp8",
            "DSV41_HEAD_FP32": "0",
            "DSV41_PREFILL_CHUNK": "256",
            "DSV41_RING": "4096",
            "DSV41_SWA_REPLAY": "1",
            "DSV41_TOPK": "6",
            "DSV41_BLOCK": "5",
            "DSV41_FAST_REFERENCE_SHAPES": "0",
            "DSV41_MEM_FLOOR_GB": "2.5",
            "DSV41_DENSE_FP8": "1",
            "DSV41_WOA_FP8": "1",
            "DSV41_FP4_DENSE_PREFILL": "kernel",
            "DSV41_FP4_DENSE_F16": "1",
            "DSV41_CB3_PREFILL": "fp4",
            "DSV41_CB3_PREFILL_MIN_P": "65",
            "DSV41_FAST": "1",
            "DSV41_GRAPHS": "1",
            "DSV41_LUT": "1",
            "DSV41_HC_KERNEL": "1",
            "DSV41_FUSED_ATTN": "0",
            "DSV41_GRAPH_SEGMENTS": "1",
            "DSV41_LEAN_STEP": "1",
            "DSV41_CYCLE_BREAK": "1",
        }
        for name, value in env.items():
            self.assertIn(f'export {name}="{value}"', launcher)

    def test_engine_pack_path_verifies_all_weights_and_uses_cb3(self):
        source = (ROOT / "engine" / "v41_engine.py").read_text()
        self.assertIn("verify_release_base(model_dir, expert_pack, self.pack_manifest)", source)
        self.assertIn('if self.expert_format == "cb3":', source)
        self.assertNotIn('if self.expert_format == "cb3" and not self.pack_manifest:', source)
        self.assertIn('self.kernel = "triton-cb3"', source)
        self.assertIn("self.store.warm_start_prepacked_cb3", source)
        self.assertIn('raise ValueError("--expert-pack does not accept --trace-stats;', source)

    def test_prepacked_loader_never_reads_or_repacks_source_experts(self):
        tree = ast.parse((ROOT / "engine" / "experts.py").read_text())
        fn = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "warm_start_prepacked_cb3")
        calls = {node.func.attr for node in ast.walk(fn)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertNotIn("read_expert", calls)
        self.assertNotIn("load_slot", calls)
        self.assertIn("load_prepacked_slot", calls)

    def test_release_preflight_hashes_all_final_weight_classes(self):
        source = (ROOT / "engine" / "prepacked_cb3.py").read_text()
        self.assertIn("for row in _base_shards(manifest[\"base_shards\"])", source)
        self.assertIn("verified_layer_path(pack_dir, entry)", source)
        self.assertIn("k154-selection.json digest mismatch", source)

    def test_server_cannot_bind_before_pack_integrity_and_warm_start(self):
        engine = (ROOT / "engine" / "v41_engine.py").read_text()
        server = (ROOT / "server" / "app.py").read_text()
        self.assertLess(engine.index("verify_release_base("), engine.index("warm_start_prepacked_cb3("))
        self.assertLess(server.index("engine = make_engine(args, tok, enc)"),
                        server.index("ThreadingHTTPServer((args.host, args.port), Handler)"))


if __name__ == "__main__":
    unittest.main()

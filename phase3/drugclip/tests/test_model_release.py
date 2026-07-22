from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from phase3.drugclip.validate_model_release import CHECKPOINT_SCHEMA, RELEASE_SCHEMA, validate_release


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class ModelReleaseValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint_path = self.root / "runs/selected/checkpoint_best.pt"
        self.init_path = self.root / "runs/phase2/checkpoint_best.pt"
        self.manifest_path = self.root / "data/random_conformer_v3/DATA_MANIFEST.json"
        self.plan_path = self.root / "runs/evaluation/plan.jsonl"
        self.single_path = self.root / "runs/evaluation/single.json"
        self.multi_path = self.root / "runs/evaluation/multi.json"
        self.summary_path = self.root / "runs/evaluation/summary.json"
        for path in (self.checkpoint_path, self.init_path, self.manifest_path, self.plan_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.init_path.write_bytes(b"phase2-init")
        self.plan_path.write_text('{"query": 1}\n', encoding="utf-8")
        self.manifest = {
            "manifest_schema": "pepclip-data-manifest-v2",
            "dataset_version": "random_conformer_v3",
            "parent_dataset": "random_conformer_v2",
            "relation_schema": "pairs-v3",
            "cache_schema": "cache-v2",
            "generator_id": "generator-v3",
            "absolute_paths_recorded": False,
        }
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self._write_checkpoint()
        checkpoint_sha = sha256(self.checkpoint_path)
        self.single = {
            "schema_version": "single-v1",
            "requested_model_label": "selected",
            "checkpoints": {"selected": {"sha256": checkpoint_sha}},
            "validation_interface_pair_ids_sha256": "1" * 64,
            "fixed_validation_plan": {"file_sha256": sha256(self.plan_path), "canonical_sha256": "2" * 64},
        }
        self.multi = {
            "schema_version": "multi-v1",
            "requested_model_label": "selected",
            "checkpoints": {"selected": {"sha256": checkpoint_sha}},
            "conformer_indices": list(range(10)),
            "conformer_zero_regression": "passed",
            "fixed_validation_plan_sha256": "2" * 64,
        }
        self._write_json(self.single_path, self.single)
        self._write_json(self.multi_path, self.multi)
        self._write_json(self.summary_path, {"result": "balanced"})
        self.descriptor_path = self.root / "release.json"
        self.descriptor = self._descriptor()
        self._save_descriptor()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_checkpoint(self, *, schema=CHECKPOINT_SCHEMA, data_version="v3", finite=True):
        value = torch.tensor([1.0, 2.0] if finite else [1.0, float("nan")])
        torch.save(
            {
                "schema_version": schema,
                "epoch": 0,
                "global_step": 12,
                "global_seed": 1,
                "best_validation_loss": 2.5,
                "model_state_dict": {"weight": value},
                "data_contract": {
                    "data_version": data_version,
                    "dataset_version": "random_conformer_v3",
                    "data_manifest_path": str(self.manifest_path),
                    "data_manifest_sha256": sha256(self.manifest_path),
                    "database_contract": "db-v3",
                    "cache_schema": "cache-v2",
                    "generator_id": "generator-v3",
                    "qc_id": "qc-v1",
                },
                "run_config": {
                    "data_version": data_version,
                    "dataset_version": "random_conformer_v3",
                    "data_manifest_sha256": sha256(self.manifest_path),
                    "phase2_checkpoint": str(self.init_path),
                },
            },
            self.checkpoint_path,
        )

    @staticmethod
    def _write_json(path: Path, value: dict):
        path.write_text(json.dumps(value), encoding="utf-8")

    def _descriptor(self):
        return {
            "schema_version": RELEASE_SCHEMA,
            "release_id": "test-release",
            "model_version": "test-model-v1",
            "release_status": "selected",
            "checkpoint": {
                "relative_path": "runs/selected/checkpoint_best.pt",
                "sha256": sha256(self.checkpoint_path),
                "bytes": self.checkpoint_path.stat().st_size,
                "schema_version": CHECKPOINT_SCHEMA,
                "epoch": 0,
                "global_step": 12,
                "global_seed": 1,
                "best_validation_loss": 2.5,
                "model_state": {
                    "state_dict_key": "model_state_dict",
                    "tensor_count": 1,
                    "parameter_and_buffer_numel": 2,
                    "all_floating_and_complex_tensors_finite": True,
                },
            },
            "data_contract": {
                "data_version": "v3",
                "dataset_version": "random_conformer_v3",
                "manifest_relative_path": "data/random_conformer_v3/DATA_MANIFEST.json",
                "manifest_sha256": sha256(self.manifest_path),
                "manifest_contract": self.manifest,
                "checkpoint_contract": {
                    "database_contract": "db-v3",
                    "cache_schema": "cache-v2",
                    "generator_id": "generator-v3",
                    "qc_id": "qc-v1",
                },
            },
            "initialization": {
                "checkpoint_relative_path": "runs/phase2/checkpoint_best.pt",
                "checkpoint_sha256": sha256(self.init_path),
                "checkpoint_bytes": self.init_path.stat().st_size,
            },
            "code_baseline": {"training_and_evaluation_git_commit": "a" * 40},
            "evaluation": {
                "model_label": "selected",
                "validation_interface_pair_ids_sha256": "1" * 64,
                "conformer_indices": list(range(10)),
                "fixed_plan": {
                    "relative_path": "runs/evaluation/plan.jsonl",
                    "file_sha256": sha256(self.plan_path),
                    "canonical_sha256": "2" * 64,
                },
                "reports": {
                    "single_conformer": {
                        "relative_path": "runs/evaluation/single.json",
                        "sha256": sha256(self.single_path),
                        "schema_version": "single-v1",
                    },
                    "multi_conformer_config": {
                        "relative_path": "runs/evaluation/multi.json",
                        "sha256": sha256(self.multi_path),
                        "schema_version": "multi-v1",
                    },
                    "multi_conformer_summary": {
                        "relative_path": "runs/evaluation/summary.json",
                        "sha256": sha256(self.summary_path),
                    },
                },
            },
        }

    def _save_descriptor(self):
        self._write_json(self.descriptor_path, self.descriptor)

    def test_valid_release_passes(self):
        result = validate_release(self.descriptor_path, self.root)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["evaluation_reports_checked"], 3)

    def test_checkpoint_sha_mismatch_fails(self):
        self.descriptor["checkpoint"]["sha256"] = "F" * 64
        self._save_descriptor()
        result = validate_release(self.descriptor_path, self.root)
        self.assertIn("checkpoint:sha256", result["errors"])

    def test_checkpoint_schema_mismatch_fails_after_hash_is_updated(self):
        self._write_checkpoint(schema="wrong-schema")
        self.descriptor["checkpoint"]["sha256"] = sha256(self.checkpoint_path)
        self.descriptor["checkpoint"]["bytes"] = self.checkpoint_path.stat().st_size
        self.single["checkpoints"]["selected"]["sha256"] = self.descriptor["checkpoint"]["sha256"]
        self.multi["checkpoints"]["selected"]["sha256"] = self.descriptor["checkpoint"]["sha256"]
        self._write_json(self.single_path, self.single)
        self._write_json(self.multi_path, self.multi)
        self.descriptor["evaluation"]["reports"]["single_conformer"]["sha256"] = sha256(self.single_path)
        self.descriptor["evaluation"]["reports"]["multi_conformer_config"]["sha256"] = sha256(self.multi_path)
        self._save_descriptor()
        result = validate_release(self.descriptor_path, self.root)
        self.assertIn("checkpoint:schema_version", result["errors"])

    def test_data_version_and_manifest_contract_mismatch_fail(self):
        self._write_checkpoint(data_version="v2")
        self.descriptor["checkpoint"]["sha256"] = sha256(self.checkpoint_path)
        self.descriptor["checkpoint"]["bytes"] = self.checkpoint_path.stat().st_size
        self.manifest["dataset_version"] = "random_conformer_v2"
        self._write_json(self.manifest_path, self.manifest)
        self.descriptor["data_contract"]["manifest_sha256"] = sha256(self.manifest_path)
        self.descriptor["data_contract"]["manifest_contract"]["dataset_version"] = "random_conformer_v3"
        self._save_descriptor()
        result = validate_release(self.descriptor_path, self.root)
        self.assertIn("checkpoint:data_contract:data_version", result["errors"])
        self.assertIn("manifest:dataset_version", result["errors"])

    def test_nonfinite_model_state_fails(self):
        self._write_checkpoint(finite=False)
        self.descriptor["checkpoint"]["sha256"] = sha256(self.checkpoint_path)
        self.descriptor["checkpoint"]["bytes"] = self.checkpoint_path.stat().st_size
        self._save_descriptor()
        result = validate_release(self.descriptor_path, self.root)
        self.assertIn("checkpoint:model_state:finite", result["errors"])

    def test_absolute_checkpoint_path_is_rejected(self):
        self.descriptor["checkpoint"]["relative_path"] = str(self.checkpoint_path)
        self._save_descriptor()
        result = validate_release(self.descriptor_path, self.root)
        self.assertIn("checkpoint:path_must_be_repo_relative", result["errors"])


if __name__ == "__main__":
    unittest.main()

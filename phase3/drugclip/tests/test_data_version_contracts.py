from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from phase3.drugclip.data_contract import sha256_file
from phase3.drugclip.io_utils import read_jsonl, write_jsonl
from phase3.drugclip.random_augmentation_dataset import Phase3RandomConformerDataset
from phase3.drugclip.tests.fixture_data import build_fixture, build_versioned_fixture
from phase3.drugclip.training_state import load_training_checkpoint, make_grad_scaler, save_training_checkpoint


def _dataset(paths: dict[str, Path], version: str) -> Phase3RandomConformerDataset:
    return Phase3RandomConformerDataset(
        paths["pairs"],
        paths["cache"],
        paths["biological"],
        paths["splits"],
        split="train",
        mode="train_random",
        global_seed=7,
        data_version=version,
        dataset_root=paths.get("root") if version == "v3" else None,
    )


class DataVersionContractTests(unittest.TestCase):
    def test_v3_explicit_accepts_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_versioned_fixture(Path(tmp), "v3")
            dataset = _dataset(paths, "v3")
            summary = dataset.mapping_summary()
            self.assertEqual(summary["data_version"], "v3")
            self.assertEqual(summary["dataset_version"], "random_conformer_v3")
            self.assertEqual(summary["generator_id"], "internal-coordinate-rama-v2-clash15")
            self.assertEqual(len(summary["manifest_sha256"]), 64)

    def test_v2_still_accepts_legacy_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            dataset = _dataset(paths, "v2")
            summary = dataset.mapping_summary()
            self.assertEqual(summary["data_version"], "v2")
            self.assertEqual(summary["generator_id"], "internal-coordinate-rama-v1")

    def test_v2_and_v3_contracts_reject_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v2 = build_fixture(root / "v2")
            v3 = build_versioned_fixture(root / "v3", "v3")
            with self.assertRaisesRegex(ValueError, "unsupported_random_pair_schema|unsupported_cache_schema"):
                _dataset(v3, "v2")
            with self.assertRaisesRegex(ValueError, "v3_path_not_manifest_formal_file|unsupported_random_pair_schema"):
                Phase3RandomConformerDataset(
                    v2["pairs"], v2["cache"], v2["biological"], v2["splits"],
                    split="train", mode="train_random", data_version="v3", dataset_root=v3["root"],
                )

    def test_manifest_missing_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_versioned_fixture(Path(tmp), "v3")
            paths["manifest"].unlink()
            with self.assertRaisesRegex(FileNotFoundError, "v3_manifest_missing"):
                _dataset(paths, "v3")

    def test_manifest_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_versioned_fixture(Path(tmp), "v3")
            with self.assertRaisesRegex(ValueError, "manifest_sha256_mismatch"):
                Phase3RandomConformerDataset(
                    paths["pairs"], paths["cache"], paths["biological"], paths["splits"],
                    split="train", data_version="v3", dataset_root=paths["root"],
                    expected_manifest_sha256="0" * 64,
                )

    def test_generator_qc_id_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_versioned_fixture(Path(tmp), "v3")
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            manifest["generator_id"] = "bad-generator"
            paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "v3_generator_id_mismatch"):
                _dataset(paths, "v3")

    def test_v3_cache_requires_strictly_ten_conformers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_versioned_fixture(Path(tmp), "v3")
            rows = list(read_jsonl(paths["cache"]))
            rows[0]["conformers"].pop()
            write_jsonl(paths["cache"], rows)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            for row in manifest["formal_files"]:
                if row["relative_path"] == "03_random_conformer_cache/random_conformer_cache.jsonl":
                    row["sha256"] = sha256_file(paths["cache"])
                    row["bytes"] = paths["cache"].stat().st_size
            paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly_10"):
                _dataset(paths, "v3")

    def test_checkpoint_data_version_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
            scaler = make_grad_scaler(torch.device("cpu"), False)
            config = {
                "phase2_checkpoint": "phase2.pt",
                "data_version": "v3",
                "dataset_version": "random_conformer_v3",
                "dataset_root": "root",
                "data_manifest_path": "root/DATA_MANIFEST.json",
                "data_manifest_sha256": "A" * 64,
                "database_contract": "drugclip-exact-peptide-random-conformer-v3",
                "cache_schema": "drugclip-random-conformer-cache-v2",
                "generator_id": "internal-coordinate-rama-v2-clash15",
                "qc_id": "clash15-nca-c-gap2-distance-lt1.5-v1",
                "random_pairs_sha256": "B" * 64,
                "random_conformer_cache_sha256": "C" * 64,
                "pair_splits_sha256": "D" * 64,
                "relation_schema": "drugclip-random-augmentation-pairs-v3",
                "random_pairs_jsonl": "pairs.jsonl",
                "valid_random_pairs_jsonl": "pairs.jsonl",
                "random_conformer_cache_jsonl": "cache.jsonl",
                "biological_pairs_jsonl": "bio.jsonl",
                "biological_pairs_sha256": "E" * 64,
                "pair_splits_jsonl": "splits.jsonl",
                "freeze_configuration": {},
                "global_seed": 7,
                "sampling_unit": "interface_pair",
                "train_interface_pair_ids": ["a"],
                "valid_interface_pair_ids": ["b"],
                "train_interface_pair_ids_sha256": "F" * 64,
                "valid_interface_pair_ids_sha256": "1" * 64,
                "fixed_validation_plan_sha256": "2" * 64,
                "total_train_steps": 1,
                "warmup_fraction": 0.0,
                "warmup_steps": 0,
                "scheduler_kind": "linear_warmup_constant",
            }
            path = Path(tmp) / "ckpt.pt"
            save_training_checkpoint(
                path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=0, global_step=0, global_seed=7, best_validation_loss=1.0,
                run_config=config, sampler_state={}, history=[],
            )
            mismatch = dict(config)
            mismatch["data_version"] = "v2"
            with self.assertRaisesRegex(ValueError, "data_version"):
                load_training_checkpoint(
                    path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    expected_run_config=mismatch, device=torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()

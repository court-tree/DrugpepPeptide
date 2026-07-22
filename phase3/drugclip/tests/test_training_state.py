from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import torch

from phase3.drugclip.training_state import (
    CHECKPOINT_SCHEMA,
    amp_is_enabled,
    load_training_checkpoint,
    make_grad_scaler,
    save_training_checkpoint,
)
from phase3.drugclip.train import _build_warmup_scheduler, _step_checkpoint_path


def config() -> dict:
    return {
        "phase2_checkpoint": "phase2.pt",
        "relation_schema": "schema",
        "random_pairs_jsonl": "pairs.jsonl",
        "valid_random_pairs_jsonl": "pairs.jsonl",
        "random_conformer_cache_jsonl": "cache.jsonl",
        "biological_pairs_jsonl": "biological.jsonl",
        "biological_pairs_sha256": "A" * 64,
        "pair_splits_jsonl": "splits.jsonl",
        "freeze_configuration": {"one_d": 1, "three_d": 1},
        "global_seed": 17,
        "sampling_unit": "interface_pair",
        "train_interface_pair_ids": ["pair:bio:1:0", "pair:bio:1:1"],
        "valid_interface_pair_ids": ["pair:bio:3:0"],
    }


class TrainingStateTests(unittest.TestCase):
    def test_linear_warmup_reaches_base_learning_rate(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler, warmup_steps = _build_warmup_scheduler(
            optimizer, total_train_steps=100, warmup_fraction=0.05
        )
        self.assertEqual(warmup_steps, 5)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.2)
        for _ in range(4):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0)

    def test_cpu_disables_amp(self):
        self.assertFalse(amp_is_enabled(torch.device("cpu"), True))

    def test_checkpoint_restores_model_optimizer_scheduler_scaler_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
            scaler = make_grad_scaler(torch.device("cpu"), False)
            loss = model(torch.ones(2, 3)).square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            saved_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
            random.seed(91)
            torch.manual_seed(91)
            path = Path(tmp) / "checkpoint.pt"
            save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=2,
                global_step=11,
                global_seed=17,
                best_validation_loss=0.25,
                run_config=config(),
                sampler_state={"next_epoch": 3, "next_train_plan_hash": "ABC"},
                history=[{"epoch": 2}],
            )
            expected_python = random.random()
            expected_torch = torch.rand(3)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1.0)
            random.random()
            torch.rand(3)
            restored = load_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                expected_run_config=config(),
                device=torch.device("cpu"),
            )
            self.assertEqual(restored["schema_version"], CHECKPOINT_SCHEMA)
            self.assertEqual(restored["epoch"], 2)
            self.assertEqual(restored["global_step"], 11)
            self.assertEqual(restored["sampler_state"]["next_train_plan_hash"], "ABC")
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, saved_weights[key]))
            self.assertTrue(optimizer.state_dict()["state"])
            self.assertEqual(scheduler.last_epoch, 1)
            self.assertEqual(random.random(), expected_python)
            self.assertTrue(torch.equal(torch.rand(3), expected_torch))

    def test_checkpoint_rejects_data_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
            scaler = make_grad_scaler(torch.device("cpu"), False)
            path = Path(tmp) / "checkpoint.pt"
            save_training_checkpoint(
                path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=0, global_step=0, global_seed=17, best_validation_loss=1.0,
                run_config=config(), sampler_state={}, history=[],
            )
            mismatch = config()
            mismatch["biological_pairs_sha256"] = "B" * 64
            with self.assertRaisesRegex(ValueError, "biological_pairs_sha256"):
                load_training_checkpoint(
                    path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    expected_run_config=mismatch, device=torch.device("cpu"),
                )

    def test_step_032_checkpoint_contains_v3_contract_and_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            scaler = make_grad_scaler(torch.device("cpu"), False)
            loss = model(torch.ones(1, 2)).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            run_config = config()
            run_config.update({
                "data_version": "v3",
                "dataset_version": "random_conformer_v3",
                "dataset_root": "dataset-v3",
                "data_manifest_path": "dataset-v3/DATA_MANIFEST.json",
                "data_manifest_sha256": "043278F18EFC9B9C3238788D4C6B34C35641C9C26895E5045D8598FA99D5C309",
                "database_contract": "db-v3",
                "cache_schema": "cache-v3",
                "generator_id": "generator-v3",
                "qc_id": "qc-v3",
                "random_pairs_sha256": "1" * 64,
                "random_conformer_cache_sha256": "2" * 64,
                "pair_splits_sha256": "3" * 64,
            })
            path = _step_checkpoint_path(Path(tmp), 32)
            self.assertEqual(path.name, "step_032.pt")
            save_training_checkpoint(
                path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=0, global_step=32, global_seed=17, best_validation_loss=0.5,
                run_config=run_config,
                sampler_state={"current_epoch": 0, "batches_completed_in_epoch": 32},
                history=[],
            )
            raw = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(raw["global_step"], 32)
            self.assertEqual(raw["data_contract"]["data_version"], "v3")
            self.assertEqual(raw["data_contract"]["data_manifest_sha256"], run_config["data_manifest_sha256"])
            restored = load_training_checkpoint(
                path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                expected_run_config=run_config, device=torch.device("cpu"),
            )
            self.assertEqual(restored["sampler_state"]["batches_completed_in_epoch"], 32)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from phase3.drugclip.batching import UniquePeptideBatchSampler, collate_phase3
from phase3.drugclip.io_utils import read_jsonl, write_jsonl
from phase3.drugclip.random_augmentation_dataset import Phase3RandomConformerDataset
from phase3.drugclip.tests.fixture_data import build_fixture


def make_dataset(paths: dict[str, Path], **kwargs) -> Phase3RandomConformerDataset:
    return Phase3RandomConformerDataset(
        paths["pairs"],
        paths["cache"],
        paths["biological"],
        paths["splits"],
        split="train",
        mode=kwargs.pop("mode", "train_random"),
        global_seed=kwargs.pop("global_seed", 1776),
        **kwargs,
    )


class BiologicalMappingTests(unittest.TestCase):
    def test_mapping_is_exact_unique_and_summary_is_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            dataset = make_dataset(paths)
            summary = dataset.mapping_summary()
            self.assertEqual(summary["biological_relations"], 4)
            self.assertEqual(summary["interface_rows"], 5)
            self.assertEqual(summary["missing_mappings"], 0)
            self.assertEqual(summary["ambiguous_mappings"], 0)
            self.assertEqual(len(summary["biological_pairs_sha256"]), 64)
            output = Path(tmp) / "mapping_summary.json"
            dataset.write_mapping_summary(output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), summary)

    def test_missing_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            biological = list(read_jsonl(paths["biological"]))
            write_jsonl(paths["biological"], biological[1:])
            with self.assertRaisesRegex(ValueError, "missing_biological_relation_mapping"):
                make_dataset(paths)

    def test_ambiguous_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            biological = list(read_jsonl(paths["biological"]))
            duplicate = dict(biological[0])
            duplicate["pair_id"] = "bio:ambiguous"
            write_jsonl(paths["biological"], [*biological, duplicate])
            with self.assertRaisesRegex(ValueError, "ambiguous_biological_relation_mapping"):
                make_dataset(paths)


class InterfacePairSamplingTests(unittest.TestCase):
    def test_every_interface_pair_once_and_multi_interface_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            dataset = make_dataset(paths)
            plan = dataset.epoch_plan()
            relation_ids = [row["biological_pair_id"] for row in plan]
            interface_pair_ids = [row["interface_pair_id"] for row in plan]
            self.assertEqual(len(plan), 5)
            self.assertEqual(len(interface_pair_ids), len(set(interface_pair_ids)))
            self.assertEqual(set(interface_pair_ids), set(dataset.interface_pair_ids))
            self.assertEqual(relation_ids.count("bio:1"), 2)
            self.assertEqual(
                {row["interface_pair_id"] for row in plan if row["biological_pair_id"] == "bio:1"},
                {"pair:bio:1:0", "pair:bio:1:1"},
            )

    def test_seed_and_epoch_are_reproducible_and_epochs_can_change_choices(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            first = make_dataset(paths, global_seed=31337)
            second = make_dataset(paths, global_seed=31337)
            first.set_epoch(7)
            second.set_epoch(7)
            self.assertEqual(first.epoch_plan(), second.epoch_plan())

            conformer_choices = set()
            for epoch in range(32):
                first.set_epoch(epoch)
                selected = {
                    row["interface_pair_id"]: row for row in first.epoch_plan()
                }["pair:bio:1:0"]
                conformer_choices.add(selected["conformer_index"])
            self.assertGreater(len(conformer_choices), 1)

    def test_interface_is_never_randomly_selected_within_relation(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            dataset = make_dataset(paths, global_seed=31337)
            expected = set(dataset.interface_pair_ids)
            for epoch in range(8):
                dataset.set_epoch(epoch)
                plan = dataset.epoch_plan()
                self.assertEqual({row["interface_pair_id"] for row in plan}, expected)
                self.assertEqual(
                    sum(row["biological_pair_id"] == "bio:1" for row in plan), 2
                )

    def test_fixed_mode_is_identical_across_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            dataset = make_dataset(
                paths,
                mode="fixed",
                fixed_conformer_index=7,
            )
            initial = dataset.epoch_plan()
            initial_atoms = [dataset[index]["peptide_atoms"] for index in range(len(dataset))]
            dataset.set_epoch(999)
            self.assertEqual(dataset.epoch_plan(), initial)
            self.assertEqual(
                [dataset[index]["peptide_atoms"] for index in range(len(dataset))],
                initial_atoms,
            )
            self.assertEqual({row["conformer_index"] for row in initial}, {7})


class CacheContractTests(unittest.TestCase):
    def _assert_cache_error(self, mutate, pattern: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            rows = list(read_jsonl(paths["cache"]))
            mutate(rows[0])
            write_jsonl(paths["cache"], rows)
            with self.assertRaisesRegex((ValueError, KeyError, TypeError), pattern):
                make_dataset(paths)

    def test_cache_contract_errors_are_rejected(self):
        cases = [
            (lambda row: row.update(schema_version="bad"), "unsupported_cache_schema"),
            (lambda row: row.update(generator_id="bad"), "unsupported_cache_generator"),
            (lambda row: row.update(max_conformers_requested=9), "max_conformers"),
            (lambda row: row["conformers"].pop(), "exactly_10"),
            (
                lambda row: row["conformers"][0].update(conformer_index=1),
                "indices_must_be_0_to_9",
            ),
            (
                lambda row: row["conformers"][1].update(
                    conformer_id=row["conformers"][0]["conformer_id"]
                ),
                "duplicate_conformer_id",
            ),
            (
                lambda row: row["conformers"][0]["backbone_atoms"].pop(),
                "backbone_atom_count",
            ),
            (
                lambda row: row["conformers"][0]["backbone_atoms"][0].update(x=float("nan")),
                "nonfinite_coordinate",
            ),
        ]
        for mutate, pattern in cases:
            with self.subTest(pattern=pattern):
                self._assert_cache_error(mutate, pattern)


class UniqueBatchingTests(unittest.TestCase):
    def test_duplicate_peptide_is_delayed_without_interface_pair_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            dataset = make_dataset(paths)
            sampler = UniquePeptideBatchSampler(dataset, batch_size=3, seed=19)
            batches = list(sampler)
            flattened = [index for batch in batches for index in batch]
            self.assertEqual(len(flattened), len(dataset))
            self.assertEqual(len(set(flattened)), len(dataset))
            for batch in batches:
                peptides = [dataset.peptide_sequence_for_index(index) for index in batch]
                self.assertEqual(len(peptides), len(set(peptides)))
            summary = sampler.summary()
            self.assertEqual(summary["interface_pair_loss_count"], 0)
            self.assertEqual(summary["interface_pair_duplicate_count"], 0)
            self.assertEqual(summary["peptide_uniqueness_violations"], 0)

    def test_collate_is_finite_and_compatibility_fields_are_neutral(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            dataset = make_dataset(paths)
            sampler = UniquePeptideBatchSampler(dataset, batch_size=3, seed=19)
            indices = next(iter(sampler))
            batch = collate_phase3([dataset[index] for index in indices])
            self.assertTrue(torch.isfinite(batch["one_d"]["receptor_tokens"].float()).all())
            self.assertTrue(torch.isfinite(batch["three_d"]["receptor_coords"]).all())
            self.assertEqual(batch["one_d"]["peptide_homology_80_id"], [""] * len(indices))
            self.assertEqual(batch["one_d"]["receptor_family_30_id"], [""] * len(indices))


if __name__ == "__main__":
    unittest.main()

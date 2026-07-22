import tempfile
import unittest
from pathlib import Path

from phase3.drugclip.batching import collate_phase3
from phase3.drugclip.io_utils import write_jsonl
from phase3.drugclip.random_augmentation_dataset import Phase3RandomConformerDataset
from phase3.drugclip.random_conformers import generate_conformer
from phase3.drugclip.tests.fixture_data import build_fixture


class RandomConformerTests(unittest.TestCase):
    def test_generator_is_deterministic_and_sequence_complete(self):
        first = generate_conformer("ACDEFGHI", "train", 0)
        second = generate_conformer("ACDEFGHI", "train", 0)
        self.assertEqual(first["seed"], second["seed"])
        self.assertEqual(first["backbone_atoms"], second["backbone_atoms"])
        self.assertEqual(len(first["backbone_atoms"]), 24)
        third = generate_conformer("ACDEFGHI", "train", 1)
        self.assertNotEqual(first["backbone_atoms"], third["backbone_atoms"])

    def test_dataset_and_collate_use_interface_pair_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            dataset = Phase3RandomConformerDataset(
                paths["pairs"], paths["cache"], paths["biological"], paths["splits"],
                split="train", mode="train_random", global_seed=1,
            )
            self.assertEqual(len(dataset), 5)
            self.assertEqual(dataset.interface_row_count, 5)
            items = [dataset[index] for index in range(len(dataset))]
            unique_items = []
            seen = set()
            for item in items:
                if item["peptide_sequence"] not in seen:
                    unique_items.append(item)
                    seen.add(item["peptide_sequence"])
            batch = collate_phase3(unique_items)
            self.assertEqual(batch["one_d"]["sample_id"], batch["three_d"]["sample_id"])
            self.assertEqual(batch["one_d"]["peptide_homology_80_id"], [""] * len(unique_items))
            self.assertEqual(batch["one_d"]["receptor_family_30_id"], [""] * len(unique_items))

    def test_dataset_rejects_superseded_v1_relations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = build_fixture(root)
            write_jsonl(
                paths["pairs"],
                [{
                    "schema_version": "drugclip-random-augmentation-pairs-v1",
                    "database_contract": "old",
                    "split": "train",
                    "pair": {"pair_id": "pair:bio:1:0"},
                }],
            )
            with self.assertRaisesRegex(ValueError, "v1 is superseded"):
                Phase3RandomConformerDataset(
                    paths["pairs"], paths["cache"], paths["biological"], paths["splits"],
                    split="train", mode="train_random", global_seed=1,
                )


if __name__ == "__main__":
    unittest.main()

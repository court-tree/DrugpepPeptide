from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from phase3.drugclip.audit_random_conformer_v3 import independent_clash15
from phase3.drugclip.random_conformer_v3 import (
    CACHE_SCHEMA,
    DATABASE_CONTRACT,
    DATASET_VERSION,
    GENERATOR_ID,
    PARENT_DATASET,
    RELATION_SCHEMA,
    attempt_seed,
    clash15_details,
    coordinate_sha256,
    generate_from_seed,
)
from phase3.drugclip.random_conformers import (
    GENERATOR_ID as V2_GENERATOR_ID,
    SCHEMA_VERSION as V2_CACHE_SCHEMA,
)


def atom(residue: int, name: str, x: float, y: float = 0.0, z: float = 0.0) -> dict:
    return {"residue_id": f"P:{residue}", "atom_name": name, "x": x, "y": y, "z": z}


class Clash15BoundaryTests(unittest.TestCase):
    def test_149_is_rejected(self) -> None:
        atoms = [atom(1, "N", 0.0), atom(3, "CA", 1.49)]
        self.assertTrue(clash15_details(atoms)["has_clash"])
        self.assertEqual(independent_clash15(atoms)[0], 1)

    def test_150_is_accepted(self) -> None:
        atoms = [atom(1, "N", 0.0), atom(3, "CA", 1.50)]
        self.assertFalse(clash15_details(atoms)["has_clash"])
        self.assertEqual(independent_clash15(atoms)[0], 0)

    def test_adjacent_residues_are_ignored(self) -> None:
        atoms = [atom(1, "C", 0.0), atom(2, "N", 0.1)]
        self.assertFalse(clash15_details(atoms)["has_clash"])
        self.assertEqual(independent_clash15(atoms)[0], 0)

    def test_residue_gap_two_is_checked(self) -> None:
        atoms = [atom(4, "C", 0.0), atom(6, "N", 0.2)]
        self.assertTrue(clash15_details(atoms)["has_clash"])


class VersionAndDeterminismTests(unittest.TestCase):
    def test_v2_and_v3_are_distinguishable(self) -> None:
        self.assertNotEqual(GENERATOR_ID, V2_GENERATOR_ID)
        self.assertNotEqual(CACHE_SCHEMA, V2_CACHE_SCHEMA)
        self.assertTrue(RELATION_SCHEMA.endswith("v3"))
        self.assertTrue(DATABASE_CONTRACT.endswith("v3"))

    def test_attempt_seed_is_deterministic_and_attempt_specific(self) -> None:
        first = attempt_seed("train", "ACDEFGHI", 2, 1)
        self.assertEqual(first, attempt_seed("train", "ACDEFGHI", 2, 1))
        self.assertNotEqual(first, attempt_seed("train", "ACDEFGHI", 2, 2))

    def test_generation_and_coordinate_hash_are_reproducible(self) -> None:
        first = generate_from_seed("ACDEFGHI", 123456)
        second = generate_from_seed("ACDEFGHI", 123456)
        self.assertEqual(first, second)
        self.assertEqual(coordinate_sha256(first), coordinate_sha256(second))


class ParentReadOnlyGuardTests(unittest.TestCase):
    def test_builder_rejects_output_inside_parent(self) -> None:
        from phase3.drugclip.build_random_conformer_v3 import build
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "random_conformer_v2"
            parent.mkdir()
            with self.assertRaises(ValueError):
                build(parent, parent / "random_conformer_v3")


class ReleaseDescriptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(__file__).resolve().parents[1] / "releases" / "random_conformer_v3.json"
        cls.text = cls.path.read_text(encoding="utf-8")
        cls.descriptor = json.loads(cls.text)

    def test_descriptor_matches_source_contract(self) -> None:
        self.assertEqual(self.descriptor["dataset_version"], DATASET_VERSION)
        self.assertEqual(self.descriptor["parent_dataset"], PARENT_DATASET)
        self.assertEqual(self.descriptor["cache_schema"], CACHE_SCHEMA)
        self.assertEqual(self.descriptor["relation_schema"], RELATION_SCHEMA)
        self.assertEqual(self.descriptor["generator_id"], GENERATOR_ID)
        self.assertIn("clash15", self.descriptor["generator_id"])
        self.assertEqual(self.descriptor["independent_clash15_failures"], 0)
        self.assertEqual(self.descriptor["determinism_mismatches"], 0)

    def test_manifest_hash_is_uppercase_sha256(self) -> None:
        self.assertRegex(self.descriptor["manifest_sha256"], re.compile(r"^[0-9A-F]{64}$"))

    def test_frozen_release_counts(self) -> None:
        self.assertEqual(
            self.descriptor["counts"],
            {"pairs": 24633, "caches": 6979, "conformers": 69790, "replaced": 516},
        )

    def test_descriptor_contains_no_absolute_paths(self) -> None:
        self.assertNotRegex(self.text, re.compile(r"[A-Za-z]:[\\/]"))
        self.assertNotRegex(self.text, re.compile(r"/(?:home|mnt|Users)/"))


if __name__ == "__main__":
    unittest.main()

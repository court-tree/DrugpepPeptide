from __future__ import annotations

import inspect
import unittest

from phase3.drugclip.constrained_full_atom_conformer_prototype import (
    BACKBONE_COORDINATE_TOLERANCE_ANGSTROM,
    MAX_COMPLETION_ATTEMPTS,
    MMFF_MAX_ITERATIONS,
    conformer_atoms,
    generate_constrained_full_atom_conformers,
)
from phase3.drugclip.full_atom_conformer_prototype import (
    REQUIRED_HEAVY_ATOMS,
    UnsupportedPeptideChemistry,
)
from phase3.drugclip.validate_constrained_full_atom_prototype import (
    PANEL,
    SPECIAL_CHEMISTRY_CLASSES,
    _final_classification,
    cpu_egnn_forward_all,
    validate_payload,
)
from phase3.drugclip.random_conformer_v3 import (
    coordinate_sha256 as backbone_coordinate_sha256,
)
from phase3.drugclip.random_conformers import generate_conformer


def _seed_plan(sequence: str, count: int) -> list[dict[str, object]]:
    output = []
    for index in range(count):
        conformer = generate_conformer(
            sequence, "constrained-prototype-test", index
        )
        output.append({
            "conformer_index": index,
            "seed": conformer["seed"],
            "attempt_index": 0,
            "split": "constrained-prototype-test",
            "backbone_coordinate_sha256": backbone_coordinate_sha256(
                conformer["backbone_atoms"]
            ),
        })
    return output


class ConstrainedFullAtomPrototypeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = generate_constrained_full_atom_conformers(
            "SAVTTVVN",
            num_conformers=1,
            backbone_seed_plan=_seed_plan("SAVTTVVN", 1),
        )

    def test_generates_complete_conformer(self) -> None:
        result = validate_payload(self.payload, expected_conformers=1)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["conformer_count"], 1)
        self.assertEqual(result["unique_coordinate_hashes"], 1)

    def test_n_ca_c_backbone_is_exactly_fixed(self) -> None:
        for conformer in self.payload["conformers"]:
            self.assertEqual(
                conformer["input_backbone_coordinate_sha256"],
                conformer["output_backbone_coordinate_sha256"],
            )
            self.assertLessEqual(
                conformer["backbone_deviation_after_embedding"][
                    "maximum_angstrom"
                ],
                BACKBONE_COORDINATE_TOLERANCE_ANGSTROM,
            )
            self.assertLessEqual(
                conformer["backbone_deviation_after_optimization"][
                    "maximum_angstrom"
                ],
                BACKBONE_COORDINATE_TOLERANCE_ANGSTROM,
            )

    def test_only_converged_local_optimization_is_accepted(self) -> None:
        self.assertEqual(MMFF_MAX_ITERATIONS, 500)
        self.assertEqual(MAX_COMPLETION_ATTEMPTS, 10)
        for conformer in self.payload["conformers"]:
            self.assertEqual(conformer["mmff_status"], 0)
            self.assertIsNone(
                conformer["completion_attempts"][-1]["rejection_reason"]
            )

    def test_full_standard_heavy_atom_identity_is_complete(self) -> None:
        by_residue = {}
        for atom in self.payload["atom_identity"]:
            by_residue.setdefault(
                (atom["residue_index"], atom["residue_name"]), set()
            ).add(atom["atom_name"])
        for (residue_index, residue_name), names in by_residue.items():
            required = set(REQUIRED_HEAVY_ATOMS[residue_name])
            if residue_index == len(self.payload["peptide_sequence"]):
                required.add("OXT")
            self.assertTrue(required <= names)

    def test_atom_identity_and_order_are_shared(self) -> None:
        expected = self.payload["atom_identity"]
        for index in range(self.payload["conformer_count"]):
            atoms = conformer_atoms(self.payload, index)
            observed = [
                {key: atom[key] for key in expected[0]}
                for atom in atoms
            ]
            self.assertEqual(observed, expected)

    def test_fixed_seed_is_deterministic(self) -> None:
        first = generate_constrained_full_atom_conformers(
            "SAVTTVVN",
            num_conformers=1,
            base_seed=123,
            backbone_seed_plan=_seed_plan("SAVTTVVN", 1),
        )
        second = generate_constrained_full_atom_conformers(
            "SAVTTVVN",
            num_conformers=1,
            base_seed=123,
            backbone_seed_plan=_seed_plan("SAVTTVVN", 1),
        )
        self.assertEqual(
            first["atom_identity_sha256"], second["atom_identity_sha256"]
        )
        self.assertEqual(
            first["canonical_coordinate_set_sha256"],
            second["canonical_coordinate_set_sha256"],
        )

    def test_current_egnn_cpu_forward_accepts_conformer(self) -> None:
        result = cpu_egnn_forward_all(self.payload)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["embedding_shape"], [1, 16])
        self.assertEqual(result["tensorization_unk_count"], 0)

    def test_generator_api_has_no_target_context(self) -> None:
        parameters = set(
            inspect.signature(
                generate_constrained_full_atom_conformers
            ).parameters
        )
        self.assertEqual(parameters, {
            "sequence",
            "num_conformers",
            "base_seed",
            "chemistry_class",
            "backbone_seed_plan",
            "progress_callback",
        })
        self.assertFalse(
            self.payload["dependency_contract"]["target_bound_inputs_used"]
        )

    def test_all_special_chemistry_classes_are_rejected(self) -> None:
        for classification in SPECIAL_CHEMISTRY_CLASSES:
            sequence = (
                "ACDC"
                if classification == "multiple_cys_unknown"
                else "SAVTTVVN"
            )
            with self.assertRaises(UnsupportedPeptideChemistry):
                generate_constrained_full_atom_conformers(
                    sequence,
                    num_conformers=1,
                    backbone_seed_plan=[],
                    chemistry_class=(
                        "ordinary_linear_unmodified_standard_peptide"
                        if classification == "multiple_cys_unknown"
                        else classification
                    ),
                )

    def test_panel_is_exact_and_contains_three_prior_blocks(self) -> None:
        self.assertEqual(len(PANEL), 5)
        sequences = {row["peptide_sequence"] for row in PANEL}
        self.assertTrue({
            "KVSKAAADLMAYCEAHAKE",
            "DDFTNELKAELDRYKRENQ",
            "ENYFQAEAYNLDKVLDEFEQ",
        } <= sequences)
        self.assertTrue(any(row["sequence_length"] <= 8 for row in PANEL))
        self.assertTrue(
            any(15 <= row["sequence_length"] <= 17 for row in PANEL)
        )

    def test_timeout_has_final_classification_precedence(self) -> None:
        sequence_results = [{
            "runs": [{
                "status": "FAIL",
                "timed_out": True,
                "classification": "OPTIMIZATION_COVERAGE_FAIL",
            }],
            "deterministic_double_run": False,
            "cpu_egnn_forward_status": None,
        }]
        self.assertEqual(
            _final_classification(sequence_results, []),
            "PERFORMANCE_BLOCKED",
        )


if __name__ == "__main__":
    unittest.main()

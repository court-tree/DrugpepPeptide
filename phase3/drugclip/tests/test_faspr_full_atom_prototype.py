from __future__ import annotations

import inspect
import math
from pathlib import Path
import tempfile
import unittest

from phase3.drugclip.faspr_full_atom_conformer_prototype import (
    CARBONYL_C_O_LENGTH_ANGSTROM,
    FASPR_CONFORMER_TIMEOUT_SECONDS,
    FASPRInputContractError,
    PackingCoverageError,
    _terminal_oxt,
    _windows_to_wsl,
    generate_faspr_full_atom_conformers,
    parse_faspr_pdb,
    reconstruct_backbone_oxygen,
    write_backbone_pdb,
)
from phase3.drugclip.full_atom_conformer_prototype import (
    UnsupportedPeptideChemistry,
    classify_sequence,
)
from phase3.drugclip.random_conformer_v3 import coordinate_sha256
from phase3.drugclip.random_conformers import generate_conformer
from phase3.drugclip.validate_faspr_full_atom_prototype import (
    EXPECTED_FASPR_BINARY_SHA256,
    EXPECTED_FASPR_COMMIT,
    PANEL,
    SPECIAL_CHEMISTRY_CLASSES,
    _final_classification,
)


def _backbone(sequence: str) -> list[dict[str, object]]:
    return generate_conformer(
        sequence, "faspr-prototype-test", 0
    )["backbone_atoms"]


class FASPRFullAtomPrototypeTest(unittest.TestCase):
    def test_reconstructs_one_oxygen_per_residue(self) -> None:
        atoms, audit = reconstruct_backbone_oxygen("SAVTTVVN", _backbone("SAVTTVVN"))
        self.assertEqual(len(atoms), 4 * 8)
        self.assertEqual(sum(row["atom_name"] == "O" for row in atoms), 8)
        self.assertEqual(audit["status"], "PASS")
        self.assertFalse(audit["bound_coordinates_used"])

    def test_oxygen_reconstruction_is_deterministic(self) -> None:
        backbone = _backbone("SAVTTVVN")
        first = reconstruct_backbone_oxygen("SAVTTVVN", backbone)
        second = reconstruct_backbone_oxygen("SAVTTVVN", backbone)
        self.assertEqual(first, second)

    def test_oxygen_reconstruction_keeps_n_ca_c_exact(self) -> None:
        backbone = _backbone("SAVTTVVN")
        atoms, _ = reconstruct_backbone_oxygen("SAVTTVVN", backbone)
        observed = [
            row for row in atoms if row["atom_name"] in {"N", "CA", "C"}
        ]
        self.assertEqual(coordinate_sha256(backbone), coordinate_sha256(observed))

    def test_carbonyl_and_peptide_geometry_is_ideal(self) -> None:
        _, audit = reconstruct_backbone_oxygen("SAVTTVVN", _backbone("SAVTTVVN"))
        for row in audit["residues"]:
            self.assertAlmostEqual(
                row["c_o_length_angstrom"],
                CARBONYL_C_O_LENGTH_ANGSTROM,
                places=12,
            )
            self.assertTrue(90.0 <= row["ca_c_o_angle_degrees"] <= 150.0)
            if "peptide_c_n_length_angstrom" in row:
                self.assertTrue(
                    1.20 <= row["peptide_c_n_length_angstrom"] <= 1.45
                )

    def test_terminal_oxt_has_contract_length(self) -> None:
        oxt = _terminal_oxt([0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.0, 1.0, 0.0])
        self.assertTrue(all(math.isfinite(value) for value in oxt))
        self.assertAlmostEqual(math.dist([1.5, 0.0, 0.0], oxt), 1.25, places=12)

    def test_pdb_writer_parser_preserves_backbone_identity(self) -> None:
        atoms, _ = reconstruct_backbone_oxygen("SAVTTVVN", _backbone("SAVTTVVN"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backbone.pdb"
            write_backbone_pdb(path, atoms)
            parsed = parse_faspr_pdb(path)
        self.assertEqual(len(parsed), len(atoms))
        self.assertEqual(
            [(row["residue_index"], row["atom_name"]) for row in parsed],
            [(row["residue_index"], row["atom_name"]) for row in atoms],
        )

    def test_generator_api_has_no_target_context(self) -> None:
        parameters = set(
            inspect.signature(generate_faspr_full_atom_conformers).parameters
        )
        self.assertEqual(parameters, {
            "sequence",
            "backbone_seed_plan",
            "work_dir",
            "faspr_executable",
            "faspr_commit_sha",
            "faspr_binary_sha256",
            "chemistry_class",
            "num_conformers",
            "progress_callback",
        })
        forbidden = {"receptor", "interface", "contact", "evidence", "bound"}
        self.assertFalse(parameters & forbidden)

    def test_all_special_chemistry_classes_are_rejected(self) -> None:
        for classification in SPECIAL_CHEMISTRY_CLASSES:
            sequence = "ACDC" if classification == "multiple_cys_unknown" else "SAVTTVVN"
            with self.assertRaises(UnsupportedPeptideChemistry):
                classify_sequence(
                    sequence,
                    chemistry_class=(
                        "ordinary_linear_unmodified_standard_peptide"
                        if classification == "multiple_cys_unknown"
                        else classification
                    ),
                )

    def test_panel_is_exact(self) -> None:
        self.assertEqual(
            [row["peptide_sequence"] for row in PANEL],
            [
                "SAVTTVVN",
                "TLAPADGPTTDEVTLQV",
                "KVSKAAADLMAYCEAHAKE",
                "DDFTNELKAELDRYKRENQ",
                "ENYFQAEAYNLDKVLDEFEQ",
            ],
        )

    def test_time_limit_and_tool_commit_are_fixed(self) -> None:
        self.assertEqual(FASPR_CONFORMER_TIMEOUT_SECONDS, 30)
        self.assertEqual(EXPECTED_FASPR_COMMIT, "0d55732fd6307f373018c6bddd842291c355c5f7")
        self.assertEqual(len(EXPECTED_FASPR_BINARY_SHA256), 64)

    def test_windows_path_translation_is_deterministic(self) -> None:
        translated = _windows_to_wsl(Path("E:/pep/input.pdb"))
        self.assertEqual(translated, "/mnt/e/pep/input.pdb")

    def test_missing_tool_is_explicit_input_contract_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FASPRInputContractError):
                generate_faspr_full_atom_conformers(
                    "SAVTTVVN",
                    backbone_seed_plan=[],
                    work_dir=Path(directory) / "work",
                    faspr_executable=Path(directory) / "FASPR",
                    faspr_commit_sha=EXPECTED_FASPR_COMMIT,
                    faspr_binary_sha256=EXPECTED_FASPR_BINARY_SHA256,
                    num_conformers=1,
                )

    def test_timeout_has_final_classification_precedence(self) -> None:
        result = _final_classification(
            [{
                "runs": [{
                    "status": "FAIL",
                    "timed_out": True,
                    "classification": "PACKING_COVERAGE_FAIL",
                }],
                "deterministic_double_run": False,
                "cpu_egnn_forward_status": None,
            }],
            [],
        )
        self.assertEqual(result, "PERFORMANCE_BLOCKED")

    def test_packing_failure_class_is_distinct(self) -> None:
        self.assertTrue(issubclass(PackingCoverageError, RuntimeError))


if __name__ == "__main__":
    unittest.main()

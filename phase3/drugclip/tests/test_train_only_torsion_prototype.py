from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

from phase3.drugclip.full_atom_conformer_prototype import (
    UnsupportedPeptideChemistry,
    classify_sequence,
)
from phase3.drugclip.train_only_torsion_prior_prototype import (
    DIHEDRAL_TOLERANCE_DEGREES,
    EXPECTED_CONFORMERS,
    GENERATOR_VERSION,
    angular_error_degrees,
    build_torsion_prior,
    conformer_seed,
    context_key,
    generate_backbone,
    generate_train_only_faspr_conformers,
    load_torsion_prior,
)
from phase3.drugclip.validate_train_only_torsion_prototype import (
    FIXED_PANEL,
    MIN_CONTEXT_OBSERVATIONS,
    SPECIAL_CHEMISTRY_CLASSES,
)


def _observation(context: str, residue: str, index: int) -> dict[str, object]:
    return {
        "context_key": context,
        "residue_letter": residue,
        "pdb_id": f"{index:04d}",
        "chain_id": "P",
        "residue_id": str(index),
        "residue_index": index,
        "phi_degrees": -65.0,
        "psi_degrees": -40.0,
        "omega_degrees": 180.0,
        "source_file": f"C:/source/{index}.cif",
        "source_file_sha256": f"{index:064X}",
    }


class TrainOnlyTorsionPrototypeTest(unittest.TestCase):
    def test_panel_is_exact(self) -> None:
        self.assertEqual(
            [row["peptide_sequence"] for row in FIXED_PANEL],
            [
                "SAVTTVVN",
                "TLAPADGPTTDEVTLQV",
                "KVSKAAADLMAYCEAHAKE",
                "DDFTNELKAELDRYKRENQ",
                "ENYFQAEAYNLDKVLDEFEQ",
            ],
        )

    def test_coverage_threshold_is_fixed(self) -> None:
        self.assertEqual(MIN_CONTEXT_OBSERVATIONS, 500)
        self.assertEqual(EXPECTED_CONFORMERS, 10)

    def test_context_contract_distinguishes_pro_gly_and_prepro(self) -> None:
        self.assertEqual(context_key("APG", 1), "PRO")
        self.assertEqual(context_key("GAA", 0), "GLY")
        self.assertEqual(context_key("APG", 0), "A_PRE_PRO")
        self.assertEqual(context_key("AAA", 1), "A")

    def test_seed_uses_manifest_sequence_and_conformer(self) -> None:
        first = conformer_seed("A" * 64, "SAVTTVVN", 0)
        self.assertEqual(first, conformer_seed("A" * 64, "SAVTTVVN", 0))
        self.assertNotEqual(first, conformer_seed("B" * 64, "SAVTTVVN", 0))
        self.assertNotEqual(first, conformer_seed("A" * 64, "SAVTTVVN", 1))

    def test_angular_error_wraps_at_180(self) -> None:
        self.assertAlmostEqual(angular_error_degrees(179.0, -179.0), 2.0)

    def test_generator_api_has_no_target_context(self) -> None:
        parameters = set(
            inspect.signature(generate_train_only_faspr_conformers).parameters
        )
        forbidden = {
            "receptor",
            "interface",
            "contact",
            "evidence",
            "bound_coordinates",
        }
        self.assertFalse(parameters & forbidden)

    def test_special_chemistry_is_explicitly_rejected(self) -> None:
        for classification in SPECIAL_CHEMISTRY_CLASSES:
            sequence = "ACDC" if classification == "multiple_cys_unknown" else "AAAA"
            declared = (
                "ordinary_linear_unmodified_standard_peptide"
                if classification == "multiple_cys_unknown"
                else classification
            )
            with self.assertRaises(UnsupportedPeptideChemistry):
                classify_sequence(sequence, chemistry_class=declared)

    def test_prior_manifest_round_trip_and_joint_torsions(self) -> None:
        observations = [
            _observation("A", "A", 1),
            _observation("PRO", "P", 2),
            _observation("GLY", "G", 3),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation_path = root / "observations.jsonl"
            observation_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in observations
                ),
                encoding="utf-8",
            )
            from phase3.drugclip.train_only_torsion_prior_prototype import (
                canonical_json_sha256,
            )

            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "coverage_threshold_pass": True,
                        "observation_canonical_sha256": canonical_json_sha256(
                            observations
                        ),
                    }
                ),
                encoding="utf-8",
            )
            manifest = build_torsion_prior(
                observation_path, summary_path, root / "prior"
            )
            groups, loaded = load_torsion_prior(
                root / "prior" / "torsion_prior.jsonl",
                root / "prior" / "torsion_prior_manifest.json",
            )
        self.assertEqual(manifest, loaded)
        self.assertEqual(set(groups), {"A", "PRO", "GLY"})
        self.assertIn("joint phi/psi/omega", manifest["sampling_unit"])

    def test_backbone_recomputes_sampled_dihedrals(self) -> None:
        groups = {
            key: [_observation(key, key[0], index)]
            for index, key in enumerate(
                ["A", "S", "V", "T", "N"], start=1
            )
        }
        manifest = {"manifest_canonical_sha256": "A" * 64}
        backbone, audit = generate_backbone("SAVTTVVN", 0, groups, manifest)
        self.assertEqual(len(backbone), 3 * len("SAVTTVVN"))
        self.assertLessEqual(
            audit["dihedral_convention_audit"][
                "maximum_angular_error_degrees"
            ],
            DIHEDRAL_TOLERANCE_DEGREES,
        )

    def test_backbone_is_deterministic_and_conformer_namespaced(self) -> None:
        groups = {
            key: [
                _observation(key, key[0], index),
                {
                    **_observation(key, key[0], index + 100),
                    "phi_degrees": -120.0,
                    "psi_degrees": 120.0,
                },
            ]
            for index, key in enumerate(
                ["A", "S", "V", "T", "N"], start=1
            )
        }
        manifest = {"manifest_canonical_sha256": "C" * 64}
        first = generate_backbone("SAVTTVVN", 0, groups, manifest)
        second = generate_backbone("SAVTTVVN", 0, groups, manifest)
        self.assertEqual(first[0], second[0])
        self.assertEqual(
            first[1]["backbone_coordinate_sha256"],
            second[1]["backbone_coordinate_sha256"],
        )
        self.assertNotEqual(
            first[1]["seed"],
            generate_backbone("SAVTTVVN", 1, groups, manifest)[1]["seed"],
        )

    def test_generator_version_is_new_namespace(self) -> None:
        self.assertEqual(
            GENERATOR_VERSION,
            "phase3-v2-train-only-residue-context-trans-v1",
        )
        self.assertNotIn("formal-v3", GENERATOR_VERSION)

    def test_nonstandard_sequence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_backbone(
                "AX",
                0,
                {},
                {"manifest_canonical_sha256": "D" * 64},
            )


if __name__ == "__main__":
    unittest.main()

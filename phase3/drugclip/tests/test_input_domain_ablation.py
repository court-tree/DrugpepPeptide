from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from phase3.drugclip.evaluate_input_domain_ablation import (
    _input_sha,
    arithmetic_mean_score,
    assert_1d_embeddings_identical,
    assert_finite,
    atom_cap_audit,
    call_structure_extractor,
    canonical_atom_sha,
    canonical_ncac_subset,
    model_state_sha,
    paired_bootstrap,
    select_exact_evidence,
    validate_candidate_bank,
    validate_named_hashes,
)
from phase3.drugclip.structure_qc import extract_bound_peptide_atoms


def _atom(serial: int, name: str, alt: str, residue: str, chain: str, residue_id: int,
          xyz: tuple[float, float, float], element: str) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {name:>4}{alt:1}{residue:>3} {chain:1}{residue_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          {element:>2}\n"
    )


def _peptide_pdb(chain: str = "B") -> str:
    return "".join([
        _atom(1, "N", "", "ALA", chain, 1, (0.0, 0.0, 0.0), "N"),
        _atom(2, "CA", "A", "ALA", chain, 1, (1.0, 0.0, 0.0), "C"),
        _atom(3, "CA", "B", "ALA", chain, 1, (99.0, 0.0, 0.0), "C"),
        _atom(4, "C", "", "ALA", chain, 1, (2.0, 0.0, 0.0), "C"),
        _atom(5, "O", "", "ALA", chain, 1, (3.0, 0.0, 0.0), "O"),
        _atom(6, "CB", "", "ALA", chain, 1, (1.0, 1.0, 0.0), "C"),
        _atom(7, "H", "", "ALA", chain, 1, (-1.0, 0.0, 0.0), "H"),
        "TER\nEND\n",
    ])


class InputDomainAblationTest(unittest.TestCase):
    def test_exact_evidence_join_requires_one_record(self) -> None:
        row = {"evidence_id": "e1"}
        self.assertIs(select_exact_evidence([row], "e1", "p1"), row)
        with self.assertRaisesRegex(ValueError, "not 1:1"):
            select_exact_evidence([], "e1", "p1")
        with self.assertRaisesRegex(ValueError, "not 1:1"):
            select_exact_evidence([row, dict(row)], "e1", "p1")

    def test_qbiolip_separate_peptide_file_uses_whole_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receptor = root / "rec.pdb"; peptide = root / "pep.pdb"
            receptor.write_text(_peptide_pdb("R"), encoding="ascii")
            peptide.write_text(_peptide_pdb("P"), encoding="ascii")
            row = {
                "source_database": "Q-BioLiP_PIII", "receptor_structure_file": str(receptor),
                "peptide_structure_file": str(peptide), "peptide_chain_id": "IGNORED",
                "peptide_sequence": "A",
            }
            result = extract_bound_peptide_atoms(row, root, root)
            self.assertEqual(result["observed_sequence"], "A")
            self.assertIsNone(result["peptide_chain"])

    def test_complex_pdb_chain_altloc_hydrogen_and_exact_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); complex_path = root / "complex.pdb"
            complex_path.write_text(_peptide_pdb("B"), encoding="ascii")
            row = {
                "source_database": "BioLiP2", "complex_structure_file": str(complex_path),
                "peptide_chain_id": "B", "peptide_sequence": "A",
            }
            result = extract_bound_peptide_atoms(row, root, root)
            all_atoms = result["all_heavy_atoms"]
            backbone = result["backbone_ncac_atoms"]
            self.assertEqual([atom["atom_name"] for atom in all_atoms], ["N", "CA", "C", "O", "CB"])
            self.assertEqual([atom["atom_name"] for atom in backbone], ["N", "CA", "C"])
            self.assertEqual(backbone[1]["x"], 1.0)
            expected = [atom for atom in all_atoms if atom["atom_name"] in {"N", "CA", "C"}]
            self.assertEqual(canonical_atom_sha(backbone), canonical_atom_sha(expected))
            self.assertTrue(result["source_backbone_order_canonical"])
            self.assertEqual(result["reordered_backbone_residue_count"], 0)

    def test_noncanonical_source_backbone_order_is_normalized_without_coordinate_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / "noncanonical.pdb"
            path.write_text("".join([
                _atom(1, "CA", "", "ALA", "B", 1, (1.0, 2.0, 3.0), "C"),
                _atom(2, "CB", "", "ALA", "B", 1, (4.0, 5.0, 6.0), "C"),
                _atom(3, "N", "", "ALA", "B", 1, (7.0, 8.0, 9.0), "N"),
                _atom(4, "C", "", "ALA", "B", 1, (10.0, 11.0, 12.0), "C"),
                _atom(5, "O", "", "ALA", "B", 1, (13.0, 14.0, 15.0), "O"),
                "TER\nEND\n",
            ]), encoding="ascii")
            row = {"source_database": "BioLiP2", "complex_structure_file": str(path),
                   "peptide_chain_id": "B", "peptide_sequence": "A"}
            result = extract_bound_peptide_atoms(row, root, root)
            backbone = result["backbone_ncac_atoms"]
            self.assertEqual([atom["atom_name"] for atom in backbone], ["N", "CA", "C"])
            self.assertEqual([(atom["x"], atom["y"], atom["z"]) for atom in backbone], [
                (7.0, 8.0, 9.0), (1.0, 2.0, 3.0), (10.0, 11.0, 12.0),
            ])
            self.assertFalse(result["source_backbone_order_canonical"])
            self.assertEqual(result["reordered_backbone_residue_count"], 1)
            self.assertEqual(result["reordered_backbone_residues"], [{
                "chain_id": "B", "residue_id": "1", "residue_name": "ALA",
                "source_filtered_order": ["CA", "N", "C"],
                "canonical_order": ["N", "CA", "C"],
            }])
            expected = canonical_ncac_subset(result["all_heavy_atoms"])
            self.assertEqual(canonical_atom_sha(backbone), canonical_atom_sha(expected))

    def test_sequence_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / "complex.pdb"
            path.write_text(_peptide_pdb("B"), encoding="ascii")
            row = {"source_database": "BioLiP2", "complex_structure_file": str(path),
                   "peptide_chain_id": "B", "peptide_sequence": "G"}
            with self.assertRaisesRegex(ValueError, "sequence_mismatch"):
                extract_bound_peptide_atoms(row, root, root)

    def test_extra_incomplete_residue_is_excluded_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / "complex.pdb"
            extra = "".join([
                _atom(8, "CA", "", "GLY", "B", 2, (8.0, 0.0, 0.0), "C"),
                _atom(9, "O", "", "GLY", "B", 2, (9.0, 0.0, 0.0), "O"),
                "TER\nEND\n",
            ])
            path.write_text(_peptide_pdb("B").replace("TER\nEND\n", "") + extra, encoding="ascii")
            row = {"source_database": "BioLiP2", "complex_structure_file": str(path),
                   "peptide_chain_id": "B", "peptide_sequence": "A"}
            result = extract_bound_peptide_atoms(row, root, root)
            self.assertEqual(result["observed_sequence"], "A")
            self.assertEqual(result["excluded_incomplete_residue_count"], 1)
            self.assertEqual(result["excluded_incomplete_residues"], [{
                "chain_id": "B", "residue_id": "2", "residue_name": "GLY",
                "present_atom_names": ["CA", "O"],
                "missing_backbone_atom_names": ["N", "C"],
            }])
            self.assertEqual([atom["residue_id"] for atom in result["all_heavy_atoms"]], ["B:1"] * 5)
            self.assertEqual([atom["atom_name"] for atom in result["backbone_ncac_atoms"]], ["N", "CA", "C"])

    def test_expected_residue_with_incomplete_backbone_is_sequence_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / "incomplete.pdb"
            path.write_text(
                _atom(1, "CA", "", "ALA", "B", 1, (1.0, 0.0, 0.0), "C") + "TER\nEND\n",
                encoding="ascii",
            )
            row = {"source_database": "BioLiP2", "complex_structure_file": str(path),
                   "peptide_chain_id": "B", "peptide_sequence": "A"}
            with self.assertRaisesRegex(ValueError, "peptide_coordinate_sequence_mismatch"):
                extract_bound_peptide_atoms(row, root, root)

    def test_structure_exception_has_pair_and_evidence_context(self) -> None:
        def fail() -> None:
            raise ValueError("raw_failure")

        with self.assertRaisesRegex(
            RuntimeError,
            "interface_pair_id=pair-7;evidence_id=evidence-9;original_exception=ValueError: raw_failure",
        ):
            call_structure_extractor(fail, "pair-7", "evidence-9")

        def fail_with_unlisted_exception() -> None:
            raise IndexError("unexpected_parser_failure")

        with self.assertRaisesRegex(
            RuntimeError,
            "interface_pair_id=pair-8;evidence_id=evidence-10;"
            "original_exception=IndexError: unexpected_parser_failure",
        ):
            call_structure_extractor(
                fail_with_unlisted_exception, "pair-8", "evidence-10"
            )

    def test_max_peptide_atoms_audit(self) -> None:
        atoms = [{}] * 193
        self.assertEqual(atom_cap_audit(atoms, 192), {
            "before": 193, "after": 192, "touched_cap": True, "truncated": True,
        })
        self.assertTrue(atom_cap_audit([{}] * 192, 192)["touched_cap"])
        self.assertFalse(atom_cap_audit([{}] * 192, 192)["truncated"])

    def test_receptor_and_one_d_sha_are_variant_invariant(self) -> None:
        atom = {"atom_name": "CA", "element": "C", "residue_id": "A:1", "residue_name": "ALA",
                "x": 1.0, "y": 2.0, "z": 3.0}
        base = {"receptor_atoms": [atom], "receptor_patch_sequence": "A", "peptide_sequence": "G"}
        changed_peptide = {**base, "peptide_atoms": [dict(atom, x=9.0)]}
        self.assertEqual(_input_sha(base), _input_sha(changed_peptide))

    def test_1d_embedding_identity_is_exact(self) -> None:
        tensor = torch.tensor([[1.0, 2.0]])
        assert_1d_embeddings_identical(tensor, tensor.clone())
        with self.assertRaisesRegex(RuntimeError, "1D embedding"):
            assert_1d_embeddings_identical(tensor, tensor + 1e-7)

    def test_candidate_bank_drift_is_rejected(self) -> None:
        from phase3.drugclip.evaluate_input_domain_ablation import _sequence_hash
        expected = _sequence_hash(["a", "b"])
        with patch("phase3.drugclip.evaluate_input_domain_ablation.EXPECTED", {"r2p_candidate_ids_sha256": expected}):
            validate_candidate_bank(["a", "b"], "r2p")
            with self.assertRaisesRegex(RuntimeError, "candidate bank drift"):
                validate_candidate_bank(["b", "a"], "r2p")

    def test_cmean10_is_arithmetic_mean_of_scores(self) -> None:
        matrices = [torch.full((2, 2), float(index)) for index in range(10)]
        self.assertTrue(torch.equal(arithmetic_mean_score(matrices), torch.full((2, 2), 4.5)))
        with self.assertRaisesRegex(ValueError, "exactly ten"):
            arithmetic_mean_score(matrices[:9])

    def test_checkpoint_and_manifest_hash_rejection(self) -> None:
        validate_named_hashes({"checkpoint": "A", "manifest": "B"}, {"checkpoint": "A", "manifest": "B"})
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch:manifest"):
            validate_named_hashes({"checkpoint": "A", "manifest": "X"}, {"checkpoint": "A", "manifest": "B"})

    def test_nonfinite_values_are_rejected(self) -> None:
        with self.assertRaises(FloatingPointError):
            assert_finite(torch.tensor([math.nan]))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            arithmetic_mean_score([torch.tensor([[math.inf]]) for _ in range(10)])

    def test_model_state_read_only_hash(self) -> None:
        model = torch.nn.Linear(2, 2)
        before = model_state_sha(model)
        with torch.inference_mode():
            model(torch.ones(1, 2))
        self.assertEqual(before, model_state_sha(model))
        with torch.no_grad():
            model.weight.add_(1.0)
        self.assertNotEqual(before, model_state_sha(model))

    def test_paired_bootstrap_is_deterministic_and_paired(self) -> None:
        indices = np.random.default_rng(7).integers(0, 4, size=(200, 4), dtype=np.int32)
        first = paired_bootstrap([1, 2, 3, 4], [2, 3, 4, 5], indices, 7)
        second = paired_bootstrap([1, 2, 3, 4], [2, 3, 4, 5], indices, 7)
        self.assertEqual(first, second)
        self.assertLess(first["later_minus_earlier"]["mean_rank"]["point_estimate"], 0)


if __name__ == "__main__":
    unittest.main()

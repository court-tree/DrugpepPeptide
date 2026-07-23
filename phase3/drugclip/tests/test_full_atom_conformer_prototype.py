from __future__ import annotations

import inspect
import math
import tempfile
import unittest
from pathlib import Path

from phase2.pepclip.data import ATOM_NAME_TO_ID, ELEMENT_TO_ID, RESIDUE_NAME_TO_ID
from phase3.drugclip.full_atom_conformer_prototype import (
    AA1_TO_3,
    CHEMISTRY_CLASS,
    REQUIRED_HEAVY_ATOMS,
    UnsupportedPeptideChemistry,
    _base_molecule,
    _validate_topology,
    conformer_atoms,
    generate_full_heavy_conformers,
)
from phase3.drugclip.validate_full_atom_conformer_prototype import (
    audit_one_chemistry,
    cpu_forward_smoke,
    select_generation_panel,
    validate_payload,
    verify_determinism,
)


def _pdb_atom(
    serial: int,
    atom_name: str,
    residue_name: str,
    chain: str,
    residue_id: int,
    xyz: tuple[float, float, float],
    element: str,
) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {atom_name:>4} {residue_name:>3} {chain:1}"
        f"{residue_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.0:6.2f}{20.0:6.2f}          {element:>2}\n"
    )


def _ag_complex(receptor_x: float = 20.0) -> str:
    rows = [
        _pdb_atom(1, "N", "ALA", "P", 1, (0.0, 0.0, 0.0), "N"),
        _pdb_atom(2, "CA", "ALA", "P", 1, (1.45, 0.0, 0.0), "C"),
        _pdb_atom(3, "C", "ALA", "P", 1, (2.50, 1.0, 0.0), "C"),
        _pdb_atom(4, "O", "ALA", "P", 1, (2.30, 2.2, 0.0), "O"),
        _pdb_atom(5, "CB", "ALA", "P", 1, (1.50, -1.5, 0.0), "C"),
        _pdb_atom(6, "N", "GLY", "P", 2, (3.83, 0.9, 0.0), "N"),
        _pdb_atom(7, "CA", "GLY", "P", 2, (4.70, 2.0, 0.0), "C"),
        _pdb_atom(8, "C", "GLY", "P", 2, (6.10, 1.5, 0.0), "C"),
        _pdb_atom(9, "O", "GLY", "P", 2, (6.50, 0.3, 0.0), "O"),
        _pdb_atom(10, "CA", "ALA", "R", 1, (receptor_x, 0.0, 0.0), "C"),
    ]
    return "".join(rows) + "TER\nEND\n"


class FullAtomConformerPrototypeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = generate_full_heavy_conformers("AG", num_conformers=10)

    def test_standard_linear_peptide_generates_ten_conformers(self) -> None:
        result = validate_payload(self.payload)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["conformer_count"], 10)

    def test_complete_heavy_atoms_include_backbone_o_sidechain_and_oxt(self) -> None:
        by_residue = {}
        for atom in self.payload["atom_identity"]:
            by_residue.setdefault(atom["residue_index"], set()).add(atom["atom_name"])
        self.assertTrue(REQUIRED_HEAVY_ATOMS["ALA"] <= by_residue[1])
        self.assertTrue(REQUIRED_HEAVY_ATOMS["GLY"] | {"OXT"} <= by_residue[2])

    def test_atom_identity_order_is_shared_by_all_conformers(self) -> None:
        expected = self.payload["atom_identity"]
        for index in range(10):
            atoms = conformer_atoms(self.payload, index)
            observed = [
                {key: atom[key] for key in expected[0]}
                for atom in atoms
            ]
            self.assertEqual(observed, expected)

    def test_coordinates_are_finite_and_geometry_passes(self) -> None:
        for conformer in self.payload["conformers"]:
            self.assertTrue(all(
                math.isfinite(value)
                for xyz in conformer["coordinates"]
                for value in xyz
            ))
            self.assertEqual(conformer["mmff_status"], 0)
            self.assertEqual(conformer["geometry_audit"]["status"], "PASS")
            self.assertTrue(conformer["geometry_audit"]["coordinate_chirality_match"])
            self.assertGreater(
                conformer["geometry_audit"]["minimum_heavy_bond_length_angstrom"],
                0.90,
            )
            self.assertGreaterEqual(conformer["embedding_seconds"], 0.0)
            self.assertGreaterEqual(conformer["mmff_seconds"], 0.0)
            self.assertTrue(conformer["attempt_records"])
            self.assertIsNone(
                conformer["attempt_records"][-1]["rejection_reason"]
            )

    def test_ten_conformers_are_not_identical(self) -> None:
        hashes = {
            conformer["coordinate_sha256"]
            for conformer in self.payload["conformers"]
        }
        self.assertEqual(len(hashes), 10)

    def test_fixed_seed_is_exactly_deterministic(self) -> None:
        self.assertEqual(
            verify_determinism("AG", num_conformers=2, base_seed=12345)["status"],
            "PASS",
        )

    def test_all_standard_residue_topologies_are_complete_and_in_vocab(self) -> None:
        sequence = "".join(AA1_TO_3)
        identities = _validate_topology(_base_molecule(sequence), sequence)
        self.assertTrue(identities)
        for atom in identities:
            self.assertIn(atom["atom_name"], ATOM_NAME_TO_ID)
            self.assertIn(atom["element"], ELEMENT_TO_ID)
            self.assertIn(atom["residue_name"], RESIDUE_NAME_TO_ID)

    def test_multiple_cysteines_are_explicitly_rejected(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedPeptideChemistry,
            "multiple_cysteines_require_explicit_disulfide_state",
        ):
            generate_full_heavy_conformers("ACDC", num_conformers=1)

    def test_nonstandard_or_ambiguous_residues_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedPeptideChemistry,
            "nonstandard_or_ambiguous_residues",
        ):
            generate_full_heavy_conformers("ABZ", num_conformers=1)

    def test_cyclic_or_modified_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedPeptideChemistry,
            "unsupported_chemistry_class",
        ):
            generate_full_heavy_conformers(
                "AG",
                num_conformers=1,
                chemistry_class="cyclic_or_modified",
            )

    def test_api_cannot_accept_target_bound_context(self) -> None:
        parameters = set(inspect.signature(generate_full_heavy_conformers).parameters)
        self.assertEqual(
            parameters,
            {"sequence", "num_conformers", "base_seed", "chemistry_class"},
        )
        self.assertFalse(self.payload["dependency_contract"]["target_bound_inputs_used"])

    def test_current_pepclip_egnn_cpu_forward(self) -> None:
        result = cpu_forward_smoke(self.payload)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["embedding_shape"], [1, 16])

    def test_structure_chemistry_audit_requires_connectivity_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "complex.pdb"
            path.write_text(_ag_complex(), encoding="ascii")
            row = audit_one_chemistry(
                {
                    "interface_pair_id": "pair-1",
                    "peptide_sequence": "AG",
                },
                {
                    "evidence_id": "evidence-1",
                    "source_database": "BioLiP2",
                    "structure_type": "pdb",
                    "peptide_chain": "P",
                },
                {
                    "evidence_id": "evidence-1",
                    "source_database": "BioLiP2",
                    "complex_structure_file": str(path),
                },
                Path(temporary),
                Path(temporary),
            )
        self.assertEqual(
            row["chemistry_classification"],
            "ordinary_linear_standard",
        )
        self.assertTrue(row["continuous_adjacent_peptide_bonds"])
        self.assertTrue(row["terminal_state_determined"])
        self.assertFalse(row["peptide_receptor_covalent_connection_detected"])

    def test_structure_chemistry_audit_detects_receptor_covalent_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "complex.pdb"
            # Receptor CA is placed 1.4 A from the peptide N.
            path.write_text(_ag_complex(receptor_x=1.4), encoding="ascii")
            row = audit_one_chemistry(
                {
                    "interface_pair_id": "pair-2",
                    "peptide_sequence": "AG",
                },
                {
                    "evidence_id": "evidence-2",
                    "source_database": "BioLiP2",
                    "structure_type": "pdb",
                    "peptide_chain": "P",
                },
                {
                    "evidence_id": "evidence-2",
                    "source_database": "BioLiP2",
                    "complex_structure_file": str(path),
                },
                Path(temporary),
                Path(temporary),
            )
        self.assertEqual(row["chemistry_classification"], "receptor_covalent")
        self.assertTrue(row["peptide_receptor_covalent_connection_detected"])

    def test_generation_panel_is_fixed512_derived_and_stratified(self) -> None:
        sequences = [
            "ACDEFGHI",
            "KLMNPQRS",
            "TVWYAAAA",
            "ACDEFGHIK",
            "KLMNPQRSTV",
            "TVWYACDEFGH",
            "ACDEFGHIKLMN",
            "KLMNPQRSTVWYA",
            "TVWYACDEFGHIKL",
            "ACDEFGHIKLMNPQR",
            "KLMNPQRSTVWYACDEF",
            "TVWYACDEFGHIKLMNPQR",
        ]
        rows = []
        for index, sequence in enumerate(sequences):
            rows.append({
                "peptide_sequence": sequence,
                "sequence_length": len(sequence),
                "theoretical_heavy_atom_count": 50 + index * 10,
                "interface_pair_id": f"pair-{index:02d}",
                "evidence_id": f"evidence-{index:02d}",
                "cys_count": sequence.count("C"),
                "chemistry_classification": "ordinary_linear_standard",
            })
        panel = select_generation_panel(rows, minimum_size=8)
        self.assertGreaterEqual(len(panel), 8)
        reasons = {
            reason for row in panel for reason in row["selection_reasons"]
        }
        self.assertIn("fixed512_safe_shortest", reasons)
        self.assertIn("fixed512_safe_longest", reasons)
        self.assertIn("fixed512_safe_maximum_heavy_atom_count", reasons)
        self.assertIn("fixed512_safe_single_cysteine", reasons)


if __name__ == "__main__":
    unittest.main()

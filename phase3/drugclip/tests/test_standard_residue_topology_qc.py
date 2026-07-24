import json
import math
from pathlib import Path
import unittest

from phase3.drugclip.full_atom_conformer_prototype import (
    AA1_TO_3,
    ConformerGenerationError,
    REQUIRED_HEAVY_ATOMS,
    _base_molecule,
    _geometry_audit,
    _validate_topology,
)
from phase3.drugclip.replay_safe265_topology_qc import (
    FAILED_SEQUENCE,
    _coordinates_from_saved_faspr,
    _rdkit_residue_contract,
)
from phase3.drugclip.standard_residue_topology import (
    STANDARD_RESIDUE_ATOMS,
    canonical_chirality_audit,
    canonical_peptide_graph,
    residue_bonds,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SAVED_ATTEMPT_00 = (
    REPO_ROOT
    / "phase3" / "runs" / "drugclip"
    / "v3_safe265_full_atom_conformer_coverage_prototype_v1"
    / "work" / "5C90EF6ECDE854A7A9C1"
    / "slot_00" / "attempt_00" / "conformer_00.faspr.pdb"
)
FIVE_SEQUENCE_SUMMARY = (
    REPO_ROOT
    / "phase3" / "runs" / "drugclip"
    / "v3_train_only_torsion_rejection_sampling_prototype_v1"
    / "train_only_torsion_panel_summary.json"
)


def _saved_attempt_coordinates():
    if not SAVED_ATTEMPT_00.is_file():
        raise unittest.SkipTest("saved safe265 attempt_00 authority unavailable")
    molecule = _base_molecule(FAILED_SEQUENCE)
    identities = _validate_topology(molecule, FAILED_SEQUENCE)
    coordinates = _coordinates_from_saved_faspr(
        SAVED_ATTEMPT_00, identities, FAILED_SEQUENCE
    )
    return molecule, identities, coordinates


def _reflect_across_plane(point, origin, plane_first, plane_second):
    first = [plane_first[i] - origin[i] for i in range(3)]
    second = [plane_second[i] - origin[i] for i in range(3)]
    normal = [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
    length = math.sqrt(sum(value * value for value in normal))
    unit = [value / length for value in normal]
    offset = [point[i] - origin[i] for i in range(3)]
    distance = sum(offset[i] * unit[i] for i in range(3))
    return [point[i] - 2.0 * distance * unit[i] for i in range(3)]


def _rotate_around_axis(point, axis_start, axis_end, radians):
    unit = [
        axis_end[i] - axis_start[i] for i in range(3)
    ]
    length = math.sqrt(sum(value * value for value in unit))
    unit = [value / length for value in unit]
    vector = [point[i] - axis_start[i] for i in range(3)]
    cross = [
        unit[1] * vector[2] - unit[2] * vector[1],
        unit[2] * vector[0] - unit[0] * vector[2],
        unit[0] * vector[1] - unit[1] * vector[0],
    ]
    dot = sum(unit[i] * vector[i] for i in range(3))
    return [
        axis_start[i]
        + vector[i] * math.cos(radians)
        + cross[i] * math.sin(radians)
        + unit[i] * dot * (1.0 - math.cos(radians))
        for i in range(3)
    ]


class StandardResidueTopologyQCTests(unittest.TestCase):
    def test_standard_ile_bonds_use_cg1_cd1(self):
        bonds = residue_bonds("ILE")
        self.assertIn(("CB", "CG1"), bonds)
        self.assertIn(("CB", "CG2"), bonds)
        self.assertIn(("CD1", "CG1"), bonds)
        self.assertNotIn(("CD1", "CG2"), bonds)
        rdkit_bonds = {
            tuple(row) for row in _rdkit_residue_contract("I")["bonds"]
        }
        self.assertIn(("CD1", "CG2"), rdkit_bonds)
        self.assertNotIn(("CD1", "CG1"), rdkit_bonds)

    def test_all_twenty_templates_are_atom_complete_and_connected(self):
        self.assertEqual(set(STANDARD_RESIDUE_ATOMS), set(AA1_TO_3.values()))
        for residue_name, atoms in STANDARD_RESIDUE_ATOMS.items():
            self.assertEqual(atoms, REQUIRED_HEAVY_ATOMS[residue_name])
            bonds = residue_bonds(residue_name)
            bonded_atoms = {atom for bond in bonds for atom in bond}
            self.assertEqual(bonded_atoms, atoms, residue_name)
            adjacency = {atom: set() for atom in atoms}
            for first, second in bonds:
                adjacency[first].add(second)
                adjacency[second].add(first)
            visited = set()
            pending = ["N"]
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                pending.extend(adjacency[current] - visited)
            self.assertEqual(visited, atoms, residue_name)

    def test_saved_attempt_00_no_longer_has_false_ile_bond(self):
        molecule, identities, coordinates = _saved_attempt_coordinates()
        result = _geometry_audit(molecule, coordinates, identities)
        self.assertEqual(result["status"], "PASS")
        self.assertLess(
            result["maximum_heavy_bond_length_angstrom"], 2.10
        )

    def test_true_ile_cg1_cd1_stretch_is_rejected(self):
        molecule, identities, coordinates = _saved_attempt_coordinates()
        graph = canonical_peptide_graph(identities)
        lookup = graph["identity_lookup"]
        cg1 = lookup[(6, "CG1")]
        cd1 = lookup[(6, "CD1")]
        direction = [
            coordinates[cd1][axis] - coordinates[cg1][axis]
            for axis in range(3)
        ]
        norm = math.sqrt(sum(value * value for value in direction))
        mutated = [list(row) for row in coordinates]
        mutated[cd1] = [
            coordinates[cg1][axis] + 3.0 * direction[axis] / norm
            for axis in range(3)
        ]
        with self.assertRaisesRegex(
            ConformerGenerationError, "illegal_heavy_bond_length_range"
        ):
            _geometry_audit(molecule, mutated, identities)

    def test_artificial_true_nonlocal_clash_is_rejected(self):
        molecule, identities, coordinates = _saved_attempt_coordinates()
        graph = canonical_peptide_graph(identities)
        lookup = graph["identity_lookup"]
        axis_start = coordinates[lookup[(3, "C")]]
        axis_end = coordinates[lookup[(4, "N")]]
        mutated = [list(row) for row in coordinates]
        for atom_index, identity in enumerate(identities):
            if int(identity["residue_index"]) < 4:
                continue
            if (
                int(identity["residue_index"]) == 4
                and identity["atom_name"] == "N"
            ):
                continue
            mutated[atom_index] = _rotate_around_axis(
                coordinates[atom_index],
                axis_start,
                axis_end,
                math.radians(153.0),
            )
        with self.assertRaisesRegex(
            ConformerGenerationError, "nonlocal_heavy_atom_clash"
        ):
            _geometry_audit(molecule, mutated, identities)

    def test_ile_cb_chirality_flip_is_rejected(self):
        _, identities, coordinates = _saved_attempt_coordinates()
        graph = canonical_peptide_graph(identities)
        lookup = graph["identity_lookup"]
        mutated = [list(row) for row in coordinates]
        cb = lookup[(6, "CB")]
        cg2 = lookup[(6, "CG2")]
        mutated[cg2] = _reflect_across_plane(
            coordinates[cg2],
            coordinates[cb],
            coordinates[lookup[(6, "CA")]],
            coordinates[lookup[(6, "CG1")]],
        )
        with self.assertRaisesRegex(ValueError, "canonical_chirality_mismatch"):
            canonical_chirality_audit(identities, mutated, graph)

    def test_thr_cb_chirality_flip_is_rejected(self):
        _, identities, coordinates = _saved_attempt_coordinates()
        graph = canonical_peptide_graph(identities)
        lookup = graph["identity_lookup"]
        mutated = [list(row) for row in coordinates]
        cb = lookup[(9, "CB")]
        cg2 = lookup[(9, "CG2")]
        mutated[cg2] = _reflect_across_plane(
            coordinates[cg2],
            coordinates[cb],
            coordinates[lookup[(9, "CA")]],
            coordinates[lookup[(9, "OG1")]],
        )
        with self.assertRaisesRegex(ValueError, "canonical_chirality_mismatch"):
            canonical_chirality_audit(identities, mutated, graph)

    def test_saved_five_sequence_accepts_do_not_regress(self):
        if not FIVE_SEQUENCE_SUMMARY.is_file():
            raise unittest.SkipTest("saved five-sequence authority unavailable")
        summary = json.loads(
            FIVE_SEQUENCE_SUMMARY.read_text(encoding="utf-8")
        )
        checked = 0
        for panel_row in summary["panel"]:
            payload_path = Path(panel_row["runs"][0]["payload_path"])
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            molecule = _base_molecule(payload["peptide_sequence"])
            identities = payload["atom_identity"]
            for conformer in payload["conformers"]:
                result = _geometry_audit(
                    molecule, conformer["coordinates"], identities
                )
                self.assertEqual(result["status"], "PASS")
                checked += 1
        self.assertEqual(checked, 50)


if __name__ == "__main__":
    unittest.main()

"""Canonical standard-PDB heavy-atom topology for peptide geometry QC.

The templates use standard PDB/FASPR atom names.  They are deliberately
independent of RDKit ``MolFromSequence`` bond perception because RDKit 2025.03
labels isoleucine atoms as standard PDB names while connecting CD1 to CG2.
"""

from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Any, Iterable


BACKBONE_ATOMS = {"N", "CA", "C", "O"}
BACKBONE_BONDS = {("N", "CA"), ("CA", "C"), ("C", "O")}
SIDECHAIN_BONDS: dict[str, set[tuple[str, str]]] = {
    "ALA": {("CA", "CB")},
    "ARG": {
        ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"),
        ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2"),
    },
    "ASN": {("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")},
    "ASP": {("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")},
    "CYS": {("CA", "CB"), ("CB", "SG")},
    "GLN": {
        ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"),
        ("CD", "NE2"),
    },
    "GLU": {
        ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"),
        ("CD", "OE2"),
    },
    "GLY": set(),
    "HIS": {
        ("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"),
        ("ND1", "CE1"), ("CD2", "NE2"), ("CE1", "NE2"),
    },
    "ILE": {
        ("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1"),
    },
    "LEU": {("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")},
    "LYS": {
        ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"),
        ("CE", "NZ"),
    },
    "MET": {("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")},
    "PHE": {
        ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
        ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"),
    },
    "PRO": {("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")},
    "SER": {("CA", "CB"), ("CB", "OG")},
    "THR": {("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")},
    "TRP": {
        ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
        ("CD1", "NE1"), ("NE1", "CE2"), ("CD2", "CE2"),
        ("CD2", "CE3"), ("CE2", "CZ2"), ("CE3", "CZ3"),
        ("CZ2", "CH2"), ("CZ3", "CH2"),
    },
    "TYR": {
        ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
        ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"),
        ("CZ", "OH"),
    },
    "VAL": {("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")},
}
STANDARD_RESIDUE_ATOMS = {
    residue: BACKBONE_ATOMS | {
        atom for bond in bonds for atom in bond
    }
    for residue, bonds in SIDECHAIN_BONDS.items()
}

BOND_LENGTH_MIN_ANGSTROM = 0.90
BOND_LENGTH_MAX_ANGSTROM = 2.10
BOND_ANGLE_MIN_DEGREES = 60.0
BOND_ANGLE_MAX_DEGREES = 180.0
NONLOCAL_CLASH_MIN_ANGSTROM = 0.75


def _normalized_bond(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((str(first), str(second))))


def residue_bonds(residue_name: str) -> set[tuple[str, str]]:
    name = str(residue_name).upper()
    if name not in SIDECHAIN_BONDS:
        raise ValueError(f"unsupported_standard_residue:{name}")
    return {
        _normalized_bond(first, second)
        for first, second in BACKBONE_BONDS | SIDECHAIN_BONDS[name]
    }


def canonical_peptide_graph(
    identities: list[dict[str, Any]],
) -> dict[str, Any]:
    by_key: dict[tuple[int, str], int] = {}
    residue_names: dict[int, str] = {}
    for position, identity in enumerate(identities):
        atom_index = int(identity["atom_index"])
        if atom_index != position:
            raise ValueError(
                f"noncanonical_atom_index:{position}:{atom_index}"
            )
        residue_index = int(identity["residue_index"])
        atom_name = str(identity["atom_name"]).upper()
        residue_name = str(identity["residue_name"]).upper()
        key = (residue_index, atom_name)
        if key in by_key:
            raise ValueError(f"duplicate_atom_identity:{key}")
        by_key[key] = atom_index
        if residue_index in residue_names and residue_names[residue_index] != residue_name:
            raise ValueError(f"inconsistent_residue_name:{residue_index}")
        residue_names[residue_index] = residue_name
    residue_indices = sorted(residue_names)
    if residue_indices != list(range(1, len(residue_indices) + 1)):
        raise ValueError("canonical_residue_index_contract_mismatch")

    bond_records: list[dict[str, Any]] = []
    for residue_index in residue_indices:
        residue_name = residue_names[residue_index]
        observed_names = {
            atom_name
            for (index, atom_name) in by_key
            if index == residue_index
        }
        expected_names = set(STANDARD_RESIDUE_ATOMS[residue_name])
        if residue_index == residue_indices[-1]:
            expected_names.add("OXT")
        if observed_names != expected_names:
            raise ValueError(
                f"canonical_residue_atom_set_mismatch:{residue_index}:"
                f"{residue_name}:missing={sorted(expected_names-observed_names)}:"
                f"extra={sorted(observed_names-expected_names)}"
            )
        for first, second in sorted(residue_bonds(residue_name)):
            bond_records.append({
                "first_index": by_key[(residue_index, first)],
                "second_index": by_key[(residue_index, second)],
                "first": {
                    "residue_index": residue_index,
                    "residue_name": residue_name,
                    "atom_name": first,
                },
                "second": {
                    "residue_index": residue_index,
                    "residue_name": residue_name,
                    "atom_name": second,
                },
                "kind": "intra_residue",
            })
        if residue_index == residue_indices[-1]:
            bond_records.append({
                "first_index": by_key[(residue_index, "C")],
                "second_index": by_key[(residue_index, "OXT")],
                "first": {
                    "residue_index": residue_index,
                    "residue_name": residue_name,
                    "atom_name": "C",
                },
                "second": {
                    "residue_index": residue_index,
                    "residue_name": residue_name,
                    "atom_name": "OXT",
                },
                "kind": "terminal_oxt",
            })
        if residue_index < residue_indices[-1]:
            next_name = residue_names[residue_index + 1]
            bond_records.append({
                "first_index": by_key[(residue_index, "C")],
                "second_index": by_key[(residue_index + 1, "N")],
                "first": {
                    "residue_index": residue_index,
                    "residue_name": residue_name,
                    "atom_name": "C",
                },
                "second": {
                    "residue_index": residue_index + 1,
                    "residue_name": next_name,
                    "atom_name": "N",
                },
                "kind": "peptide",
            })
    adjacency: dict[int, set[int]] = defaultdict(set)
    for row in bond_records:
        first = int(row["first_index"])
        second = int(row["second_index"])
        adjacency[first].add(second)
        adjacency[second].add(first)
    return {
        "bonds": bond_records,
        "adjacency": {index: set(adjacency[index]) for index in range(len(identities))},
        "identity_lookup": by_key,
        "residue_names": residue_names,
    }


def _signed_volume(
    coordinates: list[list[float]],
    center: int,
    first: int,
    second: int,
    third: int,
) -> float:
    vectors = [
        [
            coordinates[index][axis] - coordinates[center][axis]
            for axis in range(3)
        ]
        for index in (first, second, third)
    ]
    cross = [
        vectors[1][1] * vectors[2][2] - vectors[1][2] * vectors[2][1],
        vectors[1][2] * vectors[2][0] - vectors[1][0] * vectors[2][2],
        vectors[1][0] * vectors[2][1] - vectors[1][1] * vectors[2][0],
    ]
    return sum(vectors[0][axis] * cross[axis] for axis in range(3))


def canonical_chirality_audit(
    identities: list[dict[str, Any]],
    coordinates: list[list[float]],
    graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph = graph or canonical_peptide_graph(identities)
    lookup = graph["identity_lookup"]
    rows = []
    for residue_index, residue_name in graph["residue_names"].items():
        if residue_name != "GLY":
            volume = _signed_volume(
                coordinates,
                lookup[(residue_index, "CA")],
                lookup[(residue_index, "N")],
                lookup[(residue_index, "C")],
                lookup[(residue_index, "CB")],
            )
            rows.append({
                "center": "CA",
                "residue_index": residue_index,
                "residue_name": residue_name,
                "signed_volume": volume,
                "expected_sign": "positive_for_L_standard_PDB_naming",
                "status": "PASS" if volume > 0.0 else "FAIL",
            })
        if residue_name == "ILE":
            volume = _signed_volume(
                coordinates,
                lookup[(residue_index, "CB")],
                lookup[(residue_index, "CA")],
                lookup[(residue_index, "CG1")],
                lookup[(residue_index, "CG2")],
            )
            rows.append({
                "center": "CB",
                "residue_index": residue_index,
                "residue_name": residue_name,
                "signed_volume": volume,
                "expected_sign": "positive_for_L_isoleucine_standard_PDB_naming",
                "status": "PASS" if volume > 0.0 else "FAIL",
            })
        if residue_name == "THR":
            volume = _signed_volume(
                coordinates,
                lookup[(residue_index, "CB")],
                lookup[(residue_index, "CA")],
                lookup[(residue_index, "OG1")],
                lookup[(residue_index, "CG2")],
            )
            rows.append({
                "center": "CB",
                "residue_index": residue_index,
                "residue_name": residue_name,
                "signed_volume": volume,
                "expected_sign": "positive_for_L_threonine_standard_PDB_naming",
                "status": "PASS" if volume > 0.0 else "FAIL",
            })
    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        first = failures[0]
        raise ValueError(
            f"canonical_chirality_mismatch:{first['residue_index']}:"
            f"{first['residue_name']}:{first['center']}:"
            f"{first['signed_volume']:.6f}"
        )
    return {
        "status": "PASS",
        "center_count": len(rows),
        "ca_center_count": sum(row["center"] == "CA" for row in rows),
        "ile_cb_center_count": sum(
            row["center"] == "CB" and row["residue_name"] == "ILE"
            for row in rows
        ),
        "thr_cb_center_count": sum(
            row["center"] == "CB" and row["residue_name"] == "THR"
            for row in rows
        ),
        "minimum_signed_volume": min(
            (float(row["signed_volume"]) for row in rows), default=None
        ),
        "rows": rows,
    }


def _graph_distances(adjacency: dict[int, set[int]], start: int) -> dict[int, int]:
    distances = {start: 0}
    pending = deque([start])
    while pending:
        current = pending.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                pending.append(neighbor)
    return distances


def canonical_geometry_audit(
    identities: list[dict[str, Any]],
    coordinates: list[list[float]],
) -> dict[str, Any]:
    if len(identities) != len(coordinates):
        raise ValueError("canonical_coordinate_identity_count_mismatch")
    if any(not math.isfinite(value) for xyz in coordinates for value in xyz):
        raise ValueError("nonfinite_coordinates")
    graph = canonical_peptide_graph(identities)

    def distance(first: int, second: int) -> float:
        return math.dist(coordinates[first], coordinates[second])

    bond_rows = []
    for bond in graph["bonds"]:
        row = {**bond, "length_angstrom": distance(
            int(bond["first_index"]), int(bond["second_index"])
        )}
        bond_rows.append(row)
    bond_lengths = [float(row["length_angstrom"]) for row in bond_rows]
    if (
        not bond_lengths
        or min(bond_lengths) < BOND_LENGTH_MIN_ANGSTROM
        or max(bond_lengths) > BOND_LENGTH_MAX_ANGSTROM
    ):
        longest = max(bond_rows, key=lambda row: row["length_angstrom"])
        raise ValueError(
            f"illegal_heavy_bond_length_range:{min(bond_lengths):.6f}:"
            f"{max(bond_lengths):.6f}:longest="
            f"{longest['first']['residue_index']}:{longest['first']['atom_name']}-"
            f"{longest['second']['residue_index']}:{longest['second']['atom_name']}"
        )

    angles = []
    for center, neighbors in graph["adjacency"].items():
        ordered = sorted(neighbors)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1:]:
                first = [
                    coordinates[left][axis] - coordinates[center][axis]
                    for axis in range(3)
                ]
                second = [
                    coordinates[right][axis] - coordinates[center][axis]
                    for axis in range(3)
                ]
                denominator = math.sqrt(sum(value * value for value in first))
                denominator *= math.sqrt(sum(value * value for value in second))
                if denominator == 0.0:
                    raise ValueError("zero_length_bond_in_angle")
                cosine = max(
                    -1.0,
                    min(
                        1.0,
                        sum(a * b for a, b in zip(first, second)) / denominator,
                    ),
                )
                angles.append(math.degrees(math.acos(cosine)))
    if (
        not angles
        or min(angles) < BOND_ANGLE_MIN_DEGREES
        or max(angles) > BOND_ANGLE_MAX_DEGREES
    ):
        raise ValueError(
            f"illegal_heavy_bond_angle_range:{min(angles):.6f}:"
            f"{max(angles):.6f}"
        )

    minimum_nonlocal = math.inf
    minimum_pair = None
    for first in range(len(identities)):
        distances = _graph_distances(graph["adjacency"], first)
        for second in range(first + 1, len(identities)):
            if distances.get(second, math.inf) <= 2:
                continue
            value = distance(first, second)
            if value < minimum_nonlocal:
                minimum_nonlocal = value
                minimum_pair = (first, second)
    if minimum_nonlocal < NONLOCAL_CLASH_MIN_ANGSTROM:
        first, second = minimum_pair
        raise ValueError(
            f"nonlocal_heavy_atom_clash:{minimum_nonlocal:.6f}:"
            f"{identities[first]['residue_index']}:{identities[first]['atom_name']}-"
            f"{identities[second]['residue_index']}:{identities[second]['atom_name']}"
        )
    chirality = canonical_chirality_audit(identities, coordinates, graph)
    longest = max(bond_rows, key=lambda row: row["length_angstrom"])
    return {
        "status": "PASS",
        "topology_contract": "standard-pdb-heavy-atom-bond-templates-v1",
        "minimum_heavy_bond_length_angstrom": min(bond_lengths),
        "maximum_heavy_bond_length_angstrom": max(bond_lengths),
        "maximum_heavy_bond": {
            "first": longest["first"],
            "second": longest["second"],
            "length_angstrom": longest["length_angstrom"],
        },
        "minimum_heavy_bond_angle_degrees": min(angles),
        "maximum_heavy_bond_angle_degrees": max(angles),
        "minimum_nonlocal_heavy_atom_distance_angstrom": (
            None if math.isinf(minimum_nonlocal) else minimum_nonlocal
        ),
        "minimum_nonlocal_pair": (
            None
            if minimum_pair is None
            else {
                "first": identities[minimum_pair[0]],
                "second": identities[minimum_pair[1]],
            }
        ),
        "assigned_chiral_center_count": chirality["center_count"],
        "coordinate_chirality_match": True,
        "chirality_audit": chirality,
        "canonical_bond_count": len(bond_rows),
        "canonical_angle_count": len(angles),
    }

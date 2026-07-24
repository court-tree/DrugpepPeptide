"""Replay saved safe265 FASPR PDBs through canonical topology QC.

This command never executes FASPR and never samples a backbone.  It accepts
only an existing failed-coverage directory and saved FASPR source checkout.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from rdkit import Chem, rdBase

from phase3.drugclip.faspr_full_atom_conformer_prototype import (
    _geometry_audit,
    _lookup_atoms,
    _molecule_with_coordinates,
    _terminal_oxt,
    _xyz,
    parse_faspr_pdb,
)
from phase3.drugclip.full_atom_conformer_prototype import (
    AA1_TO_3,
    _base_molecule,
    _validate_topology,
    atom_identity_sha256,
)
from phase3.drugclip.standard_residue_topology import (
    STANDARD_RESIDUE_ATOMS,
    canonical_peptide_graph,
    residue_bonds,
)
from phase3.drugclip.train_only_torsion_prior_prototype import (
    canonical_json_sha256,
    file_sha256,
)
from phase3.drugclip.validate_constrained_full_atom_prototype import (
    cpu_egnn_forward_all,
)
from phase3.drugclip.validate_train_only_torsion_prototype import (
    _proline_planarity_audit,
)


SCHEMA_VERSION = "phase3-v2-standard-residue-topology-qc-replay-v1"
FAILED_SEQUENCE = "AERKRILPTWML"
FAILED_SEQUENCE_CACHE_KEY = "5C90EF6ECDE854A7A9C1"
EXPECTED_ATTEMPTS = 25


def _atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"replay_output_already_exists:{path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"replay_temporary_exists:{temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(
                (
                    json.dumps(
                        value, ensure_ascii=False, indent=2, sort_keys=True
                    )
                    + "\n"
                ).encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalized_bond(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((str(first), str(second))))


def _rdkit_residue_contract(one_letter: str) -> dict[str, Any]:
    molecule = _base_molecule(one_letter)
    identities = _validate_topology(molecule, one_letter)
    names = {int(row["atom_index"]): str(row["atom_name"]) for row in identities}
    bonds = {
        _normalized_bond(
            names[bond.GetBeginAtomIdx()], names[bond.GetEndAtomIdx()]
        )
        for bond in molecule.GetBonds()
        if bond.GetBeginAtomIdx() in names and bond.GetEndAtomIdx() in names
    }
    return {
        "atom_names": sorted(names.values()),
        "bonds": sorted([list(row) for row in bonds]),
    }


def _faspr_sidechain_atom_names(source: str) -> dict[str, list[str]]:
    output = {}
    for match in re.finditer(
        r"(?P<body>ttmp\.nchi=.*?)sidechainTopo\['(?P<aa>[A-Z])'\]=ttmp;",
        source,
        flags=re.DOTALL,
    ):
        output[match.group("aa")] = [
            name.strip().upper()
            for name in re.findall(
                r'atnames\.push_back\("([^"]+)"\)', match.group("body")
            )
        ]
    output["G"] = []
    return output


def audit_twenty_residue_topologies(
    faspr_source_root: Path,
) -> dict[str, Any]:
    source_path = Path(faspr_source_root).resolve() / "src" / "RotamerBuilder.cpp"
    source = source_path.read_text(encoding="utf-8")
    faspr_sidechains = _faspr_sidechain_atom_names(source)
    rows = []
    for one_letter, residue_name in AA1_TO_3.items():
        canonical_atoms = sorted(STANDARD_RESIDUE_ATOMS[residue_name])
        canonical_bonds = set(residue_bonds(residue_name))
        rdkit = _rdkit_residue_contract(one_letter)
        rdkit_bonds = {tuple(row) for row in rdkit["bonds"]}
        rdkit_nonterminal_bonds = {
            row for row in rdkit_bonds if "OXT" not in row
        }
        faspr_atoms = sorted(
            {"N", "CA", "C", "O"} | set(faspr_sidechains[one_letter])
        )
        missing_bonds = sorted(canonical_bonds - rdkit_nonterminal_bonds)
        extra_bonds = sorted(rdkit_nonterminal_bonds - canonical_bonds)
        mismatch = bool(missing_bonds or extra_bonds)
        rows.append({
            "one_letter": one_letter,
            "residue_name": residue_name,
            "canonical_atom_names": canonical_atoms,
            "canonical_heavy_atom_bonds": [
                list(row) for row in sorted(canonical_bonds)
            ],
            "rdkit_version": rdBase.rdkitVersion,
            "rdkit_pdb_atom_names": rdkit["atom_names"],
            "rdkit_heavy_atom_bonds": rdkit["bonds"],
            "faspr_standard_atom_names": faspr_atoms,
            "faspr_emits_oxt": False,
            "prototype_adds_and_connects_terminal_oxt": True,
            "rdkit_missing_canonical_bonds": [
                list(row) for row in missing_bonds
            ],
            "rdkit_extra_noncanonical_bonds": [
                list(row) for row in extra_bonds
            ],
            "atom_name_mismatch": (
                set(rdkit["atom_names"]) - {"OXT"}
                != set(canonical_atoms)
                or set(faspr_atoms) != set(canonical_atoms)
            ),
            "bond_graph_mismatch": mismatch,
            "affects_bond_qc": mismatch,
            "affects_angle_qc": mismatch,
            "affects_graph_distance_clash_qc": mismatch,
            "affects_chirality_qc": residue_name == "ILE" and mismatch,
        })
    mismatch_rows = [row for row in rows if row["bond_graph_mismatch"]]
    if [row["residue_name"] for row in mismatch_rows] != ["ILE"]:
        raise ValueError(
            "unexpected_rdkit_standard_residue_topology_mismatches:"
            f"{[row['residue_name'] for row in mismatch_rows]}"
        )
    ile = mismatch_rows[0]
    if ile["rdkit_missing_canonical_bonds"] != [["CD1", "CG1"]]:
        raise ValueError("rdkit_ile_missing_bond_contract_changed")
    if ile["rdkit_extra_noncanonical_bonds"] != [["CD1", "CG2"]]:
        raise ValueError("rdkit_ile_extra_bond_contract_changed")
    return {
        "status": "PASS",
        "standard_residue_count": len(rows),
        "rdkit_topology_mismatch_count": len(mismatch_rows),
        "rdkit_topology_mismatch_residues": [
            row["residue_name"] for row in mismatch_rows
        ],
        "faspr_source": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
        },
        "rows": rows,
    }


def _coordinates_from_saved_faspr(
    path: Path,
    identities: list[dict[str, Any]],
    sequence: str,
) -> list[list[float]]:
    lookup = _lookup_atoms(parse_faspr_pdb(path))
    last = len(sequence)
    coordinates = []
    for identity in identities:
        key = (int(identity["residue_index"]), str(identity["atom_name"]))
        if key[1] == "OXT":
            xyz = _terminal_oxt(
                _xyz(lookup[(last, "CA")]),
                _xyz(lookup[(last, "C")]),
                _xyz(lookup[(last, "O")]),
            )
        else:
            xyz = _xyz(lookup[key])
        coordinates.append(xyz)
    return coordinates


def _legacy_rdkit_chirality_mismatches(
    molecule: Chem.Mol,
    identities: list[dict[str, Any]],
    coordinates: list[list[float]],
) -> list[dict[str, Any]]:
    expected = Chem.FindMolChiralCenters(
        molecule, includeUnassigned=True, useLegacyImplementation=False
    )
    coordinate_molecule = _molecule_with_coordinates(molecule, coordinates)
    for atom in coordinate_molecule.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    Chem.AssignAtomChiralTagsFromStructure(
        coordinate_molecule, confId=0, replaceExistingTags=True
    )
    Chem.AssignStereochemistry(
        coordinate_molecule, cleanIt=True, force=True
    )
    observed = dict(Chem.FindMolChiralCenters(
        coordinate_molecule,
        includeUnassigned=True,
        useLegacyImplementation=False,
    ))
    identity_by_index = {
        int(row["atom_index"]): row for row in identities
    }
    return [
        {
            "atom_index": atom_index,
            "residue_index": identity_by_index[atom_index]["residue_index"],
            "residue_name": identity_by_index[atom_index]["residue_name"],
            "atom_name": identity_by_index[atom_index]["atom_name"],
            "rdkit_expected": expected_label,
            "rdkit_observed_from_standard_faspr_coordinates": observed.get(
                atom_index
            ),
        }
        for atom_index, expected_label in expected
        if observed.get(atom_index) != expected_label
    ]


def replay_saved_failure(
    failure_root: Path,
    faspr_source_root: Path,
) -> dict[str, Any]:
    root = Path(failure_root).resolve()
    failure = json.loads(
        (root / "build_failure.json").read_text(encoding="utf-8")
    )
    if failure["details"]["sequence"] != FAILED_SEQUENCE:
        raise ValueError("saved_failure_sequence_mismatch")
    attempts = failure["details"]["attempt_audit"]
    if len(attempts) != EXPECTED_ATTEMPTS:
        raise ValueError("saved_failure_attempt_count_mismatch")
    topology_audit = audit_twenty_residue_topologies(faspr_source_root)
    molecule = _base_molecule(FAILED_SEQUENCE)
    identities = _validate_topology(molecule, FAILED_SEQUENCE)
    graph = canonical_peptide_graph(identities)
    ile_index = FAILED_SEQUENCE.index("I") + 1
    attempt_rows = []
    payload_conformers = []
    old_wrong_lengths = []
    old_correct_lengths = []
    for attempt in attempts:
        attempt_index = int(attempt["attempt_index"])
        path = (
            root / "work" / FAILED_SEQUENCE_CACHE_KEY / "slot_00"
            / f"attempt_{attempt_index:02d}" / "conformer_00.faspr.pdb"
        )
        if not path.is_file():
            raise FileNotFoundError(f"saved_faspr_output_missing:{path}")
        if file_sha256(path) != attempt["faspr_output_sha256"]:
            raise ValueError(f"saved_faspr_output_sha_mismatch:{attempt_index}")
        coordinates = _coordinates_from_saved_faspr(
            path, identities, FAILED_SEQUENCE
        )
        lookup = graph["identity_lookup"]
        correct = math.dist(
            coordinates[lookup[(ile_index, "CG1")]],
            coordinates[lookup[(ile_index, "CD1")]],
        )
        wrong = math.dist(
            coordinates[lookup[(ile_index, "CG2")]],
            coordinates[lookup[(ile_index, "CD1")]],
        )
        old_correct_lengths.append(correct)
        old_wrong_lengths.append(wrong)
        try:
            geometry = _geometry_audit(molecule, coordinates, identities)
            status = "PASS"
            reason = None
        except Exception as error:
            geometry = None
            status = "FAIL"
            reason = f"{type(error).__name__}:{error}"
        attempt_rows.append({
            "attempt_index": attempt_index,
            "saved_faspr_pdb": str(path),
            "saved_faspr_pdb_sha256": file_sha256(path),
            "original_faspr_exit_contract": 0,
            "old_rdkit_wrong_ile6_cg2_cd1_angstrom": wrong,
            "canonical_ile6_cg1_cd1_angstrom": correct,
            "status": status,
            "failure_reason": reason,
            "geometry_audit": geometry,
        })
        if geometry is not None:
            payload_conformers.append({
                "conformer_index": len(payload_conformers),
                "coordinates": coordinates,
            })
    passed = [row for row in attempt_rows if row["status"] == "PASS"]
    failed = [row for row in attempt_rows if row["status"] == "FAIL"]
    if passed:
        payload = {
            "peptide_sequence": FAILED_SEQUENCE,
            "atom_count": len(identities),
            "atom_identity": identities,
            "atom_identity_sha256": atom_identity_sha256(identities),
            "conformer_count": len(payload_conformers),
            "conformers": payload_conformers,
        }
        planarity = _proline_planarity_audit(payload)
        egnn = cpu_egnn_forward_all(payload)
        if planarity["status"] != "PASS":
            raise ValueError("saved_replay_proline_planarity_not_pass")
        if egnn["status"] != "PASS" or not egnn["embedding_finite"]:
            raise ValueError("saved_replay_cpu_egnn_not_finite")
    else:
        planarity = {"status": "NOT_RUN_NO_GEOMETRY_PASS"}
        egnn = {"status": "NOT_RUN_NO_GEOMETRY_PASS"}
    first_coordinates = _coordinates_from_saved_faspr(
        Path(attempt_rows[0]["saved_faspr_pdb"]),
        identities,
        FAILED_SEQUENCE,
    )
    legacy_chirality = _legacy_rdkit_chirality_mismatches(
        molecule, identities, first_coordinates
    )
    if not any(
        row["residue_name"] == "ILE"
        and row["atom_name"] == "CB"
        for row in legacy_chirality
    ):
        raise ValueError("legacy_rdkit_ile_cb_chirality_effect_not_reproduced")
    rejection_counts = Counter(
        str(row["failure_reason"]) for row in failed
    )
    maximum_correct_bond = max(
        (
            row["geometry_audit"]["maximum_heavy_bond_length_angstrom"]
            for row in passed
        ),
        default=None,
    )
    minimum_nonlocal = min(
        (
            row["geometry_audit"][
                "minimum_nonlocal_heavy_atom_distance_angstrom"
            ]
            for row in passed
            if row["geometry_audit"][
                "minimum_nonlocal_heavy_atom_distance_angstrom"
            ] is not None
        ),
        default=None,
    )
    chirality_rows = [
        row
        for attempt in passed
        for row in attempt["geometry_audit"]["chirality_audit"]["rows"]
        if row["residue_name"] in {"ILE", "THR"}
    ]
    result_core = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "classification": (
            "QC_TOPOLOGY_MAPPING_FAIL / SAFE265_COVERAGE_INCONCLUSIVE"
            if passed
            else "SAFE265_GENERATION_COVERAGE_FAIL_REPLAY_CONFIRMED"
        ),
        "source_failure_classification_withdrawn": bool(passed),
        "source_failure_root": str(root),
        "source_failure_file_sha256": file_sha256(
            root / "build_failure.json"
        ),
        "sequence": FAILED_SEQUENCE,
        "attempt_count": len(attempt_rows),
        "canonical_qc_pass_count": len(passed),
        "canonical_qc_failure_count": len(failed),
        "canonical_qc_failure_reasons": dict(sorted(rejection_counts.items())),
        "old_wrong_ile6_cg2_cd1_range_angstrom": {
            "minimum": min(old_wrong_lengths),
            "maximum": max(old_wrong_lengths),
        },
        "canonical_ile6_cg1_cd1_range_angstrom": {
            "minimum": min(old_correct_lengths),
            "maximum": max(old_correct_lengths),
        },
        "maximum_correct_covalent_bond_length_angstrom": maximum_correct_bond,
        "minimum_nonlocal_heavy_atom_distance_angstrom": minimum_nonlocal,
        "ile_thr_chirality": {
            "status": (
                "PASS"
                if chirality_rows
                and all(row["status"] == "PASS" for row in chirality_rows)
                else "NOT_ESTABLISHED"
            ),
            "rows": chirality_rows,
        },
        "legacy_rdkit_chirality_mismatches_attempt_00": legacy_chirality,
        "proline_planarity": planarity,
        "cpu_egnn_forward": egnn,
        "twenty_residue_topology_audit": topology_audit,
        "attempts": attempt_rows,
        "faspr_executed_during_replay": False,
        "backbone_sampled_during_replay": False,
        "geometry_or_clash_threshold_changed": False,
        "target_bound_inputs_used": False,
        "training_or_retrieval_run": False,
    }
    return {
        **result_core,
        "report_canonical_sha256": canonical_json_sha256(result_core),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-root", required=True)
    parser.add_argument("--faspr-source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        print(f"FileExistsError:replay_output_directory_exists:{output_dir}")
        return 2
    output_dir.mkdir(parents=True)
    try:
        report = replay_saved_failure(
            Path(args.failure_root),
            Path(args.faspr_source_root),
        )
        _atomic_write_json(output_dir / "topology_qc_replay_report.json", report)
    except Exception as error:
        _atomic_write_json(
            output_dir / "topology_qc_replay_failure.json",
            {
                "status": "FAIL",
                "exception_type": type(error).__name__,
                "exception_text": str(error),
            },
        )
        print(f"{type(error).__name__}:{error}")
        return 2
    print(json.dumps({
        "status": report["status"],
        "classification": report["classification"],
        "pass_count": report["canonical_qc_pass_count"],
        "failure_count": report["canonical_qc_failure_count"],
        "report_canonical_sha256": report["report_canonical_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validation and bounded CPU-forward smoke for the full-heavy prototype."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import gemmi
import numpy as np
import torch
from scipy.spatial import cKDTree

from phase2.pepclip.data import (
    ATOM_NAME_TO_ID,
    ELEMENT_TO_ID,
    RESIDUE_NAME_TO_ID,
    atom_tensors,
    pad_atom_clouds,
)
from phase2.pepclip.model_3d import PepCLIP3DModel
from phase3.drugclip.batching import PHASE2_MAX_PEPTIDE_ATOMS
from phase3.drugclip.full_atom_conformer_prototype import (
    AA1_TO_3,
    CHEMISTRY_CLASS,
    ConformerGenerationError,
    GENERATOR_ID,
    REQUIRED_HEAVY_ATOMS,
    SCHEMA_VERSION,
    UnsupportedPeptideChemistry,
    _base_molecule,
    conformer_atoms,
    generate_full_heavy_conformers,
)
from phase3.drugclip.build_interface_pairs import _collect_evidence
from phase3.drugclip.evaluate_full_retrieval import (
    _file_hash,
    _sequence_hash,
    _validate_pilot_contract,
)
from phase3.drugclip.evaluate_input_domain_ablation import (
    EXPECTED,
    select_exact_evidence,
)
from phase3.drugclip.io_utils import read_jsonl
from phase3.drugclip.structure_qc import resolve_coordinate_paths


FORBIDDEN_DEPENDENCY_TOKENS = {
    "bound",
    "contact",
    "evidence",
    "interface",
    "pose",
    "receptor",
    "target",
}
CHEMISTRY_CLASSES = {
    "ordinary_linear_standard",
    "multiple_cys_unknown",
    "known_disulfide",
    "cyclic_or_crosslinked",
    "modified_or_nonstandard",
    "receptor_covalent",
    "chemistry_insufficient",
}
COVALENT_RADII = {
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "P": 1.07,
    "S": 1.05,
    "SE": 1.20,
}


def _keys_recursively(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key in value] + [
            nested
            for item in value.values()
            for nested in _keys_recursively(item)
        ]
    if isinstance(value, list):
        return [nested for item in value for nested in _keys_recursively(item)]
    return []


def validate_payload(payload: dict[str, Any], *, expected_conformers: int = 10) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version_mismatch")
    if payload.get("generator_id") != GENERATOR_ID:
        raise ValueError("generator_id_mismatch")
    if payload.get("chemistry", {}).get("chemistry_class") != CHEMISTRY_CLASS:
        raise ValueError("chemistry_class_mismatch")
    if int(payload.get("conformer_count", -1)) != expected_conformers:
        raise ValueError("conformer_count_mismatch")

    identities = payload.get("atom_identity")
    if not isinstance(identities, list) or not identities:
        raise ValueError("atom_identity_missing")
    if len(identities) != int(payload.get("atom_count", -1)):
        raise ValueError("atom_count_mismatch")
    if len(identities) > PHASE2_MAX_PEPTIDE_ATOMS:
        raise ValueError("peptide_atom_cap_exceeded")

    sequence = str(payload.get("peptide_sequence") or "")
    by_residue: dict[int, set[str]] = {}
    for identity in identities:
        atom_name = str(identity["atom_name"])
        element = str(identity["element"])
        residue_name = str(identity["residue_name"])
        residue_index = int(identity["residue_index"])
        if atom_name not in ATOM_NAME_TO_ID:
            raise ValueError(f"atom_name_out_of_vocab:{atom_name}")
        if element not in ELEMENT_TO_ID:
            raise ValueError(f"element_out_of_vocab:{element}")
        if residue_name not in RESIDUE_NAME_TO_ID:
            raise ValueError(f"residue_name_out_of_vocab:{residue_name}")
        by_residue.setdefault(residue_index, set()).add(atom_name)
    if sorted(by_residue) != list(range(1, len(sequence) + 1)):
        raise ValueError("residue_indices_not_contiguous")
    for residue_index, names in by_residue.items():
        residue_name = str(identities[next(
            index for index, row in enumerate(identities)
            if int(row["residue_index"]) == residue_index
        )]["residue_name"])
        required = set(REQUIRED_HEAVY_ATOMS[residue_name])
        if residue_index == len(sequence):
            required.add("OXT")
        if not required <= names:
            raise ValueError(f"required_heavy_atoms_missing:{residue_index}")

    coordinate_hashes: set[str] = set()
    for expected_index, conformer in enumerate(payload.get("conformers", [])):
        if int(conformer.get("conformer_index", -1)) != expected_index:
            raise ValueError("conformer_index_mismatch")
        coordinates = conformer.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != len(identities):
            raise ValueError("coordinate_count_mismatch")
        if any(
            not isinstance(xyz, list)
            or len(xyz) != 3
            or any(not math.isfinite(float(value)) for value in xyz)
            for xyz in coordinates
        ):
            raise ValueError("nonfinite_or_invalid_coordinates")
        if conformer.get("mmff_status") != 0:
            raise ValueError("mmff_not_converged")
        if conformer.get("geometry_audit", {}).get("status") != "PASS":
            raise ValueError("geometry_audit_not_pass")
        coordinate_hashes.add(str(conformer.get("coordinate_sha256")))
    if len(coordinate_hashes) != expected_conformers:
        raise ValueError("conformers_not_all_distinct")

    dependency_contract = payload.get("dependency_contract", {})
    if dependency_contract.get("target_bound_inputs_used") is not False:
        raise ValueError("target_bound_dependency_not_explicitly_false")
    forbidden_keys = sorted(
        key
        for key in _keys_recursively({k: v for k, v in payload.items() if k != "dependency_contract"})
        if any(token in key.lower() for token in FORBIDDEN_DEPENDENCY_TOKENS)
    )
    if forbidden_keys:
        raise ValueError(f"forbidden_dependency_keys:{forbidden_keys}")
    return {
        "status": "PASS",
        "sequence": sequence,
        "atom_count": len(identities),
        "conformer_count": expected_conformers,
        "unique_coordinate_hashes": len(coordinate_hashes),
        "target_bound_inputs_used": False,
    }


def verify_determinism(
    sequence: str,
    *,
    num_conformers: int = 2,
    base_seed: int,
) -> dict[str, Any]:
    first = generate_full_heavy_conformers(
        sequence, num_conformers=num_conformers, base_seed=base_seed
    )
    second = generate_full_heavy_conformers(
        sequence, num_conformers=num_conformers, base_seed=base_seed
    )
    first_contract = {
        "atom_identity_sha256": first["atom_identity_sha256"],
        "coordinate_sha256": [
            row["coordinate_sha256"] for row in first["conformers"]
        ],
    }
    second_contract = {
        "atom_identity_sha256": second["atom_identity_sha256"],
        "coordinate_sha256": [
            row["coordinate_sha256"] for row in second["conformers"]
        ],
    }
    if first_contract != second_contract:
        raise ValueError("fixed_seed_generation_not_exactly_deterministic")
    return {
        "status": "PASS",
        "compared_conformers": num_conformers,
        **first_contract,
    }


def cpu_forward_smoke(payload: dict[str, Any], *, conformer_index: int = 0) -> dict[str, Any]:
    atoms = conformer_atoms(payload, conformer_index)
    tensors = atom_tensors(atoms)
    padded = pad_atom_clouds(
        [tensors["coords"]],
        [tensors["elements"]],
        [tensors["atom_names"]],
        [tensors["residue_names"]],
    )
    model = PepCLIP3DModel(
        num_elements=max(ELEMENT_TO_ID.values()) + 1,
        num_atom_names=max(ATOM_NAME_TO_ID.values()) + 1,
        num_residue_names=max(RESIDUE_NAME_TO_ID.values()) + 1,
        encoder_type="egnn",
        element_dim=16,
        hidden_dim=32,
        output_dim=16,
        dropout=0.0,
        num_layers=1,
        num_rbf=8,
        num_neighbors=8,
    ).cpu().eval()
    with torch.inference_mode():
        embedding = model.encode_peptide(
            padded["coords"],
            padded["elements"],
            padded["mask"],
            padded["atom_names"],
            padded["residue_names"],
        )
    if embedding.shape != (1, 16) or not torch.isfinite(embedding).all():
        raise ValueError("pepclip_cpu_forward_invalid_output")
    return {
        "status": "PASS",
        "encoder_type": "egnn",
        "device": "cpu",
        "input_atom_count": len(atoms),
        "embedding_shape": list(embedding.shape),
        "embedding_finite": True,
    }


def _seqid_text(residue: gemmi.Residue) -> str:
    value = str(residue.seqid.num)
    insertion = str(residue.seqid.icode).strip()
    return value + (insertion if insertion and insertion != "?" else "")


def _standard_letter(residue: gemmi.Residue) -> str | None:
    letter = str(
        gemmi.find_tabulated_residue(residue.name.strip().upper()).one_letter_code
    ).upper()
    return letter if letter in AA1_TO_3 else None


def _heavy_atoms(
    residues: list[tuple[str, gemmi.Residue]],
) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for chain_name, residue in residues:
        seen: set[str] = set()
        for atom in residue:
            atom_name = str(atom.name).strip().upper()
            altloc = str(atom.altloc).strip().upper()
            element = str(atom.element.name).strip().upper()
            if atom_name in seen or altloc not in {"", "A", "\x00"} or element == "H":
                continue
            seen.add(atom_name)
            atoms.append({
                "chain_id": chain_name,
                "residue_id": _seqid_text(residue),
                "residue_name": residue.name.strip().upper(),
                "atom_name": atom_name,
                "element": element,
                "coords": [float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)],
            })
    return atoms


def _select_exact_peptide_residues(
    structure: gemmi.Structure,
    expected_sequence: str,
    peptide_chain: str | None,
) -> tuple[list[tuple[str, gemmi.Residue]], str]:
    candidates: list[tuple[str, list[tuple[str, gemmi.Residue]]]] = []
    if len(structure) == 0:
        raise ValueError("empty_structure")
    for chain in structure[0]:
        if peptide_chain is not None and str(chain.name) != peptide_chain:
            continue
        selected: list[tuple[str, gemmi.Residue]] = []
        letters: list[str] = []
        for residue in chain:
            letter = _standard_letter(residue)
            atom_names = {
                str(atom.name).strip().upper()
                for atom in residue
                if str(atom.altloc).strip().upper() in {"", "A", "\x00"}
            }
            if letter is not None and {"N", "CA", "C"} <= atom_names:
                selected.append((str(chain.name), residue))
                letters.append(letter)
        if "".join(letters) == expected_sequence:
            candidates.append((str(chain.name), selected))
    if len(candidates) != 1:
        raise ValueError(
            f"exact_peptide_chain_selection_not_1_to_1:{len(candidates)}"
        )
    return candidates[0][1], candidates[0][0]


def _connection_partner_key(partner: Any) -> tuple[str, str, str]:
    residue_id = str(partner.res_id.seqid.num)
    insertion = str(partner.res_id.seqid.icode).strip()
    if insertion and insertion != "?":
        residue_id += insertion
    return (
        str(partner.chain_name),
        residue_id,
        str(partner.atom_name).strip().upper(),
    )


def _minimum_covalent_cross_distance(
    peptide_atoms: list[dict[str, Any]],
    other_atoms: list[dict[str, Any]],
) -> tuple[float | None, dict[str, Any] | None]:
    eligible_other = [
        atom for atom in other_atoms
        if atom["element"] in COVALENT_RADII
    ]
    if not peptide_atoms or not eligible_other:
        return None, None
    other_coords = np.asarray([atom["coords"] for atom in eligible_other], dtype=np.float64)
    tree = cKDTree(other_coords)
    best_distance = math.inf
    best_pair: dict[str, Any] | None = None
    for peptide_atom in peptide_atoms:
        if peptide_atom["element"] not in COVALENT_RADII:
            continue
        radius = COVALENT_RADII[peptide_atom["element"]] + max(COVALENT_RADII.values()) + 0.20
        for index in tree.query_ball_point(peptide_atom["coords"], radius):
            other_atom = eligible_other[index]
            distance = math.dist(peptide_atom["coords"], other_atom["coords"])
            threshold = (
                COVALENT_RADII[peptide_atom["element"]]
                + COVALENT_RADII[other_atom["element"]]
                + 0.20
            )
            if 0.50 < distance <= threshold and distance < best_distance:
                best_distance = distance
                best_pair = {
                    "peptide_atom": {
                        key: peptide_atom[key]
                        for key in ("chain_id", "residue_id", "residue_name", "atom_name", "element")
                    },
                    "other_atom": {
                        key: other_atom[key]
                        for key in ("chain_id", "residue_id", "residue_name", "atom_name", "element")
                    },
                    "distance_angstrom": distance,
                    "covalent_threshold_angstrom": threshold,
                }
    return (None, None) if best_pair is None else (best_distance, best_pair)


def audit_one_chemistry(
    plan_row: dict[str, Any],
    input_audit: dict[str, Any],
    evidence_row: dict[str, Any],
    qbiolip_root: Path,
    biolip_root: Path,
) -> dict[str, Any]:
    sequence = str(plan_row["peptide_sequence"]).upper()
    receptor_path, peptide_path, separate = resolve_coordinate_paths(
        evidence_row, qbiolip_root, biolip_root
    )
    peptide_structure = gemmi.read_structure(str(peptide_path))
    peptide_residues, resolved_chain = _select_exact_peptide_residues(
        peptide_structure,
        sequence,
        None if separate else str(input_audit.get("peptide_chain") or ""),
    )
    peptide_atoms = _heavy_atoms(peptide_residues)
    residue_names = [
        residue.name.strip().upper() for _, residue in peptide_residues
    ]
    expected_residue_names = [AA1_TO_3[letter] for letter in sequence]
    standard = residue_names == expected_residue_names
    modified_positions = [
        {
            "residue_index": index,
            "expected_residue_name": expected,
            "observed_residue_name": observed,
        }
        for index, (expected, observed) in enumerate(
            zip(expected_residue_names, residue_names), start=1
        )
        if expected != observed
    ]
    residue_keys = {
        (chain, _seqid_text(residue))
        for chain, residue in peptide_residues
    }
    atom_by_key = {
        (atom["chain_id"], atom["residue_id"], atom["atom_name"]): atom
        for atom in peptide_atoms
    }

    explicit_connections: list[dict[str, Any]] = []
    peptide_internal_connections: list[dict[str, Any]] = []
    peptide_external_connections: list[dict[str, Any]] = []
    for connection in peptide_structure.connections:
        first = _connection_partner_key(connection.partner1)
        second = _connection_partner_key(connection.partner2)
        first_in = first[:2] in residue_keys
        second_in = second[:2] in residue_keys
        if not (first_in or second_in):
            continue
        row = {
            "name": str(connection.name),
            "type": str(connection.type),
            "partner1": list(first),
            "partner2": list(second),
        }
        explicit_connections.append(row)
        if first_in and second_in:
            peptide_internal_connections.append(row)
        elif first_in != second_in:
            peptide_external_connections.append(row)

    adjacent_bonds: list[dict[str, Any]] = []
    for index in range(len(peptide_residues) - 1):
        left_chain, left_residue = peptide_residues[index]
        right_chain, right_residue = peptide_residues[index + 1]
        left = atom_by_key.get((left_chain, _seqid_text(left_residue), "C"))
        right = atom_by_key.get((right_chain, _seqid_text(right_residue), "N"))
        distance = (
            math.dist(left["coords"], right["coords"])
            if left is not None and right is not None else None
        )
        adjacent_bonds.append({
            "left_residue_index": index + 1,
            "right_residue_index": index + 2,
            "c_n_distance_angstrom": distance,
            "peptide_bond_geometry": distance is not None and distance <= 1.80,
        })
    continuous_backbone = all(
        row["peptide_bond_geometry"] for row in adjacent_bonds
    )

    first_chain, first_residue = peptide_residues[0]
    last_chain, last_residue = peptide_residues[-1]
    first_n = atom_by_key.get((first_chain, _seqid_text(first_residue), "N"))
    last_c = atom_by_key.get((last_chain, _seqid_text(last_residue), "C"))
    head_tail_distance = (
        math.dist(first_n["coords"], last_c["coords"])
        if first_n is not None and last_c is not None else None
    )
    head_to_tail = (
        len(peptide_residues) > 2
        and head_tail_distance is not None
        and head_tail_distance <= 1.80
    )

    cysteine_sg = [
        atom for atom in peptide_atoms
        if atom["residue_name"] == "CYS" and atom["atom_name"] == "SG"
    ]
    ss_bonds: list[dict[str, Any]] = []
    for first_index, first in enumerate(cysteine_sg):
        for second in cysteine_sg[first_index + 1:]:
            distance = math.dist(first["coords"], second["coords"])
            if distance <= 2.30:
                ss_bonds.append({
                    "residue1": [first["chain_id"], first["residue_id"]],
                    "residue2": [second["chain_id"], second["residue_id"]],
                    "sg_sg_distance_angstrom": distance,
                    "detected_by": "geometry",
                })
    for connection in peptide_internal_connections:
        partners = [connection["partner1"], connection["partner2"]]
        if {partners[0][2], partners[1][2]} == {"SG"}:
            signature = {tuple(partners[0][:2]), tuple(partners[1][:2])}
            if not any(
                {tuple(row["residue1"]), tuple(row["residue2"])} == signature
                for row in ss_bonds
            ):
                ss_bonds.append({
                    "residue1": partners[0][:2],
                    "residue2": partners[1][:2],
                    "sg_sg_distance_angstrom": None,
                    "detected_by": "connection_record",
                })

    nonstandard_internal_connections = []
    residue_position = {
        (chain, _seqid_text(residue)): index
        for index, (chain, residue) in enumerate(peptide_residues)
    }
    for connection in peptide_internal_connections:
        first = connection["partner1"]
        second = connection["partner2"]
        positions = (
            residue_position.get(tuple(first[:2])),
            residue_position.get(tuple(second[:2])),
        )
        is_adjacent_cn = (
            positions[0] is not None
            and positions[1] is not None
            and abs(positions[0] - positions[1]) == 1
            and {first[2], second[2]} == {"C", "N"}
        )
        is_disulfide = {first[2], second[2]} == {"SG"}
        is_head_tail = (
            {positions[0], positions[1]} == {0, len(peptide_residues) - 1}
            and {first[2], second[2]} == {"C", "N"}
        )
        if not (is_adjacent_cn or is_disulfide or is_head_tail):
            nonstandard_internal_connections.append(connection)

    if separate:
        receptor_structure = gemmi.read_structure(str(receptor_path))
        other_residues = [
            (str(chain.name), residue)
            for chain in receptor_structure[0]
            for residue in chain
        ]
    else:
        other_residues = [
            (str(chain.name), residue)
            for chain in peptide_structure[0]
            for residue in chain
            if (str(chain.name), _seqid_text(residue)) not in residue_keys
        ]
    other_atoms = _heavy_atoms(other_residues)
    minimum_external_distance, external_geometry = _minimum_covalent_cross_distance(
        peptide_atoms, other_atoms
    )
    receptor_covalent = bool(peptide_external_connections or external_geometry)
    known_disulfide = bool(ss_bonds)
    cyclic_or_crosslinked = bool(
        head_to_tail or nonstandard_internal_connections
    )
    terminal_state_determined = bool(
        continuous_backbone
        and first_n is not None
        and last_c is not None
        and not head_to_tail
        and not nonstandard_internal_connections
        and not receptor_covalent
    )
    cys_count = sequence.count("C")
    theoretical_heavy_atom_count = int(_base_molecule(sequence).GetNumAtoms())

    if not standard:
        chemistry_classification = "modified_or_nonstandard"
        exclusion_reason = "raw_structure_residue_names_do_not_match_standard_sequence_chemistry"
    elif receptor_covalent:
        chemistry_classification = "receptor_covalent"
        exclusion_reason = "explicit_or_geometry_supported_peptide_receptor_covalent_connection"
    elif known_disulfide:
        chemistry_classification = "known_disulfide"
        exclusion_reason = "intrapeptide_disulfide_detected"
    elif cyclic_or_crosslinked:
        chemistry_classification = "cyclic_or_crosslinked"
        exclusion_reason = "head_to_tail_or_noncanonical_intrapeptide_connection_detected"
    elif cys_count > 1:
        chemistry_classification = "multiple_cys_unknown"
        exclusion_reason = "multiple_cysteines_without_resolved_disulfide_chemistry"
    elif not terminal_state_determined:
        chemistry_classification = "chemistry_insufficient"
        exclusion_reason = "linear_connectivity_or_terminal_state_not_determined"
    else:
        chemistry_classification = "ordinary_linear_standard"
        exclusion_reason = None
    if chemistry_classification not in CHEMISTRY_CLASSES:
        raise AssertionError("unknown_chemistry_classification")

    return {
        "interface_pair_id": str(plan_row["interface_pair_id"]),
        "evidence_id": str(input_audit["evidence_id"]),
        "source_database": str(input_audit["source_database"]),
        "peptide_sequence": sequence,
        "sequence_length": len(sequence),
        "theoretical_heavy_atom_count": theoretical_heavy_atom_count,
        "residue_names": residue_names,
        "expected_standard_residue_names": expected_residue_names,
        "standard_residues_only": standard,
        "modified_residue_detected": bool(modified_positions),
        "modified_residue_positions": modified_positions,
        "cys_count": cys_count,
        "detectable_ss_bond": known_disulfide,
        "ss_bond_evidence": ss_bonds,
        "head_to_tail_closure_detected": head_to_tail,
        "head_to_tail_c_n_distance_angstrom": head_tail_distance,
        "noncanonical_internal_connections": nonstandard_internal_connections,
        "peptide_receptor_covalent_connection_detected": receptor_covalent,
        "peptide_receptor_explicit_connections": peptide_external_connections,
        "minimum_peptide_other_covalent_distance_angstrom": minimum_external_distance,
        "peptide_other_covalent_geometry": external_geometry,
        "continuous_adjacent_peptide_bonds": continuous_backbone,
        "adjacent_peptide_bond_audit": adjacent_bonds,
        "terminal_state_determined": terminal_state_determined,
        "terminal_state": (
            "linear_free_termini_inferred_from_structure_connectivity"
            if terminal_state_determined else "unresolved_or_nonfree"
        ),
        "chemistry_classification": chemistry_classification,
        "exclusion_reason": exclusion_reason,
        "structure_path": str(peptide_path.resolve()),
        "receptor_structure_path": str(receptor_path.resolve()),
        "structure_type": str(input_audit["structure_type"]),
        "resolved_peptide_chain": resolved_chain,
        "explicit_peptide_connection_records": explicit_connections,
        "atom_cap": PHASE2_MAX_PEPTIDE_ATOMS,
        "at_or_above_atom_cap": theoretical_heavy_atom_count >= PHASE2_MAX_PEPTIDE_ATOMS,
    }


def audit_fixed512_chemistry(
    *,
    pilot_output: Path,
    dataset_root: Path,
    input_variant_audit_path: Path,
    candidate_evidence_jsonl: Path,
    expanded_evidence_jsonl: Path,
    mmcif_root: Path,
    qbiolip_root: Path,
    biolip_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config, subset, plan = _validate_pilot_contract(pilot_output)
    plan_path = pilot_output / "validation_sampling_plan.jsonl"
    if _file_hash(plan_path) != EXPECTED["plan_file_sha256"]:
        raise ValueError("fixed512_plan_file_sha256_mismatch")
    if str(config["fixed_validation_plan_sha256"]) != EXPECTED["plan_canonical_sha256"]:
        raise ValueError("fixed512_plan_canonical_sha256_mismatch")
    if _sequence_hash([str(row["interface_pair_id"]) for row in plan]) != EXPECTED["interface_pair_sha256"]:
        raise ValueError("fixed512_interface_pair_sha256_mismatch")

    input_audits = {
        str(row["interface_pair_id"]): row
        for row in read_jsonl(input_variant_audit_path)
    }
    if set(input_audits) != {str(row["interface_pair_id"]) for row in plan}:
        raise ValueError("input_variant_audit_does_not_exactly_cover_fixed512")
    biological_rows = list(
        read_jsonl(dataset_root / "dependencies" / "biological_pairs.jsonl")
    )
    evidence = _collect_evidence(
        biological_rows,
        candidate_evidence_jsonl,
        expanded_evidence_jsonl,
        mmcif_root,
    )
    output: list[dict[str, Any]] = []
    for plan_row in plan:
        pair_id = str(plan_row["interface_pair_id"])
        audit = input_audits[pair_id]
        evidence_row = select_exact_evidence(
            evidence.get(str(plan_row["biological_pair_id"]), []),
            str(audit["evidence_id"]),
            pair_id,
        )
        output.append(
            audit_one_chemistry(
                plan_row,
                audit,
                evidence_row,
                qbiolip_root,
                biolip_root,
            )
        )

    query_counts = Counter(row["chemistry_classification"] for row in output)
    sequence_classes: dict[str, list[str]] = defaultdict(list)
    for row in output:
        sequence_classes[row["peptide_sequence"]].append(
            row["chemistry_classification"]
        )
    precedence = [
        "modified_or_nonstandard",
        "receptor_covalent",
        "known_disulfide",
        "cyclic_or_crosslinked",
        "multiple_cys_unknown",
        "chemistry_insufficient",
        "ordinary_linear_standard",
    ]
    unique_classification = {}
    for sequence, classifications in sequence_classes.items():
        unique_classification[sequence] = next(
            name for name in precedence if name in classifications
        )
    unique_counts = Counter(unique_classification.values())
    safe = [
        row for row in output
        if row["chemistry_classification"] == "ordinary_linear_standard"
        and unique_classification[row["peptide_sequence"]] == "ordinary_linear_standard"
    ]
    safe_lengths = [row["sequence_length"] for row in safe]
    safe_heavy = [row["theoretical_heavy_atom_count"] for row in safe]
    all_pair_ids = {str(row["interface_pair_id"]) for row in plan}
    all_sequences = {str(row["peptide_sequence"]) for row in plan}
    all_receptors = {str(row["receptor_interface_id"]) for row in plan}
    summary = {
        "schema_version": "phase3-v2-fixed512-full-heavy-chemistry-audit-v1",
        "status": "PASS",
        "classification_boundary": "PROTOTYPE_SMOKE_PASS",
        "fixed512_plan_path": str(plan_path.resolve()),
        "fixed512_plan_sha256": _file_hash(plan_path),
        "fixed512_interface_pair_sha256": _sequence_hash(
            [str(row["interface_pair_id"]) for row in plan]
        ),
        "input_variant_audit_path": str(input_variant_audit_path.resolve()),
        "input_variant_audit_sha256": _file_hash(input_variant_audit_path),
        "query_count": len(output),
        "unique_sequence_count": len(sequence_classes),
        "query_classification_counts": {
            name: query_counts.get(name, 0) for name in sorted(CHEMISTRY_CLASSES)
        },
        "unique_sequence_classification_counts": {
            name: unique_counts.get(name, 0) for name in sorted(CHEMISTRY_CLASSES)
        },
        "ordinary_linear_safe_query_count": len(safe),
        "ordinary_linear_safe_unique_sequence_count": len({
            row["peptide_sequence"] for row in safe
        }),
        "excluded_query_count": len(output) - len(safe),
        "excluded_query_fraction": (len(output) - len(safe)) / len(output),
        "exclusion_reason_counts": dict(sorted(Counter(
            row["exclusion_reason"] for row in output if row["exclusion_reason"]
        ).items())),
        "original_candidate_bank": {
            "interface_pair_count": len(all_pair_ids),
            "peptide_candidate_count": len(all_sequences),
            "receptor_candidate_count": len(all_receptors),
            "safe_query_targets_present": all(
                row["interface_pair_id"] in all_pair_ids
                and row["peptide_sequence"] in all_sequences
                for row in safe
            ),
            "safe_subset_uses_unchanged_full_candidate_bank": True,
        },
        "safe_sequence_length_distribution": _distribution(safe_lengths),
        "safe_heavy_atom_count_distribution": _distribution(safe_heavy),
        "fixed512_at_or_above_192_atom_count": sum(
            row["at_or_above_atom_cap"] for row in output
        ),
        "fixed512_max_theoretical_heavy_atom_count": max(
            row["theoretical_heavy_atom_count"] for row in output
        ),
        "prototype_depends_on_atom_truncation": False,
        "source_files": {
            "candidate_evidence_jsonl": str(candidate_evidence_jsonl.resolve()),
            "expanded_evidence_jsonl": str(expanded_evidence_jsonl.resolve()),
            "dataset_root": str(dataset_root.resolve()),
        },
    }
    return output, summary


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    def quantile(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]
    return {
        "count": len(values),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "max": ordered[-1],
    }


def select_generation_panel(
    chemistry_rows: list[dict[str, Any]],
    *,
    minimum_size: int = 8,
) -> list[dict[str, Any]]:
    all_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chemistry_rows:
        all_by_sequence[row["peptide_sequence"]].append(row)
    candidates = [
        {
            "peptide_sequence": sequence,
            "sequence_length": rows[0]["sequence_length"],
            "theoretical_heavy_atom_count": rows[0]["theoretical_heavy_atom_count"],
            "interface_pair_id": min(row["interface_pair_id"] for row in rows),
            "evidence_id": min(row["evidence_id"] for row in rows),
            "cys_count": rows[0]["cys_count"],
            "composition_signature": ",".join(
                f"{letter}{sequence.count(letter)}"
                for letter in sorted(set(sequence))
            ),
        }
        for sequence, rows in all_by_sequence.items()
        if all(
            row["chemistry_classification"] == "ordinary_linear_standard"
            for row in rows
        )
    ]
    if not candidates:
        raise ValueError("no_ordinary_linear_standard_sequences")
    ordered = sorted(
        candidates,
        key=lambda row: (
            row["sequence_length"],
            row["theoretical_heavy_atom_count"],
            row["peptide_sequence"],
        ),
    )
    selected: dict[str, dict[str, Any]] = {}

    def add(row: dict[str, Any], reason: str) -> None:
        selected.setdefault(
            row["peptide_sequence"],
            {**row, "selection_reasons": []},
        )["selection_reasons"].append(reason)

    add(ordered[0], "fixed512_safe_shortest")
    for fraction, label in (
        (0.50, "fixed512_safe_length_median"),
        (0.75, "fixed512_safe_length_p75"),
        (0.90, "fixed512_safe_length_p90"),
        (0.95, "fixed512_safe_length_p95"),
    ):
        add(ordered[round((len(ordered) - 1) * fraction)], label)
    add(ordered[-1], "fixed512_safe_longest")
    add(
        max(
            candidates,
            key=lambda row: (
                row["theoretical_heavy_atom_count"],
                row["sequence_length"],
                row["peptide_sequence"],
            ),
        ),
        "fixed512_safe_maximum_heavy_atom_count",
    )
    under_cap = [
        row for row in candidates
        if row["theoretical_heavy_atom_count"] < PHASE2_MAX_PEPTIDE_ATOMS
    ]
    add(
        max(
            under_cap,
            key=lambda row: (
                row["theoretical_heavy_atom_count"],
                row["peptide_sequence"],
            ),
        ),
        "fixed512_safe_closest_below_192_atoms",
    )
    single_cys = [row for row in candidates if row["cys_count"] == 1]
    if not single_cys:
        raise ValueError("safe_subset_has_no_single_cys_sequence")
    add(
        sorted(
            single_cys,
            key=lambda row: (
                row["sequence_length"],
                row["theoretical_heavy_atom_count"],
                row["peptide_sequence"],
            ),
        )[len(single_cys) // 2],
        "fixed512_safe_single_cysteine",
    )

    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_length[row["sequence_length"]].append(row)
    median_length = statistics.median(row["sequence_length"] for row in candidates)
    comparable_length, comparable = min(
        (
            (length, rows)
            for length, rows in by_length.items()
            if len({row["composition_signature"] for row in rows}) >= 3
        ),
        key=lambda item: (abs(item[0] - median_length), item[0]),
    )
    used_compositions: set[str] = set()
    comparable_count = 0
    for row in sorted(comparable, key=lambda row: row["peptide_sequence"]):
        if row["composition_signature"] in used_compositions:
            continue
        add(row, f"three_distinct_compositions_at_length_{comparable_length}")
        used_compositions.add(row["composition_signature"])
        comparable_count += 1
        if comparable_count == 3:
            break
    if comparable_count != 3:
        raise ValueError("unable_to_select_three_distinct_compositions")

    if len(selected) < minimum_size:
        for index in np.linspace(0, len(ordered) - 1, minimum_size, dtype=int):
            add(ordered[int(index)], "deterministic_stratified_fill")
            if len(selected) >= minimum_size:
                break
    return sorted(
        selected.values(),
        key=lambda row: (
            row["sequence_length"],
            row["theoretical_heavy_atom_count"],
            row["peptide_sequence"],
        ),
    )


def _canonical_coordinate_set_sha(payload: dict[str, Any]) -> str:
    material = "|".join(
        row["coordinate_sha256"] for row in payload["conformers"]
    )
    return hashlib.sha256(material.encode("ascii")).hexdigest().upper()


def run_worker(sequence: str, seed: int, output_path: Path) -> None:
    started = time.perf_counter()
    try:
        payload = generate_full_heavy_conformers(
            sequence,
            num_conformers=10,
            base_seed=seed,
        )
    except ConformerGenerationError as error:
        _write_json(output_path, {
            "status": "FAIL",
            "sequence": sequence,
            "sequence_length": len(sequence),
            "failure_type": type(error).__name__,
            "failure_text": str(error),
            "failure_details": error.details,
            "elapsed_seconds": time.perf_counter() - started,
            "target_bound_inputs_used": False,
        })
        return
    validation = validate_payload(payload, expected_conformers=10)
    cpu_forward = cpu_forward_smoke(payload)
    output = {
        "status": "PASS",
        "sequence": sequence,
        "sequence_length": len(sequence),
        "heavy_atom_count": payload["atom_count"],
        "accepted_conformer_count": payload["conformer_count"],
        "total_seconds_to_10_of_10": time.perf_counter() - started,
        "atom_identity_sha256": payload["atom_identity_sha256"],
        "canonical_coordinate_set_sha256": _canonical_coordinate_set_sha(payload),
        "conformers": [
            {
                key: conformer[key]
                for key in (
                    "conformer_index",
                    "attempt_index",
                    "random_seed",
                    "embedding_seconds",
                    "mmff_seconds",
                    "mmff_status",
                    "mmff_energy",
                    "coordinate_sha256",
                    "geometry_audit",
                    "attempt_records",
                )
            }
            for conformer in payload["conformers"]
        ],
        "validation": validation,
        "cpu_egnn_forward": cpu_forward,
        "target_bound_inputs_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def launch_worker(
    sequence: str,
    seed: int,
    output_path: Path,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    stdout_path = output_path.with_suffix(".stdout.log")
    stderr_path = output_path.with_suffix(".stderr.log")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "phase3.drugclip.validate_full_atom_conformer_prototype",
        "--worker-sequence",
        sequence,
        "--worker-seed",
        str(seed),
        "--worker-output",
        str(output_path),
    ]
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=stdout,
            stderr=stderr,
        )
        pid = process.pid
        timed_out = False
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            exit_code = process.wait(timeout=30)
    elapsed = time.perf_counter() - started
    leftover_process = process.poll() is None
    result: dict[str, Any] = {
        "sequence": sequence,
        "pid": pid,
        "command": command,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "leftover_process": leftover_process,
        "output_path": str(output_path.resolve()),
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
        "status": "FAIL",
    }
    if not timed_out and exit_code == 0 and output_path.is_file():
        result["worker_result"] = json.loads(output_path.read_text(encoding="utf-8"))
        result["status"] = result["worker_result"].get("status", "FAIL")
    else:
        result["stderr"] = stderr_path.read_text(encoding="utf-8")
    return result


def run_generation_panel(
    panel: list[dict[str, Any]],
    output_dir: Path,
    *,
    seed: int,
    timeout_seconds: int,
    parallel_workers: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    sequence_results: list[dict[str, Any]] = []
    first_results: dict[int, dict[str, Any]] = {}
    second_results: dict[int, dict[str, Any]] = {}

    def launch(panel_index: int, repeat: int) -> dict[str, Any]:
        sequence = panel[panel_index]["peptide_sequence"]
        sequence_dir = output_dir / "workers" / f"{panel_index:02d}_{sequence}"
        return launch_worker(
            sequence,
            seed,
            sequence_dir / f"run{repeat}.json",
            timeout_seconds=timeout_seconds,
        )

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(launch, panel_index, 1): panel_index
            for panel_index in range(len(panel))
        }
        for future in as_completed(futures):
            panel_index = futures[future]
            first_results[panel_index] = future.result()
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(launch, panel_index, 2): panel_index
            for panel_index, first in first_results.items()
            if first["status"] == "PASS"
        }
        for future in as_completed(futures):
            panel_index = futures[future]
            second_results[panel_index] = future.result()

    for panel_index, panel_row in enumerate(panel):
        first = first_results[panel_index]
        second = second_results.get(panel_index)
        runs.append({"panel_index": panel_index, "repeat": 1, **first})
        if second is not None:
            runs.append({"panel_index": panel_index, "repeat": 2, **second})
        deterministic = bool(
            second
            and second["status"] == "PASS"
            and first["worker_result"]["atom_identity_sha256"]
            == second["worker_result"]["atom_identity_sha256"]
            and first["worker_result"]["canonical_coordinate_set_sha256"]
            == second["worker_result"]["canonical_coordinate_set_sha256"]
        )
        sequence_results.append({
            **panel_row,
            "run1_status": first["status"],
            "run2_status": second["status"] if second else "NOT_RUN",
            "run1_timed_out": first["timed_out"],
            "run2_timed_out": second["timed_out"] if second else False,
            "deterministic_double_run": deterministic,
            "atom_identity_sha256": (
                first.get("worker_result", {}).get("atom_identity_sha256")
            ),
            "canonical_coordinate_set_sha256": (
                first.get("worker_result", {}).get("canonical_coordinate_set_sha256")
            ),
            "run1_total_seconds": (
                first.get("worker_result", {}).get("total_seconds_to_10_of_10")
            ),
            "run2_total_seconds": (
                second.get("worker_result", {}).get("total_seconds_to_10_of_10")
                if second else None
            ),
            "cpu_egnn_forward_status": (
                first.get("worker_result", {})
                .get("cpu_egnn_forward", {})
                .get("status")
            ),
        })
    summary = {
        "panel_sequence_count": len(panel),
        "parallel_worker_limit": parallel_workers,
        "panel_all_run1_pass": all(row["run1_status"] == "PASS" for row in sequence_results),
        "panel_all_run2_pass": all(row["run2_status"] == "PASS" for row in sequence_results),
        "panel_all_double_run_deterministic": all(
            row["deterministic_double_run"] for row in sequence_results
        ),
        "panel_all_cpu_egnn_forward_pass": all(
            row["cpu_egnn_forward_status"] == "PASS" for row in sequence_results
        ),
        "timed_out_run_count": sum(row["timed_out"] for row in runs),
        "leftover_process_count": sum(row["leftover_process"] for row in runs),
        "sequence_results": sequence_results,
    }
    return runs, summary


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _special_rejection_audit(
    chemistry_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_class: dict[str, dict[str, Any]] = {}
    for row in chemistry_rows:
        classification = row["chemistry_classification"]
        if classification != "ordinary_linear_standard":
            by_class.setdefault(classification, row)
    output = []
    for classification, row in sorted(by_class.items()):
        rejected = False
        exception = None
        try:
            if classification == "multiple_cys_unknown":
                generate_full_heavy_conformers(
                    row["peptide_sequence"], num_conformers=1
                )
            else:
                generate_full_heavy_conformers(
                    row["peptide_sequence"],
                    num_conformers=1,
                    chemistry_class=classification,
                )
        except UnsupportedPeptideChemistry as error:
            rejected = True
            exception = f"{type(error).__name__}:{error}"
        output.append({
            "chemistry_classification": classification,
            "interface_pair_id": row["interface_pair_id"],
            "evidence_id": row["evidence_id"],
            "peptide_sequence": row["peptide_sequence"],
            "generation_rejected": rejected,
            "exception": exception,
        })
    return output


def _final_classification(
    chemistry_summary: dict[str, Any],
    generation_summary: dict[str, Any],
    special_rejections: list[dict[str, Any]],
) -> str:
    if chemistry_summary["query_count"] != 512 or not chemistry_summary[
        "ordinary_linear_safe_query_count"
    ]:
        return "CHEMISTRY_CLASSIFICATION_BLOCKED"
    if not all(row["generation_rejected"] for row in special_rejections):
        return "CHEMISTRY_CLASSIFICATION_BLOCKED"
    if chemistry_summary["fixed512_at_or_above_192_atom_count"]:
        return "ATOM_CAP_BLOCKED"
    if generation_summary["leftover_process_count"]:
        return "PERFORMANCE_BLOCKED"
    if generation_summary["timed_out_run_count"]:
        return "PERFORMANCE_BLOCKED"
    if not (
        generation_summary["panel_all_run1_pass"]
        and generation_summary["panel_all_run2_pass"]
    ):
        return "GENERATION_COVERAGE_FAIL"
    if not generation_summary["panel_all_double_run_deterministic"]:
        return "DETERMINISM_FAIL"
    if not generation_summary["panel_all_cpu_egnn_forward_pass"]:
        return "MODEL_INPUT_FAIL"
    return "FEASIBLE_WITH_EXISTING_LOCAL_STACK"


def run_fixed512_audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not args.resume_existing_output:
            raise FileExistsError(f"prototype_output_directory_exists:{output_dir}")
        if (output_dir / "final_report.json").exists():
            raise FileExistsError("cannot_resume_completed_prototype_output")
        required = {
            "fixed512_chemistry_audit.jsonl",
            "fixed512_chemistry_summary.json",
            "generation_panel.json",
            "special_chemistry_rejection_audit.json",
        }
        present = {
            path.name for path in output_dir.iterdir() if path.is_file()
        }
        if not required <= present:
            raise ValueError(
                f"incomplete_resume_evidence_missing:{sorted(required - present)}"
            )
        chemistry_rows = list(
            read_jsonl(output_dir / "fixed512_chemistry_audit.jsonl")
        )
        chemistry_summary = json.loads(
            (output_dir / "fixed512_chemistry_summary.json").read_text(
                encoding="utf-8"
            )
        )
        panel = json.loads(
            (output_dir / "generation_panel.json").read_text(encoding="utf-8")
        )["sequences"]
        special_rejections = json.loads(
            (output_dir / "special_chemistry_rejection_audit.json").read_text(
                encoding="utf-8"
            )
        )
        if len(chemistry_rows) != 512 or chemistry_summary.get("query_count") != 512:
            raise ValueError("resume_chemistry_evidence_not_fixed512")
    else:
        output_dir.mkdir(parents=True)
        chemistry_rows, chemistry_summary = audit_fixed512_chemistry(
            pilot_output=Path(args.pilot_output).resolve(),
            dataset_root=Path(args.dataset_root).resolve(),
            input_variant_audit_path=Path(args.input_variant_audit).resolve(),
            candidate_evidence_jsonl=Path(args.candidate_evidence_jsonl).resolve(),
            expanded_evidence_jsonl=Path(args.expanded_evidence_jsonl).resolve(),
            mmcif_root=Path(args.mmcif_root).resolve(),
            qbiolip_root=Path(args.qbiolip_root).resolve(),
            biolip_root=Path(args.biolip_root).resolve(),
        )
        _write_jsonl(output_dir / "fixed512_chemistry_audit.jsonl", chemistry_rows)
        _write_json(output_dir / "fixed512_chemistry_summary.json", chemistry_summary)
        panel = select_generation_panel(chemistry_rows)
        _write_json(output_dir / "generation_panel.json", {
            "schema_version": "phase3-v2-full-heavy-generation-panel-v1",
            "selection_is_deterministic": True,
            "sequence_count": len(panel),
            "sequences": panel,
        })
        special_rejections = _special_rejection_audit(chemistry_rows)
        _write_json(
            output_dir / "special_chemistry_rejection_audit.json",
            special_rejections,
        )
    runs, generation_summary = run_generation_panel(
        panel,
        output_dir,
        seed=args.seed,
        timeout_seconds=args.per_peptide_timeout_seconds,
        parallel_workers=args.parallel_workers,
    )
    _write_jsonl(output_dir / "generation_runs.jsonl", runs)
    _write_json(output_dir / "generation_summary.json", generation_summary)
    classification = _final_classification(
        chemistry_summary,
        generation_summary,
        special_rejections,
    )
    final = {
        "schema_version": "phase3-v2-full-heavy-prototype-evidence-v1",
        "classification": classification,
        "prior_boundary": "PROTOTYPE_SMOKE_PASS",
        "fixed512_ready_for_formal_gpu_retrieval": False,
        "retrieval_improvement_claimed": False,
        "formal_data_version_published": False,
        "training_run": False,
        "gpu_retrieval_run": False,
        "chemistry_summary": chemistry_summary,
        "generation_summary": generation_summary,
        "special_chemistry_rejection_audit": special_rejections,
        "atom_cap_contract": {
            "cap": PHASE2_MAX_PEPTIDE_ATOMS,
            "fixed512_at_or_above_cap": chemistry_summary[
                "fixed512_at_or_above_192_atom_count"
            ],
            "fixed512_max_theoretical_heavy_atom_count": chemistry_summary[
                "fixed512_max_theoretical_heavy_atom_count"
            ],
            "prototype_uses_truncation": False,
            "future_full_6979_known_risk": (
                "audit_baseline_reports_at_least_one 197-heavy-atom sequence; "
                "not changed or generated in this prototype"
            ),
        },
        "target_bound_leakage": {
            "structure_used_only_for_eligibility_audit": True,
            "structure_coordinates_used_for_generation": False,
            "generator_allowed_inputs": [
                "peptide_sequence",
                "eligible_ordinary_linear_standard_chemistry",
                "fixed_seed",
                "generator_version",
            ],
            "receptor_interface_contact_bound_pose_used_for_generation": False,
        },
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "final_report.json", final)
    summary_lines = [
        "# Phase-3 v2 full-heavy conformer prototype evidence",
        "",
        f"- Classification: `{classification}`",
        f"- Fixed-512 chemistry rows: {chemistry_summary['query_count']}",
        f"- Ordinary-linear safe queries: {chemistry_summary['ordinary_linear_safe_query_count']}",
        f"- Excluded query fraction: {chemistry_summary['excluded_query_fraction']:.6f}",
        f"- Panel sequences: {generation_summary['panel_sequence_count']}",
        f"- Double-run deterministic: {generation_summary['panel_all_double_run_deterministic']}",
        f"- Per-peptide timeout count: {generation_summary['timed_out_run_count']}",
        f"- Leftover process count: {generation_summary['leftover_process_count']}",
        f"- Fixed-512 at/above 192 atoms: {chemistry_summary['fixed512_at_or_above_192_atom_count']}",
        "- Training/GPU retrieval: not run",
        "- Formal data release: not created",
        "- Retrieval improvement: not evaluated",
        "",
    ]
    (output_dir / "summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="ACDEFG")
    parser.add_argument("--conformers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--worker-sequence")
    parser.add_argument("--worker-seed", type=int, default=20260723)
    parser.add_argument("--worker-output")
    parser.add_argument("--run-fixed512-audit", action="store_true")
    parser.add_argument("--pilot-output")
    parser.add_argument("--dataset-root")
    parser.add_argument("--input-variant-audit")
    parser.add_argument("--candidate-evidence-jsonl")
    parser.add_argument("--expanded-evidence-jsonl")
    parser.add_argument("--mmcif-root")
    parser.add_argument("--qbiolip-root")
    parser.add_argument("--biolip-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--per-peptide-timeout-seconds", type=int, default=600)
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--resume-existing-output", action="store_true")
    args = parser.parse_args()
    if args.worker_sequence:
        if not args.worker_output:
            parser.error("--worker-output is required with --worker-sequence")
        run_worker(
            args.worker_sequence,
            args.worker_seed,
            Path(args.worker_output).resolve(),
        )
        return
    if args.run_fixed512_audit:
        required = [
            "pilot_output",
            "dataset_root",
            "input_variant_audit",
            "candidate_evidence_jsonl",
            "expanded_evidence_jsonl",
            "mmcif_root",
            "qbiolip_root",
            "biolip_root",
            "output_dir",
        ]
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            parser.error(f"fixed512 audit missing arguments: {missing}")
        result = run_fixed512_audit(args)
        print(json.dumps({
            "classification": result["classification"],
            "output_dir": result["output_dir"],
        }, sort_keys=True))
        return
    payload = generate_full_heavy_conformers(
        args.sequence,
        num_conformers=args.conformers,
        base_seed=args.seed,
    )
    result = {
        "status": "PASS",
        "validation": validate_payload(payload, expected_conformers=args.conformers),
        "determinism": verify_determinism(
            args.sequence,
            num_conformers=min(2, args.conformers),
            base_seed=args.seed,
        ),
        "cpu_forward": cpu_forward_smoke(payload),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

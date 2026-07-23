"""Read-only source audit for the Phase-3 train-only torsion prototype."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import gemmi
import numpy as np

from phase2.pepclip.data import ATOM_NAME_TO_ID, ELEMENT_TO_ID, RESIDUE_NAME_TO_ID
from phase3.drugclip.batching import PHASE2_MAX_PEPTIDE_ATOMS
from phase3.drugclip.faspr_full_atom_conformer_prototype import conformer_atoms
from phase3.drugclip.full_atom_conformer_prototype import (
    CHEMISTRY_CLASS,
    REQUIRED_HEAVY_ATOMS,
    UnsupportedPeptideChemistry,
    classify_sequence,
)
from phase3.drugclip.structure_qc import AA1_TO_3, resolve_coordinate_paths
from phase3.drugclip.train_only_torsion_prior_prototype import (
    EXPECTED_CONFORMERS,
    GENERATOR_VERSION,
    PANEL_SEQUENCE_TIMEOUT_SECONDS,
    generate_train_only_faspr_conformers,
    load_torsion_prior,
)
from phase3.drugclip.validate_constrained_full_atom_prototype import (
    PANEL as FIXED_PANEL,
    cpu_egnn_forward_all,
)
from phase3.drugclip.validate_faspr_full_atom_prototype import (
    EXPECTED_FASPR_BINARY_SHA256,
    EXPECTED_FASPR_COMMIT,
    verify_faspr_tool,
)
from phase3.drugclip.validate_full_atom_conformer_prototype import (
    _select_exact_peptide_residues,
    audit_one_chemistry,
)


SCHEMA_VERSION = "phase3-v2-train-only-torsion-source-audit-v1"
TRANS_MAX_DEVIATION_DEGREES = 30.0
MIN_CONTEXT_OBSERVATIONS = 500
SPECIAL_CHEMISTRY_CLASSES = [
    "receptor_covalent",
    "modified_or_nonstandard",
    "chemistry_insufficient",
    "known_disulfide",
    "multiple_cys_unknown",
    "cyclic_or_crosslinked",
]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _dihedral_degrees(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
) -> float:
    b0 = -(second - first)
    b1 = third - second
    b2 = fourth - third
    norm = float(np.linalg.norm(b1))
    if norm <= 1e-12:
        raise ValueError("zero_length_central_dihedral_bond")
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if float(np.linalg.norm(v)) <= 1e-12 or float(np.linalg.norm(w)) <= 1e-12:
        raise ValueError("degenerate_dihedral_plane")
    return _wrap_degrees(math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _atom_xyz(residue: gemmi.Residue, atom_name: str) -> np.ndarray:
    matches = [
        atom
        for atom in residue
        if str(atom.name).strip().upper() == atom_name
        and str(atom.altloc).strip().upper() in {"", "A", "\x00"}
    ]
    if len(matches) != 1:
        raise ValueError(f"backbone_atom_not_1_to_1:{atom_name}:{len(matches)}")
    atom = matches[0]
    return np.asarray([float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)])


def _residue_id(residue: gemmi.Residue) -> str:
    insertion = str(residue.seqid.icode).strip()
    return str(residue.seqid.num) + (
        insertion if insertion and insertion != "?" else ""
    )


def _pdb_id_from_path(path: Path) -> str:
    name = path.name.lower()
    for suffix in (".cif.gz", ".mmcif.gz", ".pdb.gz", ".cif", ".mmcif", ".pdb"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem.lower()


def _existing_coordinate_path(
    row: dict[str, Any],
    mmcif_root: Path,
    qbiolip_root: Path,
    biolip_root: Path,
) -> Path | None:
    if str(row.get("source_database") or "") == "Q-BioLiP_PIII":
        raw = Path(str(row.get("peptide_structure_file") or ""))
        if raw.is_file():
            return raw.resolve()
        basename = raw.name
        candidate = qbiolip_root / "extracted" / "nonredund_lig" / basename
        return candidate.resolve() if candidate.is_file() else None
    raw = Path(str(row.get("complex_structure_file") or ""))
    if raw.is_file():
        return raw.resolve()
    pdb_id = str(row.get("pdb_id") or "").lower()
    candidate = mmcif_root / f"{pdb_id}.cif.gz"
    if candidate.is_file():
        return candidate.resolve()
    candidate = biolip_root / f"{pdb_id}.pdb"
    return candidate.resolve() if candidate.is_file() else None


def _evidence_index(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    output: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for row in read_jsonl(path):
            evidence_id = str(row.get("evidence_id") or "")
            if evidence_id:
                output[evidence_id].append(row)
    return dict(output)


def _select_evidence_row(
    evidence_id: str,
    sequence: str,
    rows: list[dict[str, Any]],
    mmcif_root: Path,
    qbiolip_root: Path,
    biolip_root: Path,
) -> dict[str, Any]:
    compatible = [
        row
        for row in rows
        if str(row.get("peptide_sequence") or "").upper() == sequence
        and row.get("pdb_id")
        and row.get("peptide_chain_id")
    ]
    if not compatible:
        raise ValueError(f"evidence_not_resolved:{evidence_id}")

    def score(row: dict[str, Any]) -> tuple[int, int]:
        path = _existing_coordinate_path(
            row, mmcif_root, qbiolip_root, biolip_root
        )
        populated = sum(
            bool(row.get(field))
            for field in (
                "complex_structure_file",
                "peptide_structure_file",
                "receptor_structure_file",
                "resolution",
                "experimental_method",
                "structure_method",
            )
        )
        return (int(path is not None), populated)

    selected = dict(max(compatible, key=score))
    path = _existing_coordinate_path(
        selected, mmcif_root, qbiolip_root, biolip_root
    )
    if path is not None and str(selected.get("source_database") or "") != "Q-BioLiP_PIII":
        selected["complex_structure_file"] = str(path)
    return selected


def _wildcard_signatures(sequences: Iterable[str]) -> set[str]:
    output: set[str] = set()
    for sequence in sequences:
        sequence = sequence.upper()
        for start in range(max(0, len(sequence) - 7)):
            window = sequence[start : start + 8]
            if len(window) != 8:
                continue
            output.add(window)
            output.update(window[:index] + "*" + window[index + 1 :] for index in range(8))
    return output


def _has_similar_eightmer(sequence: str, signatures: set[str]) -> bool:
    for start in range(max(0, len(sequence) - 7)):
        window = sequence[start : start + 8]
        if len(window) != 8:
            continue
        if window in signatures:
            return True
        if any(
            window[:index] + "*" + window[index + 1 :] in signatures
            for index in range(8)
        ):
            return True
    return False


def _source_key(row: dict[str, Any], sequence: str) -> tuple[str, str, str]:
    return (
        str(row.get("pdb_id") or "").lower(),
        str(row.get("peptide_chain_id") or ""),
        sequence,
    )


def _context_key(sequence: str, residue_index: int) -> str:
    residue = sequence[residue_index]
    pre_pro = residue_index + 1 < len(sequence) and sequence[residue_index + 1] == "P"
    if residue == "P":
        return "PRO"
    if pre_pro:
        return f"{residue}_PRE_PRO"
    if residue == "G":
        return "GLY"
    return residue


def _extract_observations(
    evidence_row: dict[str, Any],
    sequence: str,
    source_path: Path,
    source_sha256: str,
    qbiolip_root: Path,
    biolip_root: Path,
) -> tuple[list[dict[str, Any]], int]:
    _, peptide_path, separate = resolve_coordinate_paths(
        evidence_row, qbiolip_root, biolip_root
    )
    structure = gemmi.read_structure(str(peptide_path))
    residues, chain_id = _select_exact_peptide_residues(
        structure,
        sequence,
        None if separate else str(evidence_row.get("peptide_chain_id") or ""),
    )
    observations: list[dict[str, Any]] = []
    non_trans = 0
    for index in range(1, len(residues) - 1):
        previous = residues[index - 1][1]
        current = residues[index][1]
        following = residues[index + 1][1]
        phi = _dihedral_degrees(
            _atom_xyz(previous, "C"),
            _atom_xyz(current, "N"),
            _atom_xyz(current, "CA"),
            _atom_xyz(current, "C"),
        )
        psi = _dihedral_degrees(
            _atom_xyz(current, "N"),
            _atom_xyz(current, "CA"),
            _atom_xyz(current, "C"),
            _atom_xyz(following, "N"),
        )
        omega = _dihedral_degrees(
            _atom_xyz(previous, "CA"),
            _atom_xyz(previous, "C"),
            _atom_xyz(current, "N"),
            _atom_xyz(current, "CA"),
        )
        if abs(abs(omega) - 180.0) > TRANS_MAX_DEVIATION_DEGREES:
            non_trans += 1
            continue
        residue = sequence[index]
        observations.append(
            {
                "pdb_id": str(evidence_row["pdb_id"]).lower(),
                "chain_id": chain_id,
                "residue_id": _residue_id(current),
                "residue_index": index + 1,
                "residue_name": AA1_TO_3[residue],
                "residue_letter": residue,
                "next_residue_is_pro": sequence[index + 1] == "P",
                "context_key": _context_key(sequence, index),
                "phi_degrees": phi,
                "psi_degrees": psi,
                "omega_degrees": omega,
                "source_file": str(source_path),
                "source_file_sha256": source_sha256,
            }
        )
    return observations, non_trans


def run_source_audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    mmcif_root = Path(args.mmcif_root).resolve()
    qbiolip_root = Path(args.qbiolip_root).resolve()
    biolip_root = Path(args.biolip_root).resolve()
    train_rows = list(read_jsonl(dataset_root / "02_leakage_safe_split" / "train.jsonl"))
    valid_rows = list(read_jsonl(dataset_root / "02_leakage_safe_split" / "valid.jsonl"))
    test_rows = list(read_jsonl(dataset_root / "02_leakage_safe_split" / "test.jsonl"))
    fixed_rows = list(read_jsonl(Path(args.fixed512_input_audit).resolve()))
    fixed_plan = list(read_jsonl(Path(args.fixed512_plan).resolve()))
    evidence = _evidence_index(
        [
            Path(args.candidate_evidence_jsonl).resolve(),
            Path(args.expanded_evidence_jsonl).resolve(),
        ]
    )

    valid_pdbs = {
        str(pdb_id).lower()
        for row in valid_rows
        for pdb_id in row.get("structure_pdb_ids", [])
    }
    test_pdbs = {
        str(pdb_id).lower()
        for row in test_rows
        for pdb_id in row.get("structure_pdb_ids", [])
    }
    fixed_pdbs = {
        _pdb_id_from_path(Path(str(row["structure_path"]))) for row in fixed_rows
    } | {
        str(row.get("evidence_id") or "").split(":", 1)[0].lower()
        for row in fixed_rows
        if row.get("evidence_id")
    }
    evaluation_sequences = {
        str(row["peptide_sequence"]).upper() for row in valid_rows + test_rows
    } | {str(row["peptide_sequence"]).upper() for row in fixed_plan}
    similar_signatures = _wildcard_signatures(evaluation_sequences)

    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    unresolved_evidence = 0
    for pair in train_rows:
        sequence = str(pair["peptide_sequence"]).upper()
        for evidence_id in pair.get("evidence_ids", []):
            try:
                row = _select_evidence_row(
                    str(evidence_id),
                    sequence,
                    evidence.get(str(evidence_id), []),
                    mmcif_root,
                    qbiolip_root,
                    biolip_root,
                )
            except ValueError:
                unresolved_evidence += 1
                continue
            row["_evidence_id"] = str(evidence_id)
            candidates.setdefault(_source_key(row, sequence), row)

    shared_valid = {
        key for key in candidates if key[0] in valid_pdbs
    }
    shared_test = {
        key for key in candidates if key[0] in test_pdbs
    }
    shared_fixed = {
        key for key in candidates if key[0] in fixed_pdbs
    }
    exact_evaluation = {
        key for key in candidates if key[2] in evaluation_sequences
    }
    similar_evaluation = {
        key
        for key in candidates
        if key not in exact_evaluation
        and _has_similar_eightmer(key[2], similar_signatures)
    }

    source_audits: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    file_hash_cache: dict[Path, str] = {}
    exclusion_counts: Counter[str] = Counter()
    non_trans_observations = 0
    accepted_source_paths: set[str] = set()
    accepted_pdbs: set[str] = set()
    accepted_sequences: set[str] = set()
    for key in sorted(candidates):
        row = candidates[key]
        sequence = key[2]
        reason: str | None = None
        if key in shared_valid:
            reason = "shared_validation_pdb"
        elif key in shared_test:
            reason = "shared_test_pdb"
        elif key in shared_fixed:
            reason = "shared_fixed512_pdb"
        elif key in exact_evaluation:
            reason = "exact_evaluation_peptide_sequence"
        elif key in similar_evaluation:
            reason = "evaluation_eightmer_at_least_80pct_identity"
        source_path = _existing_coordinate_path(
            row, mmcif_root, qbiolip_root, biolip_root
        )
        chemistry: dict[str, Any] | None = None
        extracted = 0
        local_non_trans = 0
        if reason is None and source_path is None:
            reason = "missing_local_structure"
        if reason is None:
            try:
                chemistry = audit_one_chemistry(
                    {
                        "interface_pair_id": f"train_source:{key[0]}:{key[1]}",
                        "peptide_sequence": sequence,
                    },
                    {
                        "evidence_id": row["_evidence_id"],
                        "source_database": str(row.get("source_database") or ""),
                        "peptide_chain": str(row.get("peptide_chain_id") or ""),
                        "structure_type": source_path.suffix,
                    },
                    row,
                    qbiolip_root,
                    biolip_root,
                )
            except Exception as exc:
                reason = f"chemistry_audit_error:{type(exc).__name__}:{exc}"
            else:
                if chemistry["chemistry_classification"] != "ordinary_linear_standard":
                    reason = f"chemistry:{chemistry['chemistry_classification']}"
        if reason is None and source_path is not None:
            source_sha = file_hash_cache.setdefault(source_path, file_sha256(source_path))
            try:
                extracted_rows, local_non_trans = _extract_observations(
                    row,
                    sequence,
                    source_path,
                    source_sha,
                    qbiolip_root,
                    biolip_root,
                )
            except Exception as exc:
                reason = f"torsion_extraction_error:{type(exc).__name__}:{exc}"
            else:
                observations.extend(extracted_rows)
                extracted = len(extracted_rows)
                non_trans_observations += local_non_trans
                accepted_source_paths.add(str(source_path))
                accepted_pdbs.add(key[0])
                accepted_sequences.add(sequence)
        if reason is not None:
            exclusion_counts[reason.split(":", 1)[0] if reason.startswith("chemistry:") else reason] += 1
        source_audits.append(
            {
                "pdb_id": key[0],
                "peptide_chain_id": key[1],
                "peptide_sequence": sequence,
                "evidence_id": row["_evidence_id"],
                "source_database": str(row.get("source_database") or ""),
                "source_path": str(source_path) if source_path else None,
                "included": reason is None,
                "exclusion_reason": reason,
                "chemistry_classification": (
                    chemistry["chemistry_classification"] if chemistry else None
                ),
                "trans_observation_count": extracted,
                "non_trans_observation_count": local_non_trans,
            }
        )

    residue_counts = Counter(row["residue_letter"] for row in observations)
    context_counts = Counter(row["context_key"] for row in observations)
    pro_count = residue_counts["P"]
    gly_count = residue_counts["G"]
    pre_pro_count = sum(row["next_residue_is_pro"] for row in observations)
    coverage_pass = min(pro_count, gly_count, pre_pro_count) >= MIN_CONTEXT_OBSERVATIONS
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if coverage_pass else "TORSION_PRIOR_COVERAGE_BLOCKED",
        "classification": (
            "SOURCE_AUDIT_PASS"
            if coverage_pass
            else "TORSION_PRIOR_COVERAGE_BLOCKED"
        ),
        "trans_max_deviation_degrees": TRANS_MAX_DEVIATION_DEGREES,
        "minimum_required_context_observations": MIN_CONTEXT_OBSERVATIONS,
        "formal_split_counts": {
            "train_pairs": len(train_rows),
            "valid_pairs": len(valid_rows),
            "test_pairs": len(test_rows),
            "train_unique_sequences": len({
                str(row["peptide_sequence"]).upper() for row in train_rows
            }),
        },
        "candidate_train_source_count": len(candidates),
        "candidate_train_pdb_count": len({key[0] for key in candidates}),
        "unresolved_evidence_count": unresolved_evidence,
        "exclusion_overlap_counts": {
            "validation_pdb": len(shared_valid),
            "test_pdb": len(shared_test),
            "fixed512_pdb": len(shared_fixed),
            "exact_evaluation_sequence": len(exact_evaluation),
            "evaluation_eightmer_at_least_80pct_identity": len(similar_evaluation),
        },
        "exclusive_exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "accepted_source_count": sum(row["included"] for row in source_audits),
        "accepted_pdb_count": len(accepted_pdbs),
        "accepted_structure_file_count": len(accepted_source_paths),
        "accepted_unique_peptide_sequence_count": len(accepted_sequences),
        "trans_observation_count": len(observations),
        "non_trans_observation_count": non_trans_observations,
        "residue_observation_counts": {
            residue: residue_counts.get(residue, 0) for residue in sorted(AA1_TO_3)
        },
        "context_observation_counts": dict(sorted(context_counts.items())),
        "pro_observation_count": pro_count,
        "gly_observation_count": gly_count,
        "pre_pro_observation_count": pre_pro_count,
        "coverage_threshold_pass": coverage_pass,
        "source_audit_canonical_sha256": canonical_json_sha256(source_audits),
        "observation_canonical_sha256": canonical_json_sha256(observations),
        "input_files": {
            "dataset_manifest": {
                "path": str(dataset_root / "DATA_MANIFEST.json"),
                "sha256": file_sha256(dataset_root / "DATA_MANIFEST.json"),
            },
            "train_split": {
                "path": str(dataset_root / "02_leakage_safe_split" / "train.jsonl"),
                "sha256": file_sha256(dataset_root / "02_leakage_safe_split" / "train.jsonl"),
            },
            "valid_split": {
                "path": str(dataset_root / "02_leakage_safe_split" / "valid.jsonl"),
                "sha256": file_sha256(dataset_root / "02_leakage_safe_split" / "valid.jsonl"),
            },
            "test_split": {
                "path": str(dataset_root / "02_leakage_safe_split" / "test.jsonl"),
                "sha256": file_sha256(dataset_root / "02_leakage_safe_split" / "test.jsonl"),
            },
            "fixed512_plan": {
                "path": str(Path(args.fixed512_plan).resolve()),
                "sha256": file_sha256(Path(args.fixed512_plan).resolve()),
            },
            "fixed512_input_audit": {
                "path": str(Path(args.fixed512_input_audit).resolve()),
                "sha256": file_sha256(Path(args.fixed512_input_audit).resolve()),
            },
        },
    }
    write_jsonl(output_dir / "train_source_audit.jsonl", source_audits)
    write_jsonl(output_dir / "train_trans_torsion_observations.jsonl", observations)
    write_json(output_dir / "train_source_audit_summary.json", summary)
    return summary


def _angle_degrees(first: np.ndarray, center: np.ndarray, third: np.ndarray) -> float:
    left = first - center
    right = third - center
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        raise ValueError("zero_length_angle_vector")
    cosine = max(-1.0, min(1.0, float(np.dot(left, right)) / denominator))
    return math.degrees(math.acos(cosine))


def _proline_planarity_audit(payload: dict[str, Any]) -> dict[str, Any]:
    identities = payload["atom_identity"]
    proline_positions = [
        index for index, letter in enumerate(payload["peptide_sequence"], start=1)
        if letter == "P"
    ]
    rows = []
    for conformer in payload["conformers"]:
        coordinates = conformer["coordinates"]
        lookup = {
            (int(identity["residue_index"]), str(identity["atom_name"])): np.asarray(
                coordinates[int(identity["atom_index"])], dtype=np.float64
            )
            for identity in identities
        }
        for position in proline_positions:
            if position == 1:
                continue
            c_prev = lookup[(position - 1, "C")]
            n_atom = lookup[(position, "N")]
            ca_atom = lookup[(position, "CA")]
            cd_atom = lookup[(position, "CD")]
            angle_c_ca = _angle_degrees(c_prev, n_atom, ca_atom)
            angle_ca_cd = _angle_degrees(ca_atom, n_atom, cd_atom)
            angle_cd_c = _angle_degrees(cd_atom, n_atom, c_prev)
            sum_degrees = angle_c_ca + angle_ca_cd + angle_cd_c
            residual = abs(360.0 - sum_degrees)
            if residual > 10.0:
                raise ValueError(
                    f"proline_amide_nitrogen_nonplanar:{position}:"
                    f"{conformer['conformer_index']}:{residual}"
                )
            rows.append(
                {
                    "conformer_index": conformer["conformer_index"],
                    "residue_index": position,
                    "cprev_n_ca_degrees": angle_c_ca,
                    "ca_n_cd_degrees": angle_ca_cd,
                    "cd_n_cprev_degrees": angle_cd_c,
                    "angle_sum_degrees": sum_degrees,
                    "planarity_residual_degrees": residual,
                }
            )
    return {
        "status": "PASS",
        "proline_residue_count": len(proline_positions),
        "checked_proline_conformer_count": len(rows),
        "maximum_planarity_residual_degrees": max(
            (row["planarity_residual_degrees"] for row in rows), default=0.0
        ),
        "maximum_allowed_planarity_residual_degrees": 10.0,
        "rows": rows,
    }


def validate_panel_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["generator_id"] != GENERATOR_VERSION:
        raise ValueError("train_only_generator_version_mismatch")
    if payload["conformer_count"] != EXPECTED_CONFORMERS:
        raise ValueError("train_only_conformer_count_mismatch")
    if payload["atom_count"] > PHASE2_MAX_PEPTIDE_ATOMS:
        raise ValueError(
            f"train_only_atom_cap_exceeded:{payload['atom_count']}:"
            f"{PHASE2_MAX_PEPTIDE_ATOMS}"
        )
    required_by_residue: defaultdict[tuple[int, str], set[str]] = defaultdict(set)
    unknowns = []
    for identity in payload["atom_identity"]:
        residue_key = (
            int(identity["residue_index"]),
            str(identity["residue_name"]),
        )
        required_by_residue[residue_key].add(str(identity["atom_name"]))
        if identity["element"] not in ELEMENT_TO_ID:
            unknowns.append(f"element:{identity['element']}")
        if identity["atom_name"] not in ATOM_NAME_TO_ID:
            unknowns.append(f"atom_name:{identity['atom_name']}")
        if identity["residue_name"] not in RESIDUE_NAME_TO_ID:
            unknowns.append(f"residue_name:{identity['residue_name']}")
    if unknowns:
        raise ValueError(f"train_only_tensorization_unknown:{sorted(set(unknowns))}")
    for (residue_index, residue_name), names in required_by_residue.items():
        required = set(REQUIRED_HEAVY_ATOMS[residue_name])
        if residue_index == len(payload["peptide_sequence"]):
            required.add("OXT")
        if not required <= names:
            raise ValueError(
                f"train_only_required_atoms_missing:{residue_index}:"
                f"{sorted(required - names)}"
            )
    coordinate_hashes = []
    maximum_dihedral_error = 0.0
    maximum_backbone_deviation = 0.0
    maximum_faspr_seconds = 0.0
    for conformer in payload["conformers"]:
        coordinate_hashes.append(conformer["coordinate_sha256"])
        maximum_backbone_deviation = max(
            maximum_backbone_deviation,
            float(conformer["maximum_backbone_deviation_angstrom"]),
        )
        maximum_faspr_seconds = max(
            maximum_faspr_seconds,
            float(conformer["faspr"]["elapsed_seconds"]),
        )
        if int(conformer["faspr"]["exit_code"]) != 0:
            raise ValueError("train_only_faspr_nonzero_exit")
        if conformer["geometry_audit"]["status"] != "PASS":
            raise ValueError("train_only_geometry_not_pass")
        if conformer["oxygen_reconstruction_audit"]["status"] != "PASS":
            raise ValueError("train_only_oxygen_reconstruction_not_pass")
        if (
            conformer["input_backbone_coordinate_sha256"]
            != conformer["output_backbone_coordinate_sha256"]
        ):
            raise ValueError("train_only_fixed_backbone_hash_changed")
        backbone_audit = conformer["train_only_backbone_audit"]
        if backbone_audit["dihedral_convention_audit"]["status"] != "PASS":
            raise ValueError("train_only_dihedral_convention_not_pass")
        maximum_dihedral_error = max(
            maximum_dihedral_error,
            float(
                backbone_audit["dihedral_convention_audit"][
                    "maximum_angular_error_degrees"
                ]
            ),
        )
        contract = backbone_audit["dependency_contract"]
        if any(
            contract[key]
            for key in (
                "target_bound_inputs_used",
                "receptor_inputs_used",
                "interface_or_contact_inputs_used",
                "complete_fragment_matching_used",
            )
        ):
            raise ValueError("train_only_forbidden_generation_input_used")
        atoms = conformer_atoms(payload, conformer["conformer_index"])
        if any(
            not math.isfinite(float(atom[axis]))
            for atom in atoms
            for axis in ("x", "y", "z")
        ):
            raise ValueError("train_only_nonfinite_coordinate")
    if len(set(coordinate_hashes)) != EXPECTED_CONFORMERS:
        raise ValueError("train_only_conformers_not_distinct")
    if maximum_backbone_deviation != 0.0:
        raise ValueError("train_only_backbone_not_exactly_fixed")
    if payload["total_generation_seconds"] > PANEL_SEQUENCE_TIMEOUT_SECONDS:
        raise TimeoutError("train_only_sequence_performance_limit_exceeded")
    planarity = _proline_planarity_audit(payload)
    return {
        "status": "PASS",
        "conformer_count": EXPECTED_CONFORMERS,
        "atom_count": payload["atom_count"],
        "unique_coordinate_hashes": len(set(coordinate_hashes)),
        "maximum_backbone_deviation_angstrom": maximum_backbone_deviation,
        "maximum_dihedral_error_degrees": maximum_dihedral_error,
        "maximum_faspr_conformer_seconds": maximum_faspr_seconds,
        "total_generation_seconds": payload["total_generation_seconds"],
        "pepclip_tensorization_unknown_count": 0,
        "proline_planarity_audit": planarity,
        "target_bound_inputs_used": False,
    }


def _special_chemistry_rejections() -> list[dict[str, Any]]:
    rows = []
    for classification in SPECIAL_CHEMISTRY_CLASSES:
        sequence = "ACDC" if classification == "multiple_cys_unknown" else "SAVTTVVN"
        declared = (
            CHEMISTRY_CLASS
            if classification == "multiple_cys_unknown"
            else classification
        )
        try:
            classify_sequence(sequence, chemistry_class=declared)
        except UnsupportedPeptideChemistry as error:
            rows.append(
                {
                    "chemistry_classification": classification,
                    "status": "PASS",
                    "exception": f"{type(error).__name__}:{error}",
                }
            )
        else:
            raise ValueError(f"special_chemistry_not_rejected:{classification}")
    return rows


def _classification_for_error(error: BaseException) -> str:
    text = f"{type(error).__name__}:{error}".lower()
    if isinstance(error, TimeoutError):
        return "PERFORMANCE_BLOCKED"
    if "dihedral" in text:
        return "DIHEDRAL_CONVENTION_FAIL"
    if "proline" in text:
        return "PROLINE_BACKBONE_FAIL"
    if "faspr" in text or "pack" in text or "geometry" in text:
        return "FASPR_PACKING_FAIL"
    if "determin" in text or "hash" in text:
        return "DETERMINISM_FAIL"
    if "tensor" in text or "atom_cap" in text or "egnn" in text:
        return "MODEL_INPUT_FAIL"
    return "FASPR_PACKING_FAIL"


def run_panel(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_groups, manifest = load_torsion_prior(
        Path(args.prior_jsonl), Path(args.prior_manifest)
    )
    tool = verify_faspr_tool(Path(args.faspr_root))
    faspr_executable = Path(tool["binary_path"])
    panel_rows = []
    final_classification = "TRAIN_ONLY_TORSION_FASPR_PASS"
    for panel_entry in FIXED_PANEL:
        sequence = panel_entry["peptide_sequence"]
        runs = []
        try:
            for run_index in (1, 2):
                payload = generate_train_only_faspr_conformers(
                    sequence,
                    torsion_prior_groups=prior_groups,
                    torsion_prior_manifest=manifest,
                    work_dir=(
                        output_dir / "panel_work" / sequence / f"run_{run_index}"
                    ),
                    faspr_executable=faspr_executable,
                    faspr_commit_sha=tool["commit_sha"],
                    faspr_binary_sha256=tool["binary_sha256"],
                )
                validation = validate_panel_payload(payload)
                egnn = cpu_egnn_forward_all(payload)
                run_path = output_dir / f"{sequence}.run_{run_index}.json"
                write_json(run_path, payload)
                runs.append(
                    {
                        "run_index": run_index,
                        "status": "PASS",
                        "payload_path": str(run_path),
                        "atom_identity_sha256": payload["atom_identity_sha256"],
                        "coordinate_set_sha256": payload[
                            "canonical_coordinate_set_sha256"
                        ],
                        "validation": validation,
                        "cpu_egnn_forward": egnn,
                    }
                )
            deterministic = (
                runs[0]["atom_identity_sha256"]
                == runs[1]["atom_identity_sha256"]
                and runs[0]["coordinate_set_sha256"]
                == runs[1]["coordinate_set_sha256"]
            )
            if not deterministic:
                raise ValueError(f"determinism_hash_mismatch:{sequence}")
            status = "PASS"
            error_text = None
        except Exception as error:
            deterministic = False
            status = "FAIL"
            error_text = f"{type(error).__name__}:{error}"
            error_details = getattr(error, "details", None)
            final_classification = _classification_for_error(error)
        else:
            error_details = None
        panel_rows.append(
            {
                **panel_entry,
                "status": status,
                "runs": runs,
                "deterministic_double_run": deterministic,
                "error": error_text,
                "error_details": error_details,
            }
        )
        write_json(output_dir / "panel_progress.json", panel_rows)
        if status != "PASS":
            break
    special = _special_chemistry_rejections()
    summary = {
        "schema_version": "phase3-v2-train-only-torsion-panel-summary-v1",
        "status": (
            "PASS"
            if final_classification == "TRAIN_ONLY_TORSION_FASPR_PASS"
            else "FAIL"
        ),
        "classification": final_classification,
        "torsion_prior_manifest_sha256": manifest[
            "manifest_canonical_sha256"
        ],
        "panel": panel_rows,
        "special_chemistry_rejections": special,
        "faspr_tool": tool,
        "generator_api_parameters": sorted(
            inspect.signature(
                generate_train_only_faspr_conformers
            ).parameters
        ),
        "target_bound_inputs_used": False,
        "training_or_gpu_retrieval_run": False,
    }
    write_json(output_dir / "train_only_torsion_panel_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-panel", action="store_true")
    parser.add_argument("--dataset-root")
    parser.add_argument("--candidate-evidence-jsonl")
    parser.add_argument("--expanded-evidence-jsonl")
    parser.add_argument("--mmcif-root")
    parser.add_argument("--qbiolip-root")
    parser.add_argument("--biolip-root")
    parser.add_argument("--fixed512-plan")
    parser.add_argument("--fixed512-input-audit")
    parser.add_argument("--prior-jsonl")
    parser.add_argument("--prior-manifest")
    parser.add_argument("--faspr-root")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.run_panel:
        required = ("prior_jsonl", "prior_manifest", "faspr_root")
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            raise ValueError(f"missing_panel_arguments:{missing}")
        summary = run_panel(args)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if summary["status"] == "PASS" else 2
    required = (
        "dataset_root",
        "candidate_evidence_jsonl",
        "expanded_evidence_jsonl",
        "mmcif_root",
        "qbiolip_root",
        "biolip_root",
        "fixed512_plan",
        "fixed512_input_audit",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ValueError(f"missing_source_audit_arguments:{missing}")
    summary = run_source_audit(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["coverage_threshold_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

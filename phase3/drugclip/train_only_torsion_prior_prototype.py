"""Train-only residue-context torsion prior and backbone prototype."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from phase3.drugclip.faspr_full_atom_conformer_prototype import (
    FASPR_CONFORMER_TIMEOUT_SECONDS,
    _canonical_coordinate_set_sha256,
    _pack_one,
    reconstruct_backbone_oxygen,
)
from phase3.drugclip.full_atom_conformer_prototype import (
    CHEMISTRY_CLASS,
    _base_molecule,
    _validate_topology,
    atom_identity_sha256,
    classify_sequence,
)
from phase3.drugclip.random_conformers import (
    CA_C_LENGTH,
    CA_C_N_ANGLE,
    C_N_CA_ANGLE,
    C_N_LENGTH,
    N_CA_C_ANGLE,
    N_CA_LENGTH,
    _place_atom,
)
from phase3.drugclip.structure_qc import AA1_TO_3


GENERATOR_VERSION = "phase3-v2-train-only-residue-context-trans-v1"
PRIOR_SCHEMA_VERSION = "phase3-v2-train-only-torsion-prior-v1"
BACKBONE_SCHEMA_VERSION = "phase3-v2-train-only-backbone-v1"
TRANS_MAX_DEVIATION_DEGREES = 30.0
DIHEDRAL_TOLERANCE_DEGREES = 1e-7
BACKBONE_CLASH_DISTANCE_ANGSTROM = 1.5
MAX_BACKBONE_ATTEMPTS = 100
EXPECTED_CONFORMERS = 10
PANEL_SEQUENCE_TIMEOUT_SECONDS = 60


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )


def _observation_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["context_key"]),
        str(row["pdb_id"]),
        str(row["chain_id"]),
        str(row["residue_id"]),
        int(row["residue_index"]),
        format(float(row["phi_degrees"]), ".12f"),
        format(float(row["psi_degrees"]), ".12f"),
        format(float(row["omega_degrees"]), ".12f"),
        str(row["source_file_sha256"]),
    )


def build_torsion_prior(
    observation_path: Path,
    source_audit_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Canonicalize the audited observations and bind a manifest."""
    observation_path = Path(observation_path).resolve()
    source_audit_summary_path = Path(source_audit_summary_path).resolve()
    output_dir = Path(output_dir).resolve()
    audit_summary = json.loads(source_audit_summary_path.read_text(encoding="utf-8"))
    if not audit_summary.get("coverage_threshold_pass"):
        raise ValueError("train_source_coverage_threshold_not_passed")
    source_observations = list(read_jsonl(observation_path))
    source_canonical_sha = canonical_json_sha256(source_observations)
    if source_canonical_sha != str(audit_summary["observation_canonical_sha256"]):
        raise ValueError(
            "source_observation_canonical_sha256_mismatch:"
            f"{source_canonical_sha}:{audit_summary['observation_canonical_sha256']}"
        )
    observations = sorted(source_observations, key=_observation_sort_key)
    canonical_sha = canonical_json_sha256(observations)
    if any(
        abs(abs(float(row["omega_degrees"])) - 180.0)
        > TRANS_MAX_DEVIATION_DEGREES
        for row in observations
    ):
        raise ValueError("non_trans_observation_in_prior")
    context_counts = Counter(str(row["context_key"]) for row in observations)
    residue_counts = Counter(str(row["residue_letter"]) for row in observations)
    prior_path = output_dir / "torsion_prior.jsonl"
    _write_jsonl(prior_path, observations)
    manifest_core = {
        "schema_version": PRIOR_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "sampling_unit": "one residue-context observation containing joint phi/psi/omega",
        "fragment_matching_used": False,
        "trans_only": True,
        "trans_max_deviation_degrees": TRANS_MAX_DEVIATION_DEGREES,
        "context_contract": {
            "PRO": "all proline residues",
            "GLY": "glycine not immediately preceding proline",
            "X_PRE_PRO": "residue-identity-specific pre-proline context",
            "X": "other residue identity",
        },
        "observation_count": len(observations),
        "context_observation_counts": dict(sorted(context_counts.items())),
        "residue_observation_counts": dict(sorted(residue_counts.items())),
        "prior_jsonl_sha256": file_sha256(prior_path),
        "prior_canonical_sha256": canonical_sha,
        "source_audit_summary": {
            "path": str(source_audit_summary_path),
            "sha256": file_sha256(source_audit_summary_path),
            "canonical_sha256": canonical_json_sha256(audit_summary),
        },
        "source_observation_file": {
            "path": str(observation_path),
            "sha256": file_sha256(observation_path),
            "canonical_sha256": source_canonical_sha,
        },
        "provenance_fields_per_observation": [
            "pdb_id",
            "chain_id",
            "residue_id",
            "residue_index",
            "source_file",
            "source_file_sha256",
        ],
        "allowed_generation_seed_material": [
            "generator_version",
            "torsion_prior_manifest_sha256",
            "peptide_sequence",
            "conformer_index",
        ],
        "forbidden_generation_inputs": [
            "receptor",
            "interface",
            "contact",
            "evidence_id",
            "bound_coordinates",
        ],
    }
    manifest = {
        **manifest_core,
        "manifest_canonical_sha256": canonical_json_sha256(manifest_core),
    }
    _write_json(output_dir / "torsion_prior_manifest.json", manifest)
    return manifest


def load_torsion_prior(
    prior_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = str(manifest.pop("manifest_canonical_sha256"))
    if canonical_json_sha256(manifest) != manifest_sha:
        raise ValueError("torsion_prior_manifest_canonical_sha256_mismatch")
    manifest["manifest_canonical_sha256"] = manifest_sha
    prior_path = Path(prior_path).resolve()
    if file_sha256(prior_path) != str(manifest["prior_jsonl_sha256"]):
        raise ValueError("torsion_prior_file_sha256_mismatch")
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(prior_path):
        groups[str(row["context_key"])].append(row)
    for values in groups.values():
        values.sort(key=_observation_sort_key)
    actual_counts = {key: len(value) for key, value in sorted(groups.items())}
    if actual_counts != manifest["context_observation_counts"]:
        raise ValueError("torsion_prior_context_counts_mismatch")
    return dict(groups), manifest


def context_key(sequence: str, residue_index: int) -> str:
    residue = sequence[residue_index]
    pre_pro = residue_index + 1 < len(sequence) and sequence[residue_index + 1] == "P"
    if residue == "P":
        return "PRO"
    if pre_pro:
        return f"{residue}_PRE_PRO"
    if residue == "G":
        return "GLY"
    return residue


def conformer_seed(
    manifest_sha256: str,
    peptide_sequence: str,
    conformer_index: int,
) -> int:
    material = "|".join(
        [
            GENERATOR_VERSION,
            str(manifest_sha256).upper(),
            peptide_sequence.upper(),
            str(int(conformer_index)),
        ]
    )
    return int(hashlib.sha256(material.encode("ascii")).hexdigest()[:16], 16)


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def angular_error_degrees(measured: float, target: float) -> float:
    return abs(_wrap_degrees(float(measured) - float(target)))


def dihedral_degrees(
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
    return _wrap_degrees(
        math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))
    )


def _sample_torsions(
    sequence: str,
    groups: dict[str, list[dict[str, Any]]],
    rng: random.Random,
) -> tuple[list[float], list[float], list[float], list[dict[str, Any]]]:
    phis: list[float] = []
    psis: list[float] = []
    omegas: list[float] = []
    provenance: list[dict[str, Any]] = []
    for index in range(len(sequence)):
        key = context_key(sequence, index)
        choices = groups.get(key, [])
        if not choices:
            raise ValueError(f"torsion_context_has_no_observations:{key}")
        selected = choices[rng.randrange(len(choices))]
        phis.append(float(selected["phi_degrees"]))
        psis.append(float(selected["psi_degrees"]))
        omegas.append(float(selected["omega_degrees"]))
        provenance.append(
            {
                field: selected[field]
                for field in (
                    "context_key",
                    "pdb_id",
                    "chain_id",
                    "residue_id",
                    "source_file_sha256",
                )
            }
        )
    return phis, psis, omegas, provenance


def _coordinates_from_torsions(
    sequence: str,
    phis: list[float],
    psis: list[float],
    omegas: list[float],
) -> list[dict[str, Any]]:
    n_atoms = [np.asarray([0.0, 0.0, 0.0], dtype=np.float64)]
    ca_atoms = [np.asarray([N_CA_LENGTH, 0.0, 0.0], dtype=np.float64)]
    initial_direction = math.pi - math.radians(N_CA_C_ANGLE)
    c_atoms = [
        ca_atoms[0]
        + CA_C_LENGTH
        * np.asarray([math.cos(initial_direction), math.sin(initial_direction), 0.0])
    ]
    for index in range(len(sequence) - 1):
        next_n = _place_atom(
            n_atoms[index],
            ca_atoms[index],
            c_atoms[index],
            C_N_LENGTH,
            CA_C_N_ANGLE,
            -psis[index],
        )
        next_ca = _place_atom(
            ca_atoms[index],
            c_atoms[index],
            next_n,
            N_CA_LENGTH,
            C_N_CA_ANGLE,
            -omegas[index + 1],
        )
        next_c = _place_atom(
            c_atoms[index],
            next_n,
            next_ca,
            CA_C_LENGTH,
            N_CA_C_ANGLE,
            -phis[index + 1],
        )
        n_atoms.append(next_n)
        ca_atoms.append(next_ca)
        c_atoms.append(next_c)
    output: list[dict[str, Any]] = []
    for residue_index, residue in enumerate(sequence, start=1):
        for atom_name, element, xyz in (
            ("N", "N", n_atoms[residue_index - 1]),
            ("CA", "C", ca_atoms[residue_index - 1]),
            ("C", "C", c_atoms[residue_index - 1]),
        ):
            output.append(
                {
                    "residue_id": f"P:{residue_index}",
                    "residue_index": residue_index,
                    "residue_name": AA1_TO_3[residue],
                    "atom_name": atom_name,
                    "element": element,
                    "x": float(xyz[0]),
                    "y": float(xyz[1]),
                    "z": float(xyz[2]),
                }
            )
    return output


def _lookup_ncac(
    backbone: list[dict[str, Any]], sequence_length: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    n_atoms, ca_atoms, c_atoms = [], [], []
    for index in range(sequence_length):
        rows = backbone[index * 3 : index * 3 + 3]
        if [row["atom_name"] for row in rows] != ["N", "CA", "C"]:
            raise ValueError(f"noncanonical_backbone_order:{index + 1}")
        xyz = [
            np.asarray([row["x"], row["y"], row["z"]], dtype=np.float64)
            for row in rows
        ]
        n_atoms.append(xyz[0])
        ca_atoms.append(xyz[1])
        c_atoms.append(xyz[2])
    return n_atoms, ca_atoms, c_atoms


def measured_torsions(
    backbone: list[dict[str, Any]], sequence_length: int
) -> dict[str, list[float | None]]:
    n_atoms, ca_atoms, c_atoms = _lookup_ncac(backbone, sequence_length)
    phi: list[float | None] = [None]
    psi: list[float | None] = []
    omega: list[float | None] = [None]
    for index in range(sequence_length):
        if index > 0:
            phi.append(
                dihedral_degrees(
                    c_atoms[index - 1],
                    n_atoms[index],
                    ca_atoms[index],
                    c_atoms[index],
                )
            )
            omega.append(
                dihedral_degrees(
                    ca_atoms[index - 1],
                    c_atoms[index - 1],
                    n_atoms[index],
                    ca_atoms[index],
                )
            )
        psi.append(
            dihedral_degrees(
                n_atoms[index],
                ca_atoms[index],
                c_atoms[index],
                n_atoms[index + 1],
            )
            if index + 1 < sequence_length
            else None
        )
    return {"phi": phi, "psi": psi, "omega": omega}


def _dihedral_audit(
    sampled: dict[str, list[float]],
    measured: dict[str, list[float | None]],
) -> dict[str, Any]:
    rows = []
    maximum = 0.0
    for name in ("phi", "psi", "omega"):
        for index, target in enumerate(sampled[name]):
            actual = measured[name][index]
            if actual is None:
                continue
            error = angular_error_degrees(actual, target)
            maximum = max(maximum, error)
            rows.append(
                {
                    "torsion": name,
                    "residue_index": index + 1,
                    "sampled_degrees": target,
                    "measured_degrees": actual,
                    "angular_error_degrees": error,
                }
            )
    if maximum > DIHEDRAL_TOLERANCE_DEGREES:
        raise ValueError(f"dihedral_convention_mismatch:{maximum}")
    return {
        "status": "PASS",
        "maximum_angular_error_degrees": maximum,
        "tolerance_degrees": DIHEDRAL_TOLERANCE_DEGREES,
        "rows": rows,
    }


def _backbone_clash_free(backbone: list[dict[str, Any]]) -> bool:
    for left_index, left in enumerate(backbone):
        left_residue = int(left["residue_index"])
        left_xyz = np.asarray([left["x"], left["y"], left["z"]])
        for right in backbone[left_index + 1 :]:
            right_residue = int(right["residue_index"])
            if abs(left_residue - right_residue) <= 2:
                continue
            right_xyz = np.asarray([right["x"], right["y"], right["z"]])
            if float(np.linalg.norm(left_xyz - right_xyz)) < BACKBONE_CLASH_DISTANCE_ANGSTROM:
                return False
    return True


def backbone_coordinate_sha256(backbone: list[dict[str, Any]]) -> str:
    material = "|".join(
        ":".join(
            [
                str(row["residue_index"]),
                str(row["atom_name"]),
                format(float(row["x"]), ".12f"),
                format(float(row["y"]), ".12f"),
                format(float(row["z"]), ".12f"),
            ]
        )
        for row in backbone
    )
    return hashlib.sha256(material.encode("ascii")).hexdigest().upper()


def generate_backbone(
    sequence: str,
    conformer_index: int,
    groups: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sequence = sequence.strip().upper()
    if not sequence or any(letter not in AA1_TO_3 for letter in sequence):
        raise ValueError("sequence_not_standard_amino_acids")
    seed = conformer_seed(
        str(manifest["manifest_canonical_sha256"]), sequence, conformer_index
    )
    rng = random.Random(seed)
    started = time.perf_counter()
    for attempt in range(1, MAX_BACKBONE_ATTEMPTS + 1):
        phis, psis, omegas, provenance = _sample_torsions(sequence, groups, rng)
        backbone = _coordinates_from_torsions(sequence, phis, psis, omegas)
        if not _backbone_clash_free(backbone):
            continue
        measured = measured_torsions(backbone, len(sequence))
        audit = _dihedral_audit(
            {"phi": phis, "psi": psis, "omega": omegas}, measured
        )
        ncaco, oxygen_audit = reconstruct_backbone_oxygen(sequence, backbone)
        return backbone, {
            "schema_version": BACKBONE_SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "torsion_prior_manifest_sha256": manifest[
                "manifest_canonical_sha256"
            ],
            "peptide_sequence": sequence,
            "conformer_index": int(conformer_index),
            "seed": seed,
            "attempt_index": attempt,
            "sampled_torsions": {
                "phi": phis,
                "psi": psis,
                "omega": omegas,
            },
            "sampled_observation_provenance": provenance,
            "measured_torsions": measured,
            "dihedral_convention_audit": audit,
            "backbone_coordinate_sha256": backbone_coordinate_sha256(backbone),
            "backbone_ncaco_coordinate_sha256": backbone_coordinate_sha256(ncaco),
            "oxygen_reconstruction_audit": oxygen_audit,
            "backbone_generation_seconds": time.perf_counter() - started,
            "dependency_contract": {
                "target_bound_inputs_used": False,
                "receptor_inputs_used": False,
                "interface_or_contact_inputs_used": False,
                "complete_fragment_matching_used": False,
            },
        }
    raise RuntimeError(
        f"backbone_generation_attempts_exhausted:{sequence}:{conformer_index}:"
        f"{MAX_BACKBONE_ATTEMPTS}"
    )


def generate_train_only_faspr_conformers(
    sequence: str,
    *,
    torsion_prior_groups: dict[str, list[dict[str, Any]]],
    torsion_prior_manifest: dict[str, Any],
    work_dir: Path,
    faspr_executable: Path,
    faspr_commit_sha: str,
    faspr_binary_sha256: str,
    chemistry_class: str = CHEMISTRY_CLASS,
    num_conformers: int = EXPECTED_CONFORMERS,
) -> dict[str, Any]:
    """Generate residue-context backbones and pack fixed-backbone side chains.

    The API intentionally has no receptor, evidence, interface, contact, or
    bound-coordinate input.  Its only structural prior is the already audited
    train-only torsion observation collection.
    """
    chemistry = classify_sequence(sequence, chemistry_class=chemistry_class)
    normalized = chemistry["sequence"]
    if int(num_conformers) != EXPECTED_CONFORMERS:
        raise ValueError(f"exactly_{EXPECTED_CONFORMERS}_conformers_required")
    executable = Path(faspr_executable).resolve()
    library = executable.parent / "dun2010bbdep.bin"
    if not executable.is_file() or not library.is_file():
        raise ValueError(
            f"faspr_binary_or_rotamer_library_missing:{executable}:{library}"
        )
    actual_binary_sha = file_sha256(executable)
    if actual_binary_sha != str(faspr_binary_sha256).upper():
        raise ValueError(
            f"faspr_binary_sha256_mismatch:{actual_binary_sha}:"
            f"{str(faspr_binary_sha256).upper()}"
        )
    base = _base_molecule(normalized)
    identities = _validate_topology(base, normalized)
    output_dir = Path(work_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conformers = []
    started = time.perf_counter()
    for conformer_index in range(EXPECTED_CONFORMERS):
        backbone, backbone_audit = generate_backbone(
            normalized,
            conformer_index,
            torsion_prior_groups,
            torsion_prior_manifest,
        )
        packed = _pack_one(
            normalized,
            base,
            identities,
            backbone,
            conformer_index,
            output_dir,
            executable,
        )
        conformers.append(
            {
                **packed,
                "train_only_backbone_audit": backbone_audit,
                "total_conformer_seconds": (
                    backbone_audit["backbone_generation_seconds"]
                    + packed["faspr"]["elapsed_seconds"]
                ),
            }
        )
    elapsed = time.perf_counter() - started
    if elapsed > PANEL_SEQUENCE_TIMEOUT_SECONDS:
        raise TimeoutError(
            f"sequence_generation_exceeded_{PANEL_SEQUENCE_TIMEOUT_SECONDS}s:"
            f"{normalized}:{elapsed}"
        )
    hashes = [row["coordinate_sha256"] for row in conformers]
    if len(set(hashes)) != EXPECTED_CONFORMERS:
        raise ValueError("train_only_conformer_coordinates_not_unique")
    return {
        "schema_version": "phase3-v2-train-only-torsion-faspr-panel-v1",
        "generator_id": GENERATOR_VERSION,
        "generator_version": {
            "backbone": GENERATOR_VERSION,
            "torsion_prior_manifest_sha256": torsion_prior_manifest[
                "manifest_canonical_sha256"
            ],
            "sidechain_packer": "FASPR-fixed-backbone",
            "faspr_commit_sha": str(faspr_commit_sha),
            "faspr_binary_sha256": actual_binary_sha,
            "faspr_per_conformer_timeout_seconds": (
                FASPR_CONFORMER_TIMEOUT_SECONDS
            ),
            "sequence_timeout_seconds": PANEL_SEQUENCE_TIMEOUT_SECONDS,
        },
        "peptide_sequence": normalized,
        "chemistry": chemistry,
        "atom_count": len(identities),
        "atom_identity": identities,
        "atom_identity_sha256": atom_identity_sha256(identities),
        "conformer_count": len(conformers),
        "canonical_coordinate_set_sha256": (
            _canonical_coordinate_set_sha256(conformers)
        ),
        "conformers": conformers,
        "total_generation_seconds": elapsed,
        "dependency_contract": {
            "allowed_inputs": [
                "peptide_sequence",
                "ordinary_linear_standard_chemistry",
                "train_only_torsion_prior_manifest",
                "conformer_index",
                "faspr_version_and_binary",
            ],
            "target_bound_inputs_used": False,
            "receptor_inputs_used": False,
            "interface_or_contact_inputs_used": False,
            "evaluation_fragment_matching_used": False,
        },
    }

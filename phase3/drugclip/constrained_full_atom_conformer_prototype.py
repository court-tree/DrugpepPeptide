"""Fixed-backbone, candidate-independent full-heavy peptide prototype.

The N/CA/C backbone comes only from the existing Phase-3 random-backbone
generator. RDKit supplies full peptide topology and initializes the remaining
atoms under an exact coordinate map. MMFF94s then optimizes with every N/CA/C
atom fixed.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Callable

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from phase3.drugclip.full_atom_conformer_prototype import (
    CHEMISTRY_CLASS,
    _base_molecule,
    _geometry_audit,
    _validate_topology,
    atom_identity_sha256,
    classify_sequence,
)
from phase3.drugclip.random_conformer_v3 import (
    coordinate_sha256 as backbone_coordinate_sha256,
    generate_from_seed,
)


SCHEMA_VERSION = "pepclip-constrained-full-heavy-conformer-prototype-v1"
GENERATOR_ID = "formal-v3-seeded-ncac-rdkit-constrained-completion-mmff94s-v2"
EXPECTED_CONFORMERS = 10
MAX_COMPLETION_ATTEMPTS = 10
MMFF_MAX_ITERATIONS = 500
BACKBONE_COORDINATE_TOLERANCE_ANGSTROM = 1.0e-8


class ConstrainedCompletionError(RuntimeError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class BackboneConstraintError(ConstrainedCompletionError):
    pass


class SidechainCompletionError(ConstrainedCompletionError):
    pass


class OptimizationCoverageError(ConstrainedCompletionError):
    pass


def _completion_seed(
    sequence: str,
    base_seed: int,
    conformer_index: int,
    attempt_index: int,
) -> int:
    material = "|".join([
        GENERATOR_ID,
        sequence,
        str(int(base_seed)),
        str(int(conformer_index)),
        str(int(attempt_index)),
    ]).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") % 2147483647 or 1


def _canonical_coordinate_set_sha256(conformers: list[dict[str, Any]]) -> str:
    material = "|".join(row["coordinate_sha256"] for row in conformers)
    return hashlib.sha256(material.encode("ascii")).hexdigest().upper()


def _coordinate_sha256(coordinates: list[list[float]]) -> str:
    canonical = ";".join(
        ",".join(format(value, ".12f") for value in xyz)
        for xyz in coordinates
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest().upper()


def _generate_backbone(
    sequence: str,
    conformer_index: int,
    backbone_seed_plan: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    matches = [
        row for row in backbone_seed_plan
        if int(row["conformer_index"]) == conformer_index
    ]
    if len(matches) != 1:
        raise BackboneConstraintError(
            f"backbone_seed_plan_not_1_to_1:{conformer_index}:{len(matches)}"
        )
    planned = matches[0]
    seed = int(planned["seed"])
    backbone = generate_from_seed(sequence, seed)
    actual_sha = backbone_coordinate_sha256(backbone)
    expected_sha = str(planned["backbone_coordinate_sha256"])
    if actual_sha != expected_sha:
        raise BackboneConstraintError(
            "formal_v3_backbone_seed_reproduction_failed",
            details={
                "conformer_index": conformer_index,
                "seed": seed,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
            },
        )
    return backbone, {
        "backbone_generation_seconds": time.perf_counter() - started,
        "backbone_seed": seed,
        "formal_v3_split": str(planned["split"]),
        "formal_v3_attempt_index": int(planned["attempt_index"]),
        "backbone_coordinate_sha256": actual_sha,
        "formal_v3_seed_reproduction_pass": True,
    }


def _backbone_coord_map(
    molecule: Chem.Mol,
    backbone: list[dict[str, Any]],
    sequence_length: int,
) -> tuple[dict[int, Point3D], list[dict[str, Any]]]:
    lookup: dict[tuple[int, str], int] = {}
    for atom in molecule.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        lookup[(int(info.GetResidueNumber()), info.GetName().strip().upper())] = atom.GetIdx()
    coord_map: dict[int, Point3D] = {}
    audit: list[dict[str, Any]] = []
    expected = 3 * sequence_length
    if len(backbone) != expected:
        raise BackboneConstraintError(
            f"backbone_atom_count_mismatch:{len(backbone)}:{expected}"
        )
    for residue_index in range(1, sequence_length + 1):
        for offset, atom_name in enumerate(("N", "CA", "C")):
            source = backbone[(residue_index - 1) * 3 + offset]
            if str(source["atom_name"]).upper() != atom_name:
                raise BackboneConstraintError(
                    f"backbone_atom_order_mismatch:{residue_index}:{atom_name}"
                )
            atom_index = lookup.get((residue_index, atom_name))
            if atom_index is None:
                raise BackboneConstraintError(
                    f"rdkit_backbone_atom_mapping_missing:{residue_index}:{atom_name}"
                )
            xyz = [float(source[axis]) for axis in ("x", "y", "z")]
            coord_map[atom_index] = Point3D(*xyz)
            audit.append({
                "atom_index": atom_index,
                "residue_index": residue_index,
                "atom_name": atom_name,
                "input_coordinates": xyz,
            })
    return coord_map, audit


def _backbone_deviation(
    molecule: Chem.Mol,
    coord_map: dict[int, Point3D],
) -> dict[str, float]:
    conformer = molecule.GetConformer()
    distances = []
    for atom_index, expected in coord_map.items():
        observed = conformer.GetAtomPosition(atom_index)
        distances.append(math.dist(
            (expected.x, expected.y, expected.z),
            (observed.x, observed.y, observed.z),
        ))
    return {
        "maximum_angstrom": max(distances, default=0.0),
        "rms_angstrom": (
            math.sqrt(sum(value * value for value in distances) / len(distances))
            if distances else 0.0
        ),
    }


def _attempt_completion(
    base: Chem.Mol,
    sequence: str,
    backbone: list[dict[str, Any]],
    *,
    base_seed: int,
    conformer_index: int,
    attempt_index: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    molecule = Chem.AddHs(Chem.Mol(base))
    coord_map, mapping_audit = _backbone_coord_map(
        molecule, backbone, len(sequence)
    )
    seed = _completion_seed(
        sequence, base_seed, conformer_index, attempt_index
    )
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.numThreads = 1
    parameters.useRandomCoords = True
    parameters.clearConfs = True
    parameters.SetCoordMap(coord_map)
    completion_started = time.perf_counter()
    embed_status = int(AllChem.EmbedMolecule(molecule, parameters))
    completion_seconds = time.perf_counter() - completion_started
    attempt: dict[str, Any] = {
        "attempt_index": attempt_index,
        "completion_seed": seed,
        "sidechain_completion_seconds": completion_seconds,
        "embedding_status": embed_status,
        "optimization_seconds": None,
        "mmff_status": None,
        "rejection_reason": None,
    }
    if embed_status != 0:
        attempt["rejection_reason"] = "constrained_embedding_failed"
        return None, attempt

    before = _backbone_deviation(molecule, coord_map)
    attempt["backbone_deviation_after_embedding"] = before
    if before["maximum_angstrom"] > BACKBONE_COORDINATE_TOLERANCE_ANGSTROM:
        raise BackboneConstraintError(
            "constrained_embedding_moved_backbone",
            details={
                "conformer_index": conformer_index,
                "attempt_index": attempt_index,
                "deviation": before,
                "tolerance_angstrom": BACKBONE_COORDINATE_TOLERANCE_ANGSTROM,
            },
        )
    if not AllChem.MMFFHasAllMoleculeParams(molecule):
        raise SidechainCompletionError(
            "mmff_parameters_incomplete",
            details={"conformer_index": conformer_index},
        )
    properties = AllChem.MMFFGetMoleculeProperties(
        molecule, mmffVariant="MMFF94s"
    )
    force_field = AllChem.MMFFGetMoleculeForceField(
        molecule, properties, confId=0
    )
    for atom_index in coord_map:
        force_field.AddFixedPoint(atom_index)
    optimization_started = time.perf_counter()
    mmff_status = int(force_field.Minimize(maxIts=MMFF_MAX_ITERATIONS))
    optimization_seconds = time.perf_counter() - optimization_started
    attempt["optimization_seconds"] = optimization_seconds
    attempt["mmff_status"] = mmff_status
    after = _backbone_deviation(molecule, coord_map)
    attempt["backbone_deviation_after_optimization"] = after
    if after["maximum_angstrom"] > BACKBONE_COORDINATE_TOLERANCE_ANGSTROM:
        raise BackboneConstraintError(
            "fixed_point_optimization_moved_backbone",
            details={
                "conformer_index": conformer_index,
                "attempt_index": attempt_index,
                "deviation": after,
                "tolerance_angstrom": BACKBONE_COORDINATE_TOLERANCE_ANGSTROM,
            },
        )
    if mmff_status != 0:
        attempt["rejection_reason"] = f"local_mmff_not_converged:{mmff_status}"
        return None, attempt

    heavy_atom_count = base.GetNumAtoms()
    conformer = molecule.GetConformer()
    coordinates = [
        [
            float(conformer.GetAtomPosition(atom_index).x),
            float(conformer.GetAtomPosition(atom_index).y),
            float(conformer.GetAtomPosition(atom_index).z),
        ]
        for atom_index in range(heavy_atom_count)
    ]
    try:
        geometry = _geometry_audit(molecule, coordinates)
    except Exception as error:
        attempt["rejection_reason"] = (
            f"geometry_rejected:{type(error).__name__}:{error}"
        )
        return None, attempt
    output_backbone = []
    for row in mapping_audit:
        xyz = coordinates[row["atom_index"]]
        output_backbone.append({
            "residue_id": f"P:{row['residue_index']}",
            "atom_name": row["atom_name"],
            "x": xyz[0],
            "y": xyz[1],
            "z": xyz[2],
        })
    output_backbone_sha = backbone_coordinate_sha256(output_backbone)
    input_backbone_sha = backbone_coordinate_sha256(backbone)
    if output_backbone_sha != input_backbone_sha:
        raise BackboneConstraintError(
            "backbone_coordinate_hash_changed",
            details={
                "input_sha256": input_backbone_sha,
                "output_sha256": output_backbone_sha,
            },
        )
    return {
        "conformer_index": conformer_index,
        "completion_attempt_index": attempt_index,
        "completion_seed": seed,
        "sidechain_completion_seconds": completion_seconds,
        "optimization_seconds": optimization_seconds,
        "mmff_status": mmff_status,
        "mmff_energy": float(force_field.CalcEnergy()),
        "backbone_deviation_after_embedding": before,
        "backbone_deviation_after_optimization": after,
        "input_backbone_coordinate_sha256": input_backbone_sha,
        "output_backbone_coordinate_sha256": output_backbone_sha,
        "coordinate_sha256": _coordinate_sha256(coordinates),
        "geometry_audit": geometry,
        "coordinates": coordinates,
    }, attempt


def generate_constrained_full_atom_conformers(
    sequence: str,
    *,
    num_conformers: int = EXPECTED_CONFORMERS,
    base_seed: int = 20260723,
    chemistry_class: str = CHEMISTRY_CLASS,
    backbone_seed_plan: list[dict[str, Any]],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not 1 <= int(num_conformers) <= EXPECTED_CONFORMERS:
        raise ValueError("num_conformers_must_be_between_1_and_10")
    chemistry = classify_sequence(
        sequence, chemistry_class=chemistry_class
    )
    normalized = chemistry["sequence"]
    base = _base_molecule(normalized)
    identities = _validate_topology(base, normalized)
    identity_sha = atom_identity_sha256(identities)
    if len(backbone_seed_plan) < int(num_conformers):
        raise BackboneConstraintError("backbone_seed_plan_too_short")
    conformers: list[dict[str, Any]] = []
    started = time.perf_counter()
    for conformer_index in range(int(num_conformers)):
        backbone, backbone_audit = _generate_backbone(
            normalized, conformer_index, backbone_seed_plan
        )
        attempts: list[dict[str, Any]] = []
        accepted = None
        for attempt_index in range(MAX_COMPLETION_ATTEMPTS):
            candidate, attempt = _attempt_completion(
                base,
                normalized,
                backbone,
                base_seed=int(base_seed),
                conformer_index=conformer_index,
                attempt_index=attempt_index,
            )
            attempts.append(attempt)
            if candidate is not None:
                accepted = {
                    **candidate,
                    **backbone_audit,
                    "completion_attempts": attempts,
                    "total_conformer_seconds": (
                        backbone_audit["backbone_generation_seconds"]
                        + sum(
                            row["sidechain_completion_seconds"]
                            + float(row["optimization_seconds"] or 0.0)
                            for row in attempts
                        )
                    ),
                }
                break
        if accepted is None:
            raise OptimizationCoverageError(
                f"completion_attempts_exhausted:{conformer_index}",
                details={
                    "failed_conformer_index": conformer_index,
                    "accepted_conformer_count": len(conformers),
                    "completion_attempts": attempts,
                    "maximum_completion_attempts": MAX_COMPLETION_ATTEMPTS,
                    "mmff_max_iterations": MMFF_MAX_ITERATIONS,
                    "atom_identity_sha256": identity_sha,
                },
            )
        if accepted["coordinate_sha256"] in {
            row["coordinate_sha256"] for row in conformers
        }:
            raise SidechainCompletionError(
                f"duplicate_conformer_coordinates:{conformer_index}"
            )
        conformers.append(accepted)
        if progress_callback is not None:
            progress_callback({
                "sequence": normalized,
                "accepted_conformer_count": len(conformers),
                "latest_conformer_index": conformer_index,
                "latest_coordinate_sha256": accepted["coordinate_sha256"],
                "elapsed_seconds": time.perf_counter() - started,
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_id": GENERATOR_ID,
        "generator_version": {
            "rdkit": rdBase.rdkitVersion,
            "backbone_generator": "internal-coordinate-rama-v2-clash15",
            "backbone_source": "formal-random_conformer_v3-seed-plan",
            "sidechain_initializer": "ETKDGv3-exact-coordinate-map",
            "force_field": "MMFF94s-fixed-N-CA-C",
            "mmff_max_iterations": MMFF_MAX_ITERATIONS,
            "maximum_completion_attempts": MAX_COMPLETION_ATTEMPTS,
        },
        "peptide_sequence": normalized,
        "chemistry": chemistry,
        "base_seed": int(base_seed),
        "atom_count": len(identities),
        "atom_identity": identities,
        "atom_identity_sha256": identity_sha,
        "conformer_count": len(conformers),
        "canonical_coordinate_set_sha256": (
            _canonical_coordinate_set_sha256(conformers)
        ),
        "conformers": conformers,
        "total_generation_seconds": time.perf_counter() - started,
        "dependency_contract": {
            "allowed_inputs": [
                "peptide_sequence",
                "ordinary_linear_standard_chemistry",
                "conformer_index",
                "formal_v3_backbone_seed_and_hash",
                "fixed_seed",
                "generator_version",
            ],
            "target_bound_inputs_used": False,
        },
    }


def conformer_atoms(
    payload: dict[str, Any], conformer_index: int
) -> list[dict[str, Any]]:
    identities = payload["atom_identity"]
    coordinates = payload["conformers"][int(conformer_index)]["coordinates"]
    return [
        {
            **identity,
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
        }
        for identity, xyz in zip(identities, coordinates)
    ]

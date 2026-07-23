"""FASPR fixed-backbone full-heavy peptide conformer prototype.

Only the peptide sequence, an ordinary-linear chemistry declaration, the
formal random_conformer_v3 backbone seed plan, and a local FASPR tool contract
are accepted.  Receptor, interface, contact, evidence, and bound coordinates
are deliberately absent from the API.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from rdkit import Chem
from rdkit.Geometry import Point3D

from phase3.drugclip.constrained_full_atom_conformer_prototype import (
    _generate_backbone,
)
from phase3.drugclip.full_atom_conformer_prototype import (
    AA1_TO_3,
    CHEMISTRY_CLASS,
    _base_molecule,
    _geometry_audit,
    _validate_topology,
    atom_identity_sha256,
    classify_sequence,
)
from phase3.drugclip.random_conformer_v3 import (
    coordinate_sha256 as backbone_coordinate_sha256,
)


SCHEMA_VERSION = "pepclip-faspr-fixed-backbone-full-heavy-prototype-v1"
GENERATOR_ID = "formal-v3-ncac-ideal-o-faspr-fixed-backbone-oxt-v1"
EXPECTED_CONFORMERS = 10
FASPR_CONFORMER_TIMEOUT_SECONDS = 30
BACKBONE_COORDINATE_TOLERANCE_ANGSTROM = 0.0
CARBONYL_C_O_LENGTH_ANGSTROM = 1.231
TERMINAL_C_OXT_LENGTH_ANGSTROM = 1.250
PDB_BACKBONE_ROUNDING_TOLERANCE_ANGSTROM = 0.0009


class FASPRPrototypeError(RuntimeError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class BackboneOReconstructionError(FASPRPrototypeError):
    pass


class FASPRInputContractError(FASPRPrototypeError):
    pass


class PackingCoverageError(FASPRPrototypeError):
    pass


def _xyz(row: dict[str, Any]) -> list[float]:
    return [float(row[axis]) for axis in ("x", "y", "z")]


def _sub(first: list[float], second: list[float]) -> list[float]:
    return [first[index] - second[index] for index in range(3)]


def _add(first: list[float], second: list[float]) -> list[float]:
    return [first[index] + second[index] for index in range(3)]


def _scale(vector: list[float], factor: float) -> list[float]:
    return [value * factor for value in vector]


def _dot(first: list[float], second: list[float]) -> float:
    return sum(left * right for left, right in zip(first, second))


def _cross(first: list[float], second: list[float]) -> list[float]:
    return [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]


def _norm(vector: list[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(vector: list[float], label: str) -> list[float]:
    length = _norm(vector)
    if not math.isfinite(length) or length < 1.0e-8:
        raise BackboneOReconstructionError(f"degenerate_vector:{label}")
    return _scale(vector, 1.0 / length)


def _angle(first: list[float], center: list[float], third: list[float]) -> float:
    left = _unit(_sub(first, center), "angle_left")
    right = _unit(_sub(third, center), "angle_right")
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(left, right)))))


def _terminal_oxygen_direction(
    n_xyz: list[float],
    ca_xyz: list[float],
    c_xyz: list[float],
) -> list[float]:
    c_to_ca = _unit(_sub(ca_xyz, c_xyz), "terminal_c_to_ca")
    plane_normal = _unit(
        _cross(_sub(n_xyz, ca_xyz), _sub(c_xyz, ca_xyz)),
        "terminal_peptide_plane",
    )
    in_plane = _unit(_cross(plane_normal, c_to_ca), "terminal_in_plane")
    theta = math.radians(120.0)
    candidates = [
        _add(_scale(c_to_ca, math.cos(theta)), _scale(in_plane, sign * math.sin(theta)))
        for sign in (-1.0, 1.0)
    ]
    c_to_n = _unit(_sub(n_xyz, c_xyz), "terminal_c_to_n")
    return min(candidates, key=lambda candidate: _dot(candidate, c_to_n))


def reconstruct_backbone_oxygen(
    sequence: str,
    backbone: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically add one backbone O per residue from local geometry."""
    expected = 3 * len(sequence)
    if len(backbone) != expected:
        raise BackboneOReconstructionError(
            f"backbone_atom_count_mismatch:{len(backbone)}:{expected}"
        )
    residues: list[dict[str, list[float]]] = []
    for residue_index in range(len(sequence)):
        residue: dict[str, list[float]] = {}
        for offset, atom_name in enumerate(("N", "CA", "C")):
            row = backbone[residue_index * 3 + offset]
            if str(row["atom_name"]).upper() != atom_name:
                raise BackboneOReconstructionError(
                    f"backbone_atom_order_mismatch:{residue_index + 1}:{atom_name}"
                )
            residue[atom_name] = _xyz(row)
        residues.append(residue)

    atoms: list[dict[str, Any]] = []
    oxygen_audit = []
    for index, residue in enumerate(residues):
        if index + 1 < len(residues):
            c_to_ca = _unit(
                _sub(residue["CA"], residue["C"]),
                f"c_to_ca:{index + 1}",
            )
            c_to_next_n = _unit(
                _sub(residues[index + 1]["N"], residue["C"]),
                f"c_to_next_n:{index + 1}",
            )
            direction = _unit(
                _scale(_add(c_to_ca, c_to_next_n), -1.0),
                f"carbonyl_bisector:{index + 1}",
            )
        else:
            direction = _terminal_oxygen_direction(
                residue["N"], residue["CA"], residue["C"]
            )
        oxygen = _add(
            residue["C"],
            _scale(direction, CARBONYL_C_O_LENGTH_ANGSTROM),
        )
        residue_name = AA1_TO_3[sequence[index]]
        for atom_name, element in (("N", "N"), ("CA", "C"), ("C", "C")):
            atoms.append({
                "residue_id": f"P:{index + 1}",
                "residue_index": index + 1,
                "residue_name": residue_name,
                "atom_name": atom_name,
                "element": element,
                "x": residue[atom_name][0],
                "y": residue[atom_name][1],
                "z": residue[atom_name][2],
            })
        atoms.append({
            "residue_id": f"P:{index + 1}",
            "residue_index": index + 1,
            "residue_name": residue_name,
            "atom_name": "O",
            "element": "O",
            "x": oxygen[0],
            "y": oxygen[1],
            "z": oxygen[2],
        })
        audit = {
            "residue_index": index + 1,
            "c_o_length_angstrom": math.dist(residue["C"], oxygen),
            "ca_c_o_angle_degrees": _angle(residue["CA"], residue["C"], oxygen),
            "deterministic_local_geometry_only": True,
        }
        if index + 1 < len(residues):
            audit["o_c_next_n_angle_degrees"] = _angle(
                oxygen, residue["C"], residues[index + 1]["N"]
            )
            audit["peptide_c_n_length_angstrom"] = math.dist(
                residue["C"], residues[index + 1]["N"]
            )
        oxygen_audit.append(audit)
    return atoms, {
        "status": "PASS",
        "method": "local_trigonal_carbonyl_bisector-v1",
        "c_o_ideal_length_angstrom": CARBONYL_C_O_LENGTH_ANGSTROM,
        "residues": oxygen_audit,
        "bound_coordinates_used": False,
    }


def write_backbone_pdb(path: Path, atoms: list[dict[str, Any]]) -> None:
    lines = ["REMARK candidate-independent formal-v3 backbone plus ideal O"]
    for serial, atom in enumerate(atoms, start=1):
        atom_name = str(atom["atom_name"])
        lines.append(
            f"ATOM  {serial:5d} {atom_name:^4s} {atom['residue_name']:>3s} P"
            f"{int(atom['residue_index']):4d}    "
            f"{float(atom['x']):8.3f}{float(atom['y']):8.3f}"
            f"{float(atom['z']):8.3f}  1.00  0.00          "
            f"{atom['element']:>2s}"
        )
    last = atoms[-1]
    lines.append(
        f"TER   {len(atoms) + 1:5d}      {last['residue_name']:>3s} P"
        f"{int(last['residue_index']):4d}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_faspr_pdb(path: Path) -> list[dict[str, Any]]:
    atoms = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            atoms.append({
                "atom_name": line[12:16].strip().upper(),
                "residue_name": line[17:20].strip().upper(),
                "chain_id": line[21:22].strip(),
                "residue_index": int(line[22:26]),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "element": line[76:78].strip().upper(),
            })
        except Exception as error:
            raise FASPRInputContractError(
                f"faspr_output_pdb_parse_failed:{line}:{type(error).__name__}:{error}"
            ) from error
    if not atoms:
        raise FASPRInputContractError("faspr_output_has_no_atoms")
    return atoms


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if len(drive) != 1:
        raise FASPRInputContractError(f"unsupported_windows_path:{resolved}")
    tail = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def _run_faspr(
    executable: Path,
    input_path: Path,
    output_path: Path,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = [
        "wsl.exe",
        "--exec",
        _windows_to_wsl(executable),
        "-i",
        _windows_to_wsl(input_path),
        "-o",
        _windows_to_wsl(output_path),
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=str(executable.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        process.wait(timeout=10)
        raise TimeoutError(
            f"faspr_conformer_timeout:{timeout_seconds}:pid={process.pid}"
        ) from error
    elapsed = time.perf_counter() - started
    return {
        "command": command,
        "pid": process.pid,
        "exit_code": int(process.returncode),
        "elapsed_seconds": elapsed,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": False,
    }


def _lookup_atoms(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["residue_index"]), str(row["atom_name"]).upper())
        if key in lookup:
            raise FASPRInputContractError(f"duplicate_faspr_atom:{key}")
        lookup[key] = row
    return lookup


def _terminal_oxt(
    ca_xyz: list[float],
    c_xyz: list[float],
    o_xyz: list[float],
) -> list[float]:
    c_to_ca = _unit(_sub(ca_xyz, c_xyz), "oxt_c_to_ca")
    c_to_o = _unit(_sub(o_xyz, c_xyz), "oxt_c_to_o")
    direction = _unit(
        _scale(_add(c_to_ca, c_to_o), -1.0),
        "oxt_trigonal_direction",
    )
    return _add(c_xyz, _scale(direction, TERMINAL_C_OXT_LENGTH_ANGSTROM))


def _molecule_with_coordinates(
    molecule: Chem.Mol,
    coordinates: list[list[float]],
) -> Chem.Mol:
    output = Chem.Mol(molecule)
    output.RemoveAllConformers()
    conformer = Chem.Conformer(len(coordinates))
    for atom_index, xyz in enumerate(coordinates):
        conformer.SetAtomPosition(atom_index, Point3D(*xyz))
    output.AddConformer(conformer, assignId=True)
    return output


def _coordinate_sha256(coordinates: list[list[float]]) -> str:
    canonical = ";".join(
        ",".join(format(value, ".12f") for value in xyz)
        for xyz in coordinates
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest().upper()


def _canonical_coordinate_set_sha256(conformers: list[dict[str, Any]]) -> str:
    material = "|".join(row["coordinate_sha256"] for row in conformers)
    return hashlib.sha256(material.encode("ascii")).hexdigest().upper()


def _pack_one(
    sequence: str,
    base: Chem.Mol,
    identities: list[dict[str, Any]],
    backbone: list[dict[str, Any]],
    conformer_index: int,
    work_dir: Path,
    faspr_executable: Path,
    *,
    faspr_runner: Callable[..., dict[str, Any]] = _run_faspr,
) -> dict[str, Any]:
    backbone_o_atoms, oxygen_audit = reconstruct_backbone_oxygen(sequence, backbone)
    input_path = work_dir / f"conformer_{conformer_index:02d}.input.pdb"
    output_path = work_dir / f"conformer_{conformer_index:02d}.faspr.pdb"
    write_backbone_pdb(input_path, backbone_o_atoms)
    result = faspr_runner(
        faspr_executable,
        input_path,
        output_path,
        timeout_seconds=FASPR_CONFORMER_TIMEOUT_SECONDS,
    )
    stdout_path = work_dir / f"conformer_{conformer_index:02d}.stdout.log"
    stderr_path = work_dir / f"conformer_{conformer_index:02d}.stderr.log"
    stdout_path.write_text(str(result.get("stdout", "")), encoding="utf-8")
    stderr_path.write_text(str(result.get("stderr", "")), encoding="utf-8")
    result["stdout_path"] = str(stdout_path.resolve())
    result["stderr_path"] = str(stderr_path.resolve())
    if result["exit_code"] != 0 or not output_path.exists():
        raise PackingCoverageError(
            f"faspr_execution_failed:{conformer_index}:{result['exit_code']}",
            details={"faspr": result},
        )
    packed_rows = parse_faspr_pdb(output_path)
    lookup = _lookup_atoms(packed_rows)
    backbone_lookup = _lookup_atoms(backbone_o_atoms)
    coordinates: list[list[float]] = []
    missing = []
    for identity in identities:
        key = (int(identity["residue_index"]), str(identity["atom_name"]))
        atom_name = key[1]
        if atom_name in {"N", "CA", "C"}:
            row = backbone_lookup[key]
            xyz = _xyz(row)
        elif atom_name == "OXT":
            last = len(sequence)
            xyz = _terminal_oxt(
                _xyz(backbone_lookup[(last, "CA")]),
                _xyz(backbone_lookup[(last, "C")]),
                _xyz(lookup[(last, "O")]),
            )
        elif key in lookup:
            row = lookup[key]
            if row["residue_name"] != identity["residue_name"]:
                raise FASPRInputContractError(
                    f"faspr_residue_name_mismatch:{key}:"
                    f"{row['residue_name']}:{identity['residue_name']}"
                )
            xyz = _xyz(row)
        else:
            missing.append(key)
            continue
        if any(not math.isfinite(value) for value in xyz):
            raise FASPRInputContractError(f"nonfinite_faspr_coordinate:{key}")
        coordinates.append(xyz)
    if missing:
        raise PackingCoverageError(
            f"faspr_required_atoms_missing:{missing}",
            details={"conformer_index": conformer_index, "missing": missing},
        )
    if len(coordinates) != len(identities):
        raise PackingCoverageError("canonical_coordinate_count_mismatch")

    quantized_backbone_deviation = []
    exact_backbone_deviation = []
    for residue_index in range(1, len(sequence) + 1):
        for atom_name in ("N", "CA", "C"):
            original = _xyz(backbone_lookup[(residue_index, atom_name)])
            packed = _xyz(lookup[(residue_index, atom_name)])
            quantized = [round(value, 3) for value in original]
            quantized_backbone_deviation.append(math.dist(quantized, packed))
            identity_index = next(
                row["atom_index"] for row in identities
                if row["residue_index"] == residue_index
                and row["atom_name"] == atom_name
            )
            exact_backbone_deviation.append(
                math.dist(original, coordinates[identity_index])
            )
    if max(quantized_backbone_deviation) > PDB_BACKBONE_ROUNDING_TOLERANCE_ANGSTROM:
        raise FASPRInputContractError(
            "faspr_moved_quantized_backbone",
            details={"maximum_angstrom": max(quantized_backbone_deviation)},
        )
    if max(exact_backbone_deviation) != 0.0:
        raise FASPRInputContractError(
            "canonical_output_backbone_changed",
            details={"maximum_angstrom": max(exact_backbone_deviation)},
        )
    input_sha = backbone_coordinate_sha256(backbone)
    output_backbone = []
    for residue_index in range(1, len(sequence) + 1):
        for atom_name in ("N", "CA", "C"):
            identity_index = next(
                row["atom_index"] for row in identities
                if row["residue_index"] == residue_index
                and row["atom_name"] == atom_name
            )
            xyz = coordinates[identity_index]
            output_backbone.append({
                "residue_id": f"P:{residue_index}",
                "atom_name": atom_name,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
            })
    output_sha = backbone_coordinate_sha256(output_backbone)
    if input_sha != output_sha:
        raise FASPRInputContractError(
            "canonical_output_backbone_hash_changed",
            details={"input_sha256": input_sha, "output_sha256": output_sha},
        )
    try:
        geometry = _geometry_audit(
            _molecule_with_coordinates(base, coordinates),
            coordinates,
        )
    except Exception as error:
        raise PackingCoverageError(
            f"packed_geometry_rejected:{type(error).__name__}:{error}",
            details={"conformer_index": conformer_index},
        ) from error
    return {
        "conformer_index": conformer_index,
        "input_backbone_coordinate_sha256": input_sha,
        "output_backbone_coordinate_sha256": output_sha,
        "maximum_backbone_deviation_angstrom": 0.0,
        "maximum_faspr_vs_input_pdb_backbone_deviation_angstrom": max(
            quantized_backbone_deviation
        ),
        "oxygen_reconstruction_audit": oxygen_audit,
        "oxt_contract": {
            "method": "terminal_trigonal_completion_from_fixed_CA_C_and_rebuilt_O",
            "c_oxt_length_angstrom": TERMINAL_C_OXT_LENGTH_ANGSTROM,
        },
        "faspr": result,
        "coordinate_sha256": _coordinate_sha256(coordinates),
        "geometry_audit": geometry,
        "coordinates": coordinates,
    }


def generate_faspr_full_atom_conformers(
    sequence: str,
    *,
    backbone_seed_plan: list[dict[str, Any]],
    work_dir: Path,
    faspr_executable: Path,
    faspr_commit_sha: str,
    faspr_binary_sha256: str,
    chemistry_class: str = CHEMISTRY_CLASS,
    num_conformers: int = EXPECTED_CONFORMERS,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    chemistry = classify_sequence(sequence, chemistry_class=chemistry_class)
    normalized = chemistry["sequence"]
    if not 1 <= int(num_conformers) <= EXPECTED_CONFORMERS:
        raise ValueError("num_conformers_must_be_between_1_and_10")
    executable = Path(faspr_executable).resolve()
    library = executable.parent / "dun2010bbdep.bin"
    if not executable.is_file() or not library.is_file():
        raise FASPRInputContractError(
            f"faspr_binary_or_rotamer_library_missing:{executable}:{library}"
        )
    actual_binary_sha = hashlib.sha256(executable.read_bytes()).hexdigest().upper()
    if actual_binary_sha != str(faspr_binary_sha256).upper():
        raise FASPRInputContractError(
            f"faspr_binary_sha256_mismatch:{actual_binary_sha}:"
            f"{str(faspr_binary_sha256).upper()}"
        )
    base = _base_molecule(normalized)
    identities = _validate_topology(base, normalized)
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    conformers = []
    started = time.perf_counter()
    for conformer_index in range(int(num_conformers)):
        backbone, backbone_audit = _generate_backbone(
            normalized, conformer_index, backbone_seed_plan
        )
        packed = _pack_one(
            normalized,
            base,
            identities,
            backbone,
            conformer_index,
            work_dir,
            executable,
        )
        conformers.append({
            **packed,
            **backbone_audit,
            "total_conformer_seconds": (
                backbone_audit["backbone_generation_seconds"]
                + packed["faspr"]["elapsed_seconds"]
            ),
        })
        if progress_callback:
            progress_callback({
                "sequence": normalized,
                "accepted_conformer_count": len(conformers),
                "latest_conformer_index": conformer_index,
                "latest_faspr_seconds": packed["faspr"]["elapsed_seconds"],
                "elapsed_seconds": time.perf_counter() - started,
            })
    hashes = [row["coordinate_sha256"] for row in conformers]
    if len(set(hashes)) != len(hashes):
        raise PackingCoverageError("faspr_conformer_coordinates_not_unique")
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_id": GENERATOR_ID,
        "generator_version": {
            "backbone_source": "formal-random_conformer_v3-seed-plan",
            "backbone_oxygen": "local_trigonal_carbonyl_bisector-v1",
            "sidechain_packer": "FASPR-fixed-backbone",
            "faspr_commit_sha": str(faspr_commit_sha),
            "faspr_binary_sha256": actual_binary_sha,
            "faspr_per_conformer_timeout_seconds": FASPR_CONFORMER_TIMEOUT_SECONDS,
        },
        "peptide_sequence": normalized,
        "chemistry": chemistry,
        "atom_count": len(identities),
        "atom_identity": identities,
        "atom_identity_sha256": atom_identity_sha256(identities),
        "conformer_count": len(conformers),
        "canonical_coordinate_set_sha256": _canonical_coordinate_set_sha256(
            conformers
        ),
        "conformers": conformers,
        "total_generation_seconds": time.perf_counter() - started,
        "dependency_contract": {
            "allowed_inputs": [
                "peptide_sequence",
                "ordinary_linear_standard_chemistry",
                "formal_v3_backbone_seed_and_hash",
                "faspr_version_and_binary",
            ],
            "target_bound_inputs_used": False,
        },
    }


def conformer_atoms(
    payload: dict[str, Any],
    conformer_index: int,
) -> list[dict[str, Any]]:
    coordinates = payload["conformers"][int(conformer_index)]["coordinates"]
    return [
        {**identity, "x": xyz[0], "y": xyz[1], "z": xyz[2]}
        for identity, xyz in zip(payload["atom_identity"], coordinates)
    ]

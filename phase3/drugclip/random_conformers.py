"""Sequence-only random peptide conformer generation for DrugCLIP Phase-3.

The generator deliberately has no access to PDB coordinates, receptor context,
or real-pair labels. It uses fixed internal-coordinate peptide geometry and
seeded Ramachandran-basin sampling; its variable inputs are peptide sequence,
split, conformer index, and stored seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from phase3.drugclip.io_utils import read_jsonl, write_json, write_jsonl

AA1_TO_3 = {"A":"ALA", "R":"ARG", "N":"ASN", "D":"ASP", "C":"CYS", "Q":"GLN", "E":"GLU", "G":"GLY", "H":"HIS", "I":"ILE", "L":"LEU", "K":"LYS", "M":"MET", "F":"PHE", "P":"PRO", "S":"SER", "T":"THR", "W":"TRP", "Y":"TYR", "V":"VAL"}


SCHEMA_VERSION = "drugclip-random-conformer-cache-v1"
GENERATOR_ID = "internal-coordinate-rama-v1"
BACKBONE_NAMES = ("N", "CA", "C")

N_CA_LENGTH = 1.458
CA_C_LENGTH = 1.525
C_N_LENGTH = 1.329
N_CA_C_ANGLE = 111.2
CA_C_N_ANGLE = 116.2
C_N_CA_ANGLE = 121.7


def seed_for(split: str, peptide_sequence: str, conformer_index: int, attempt: int) -> int:
    text = f"{GENERATOR_ID}|{split}|{peptide_sequence}|{conformer_index}|{attempt}"
    return (int(hashlib.sha256(text.encode("ascii")).hexdigest()[:8], 16) % 2_147_483_646) + 1


def _distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return math.sqrt(
        (left["x"] - right["x"]) ** 2
        + (left["y"] - right["y"]) ** 2
        + (left["z"] - right["z"]) ** 2
    )


def _angle(left: dict[str, Any], center: dict[str, Any], right: dict[str, Any]) -> float:
    first = [left[axis] - center[axis] for axis in ("x", "y", "z")]
    second = [right[axis] - center[axis] for axis in ("x", "y", "z")]
    norm_first = math.sqrt(sum(value * value for value in first))
    norm_second = math.sqrt(sum(value * value for value in second))
    cosine = sum(a * b for a, b in zip(first, second)) / (norm_first * norm_second)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _place_atom(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    bond_length: float,
    bond_angle: float,
    dihedral: float,
) -> np.ndarray:
    backward = second - third
    backward /= np.linalg.norm(backward)
    normal = np.cross(second - first, backward)
    normal /= np.linalg.norm(normal)
    in_plane = np.cross(normal, backward)
    theta = math.radians(bond_angle)
    phi = math.radians(dihedral)
    direction = (
        math.cos(theta) * backward
        + math.sin(theta) * (math.cos(phi) * in_plane + math.sin(phi) * normal)
    )
    return third + bond_length * direction


def _sample_torsions(sequence: str, rng: random.Random) -> tuple[list[float], list[float]]:
    basins = {
        "alpha": (-62.0, -43.0, 12.0, 14.0),
        "beta": (-132.0, 132.0, 18.0, 20.0),
        "ppii": (-75.0, 145.0, 14.0, 16.0),
        "left": (60.0, 40.0, 15.0, 18.0),
    }
    names = ("alpha", "beta", "ppii", "left")
    weights = (0.36, 0.29, 0.27, 0.08)
    previous = rng.choices(names, weights=weights, k=1)[0]
    phis: list[float] = []
    psis: list[float] = []
    for residue in sequence:
        if rng.random() > 0.65:
            local_weights = (0.30, 0.25, 0.25, 0.20) if residue == "G" else weights
            previous = rng.choices(names, weights=local_weights, k=1)[0]
        phi_mean, psi_mean, phi_sd, psi_sd = basins[previous]
        phi = rng.gauss(-65.0, 10.0) if residue == "P" else rng.gauss(phi_mean, phi_sd)
        phis.append(phi)
        psis.append(rng.gauss(psi_mean, psi_sd))
    return phis, psis


def _backbone_from_torsions(
    sequence: str,
    phis: list[float],
    psis: list[float],
    rng: random.Random,
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
            psis[index],
        )
        omega = 180.0 + rng.gauss(0.0, 3.0)
        next_ca = _place_atom(
            ca_atoms[index],
            c_atoms[index],
            next_n,
            N_CA_LENGTH,
            C_N_CA_ANGLE,
            omega,
        )
        next_c = _place_atom(
            c_atoms[index],
            next_n,
            next_ca,
            CA_C_LENGTH,
            N_CA_C_ANGLE,
            phis[index + 1],
        )
        n_atoms.append(next_n)
        ca_atoms.append(next_ca)
        c_atoms.append(next_c)

    backbone: list[dict[str, Any]] = []
    for residue_index, residue in enumerate(sequence, start=1):
        for atom_name, element, coords in (
            ("N", "N", n_atoms[residue_index - 1]),
            ("CA", "C", ca_atoms[residue_index - 1]),
            ("C", "C", c_atoms[residue_index - 1]),
        ):
            backbone.append(
                {
                    "atom_name": atom_name,
                    "residue_name": AA1_TO_3[residue],
                    "element": element,
                    "x": float(coords[0]),
                    "y": float(coords[1]),
                    "z": float(coords[2]),
                    "residue_id": f"P:{residue_index}",
                }
            )
    return backbone


def validate_backbone(backbone: list[dict[str, Any]], sequence: str) -> None:
    if len(backbone) != 3 * len(sequence):
        raise ValueError("backbone_length_mismatch")
    if any(not math.isfinite(atom[axis]) for atom in backbone for axis in ("x", "y", "z")):
        raise ValueError("nonfinite_coordinate")
    if max(abs(atom[axis]) for atom in backbone for axis in ("x", "y", "z")) > 1e4:
        raise ValueError("coordinate_out_of_range")
    for index in range(len(sequence) - 1):
        c_atom = backbone[index * 3 + 2]
        n_atom = backbone[(index + 1) * 3]
        bond = _distance(c_atom, n_atom)
        if not 1.1 <= bond <= 1.9:
            raise ValueError("peptide_bond_out_of_range")
    for index in range(len(sequence)):
        n_atom, ca_atom, c_atom = backbone[index * 3 : index * 3 + 3]
        if not 1.2 <= _distance(n_atom, ca_atom) <= 1.7:
            raise ValueError("n_ca_bond_out_of_range")
        if not 1.2 <= _distance(ca_atom, c_atom) <= 1.7:
            raise ValueError("ca_c_bond_out_of_range")
        if not 90.0 <= _angle(n_atom, ca_atom, c_atom) <= 130.0:
            raise ValueError("n_ca_c_angle_out_of_range")
        if index + 1 < len(sequence):
            next_n = backbone[(index + 1) * 3]
            if not 100.0 <= _angle(ca_atom, c_atom, next_n) <= 140.0:
                raise ValueError("ca_c_n_angle_out_of_range")
    bonded_pairs = {(index, index + 1) for index in range(len(backbone) - 1)}
    for left_index, left in enumerate(backbone):
        for right_index in range(left_index + 1, len(backbone)):
            if (left_index, right_index) in bonded_pairs:
                continue
            if _distance(left, backbone[right_index]) < 0.8:
                raise ValueError("severe_backbone_atom_overlap")


def generate_conformer(sequence: str, split: str, conformer_index: int, max_attempts: int = 5) -> dict[str, Any]:
    sequence = sequence.upper()
    if not sequence or any(residue not in AA1_TO_3 for residue in sequence):
        raise ValueError("unsupported_peptide_sequence")
    for attempt in range(max_attempts):
        seed = seed_for(split, sequence, conformer_index, attempt)
        rng = random.Random(seed)
        phis, psis = _sample_torsions(sequence, rng)
        backbone = _backbone_from_torsions(sequence, phis, psis, rng)
        try:
            validate_backbone(backbone, sequence)
        except ValueError:
            continue
        return {
            "conformer_id": f"rand:{hashlib.sha256(f'{split}|{sequence}|{conformer_index}'.encode('ascii')).hexdigest()[:20]}",
            "generator_id": GENERATOR_ID,
            "seed": seed,
            "conformer_index": conformer_index,
            "peptide_sequence": sequence,
            "backbone_atoms": backbone,
        }
    raise ValueError("generation_attempts_exhausted")


def _generate_peptide_cache(task: tuple[str, str, int, int]) -> dict[str, Any]:
    split, peptide, max_conformers, max_attempts = task
    conformers = []
    rejects = []
    for index in range(max_conformers):
        try:
            conformers.append(generate_conformer(peptide, split, index, max_attempts))
        except ValueError as exc:
            rejects.append(
                {
                    "split": split,
                    "peptide_sequence": peptide,
                    "conformer_index": index,
                    "reason": str(exc),
                }
            )
    cache = None
    if conformers:
        cache = {
            "schema_version": SCHEMA_VERSION,
            "cache_id": f"cache:{hashlib.sha256(f'{split}|{peptide}'.encode('ascii')).hexdigest()[:20]}",
            "split": split,
            "peptide_sequence": peptide,
            "generator_id": GENERATOR_ID,
            "max_conformers_requested": max_conformers,
            "conformers": conformers,
        }
    else:
        rejects.append(
            {
                "split": split,
                "peptide_sequence": peptide,
                "reason": "no_valid_random_conformer",
            }
        )
    return {
        "generator_id": GENERATOR_ID,
        "max_conformers_requested": max_conformers,
        "max_attempts": max_attempts,
        "split": split,
        "peptide_sequence": peptide,
        "cache": cache,
        "rejects": rejects,
    }


def build_cache(
    split_rows_jsonl: str | Path,
    output_dir: str | Path,
    max_conformers: int = 10,
    max_attempts: int = 5,
    workers: int | None = None,
) -> dict[str, Any]:
    if max_conformers <= 0:
        raise ValueError("max_conformers must be positive")
    if workers is not None and workers <= 0:
        raise ValueError("workers must be positive")
    rows = list(read_jsonl(split_rows_jsonl))
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        by_split[split].add(str(row["peptide_sequence"]).upper())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "random_conformer_progress.jsonl"
    completed: set[tuple[str, str]] = set()
    if progress_path.is_file():
        for row in read_jsonl(progress_path):
            if (
                row.get("generator_id") != GENERATOR_ID
                or int(row.get("max_conformers_requested", -1)) != max_conformers
                or int(row.get("max_attempts", -1)) != max_attempts
            ):
                raise ValueError("incompatible_random_conformer_progress")
            completed.add((str(row["split"]), str(row["peptide_sequence"])))

    tasks = [
        (split, peptide, max_conformers, max_attempts)
        for split in sorted(by_split)
        for peptide in sorted(by_split[split])
        if (split, peptide) not in completed
    ]
    worker_count = workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    with progress_path.open("a", encoding="utf-8", newline="\n") as progress_handle:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for result in executor.map(_generate_peptide_cache, tasks, chunksize=1):
                progress_handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                progress_handle.flush()

    cache_count = 0
    conformer_count = 0
    reject_count = 0
    cache_path = output / "random_conformer_cache.jsonl"
    reject_path = output / "random_conformer_rejects.jsonl"
    with cache_path.open("w", encoding="utf-8", newline="\n") as cache_handle, reject_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as reject_handle:
        for result in read_jsonl(progress_path):
            cache = result.get("cache")
            if cache is not None:
                cache_handle.write(json.dumps(cache, ensure_ascii=False, sort_keys=True) + "\n")
                cache_count += 1
                conformer_count += len(cache["conformers"])
            for reject in result.get("rejects", []):
                reject_handle.write(json.dumps(reject, ensure_ascii=False, sort_keys=True) + "\n")
                reject_count += 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generator_id": GENERATOR_ID,
        "max_conformers_requested": max_conformers,
        "peptides_requested": sum(len(values) for values in by_split.values()),
        "peptide_caches": cache_count,
        "conformers": conformer_count,
        "rejects": reject_count,
        "workers": worker_count,
        "resumed_peptides": len(completed),
        "split_peptide_counts": {key: len(value) for key, value in sorted(by_split.items())},
    }
    write_json(output / "random_conformer_summary.json", summary)
    progress_path.unlink()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_rows_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_conformers", type=int, default=10)
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_cache(**vars(parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

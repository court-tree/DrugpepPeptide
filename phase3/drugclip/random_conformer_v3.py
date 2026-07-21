"""Versioned random-conformer generation and clash15 QC for DrugCLIP v3.

Version 3 deliberately reuses the stored v2 seed for attempt zero.  Later
attempts use a new, documented seed namespace.  This lets the builder prove
that every accepted v2 coordinate set is reproducible before deciding whether
clash15 requires replacement.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from typing import Any

from phase3.drugclip.random_conformers import (
    AA1_TO_3,
    _backbone_from_torsions,
    _sample_torsions,
    validate_backbone,
)


DATASET_VERSION = "random_conformer_v3"
PARENT_DATASET = "random_conformer_v2"
CACHE_SCHEMA = "drugclip-random-conformer-cache-v2"
RELATION_SCHEMA = "drugclip-random-augmentation-pairs-v3"
DATABASE_CONTRACT = "drugclip-exact-peptide-random-conformer-v3"
GENERATOR_ID = "internal-coordinate-rama-v2-clash15"
CLASH_DISTANCE_ANGSTROM = 1.5
EXPECTED_CONFORMERS = 10
DEFAULT_MAX_ATTEMPTS = 1000


def attempt_seed(split: str, peptide_sequence: str, conformer_index: int, attempt_index: int) -> int:
    """Return the deterministic v3 replacement seed (attempts start at 1)."""

    if attempt_index < 1:
        raise ValueError("attempt_index must be >= 1 for replacement seeds")
    material = (
        f"{GENERATOR_ID}|{split}|{peptide_sequence.upper()}|"
        f"{conformer_index}|{attempt_index}"
    )
    return (int(hashlib.sha256(material.encode("ascii")).hexdigest()[:8], 16) % 2_147_483_646) + 1


def generate_from_seed(sequence: str, seed: int) -> list[dict[str, Any]]:
    sequence = sequence.upper()
    if not sequence or any(residue not in AA1_TO_3 for residue in sequence):
        raise ValueError("unsupported_peptide_sequence")
    rng = random.Random(int(seed))
    phis, psis = _sample_torsions(sequence, rng)
    backbone = _backbone_from_torsions(sequence, phis, psis, rng)
    validate_backbone(backbone, sequence)
    return backbone


def coordinate_sha256(backbone: list[dict[str, Any]]) -> str:
    """Hash ordered, unrounded IEEE-754 x/y/z values plus atom identities."""

    digest = hashlib.sha256()
    for atom in backbone:
        digest.update(str(atom["residue_id"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(atom["atom_name"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(struct.pack(">ddd", float(atom["x"]), float(atom["y"]), float(atom["z"])))
    return digest.hexdigest().upper()


def _residue_number(atom: dict[str, Any]) -> int:
    return int(str(atom["residue_id"]).rsplit(":", 1)[1])


def clash15_details(backbone: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the formal v3 rule: N/CA/C, residue-index gap >=2, distance <1.5 A."""

    minimum = math.inf
    minimum_pair: list[dict[str, Any]] | None = None
    clashes: list[dict[str, Any]] = []
    for left_index, left in enumerate(backbone):
        left_residue = _residue_number(left)
        for right in backbone[left_index + 1 :]:
            right_residue = _residue_number(right)
            if abs(left_residue - right_residue) < 2:
                continue
            distance = math.sqrt(sum((float(left[a]) - float(right[a])) ** 2 for a in ("x", "y", "z")))
            pair = [
                {"residue_id": str(left["residue_id"]), "atom_name": str(left["atom_name"])},
                {"residue_id": str(right["residue_id"]), "atom_name": str(right["atom_name"])},
            ]
            if distance < minimum:
                minimum = distance
                minimum_pair = pair
            if distance < CLASH_DISTANCE_ANGSTROM:
                clashes.append({"atoms": pair, "distance_angstrom": distance})
    return {
        "has_clash": bool(clashes),
        "clashes": clashes,
        "minimum_nonlocal_backbone_distance_angstrom": None if math.isinf(minimum) else minimum,
        "minimum_pair": minimum_pair,
    }


def conformer_id(split: str, peptide_sequence: str, conformer_index: int) -> str:
    material = f"{GENERATOR_ID}|{split}|{peptide_sequence.upper()}|{conformer_index}"
    return f"randv3:{hashlib.sha256(material.encode('ascii')).hexdigest()[:20]}"

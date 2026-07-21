"""Independent full-data audit for DrugCLIP random_conformer_v3.

This module intentionally does not import the v3 generator or its QC helpers.
The geometry and clash rules are implemented again here so a generator defect
cannot automatically validate itself.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


CACHE_SCHEMA = "drugclip-random-conformer-cache-v2"
RELATION_SCHEMA = "drugclip-random-augmentation-pairs-v3"
DATABASE_CONTRACT = "drugclip-exact-peptide-random-conformer-v3"
GENERATOR_ID = "internal-coordinate-rama-v2-clash15"
ATOM_NAMES = ("N", "CA", "C")


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return math.sqrt(sum((float(left[a]) - float(right[a])) ** 2 for a in ("x", "y", "z")))


def _angle(left: dict[str, Any], center: dict[str, Any], right: dict[str, Any]) -> float:
    first = [float(left[a]) - float(center[a]) for a in ("x", "y", "z")]
    second = [float(right[a]) - float(center[a]) for a in ("x", "y", "z")]
    denominator = math.sqrt(sum(x*x for x in first) * sum(x*x for x in second))
    cosine = sum(a*b for a, b in zip(first, second)) / denominator
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def independent_clash15(atoms: list[dict[str, Any]]) -> tuple[int, float | None]:
    clashes = 0
    minimum = math.inf
    for left_index, left in enumerate(atoms):
        left_residue = int(str(left["residue_id"]).split(":")[-1])
        for right in atoms[left_index + 1:]:
            right_residue = int(str(right["residue_id"]).split(":")[-1])
            if abs(left_residue - right_residue) < 2:
                continue
            distance = _distance(left, right)
            minimum = min(minimum, distance)
            clashes += int(distance < 1.5)
    return clashes, None if math.isinf(minimum) else minimum


def _semantic_pair(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": row["split"],
        "pair": row["pair"],
        "interface": row["interface"],
        "random_conformer_cache_id": row["random_conformer_cache_id"],
        "known_positive_group": row["known_positive_group"],
    }


def audit(v2_dir: str | Path, v3_dir: str | Path) -> dict[str, Any]:
    v2 = Path(v2_dir).resolve()
    v3 = Path(v3_dir).resolve()
    errors: Counter[str] = Counter()
    cache_keys: dict[str, tuple[str, str]] = {}
    conformer_ids: set[str] = set()
    split_identity: dict[tuple[str, str], str] = {}
    split_counts: Counter[str] = Counter()
    total_conformers = 0
    minimum_distance = math.inf
    for cache in _rows(v3 / "03_random_conformer_cache/random_conformer_cache.jsonl"):
        cache_id = str(cache.get("cache_id", ""))
        split = str(cache.get("split", ""))
        peptide = str(cache.get("peptide_sequence", ""))
        split_counts[split] += 1
        if cache.get("schema_version") != CACHE_SCHEMA: errors["cache_schema"] += 1
        if cache.get("generator_id") != GENERATOR_ID: errors["cache_generator"] += 1
        if cache_id in cache_keys: errors["duplicate_cache_id"] += 1
        cache_keys[cache_id] = (split, peptide)
        previous_split = split_identity.setdefault(("peptide", peptide), split)
        if previous_split != split: errors["cross_split_peptide_identity_conflict"] += 1
        conformers = cache.get("conformers", [])
        if len(conformers) != 10: errors["cache_not_exactly_10"] += 1
        if sorted(int(c.get("conformer_index", -1)) for c in conformers) != list(range(10)):
            errors["conformer_indices_not_0_to_9"] += 1
        local_coordinate_fingerprints: set[tuple[float, ...]] = set()
        for conformer in conformers:
            total_conformers += 1
            conformer_id = str(conformer.get("conformer_id", ""))
            if not conformer_id or conformer_id in conformer_ids: errors["nonunique_conformer_id"] += 1
            conformer_ids.add(conformer_id)
            if conformer.get("generator_id") != GENERATOR_ID: errors["conformer_generator"] += 1
            if "seed" not in conformer or "attempt_index" not in conformer or "base_v2_seed" not in conformer:
                errors["incomplete_seed_attempt_metadata"] += 1
            atoms = conformer.get("backbone_atoms", [])
            if len(atoms) != 3 * len(peptide): errors["atom_sequence_length_mismatch"] += 1
            fingerprint: list[float] = []
            for residue_index in range(len(peptide)):
                triplet = atoms[residue_index*3:residue_index*3+3]
                if [str(a.get("atom_name")) for a in triplet] != list(ATOM_NAMES):
                    errors["atom_order"] += 1
                for atom in triplet:
                    coordinates = [float(atom.get(a, math.nan)) for a in ("x", "y", "z")]
                    if not all(math.isfinite(x) for x in coordinates): errors["nonfinite_coordinate"] += 1
                    fingerprint.extend(coordinates)
                if len(triplet) == 3:
                    if not 1.2 <= _distance(triplet[0], triplet[1]) <= 1.7: errors["n_ca_bond"] += 1
                    if not 1.2 <= _distance(triplet[1], triplet[2]) <= 1.7: errors["ca_c_bond"] += 1
                    if not 90.0 <= _angle(triplet[0], triplet[1], triplet[2]) <= 130.0: errors["n_ca_c_angle"] += 1
                    if residue_index + 1 < len(peptide):
                        next_n = atoms[(residue_index+1)*3]
                        if not 1.1 <= _distance(triplet[2], next_n) <= 1.9: errors["c_n_bond"] += 1
                        if not 100.0 <= _angle(triplet[1], triplet[2], next_n) <= 140.0: errors["ca_c_n_angle"] += 1
            fingerprint_tuple = tuple(fingerprint)
            if fingerprint_tuple in local_coordinate_fingerprints: errors["reused_conformer_coordinates"] += 1
            local_coordinate_fingerprints.add(fingerprint_tuple)
            clash_count, local_minimum = independent_clash15(atoms)
            errors["clash15"] += clash_count
            if local_minimum is not None: minimum_distance = min(minimum_distance, local_minimum)

    v2_pair_path = v2 / "04_training_input/random_conformer_pairs.jsonl"
    v3_pair_path = v3 / "04_training_input/random_conformer_pairs.jsonl"
    v2_iter, v3_iter = iter(_rows(v2_pair_path)), iter(_rows(v3_pair_path))
    pair_count = 0
    pair_ids_v2: set[str] = set()
    pair_ids_v3: set[str] = set()
    while True:
        try: left = next(v2_iter)
        except StopIteration: left = None
        try: right = next(v3_iter)
        except StopIteration: right = None
        if left is None and right is None: break
        if left is None or right is None:
            errors["pair_total_difference"] += 1
            break
        pair_count += 1
        left_id, right_id = str(left["pair"]["pair_id"]), str(right["pair"]["pair_id"])
        pair_ids_v2.add(left_id); pair_ids_v3.add(right_id)
        if right.get("schema_version") != RELATION_SCHEMA: errors["relation_schema"] += 1
        if right.get("database_contract") != DATABASE_CONTRACT: errors["database_contract"] += 1
        if _semantic_pair(left) != _semantic_pair(right):
            for field in ("split", "pair", "interface", "random_conformer_cache_id", "known_positive_group"):
                if _semantic_pair(left)[field] != _semantic_pair(right)[field]: errors[f"pair_semantic_{field}"] += 1
        cache_id = str(right.get("random_conformer_cache_id", ""))
        expected = cache_keys.get(cache_id)
        if expected is None: errors["missing_cache_reference"] += 1
        elif expected != (str(right["split"]), str(right["pair"]["peptide_sequence"])):
            errors["cache_split_sequence_mismatch"] += 1
    errors["pair_id_set_difference"] += len(pair_ids_v2 ^ pair_ids_v3)

    result = {
        "audit_id": "drugclip-random-conformer-v3-independent-audit-v1",
        "passed": not any(errors.values()),
        "counts": {
            "caches": len(cache_keys),
            "conformers": total_conformers,
            "pairs": pair_count,
            "split_caches": dict(sorted(split_counts.items())),
            "unique_conformer_ids": len(conformer_ids),
            "minimum_nonlocal_backbone_distance_angstrom": minimum_distance,
        },
        "errors": dict(sorted(errors.items())),
        "expected": {"caches": 6979, "conformers": 69790, "pairs": 24633, "conformers_per_cache": 10, "clash15": 0},
    }
    for key, expected in (("caches", 6979), ("conformers", 69790), ("pairs", 24633)):
        if result["counts"][key] != expected:
            result["errors"][f"expected_{key}"] = abs(result["counts"][key] - expected)
            result["passed"] = False
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2_dir", required=True)
    parser.add_argument("--v3_dir", required=True)
    parser.add_argument("--output_json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = audit(args.v2_dir, args.v3_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if result["passed"] else 1)

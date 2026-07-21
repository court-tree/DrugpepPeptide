"""Build random_conformer_v3 from the immutable v2 DrugCLIP database."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from phase3.drugclip.io_utils import read_jsonl, write_json
from phase3.drugclip.random_conformer_v3 import (
    CACHE_SCHEMA,
    DATABASE_CONTRACT,
    DEFAULT_MAX_ATTEMPTS,
    EXPECTED_CONFORMERS,
    GENERATOR_ID,
    RELATION_SCHEMA,
    attempt_seed,
    clash15_details,
    conformer_id,
    coordinate_sha256,
    generate_from_seed,
)


COPY_FILES = (
    "01_interface_pairs/interface_pairs.jsonl",
    "01_interface_pairs/known_positive_groups.json",
    "01_interface_pairs/receptor_interfaces.jsonl",
    "01_interface_pairs/rejects.jsonl",
    "01_interface_pairs/summary.json",
    "02_leakage_safe_split/LEAKAGE_AUDIT.md",
    "02_leakage_safe_split/pair_splits.jsonl",
    "02_leakage_safe_split/receptors.fasta",
    "02_leakage_safe_split/receptor_cluster_all_seqs.fasta",
    "02_leakage_safe_split/receptor_cluster_cluster.tsv",
    "02_leakage_safe_split/receptor_cluster_rep_seq.fasta",
    "02_leakage_safe_split/receptor_families.jsonl",
    "02_leakage_safe_split/summary.json",
    "02_leakage_safe_split/test.jsonl",
    "02_leakage_safe_split/train.jsonl",
    "02_leakage_safe_split/valid.jsonl",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _copy_parent_files(parent: Path, output: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in COPY_FILES:
        source = parent / relative
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        if sha256_file(destination) != source_hash:
            raise RuntimeError(f"copied_file_hash_mismatch:{relative}")
        hashes[relative] = source_hash
    return hashes


def _v3_conformer(v2: dict[str, Any], split: str, peptide: str, max_attempts: int) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    index = int(v2["conformer_index"])
    v2_atoms = v2["backbone_atoms"]
    v2_hash = coordinate_sha256(v2_atoms)
    regenerated = generate_from_seed(peptide, int(v2["seed"]))
    regenerated_hash = coordinate_sha256(regenerated)
    if regenerated_hash != v2_hash:
        raise RuntimeError(f"v2_generator_nondeterminism:{split}:{peptide}:{index}:{v2_hash}:{regenerated_hash}")
    old_qc = clash15_details(regenerated)
    selected_atoms = regenerated
    selected_seed = int(v2["seed"])
    selected_attempt = 0
    selected_qc = old_qc
    if old_qc["has_clash"]:
        for attempt in range(1, max_attempts + 1):
            seed = attempt_seed(split, peptide, index, attempt)
            try:
                candidate = generate_from_seed(peptide, seed)
            except ValueError:
                # A candidate that fails the pre-existing bond/angle/overlap
                # geometry contract consumes its deterministic attempt.  The
                # formal clash threshold is never relaxed.
                continue
            qc = clash15_details(candidate)
            if not qc["has_clash"]:
                selected_atoms = candidate
                selected_seed = seed
                selected_attempt = attempt
                selected_qc = qc
                break
        else:
            raise RuntimeError(f"replacement_attempts_exhausted:{split}:{peptide}:{index}:{max_attempts}")
    new = {
        "conformer_id": conformer_id(split, peptide, index),
        "generator_id": GENERATOR_ID,
        "seed": selected_seed,
        "base_v2_seed": int(v2["seed"]),
        "attempt_index": selected_attempt,
        "conformer_index": index,
        "peptide_sequence": peptide,
        "backbone_atoms": selected_atoms,
    }
    changed = selected_attempt > 0
    audit = None
    if changed:
        audit = {
            "split": split,
            "peptide_sequence": peptide,
            "conformer_index": index,
            "v2_conformer_id": v2["conformer_id"],
            "v2_seed": int(v2["seed"]),
            "v2_coordinate_sha256": v2_hash,
            "clash_atom_pairs": old_qc["clashes"],
            "clash_minimum_distance_angstrom": old_qc["minimum_nonlocal_backbone_distance_angstrom"],
            "v3_conformer_id": new["conformer_id"],
            "v3_seed": selected_seed,
            "attempt_index": selected_attempt,
            "v3_coordinate_sha256": coordinate_sha256(selected_atoms),
            "v3_minimum_nonlocal_backbone_distance_angstrom": selected_qc["minimum_nonlocal_backbone_distance_angstrom"],
        }
    return new, audit, regenerated_hash == v2_hash


def _rewrite_pair_rows(parent: Path, output: Path) -> int:
    source = parent / "04_training_input/random_conformer_pairs.jsonl"
    target = output / "04_training_input/random_conformer_pairs.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in read_jsonl(source):
            row["schema_version"] = RELATION_SCHEMA
            row["database_contract"] = DATABASE_CONTRACT
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    write_json(target.parent / "summary.json", {
        "schema_version": RELATION_SCHEMA,
        "database_contract": DATABASE_CONTRACT,
        "input_pairs": count,
        "retained_pairs": count,
        "rejects": 0,
        "known_positive_identity": "receptor_interface_id + peptide_sequence",
        "true_bound_used_as_training_input": False,
        "peptide_similarity_rules_active": False,
    })
    (target.parent / "rejects.jsonl").write_text("", encoding="utf-8")
    return count


def build(parent_dir: str | Path, output_dir: str | Path, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> dict[str, Any]:
    parent = Path(parent_dir).resolve()
    output = Path(output_dir).resolve()
    if parent == output or output.is_relative_to(parent):
        raise ValueError("v3 output must not be inside the v2 parent")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output:{output}")
    output.mkdir(parents=True, exist_ok=True)
    parent_hashes = _copy_parent_files(parent, output)
    cache_dir = output / "03_random_conformer_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    replacement_path = output / "conformer_replacement_audit.jsonl"
    unchanged_mismatch_examples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    with (cache_dir / "random_conformer_cache.jsonl").open("w", encoding="utf-8", newline="\n") as cache_handle, replacement_path.open("w", encoding="utf-8", newline="\n") as audit_handle:
        for cache in read_jsonl(parent / "03_random_conformer_cache/random_conformer_cache.jsonl"):
            split = str(cache["split"])
            peptide = str(cache["peptide_sequence"])
            if len(cache["conformers"]) != EXPECTED_CONFORMERS:
                raise RuntimeError(f"parent_cache_not_exactly_10:{cache['cache_id']}")
            new_conformers = []
            local_hashes: set[str] = set()
            for v2 in sorted(cache["conformers"], key=lambda item: int(item["conformer_index"])):
                new, audit, reproduced = _v3_conformer(v2, split, peptide, max_attempts)
                if new["conformer_id"] in seen_ids:
                    raise RuntimeError(f"duplicate_v3_conformer_id:{new['conformer_id']}")
                seen_ids.add(new["conformer_id"])
                new_hash = coordinate_sha256(new["backbone_atoms"])
                if new_hash in local_hashes:
                    raise RuntimeError(f"duplicate_coordinates_within_cache:{cache['cache_id']}")
                local_hashes.add(new_hash)
                new_conformers.append(new)
                counts["conformers"] += 1
                if audit:
                    audit["cache_id"] = cache["cache_id"]
                    audit_handle.write(json.dumps(audit, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["replaced"] += 1
                else:
                    counts["unchanged"] += 1
                    counts["unchanged_coordinate_hash_match"] += int(reproduced)
                    if not reproduced:
                        counts["unchanged_coordinate_hash_mismatch"] += 1
                        if len(unchanged_mismatch_examples) < 20:
                            unchanged_mismatch_examples.append({"cache_id": cache["cache_id"], "conformer_index": new["conformer_index"]})
            new_cache = {
                **cache,
                "schema_version": CACHE_SCHEMA,
                "generator_id": GENERATOR_ID,
                "conformers": new_conformers,
            }
            cache_handle.write(json.dumps(new_cache, ensure_ascii=False, sort_keys=True) + "\n")
            counts["caches"] += 1
            counts[f"split_{split}_caches"] += 1
    (cache_dir / "random_conformer_rejects.jsonl").write_text("", encoding="utf-8")
    write_json(cache_dir / "random_conformer_summary.json", {
        "schema_version": CACHE_SCHEMA,
        "generator_id": GENERATOR_ID,
        "max_conformers_requested": EXPECTED_CONFORMERS,
        "max_replacement_attempts": max_attempts,
        "peptide_caches": counts["caches"],
        "conformers": counts["conformers"],
        "replaced_conformers": counts["replaced"],
        "unchanged_conformers": counts["unchanged"],
        "rejects": 0,
        "split_peptide_counts": {s: counts[f"split_{s}_caches"] for s in ("train", "valid", "test")},
    })
    pair_count = _rewrite_pair_rows(parent, output)
    unchanged_summary = {
        "unchanged_count": counts["unchanged"],
        "coordinate_hash_match_count": counts["unchanged_coordinate_hash_match"],
        "coordinate_hash_mismatch_count": counts["unchanged_coordinate_hash_mismatch"],
        "mismatch_examples": unchanged_mismatch_examples,
    }
    write_json(output / "unchanged_conformer_summary.json", unchanged_summary)
    summary = {
        "dataset_name": "PepCLIP Phase-3 DrugCLIP random conformers",
        "dataset_version": "random_conformer_v3",
        "parent_dataset": "random_conformer_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator_id": GENERATOR_ID,
        "cache_schema": CACHE_SCHEMA,
        "relation_schema": RELATION_SCHEMA,
        "database_contract": DATABASE_CONTRACT,
        "max_replacement_attempts": max_attempts,
        "cache_count": counts["caches"],
        "conformer_count": counts["conformers"],
        "replacement_count": counts["replaced"],
        "unchanged_count": counts["unchanged"],
        "pair_count": pair_count,
        "parent_core_sha256": parent_hashes,
    }
    write_json(output / "build_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(**vars(parse_args())), ensure_ascii=False, indent=2))

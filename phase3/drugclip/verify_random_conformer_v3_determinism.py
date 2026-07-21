"""Re-generate replacement and sampled unchanged v3 conformers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from phase3.drugclip.io_utils import read_jsonl, write_json
from phase3.drugclip.random_conformer_v3 import (
    attempt_seed, conformer_id, coordinate_sha256, generate_from_seed,
)


def verify(v3_dir: str | Path, minimum_total: int = 500, unchanged_per_split: int = 100) -> dict[str, object]:
    root = Path(v3_dir)
    replaced = {
        (str(row["split"]), str(row["peptide_sequence"]), int(row["conformer_index"]))
        for row in read_jsonl(root / "conformer_replacement_audit.jsonl")
    }
    candidates: defaultdict[str, list[tuple[str, str, int, dict]]] = defaultdict(list)
    replacement_rows: list[tuple[str, str, int, dict]] = []
    for cache in read_jsonl(root / "03_random_conformer_cache/random_conformer_cache.jsonl"):
        split, peptide = str(cache["split"]), str(cache["peptide_sequence"])
        for conformer in cache["conformers"]:
            item = (split, peptide, int(conformer["conformer_index"]), conformer)
            if item[:3] in replaced: replacement_rows.append(item)
            else: candidates[split].append(item)
    additional = max(0, minimum_total - len(replacement_rows))
    per_split = max(unchanged_per_split, math.ceil(additional / 3))
    selected = list(replacement_rows)
    sampled_counts = {}
    for split in ("train", "valid", "test"):
        ranked = sorted(candidates[split], key=lambda item: hashlib.sha256(f"determinism-v1|{item[0]}|{item[1]}|{item[2]}".encode()).digest())
        sample = ranked[:per_split]
        selected.extend(sample)
        sampled_counts[split] = len(sample)
    mismatches = []
    for split, peptide, index, stored in selected:
        attempt = int(stored["attempt_index"])
        expected_seed = int(stored["base_v2_seed"]) if attempt == 0 else attempt_seed(split, peptide, index, attempt)
        regenerated = generate_from_seed(peptide, expected_seed)
        checks = {
            "conformer_id": str(stored["conformer_id"]) == conformer_id(split, peptide, index),
            "seed": int(stored["seed"]) == expected_seed,
            "coordinates": stored["backbone_atoms"] == regenerated,
            "coordinate_sha256": coordinate_sha256(stored["backbone_atoms"]) == coordinate_sha256(regenerated),
        }
        if not all(checks.values()):
            mismatches.append({"split": split, "peptide_sequence": peptide, "conformer_index": index, "checks": checks})
    return {
        "audit_id": "drugclip-random-conformer-v3-determinism-v1",
        "passed": not mismatches and len(selected) >= minimum_total,
        "replacement_conformers_checked": len(replacement_rows),
        "unchanged_sampled_by_split": sampled_counts,
        "total_checked": len(selected),
        "minimum_total_required": minimum_total,
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3_dir", required=True)
    parser.add_argument("--output_json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = verify(args.v3_dir)
    if args.output_json: write_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)

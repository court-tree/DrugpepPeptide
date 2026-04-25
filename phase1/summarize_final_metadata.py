from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from common import write_json


def percentile(sorted_values: List[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def numeric_summary(values: List[float]) -> Dict[str, Any]:
    values = [float(x) for x in values]
    values.sort()
    if not values:
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": len(values),
        "min": float(values[0]),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "max": float(values[-1]),
        "mean": float(mean(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase1 final_metadata.jsonl")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    split_counts: Counter[str] = Counter()
    length_counts: Counter[int] = Counter()
    cap_counts: Counter[str] = Counter()
    source_files: set[str] = set()
    pdb_ids: set[str] = set()
    task_ids: set[str] = set()
    receptor_keys: set[str] = set()
    peptide_sequences: set[str] = set()

    peptide_lengths: List[float] = []
    patch_sizes: List[float] = []
    avg_contacts: List[float] = []
    contact_coverages: List[float] = []
    longest_runs: List[float] = []
    sampling_weights: List[float] = []

    n_rows = 0
    with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            n_rows += 1

            split_counts[str(row.get("split", "missing"))] += 1
            source_files.add(str(row.get("source_file", "")))
            pdb_ids.add(str(row.get("pdb_id", "")))
            task_ids.add(str(row.get("parent_task_id", "")))
            receptor_keys.add(
                "|".join(
                    [
                        str(row.get("source_file", "")),
                        str(row.get("receptor_chain_id", "")),
                    ]
                )
            )
            peptide_sequences.add(str(row.get("peptide_sequence", "")))

            peptide_length = int(row.get("peptide_length", 0))
            length_counts[peptide_length] += 1
            peptide_lengths.append(float(peptide_length))
            patch_sizes.append(float(row.get("track_b_patch_num_residues", 0)))
            avg_contacts.append(float(row.get("avg_contact_count", 0.0)))
            contact_coverages.append(float(row.get("contact_coverage", 0.0)))
            longest_runs.append(float(row.get("longest_contact_run", 0.0)))
            sampling_weights.append(float(row.get("sampling_weight", 0.0)))

            n_cap = bool(row.get("has_n_cap_proxy", False))
            c_cap = bool(row.get("has_c_cap_proxy", False))
            cap_counts[f"N={n_cap}|C={c_cap}"] += 1

    summary = {
        "n_rows": n_rows,
        "split_counts": dict(sorted(split_counts.items())),
        "unique_source_files": len(source_files),
        "unique_pdb_ids": len(pdb_ids),
        "unique_parent_tasks": len(task_ids),
        "unique_receptor_keys": len(receptor_keys),
        "unique_peptide_sequences": len(peptide_sequences),
        "peptide_length_counts": {str(k): v for k, v in sorted(length_counts.items())},
        "cap_proxy_counts": dict(sorted(cap_counts.items())),
        "peptide_length_summary": numeric_summary(peptide_lengths),
        "patch_num_residues_summary": numeric_summary(patch_sizes),
        "avg_contact_count_summary": numeric_summary(avg_contacts),
        "contact_coverage_summary": numeric_summary(contact_coverages),
        "longest_contact_run_summary": numeric_summary(longest_runs),
        "sampling_weight_summary": numeric_summary(sampling_weights),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

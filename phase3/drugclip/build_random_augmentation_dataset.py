"""Join DrugCLIP interface pairs with split-local random conformer caches."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from phase3.drugclip.io_utils import read_jsonl, write_json, write_jsonl


def build(args: argparse.Namespace) -> dict[str, Any]:
    split_rows = list(read_jsonl(args.interface_pair_splits_jsonl))
    interfaces = {str(row["pair_id"]): row for row in read_jsonl(args.receptor_interfaces_jsonl)}
    cache_by_key = {
        (str(row["split"]), str(row["peptide_sequence"])): row
        for row in read_jsonl(args.random_conformer_cache_jsonl)
    }
    all_pairs = list(read_jsonl(args.all_interface_pairs_jsonl))
    receptor_to_peptides: defaultdict[str, set[str]] = defaultdict(set)
    peptide_to_receptors: defaultdict[str, set[str]] = defaultdict(set)
    for row in all_pairs:
        receptor_to_peptides[str(row["receptor_interface_id"])].add(str(row["peptide_sequence"]))
        peptide_to_receptors[str(row["peptide_sequence"])].add(str(row["receptor_interface_id"]))
    output_rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for pair in split_rows:
        interface_id = str(pair["receptor_interface_id"])
        interface = interfaces.get(interface_id)
        cache = cache_by_key.get((str(pair["split"]), str(pair["peptide_sequence"])))
        if interface is None or cache is None or not cache.get("conformers"):
            rejects.append({"pair_id": pair["pair_id"], "reason": "missing_interface_or_random_cache"})
            continue
        output_rows.append(
            {
                "schema_version": "drugclip-random-augmentation-pairs-v2",
                "database_contract": "drugclip-exact-peptide-random-conformer-v2",
                "split": pair["split"],
                "pair": {
                    "pair_id": pair["pair_id"],
                    "receptor_id": interface_id,
                    "receptor_sequence": pair["receptor_sequence"],
                    "peptide_sequence": pair["peptide_sequence"],
                    "source_database": pair.get("source_database", ""),
                    "experimental_evidence": pair.get("experimental_evidence", []),
                    "evidence_ids": pair.get("evidence_ids", []),
                },
                "interface": interface,
                "random_conformer_cache_id": cache["cache_id"],
                "known_positive_group": {
                    "receptor_peptides": sorted(receptor_to_peptides[interface_id]),
                    "peptide_receptors": sorted(peptide_to_receptors[str(pair["peptide_sequence"])]),
                },
            }
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "random_conformer_pairs.jsonl", output_rows)
    write_jsonl(output / "rejects.jsonl", rejects)
    summary = {
        "schema_version": "drugclip-random-augmentation-pairs-v2",
        "database_contract": "drugclip-exact-peptide-random-conformer-v2",
        "input_pairs": len(split_rows),
        "retained_pairs": len(output_rows),
        "rejects": len(rejects),
        "split_counts": dict(sorted(Counter(row["split"] for row in output_rows).items())),
        "known_positive_identity": "receptor_interface_id + peptide_sequence",
        "true_bound_used_as_training_input": False,
        "peptide_similarity_rules_active": False,
    }
    write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface_pair_splits_jsonl", required=True)
    parser.add_argument("--all_interface_pairs_jsonl", required=True)
    parser.add_argument("--receptor_interfaces_jsonl", required=True)
    parser.add_argument("--random_conformer_cache_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

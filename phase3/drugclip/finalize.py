"""Build the formal DrugCLIP random-conformer dataset from raw structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase3.drugclip.build_interface_pairs import build as build_interface_pairs
from phase3.drugclip.build_random_augmentation_dataset import build as build_random_dataset
from phase3.drugclip.config import DEFAULT_RUN_ROOT
from phase3.drugclip.io_utils import write_json
from phase3.drugclip.random_conformers import build_cache
from phase3.drugclip.split_and_audit import run as split_pairs


def load_resume_summary(path: Path, expected_schema: str) -> dict[str, object]:
    """Load only a build layer created under the expected active contract."""

    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != expected_schema:
        raise ValueError(
            f"incompatible_resume_layer:{path}: expected {expected_schema}, "
            f"found {summary.get('schema_version')}; do not resume from v1"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biological_pairs_jsonl", required=True)
    parser.add_argument("--candidate_evidence_jsonl", required=True)
    parser.add_argument("--expanded_evidence_jsonl", required=True)
    parser.add_argument("--mmcif_root", required=True)
    parser.add_argument("--qbiolip_root", required=True)
    parser.add_argument("--biolip_root", required=True)
    parser.add_argument("--mmseqs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_conformers", type=int, default=10)
    parser.add_argument("--max_generation_attempts", type=int, default=5)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir).resolve()
    allowed_root = DEFAULT_RUN_ROOT.resolve()
    if root != allowed_root and allowed_root not in root.parents:
        raise ValueError(f"DrugCLIP outputs must stay under {allowed_root}: {root}")
    interface_dir = root / "01_interface_pairs"
    split_dir = root / "02_leakage_safe_split"
    cache_dir = root / "03_random_conformer_cache"
    training_dir = root / "04_training_input"
    interface_summary_path = interface_dir / "summary.json"
    if args.resume and interface_summary_path.is_file():
        interface_summary = load_resume_summary(
            interface_summary_path, "drugclip-interface-positive-v2"
        )
    else:
        interface_summary = build_interface_pairs(
            argparse.Namespace(
                biological_pairs_jsonl=args.biological_pairs_jsonl,
                candidate_evidence_jsonl=args.candidate_evidence_jsonl,
                expanded_evidence_jsonl=args.expanded_evidence_jsonl,
                mmcif_root=args.mmcif_root,
                qbiolip_root=args.qbiolip_root,
                biolip_root=args.biolip_root,
                output_dir=str(interface_dir),
                contact_cutoff=6.0,
                min_interface_residues=2,
            )
        )
    split_summary_path = split_dir / "summary.json"
    if args.resume and split_summary_path.is_file():
        split_summary = load_resume_summary(
            split_summary_path, "phase3-drugclip-exact-peptide-split-audit-v2"
        )
    else:
        split_summary = split_pairs(
            argparse.Namespace(
                real_pairs=str(interface_dir / "interface_pairs.jsonl"),
                output_dir=str(split_dir),
                mmseqs=args.mmseqs,
                receptor_min_identity=0.40,
                receptor_coverage=0.60,
                train_fraction=0.80,
                valid_fraction=0.10,
            )
        )
    cache_summary_path = cache_dir / "random_conformer_summary.json"
    cache_path = cache_dir / "random_conformer_cache.jsonl"
    if args.resume and cache_summary_path.is_file() and cache_path.is_file():
        cache_summary = load_resume_summary(
            cache_summary_path, "drugclip-random-conformer-cache-v1"
        )
    else:
        cache_summary = build_cache(
            split_rows_jsonl=split_dir / "pair_splits.jsonl",
            output_dir=cache_dir,
            max_conformers=args.max_conformers,
            max_attempts=args.max_generation_attempts,
            workers=args.workers,
        )
    training_summary_path = training_dir / "summary.json"
    training_pairs_path = training_dir / "random_conformer_pairs.jsonl"
    if args.resume and training_summary_path.is_file() and training_pairs_path.is_file():
        training_summary = load_resume_summary(
            training_summary_path, "drugclip-random-augmentation-pairs-v2"
        )
    else:
        training_summary = build_random_dataset(
            argparse.Namespace(
                interface_pair_splits_jsonl=str(split_dir / "pair_splits.jsonl"),
                all_interface_pairs_jsonl=str(interface_dir / "interface_pairs.jsonl"),
                receptor_interfaces_jsonl=str(interface_dir / "receptor_interfaces.jsonl"),
                random_conformer_cache_jsonl=str(cache_path),
                output_dir=str(training_dir),
            )
        )
    summary = {
        "algorithm": "drugclip-random-conformer-v2",
        "interface_pairs": interface_summary,
        "leakage_safe_split": split_summary,
        "random_conformer_cache": cache_summary,
        "training_input": training_summary,
        "formal_pair_input": str((training_dir / "random_conformer_pairs.jsonl").resolve()),
        "formal_random_cache": str((cache_dir / "random_conformer_cache.jsonl").resolve()),
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "final_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

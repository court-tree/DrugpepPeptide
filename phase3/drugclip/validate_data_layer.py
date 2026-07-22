"""Read-only validation of the formal Phase-3 interface-pair sampling layer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from phase3.drugclip.batching import UniquePeptideBatchSampler, collate_phase3
from phase3.drugclip.io_utils import read_jsonl, write_json, write_jsonl
from phase3.drugclip.random_augmentation_dataset import (
    Phase3RandomConformerDataset,
    sha256_file,
)


def audit_phase2_atom_limit(receptor_interfaces_jsonl: str | Path, limit: int = 256) -> dict[str, Any]:
    """Reproduce Phase-2's exact prefix slice and report its geometry consequences."""
    total = 0
    over_limit = 0
    before_atoms = 0
    after_atoms = 0
    incomplete_backbone = 0
    examples: list[dict[str, Any]] = []
    before_histogram: Counter[int] = Counter()
    after_histogram: Counter[int] = Counter()
    for row in read_jsonl(receptor_interfaces_jsonl):
        total += 1
        atoms = row["receptor_atoms"]
        selected = atoms[:limit]  # Exact current Phase-2 PepCLIP3DDataset rule.
        before = len(atoms)
        after = len(selected)
        before_atoms += before
        after_atoms += after
        before_histogram[before] += 1
        after_histogram[after] += 1
        over_limit += int(before > limit)

        atom_names_by_residue: defaultdict[str, set[str]] = defaultdict(set)
        for atom in selected:
            atom_names_by_residue[str(atom["residue_id"])].add(str(atom["atom_name"]))
        incomplete = sorted(
            residue_id
            for residue_id, names in atom_names_by_residue.items()
            if not {"N", "CA", "C"}.issubset(names)
        )
        if incomplete:
            incomplete_backbone += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "interface_id": str(row.get("pair_id", "")),
                        "atoms_before": before,
                        "atoms_after": after,
                        "incomplete_residue_ids": incomplete,
                    }
                )
    return {
        "status": "phase2_atom_prefix_rule_accepted",
        "source": str(Path(receptor_interfaces_jsonl).resolve()),
        "phase2_rule": "receptor_raw_atoms[:max_receptor_atoms]",
        "limit": limit,
        "interfaces": total,
        "interfaces_over_limit": over_limit,
        "atoms_before_total": before_atoms,
        "atoms_after_total": after_atoms,
        "interfaces_with_incomplete_N_CA_C_after_exact_phase2_slice": incomplete_backbone,
        "geometry_complete_after_exact_phase2_slice": incomplete_backbone == 0,
        "examples": examples,
        "before_atom_count_min": min(before_histogram, default=0),
        "before_atom_count_max": max(before_histogram, default=0),
        "after_atom_count_min": min(after_histogram, default=0),
        "after_atom_count_max": max(after_histogram, default=0),
    }


def validate_epoch(
    dataset: Phase3RandomConformerDataset,
    batch_size: int,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    sampler = UniquePeptideBatchSampler(dataset, batch_size=batch_size, seed=seed, epoch=epoch)
    plan = dataset.epoch_plan()
    interface_ids = [str(row["interface_pair_id"]) for row in plan]
    conformer_indices = [int(row["conformer_index"]) for row in plan]
    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]
    unique_violations = 0
    for batch in batches:
        peptides = [dataset.peptide_sequence_for_index(index) for index in batch]
        unique_violations += int(len(peptides) != len(set(peptides)))

    collated = collate_phase3([dataset[index] for index in batches[0]])
    finite = all(
        bool(torch.isfinite(tensor.float()).all())
        for tensor in (
            collated["one_d"]["receptor_tokens"],
            collated["one_d"]["peptide_tokens"],
            collated["three_d"]["receptor_coords"],
            collated["three_d"]["peptide_coords"],
        )
    )
    return {
        "split": dataset.split,
        "epoch": epoch,
        "global_seed": seed,
        "interface_pairs_expected": len(dataset),
        "interface_pairs_visited": len(flattened),
        "unique_interface_pairs_visited": len(set(interface_ids)),
        "interface_pair_loss_count": len(dataset) - len(set(flattened)),
        "interface_pair_duplicate_count": len(flattened) - len(set(flattened)),
        "interface_pairs_available": dataset.interface_row_count,
        "interface_pairs_planned": len(interface_ids),
        "conformer_index_histogram": dict(sorted(Counter(conformer_indices).items())),
        "batch_summary": sampler.summary(),
        "batch_peptide_uniqueness_violations": unique_violations,
        "first_batch_collate_finite": finite,
        "neutral_compatibility_fields": {
            "peptide_homology_80_id": sorted(set(collated["one_d"]["peptide_homology_80_id"])),
            "receptor_family_30_id": sorted(set(collated["one_d"]["receptor_family_30_id"])),
        },
        "plan_preview": plan[:5],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random_pairs_jsonl", required=True)
    parser.add_argument("--random_conformer_cache_jsonl", required=True)
    parser.add_argument("--biological_pairs_jsonl", required=True)
    parser.add_argument("--pair_splits_jsonl", required=True)
    parser.add_argument("--receptor_interfaces_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data-version", choices=("v2", "v3"), default="v2")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--expected_manifest_sha256", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", choices=("train_random", "fixed"), default="train_random")
    parser.add_argument("--fixed_conformer_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--epoch", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / f"{args.split}_{args.mode}"
    dataset = Phase3RandomConformerDataset(
        args.random_pairs_jsonl,
        args.random_conformer_cache_jsonl,
        args.biological_pairs_jsonl,
        args.pair_splits_jsonl,
        split=args.split,
        mode=args.mode,
        global_seed=args.seed,
        fixed_conformer_index=args.fixed_conformer_index if args.mode == "fixed" else None,
        data_version=args.data_version,
        dataset_root=args.dataset_root,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    dataset.write_mapping_summary(output_dir / "interface_pair_mapping_summary.json")
    epoch_summary = validate_epoch(dataset, args.batch_size, args.seed, args.epoch)
    epoch_zero = dataset.epoch_plan()
    plan_path = output_dir / f"epoch_{args.epoch:03d}_sampling_plan.jsonl"
    write_jsonl(plan_path, epoch_zero)
    epoch_summary["sampling_plan"] = {
        "path": str(plan_path.resolve()),
        "sha256": sha256_file(plan_path),
        "fields": sorted(epoch_zero[0]),
    }
    dataset.set_epoch(args.epoch)
    epoch_summary["same_seed_epoch_reproduced_exactly"] = dataset.epoch_plan() == epoch_zero
    dataset.set_epoch(args.epoch + 1)
    epoch_one_by_interface_pair = {
        row["interface_pair_id"]: row for row in dataset.epoch_plan()
    }
    changed_conformers = 0
    for row in epoch_zero:
        other = epoch_one_by_interface_pair[row["interface_pair_id"]]
        changed_conformers += int(row["conformer_index"] != other["conformer_index"])
    epoch_summary["next_epoch_changes"] = {
        "interface_pairs_retained": len(epoch_zero),
        "conformer_changed_interface_pairs": changed_conformers,
    }
    write_json(output_dir / "epoch_sampling_summary.json", epoch_summary)
    write_json(
        output_dir / "phase2_atom_limit_audit.json",
        audit_phase2_atom_limit(args.receptor_interfaces_jsonl),
    )


if __name__ == "__main__":
    main()

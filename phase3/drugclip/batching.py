"""Peptide-unique batching and Phase-2-compatible Phase-3 collation."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import random
from typing import Any, Iterator

import torch
from torch.utils.data import Sampler

from phase2.pepclip.data import (
    atom_tensors,
    collate_pepclip,
    collate_pepclip_3d,
    encode_sequence,
)
from phase3.drugclip.random_augmentation_dataset import Phase3RandomConformerDataset


# Formal Phase-2 v9 concat-fusion configuration; preserve its atom-level rule.
PHASE2_MAX_RECEPTOR_ATOMS = 256
PHASE2_MAX_PEPTIDE_ATOMS = 192


def _seed(seed: int, epoch: int, label: str) -> int:
    material = f"{seed}|{epoch}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


class UniquePeptideBatchSampler(Sampler[list[int]]):
    """Yield every interface pair once while delaying same-peptide conflicts."""

    def __init__(
        self,
        dataset: Phase3RandomConformerDataset,
        batch_size: int,
        seed: int = 0,
        epoch: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self._batches: list[list[int]] = []
        self.set_epoch(epoch)

    def _try_build(self, num_batches: int) -> list[list[int]] | None:
        rng = random.Random(_seed(self.seed, self.epoch, "unique_peptide_batches"))
        by_peptide: defaultdict[str, list[int]] = defaultdict(list)
        for index in range(len(self.dataset)):
            by_peptide[self.dataset.peptide_sequence_for_index(index)].append(index)
        groups = list(by_peptide.items())
        for _, indices in groups:
            rng.shuffle(indices)
        rng.shuffle(groups)
        groups.sort(key=lambda item: -len(item[1]))

        batches: list[list[int]] = [[] for _ in range(num_batches)]
        batch_peptides: list[set[str]] = [set() for _ in range(num_batches)]
        tie_break = list(range(num_batches))
        rng.shuffle(tie_break)
        tie_rank = {batch_index: rank for rank, batch_index in enumerate(tie_break)}

        for peptide, indices in groups:
            candidates = [
                batch_index
                for batch_index in range(num_batches)
                if len(batches[batch_index]) < self.batch_size
                and peptide not in batch_peptides[batch_index]
            ]
            candidates.sort(key=lambda idx: (len(batches[idx]), tie_rank[idx], idx))
            if len(candidates) < len(indices):
                return None
            for index, batch_index in zip(indices, candidates[: len(indices)]):
                batches[batch_index].append(index)
                batch_peptides[batch_index].add(peptide)

        batches = [batch for batch in batches if batch]
        for batch in batches:
            rng.shuffle(batch)
        rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.set_epoch(self.epoch)
        peptide_counts: defaultdict[str, int] = defaultdict(int)
        for index in range(len(self.dataset)):
            peptide_counts[self.dataset.peptide_sequence_for_index(index)] += 1
        minimum_batches = max(
            math.ceil(len(self.dataset) / self.batch_size),
            max(peptide_counts.values(), default=0),
        )
        batches = None
        for num_batches in range(minimum_batches, len(self.dataset) + 1):
            batches = self._try_build(num_batches)
            if batches is not None:
                break
        if batches is None:
            raise RuntimeError("unable_to_construct_unique_peptide_batches")

        flattened = [index for batch in batches for index in batch]
        if len(flattened) != len(self.dataset) or len(set(flattened)) != len(self.dataset):
            raise RuntimeError("interface_pair_loss_or_duplication_in_batch_plan")
        for batch in batches:
            peptides = [self.dataset.peptide_sequence_for_index(index) for index in batch]
            if len(peptides) != len(set(peptides)):
                raise RuntimeError("duplicate_peptide_in_batch_plan")
        self._batches = batches

    def summary(self) -> dict[str, Any]:
        sizes = [len(batch) for batch in self._batches]
        flattened = [index for batch in self._batches for index in batch]
        return {
            "epoch": self.epoch,
            "interface_pairs": len(self.dataset),
            "batches": len(self._batches),
            "batch_size_requested": self.batch_size,
            "batch_size_min": min(sizes, default=0),
            "batch_size_max": max(sizes, default=0),
            "interface_pair_loss_count": len(self.dataset) - len(set(flattened)),
            "interface_pair_duplicate_count": len(flattened) - len(set(flattened)),
            "peptide_uniqueness_violations": 0,
            "peptide_conflict_delay_interface_pairs": sum(
                max(count - 1, 0) for count in self._peptide_counts().values()
            ),
        }

    def _peptide_counts(self) -> defaultdict[str, int]:
        counts: defaultdict[str, int] = defaultdict(int)
        for index in range(len(self.dataset)):
            counts[self.dataset.peptide_sequence_for_index(index)] += 1
        return counts

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def pool_item_to_fusion_item(row: dict[str, Any]) -> dict[str, Any]:
    receptor_atoms = atom_tensors(row["receptor_atoms"][:PHASE2_MAX_RECEPTOR_ATOMS])
    peptide_atoms = atom_tensors(row["peptide_atoms"][:PHASE2_MAX_PEPTIDE_ATOMS])
    common = {
        "sample_id": str(row["real_pair_id"]),
        "pdb_id": "",
        "split": str(row.get("split", "")),
        "receptor_key": str(row["receptor_key"]),
        "peptide_key": str(row["peptide_sequence"]),
        "conformer_cluster_id": str(row["conformer_cluster_id"]),
        "peptide_sequence_id": str(row["peptide_sequence"]),
        # Required by the generic Phase-2 collator, but deliberately neutral.
        # Phase-3 sampling, masks, loss, and split logic never consume them.
        "peptide_homology_80_id": "",
        "receptor_family_30_id": "",
        "receptor_interface_key": str(row["receptor_key"]),
        "receptor_sequence": str(row["receptor_patch_sequence"]),
        "peptide_sequence": str(row["peptide_sequence"]),
    }
    one_d = {
        **common,
        "receptor_tokens": encode_sequence(str(row["receptor_patch_sequence"])),
        "peptide_tokens": encode_sequence(str(row["peptide_sequence"])),
        "peptide_length": len(str(row["peptide_sequence"])),
        "avg_contact_count": 0.0,
        "contact_coverage": 0.0,
        "known_positive_group": row.get("known_positive_group", {}),
        # Formal ``iface:...`` ID; this is the only receptor candidate ID
        # accepted by peptide_to_receptors known-positive matching.
        "receptor_interface_id": str(row["receptor_id"]),
        "receptor_id": str(row["receptor_id"]),
        "biological_pair_id": str(row["biological_pair_id"]),
        "interface_pair_id": str(row["interface_pair_id"]),
        "conformer_index": int(row["conformer_index"]),
    }
    three_d = {
        **common,
        "receptor_coords": receptor_atoms["coords"],
        "receptor_elements": receptor_atoms["elements"],
        "receptor_atom_names": receptor_atoms["atom_names"],
        "receptor_residue_names": receptor_atoms["residue_names"],
        "peptide_coords": peptide_atoms["coords"],
        "peptide_elements": peptide_atoms["elements"],
        "peptide_atom_names": peptide_atoms["atom_names"],
        "peptide_residue_names": peptide_atoms["residue_names"],
        "num_receptor_atoms": receptor_atoms["coords"].shape[0],
        "num_peptide_atoms": peptide_atoms["coords"].shape[0],
    }
    return {"one_d": one_d, "three_d": three_d}


def collate_phase3(batch: list[dict[str, Any]]) -> dict[str, Any]:
    peptide_sequences = [str(item["peptide_sequence"]) for item in batch]
    if len(peptide_sequences) != len(set(peptide_sequences)):
        raise ValueError("Phase-3 batch contains duplicate peptide_sequence")
    converted = [pool_item_to_fusion_item(item) for item in batch]
    one_d_items = [item["one_d"] for item in converted]
    three_d_items = [item["three_d"] for item in converted]
    one_d = collate_pepclip(one_d_items)
    for key in (
        "known_positive_group",
        "receptor_interface_id",
        "receptor_id",
        "biological_pair_id",
        "interface_pair_id",
        "conformer_index",
    ):
        one_d[key] = [item[key] for item in one_d_items]
    one_d["conformer_source_kind"] = [str(item["conformer_source_kind"]) for item in batch]
    one_d["conformer_cluster_id"] = [str(item["conformer_cluster_id"]) for item in batch]
    three_d = collate_pepclip_3d(three_d_items)
    if one_d["sample_id"] != three_d["sample_id"]:
        raise ValueError("Phase-3 batch one_d/three_d sample_id mismatch")
    tensor_values = [
        one_d["receptor_tokens"],
        one_d["peptide_tokens"],
        three_d["receptor_coords"],
        three_d["peptide_coords"],
    ]
    if any(not torch.isfinite(value.float()).all() for value in tensor_values):
        raise ValueError("Phase-3 collate produced nonfinite tensors")
    return {"one_d": one_d, "three_d": three_d}

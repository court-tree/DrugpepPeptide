from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
PAD_TOKEN = 0
UNK_TOKEN = 1
AA_TO_ID = {aa: i + 2 for i, aa in enumerate(AMINO_ACIDS)}


def encode_sequence(sequence: str) -> torch.Tensor:
    ids = [AA_TO_ID.get(ch.upper(), UNK_TOKEN) for ch in sequence if ch.strip()]
    if not ids:
        ids = [UNK_TOKEN]
    return torch.tensor(ids, dtype=torch.long)


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


class PepCLIPDataset(Dataset):
    def __init__(
        self,
        metadata_jsonl: str | Path,
        split: str | None = None,
        receptor_field: str = "receptor_sequence",
        peptide_field: str = "peptide_sequence",
    ) -> None:
        self.metadata_jsonl = Path(metadata_jsonl)
        self.split = split
        self.receptor_field = receptor_field
        self.peptide_field = peptide_field
        self.rows: List[Dict[str, Any]] = []

        for row in read_jsonl(self.metadata_jsonl):
            if split is not None and row.get("split") != split:
                continue
            receptor_sequence = str(row.get(receptor_field, ""))
            peptide_sequence = str(row.get(peptide_field, ""))
            if receptor_sequence and peptide_sequence:
                self.rows.append(row)

        if not self.rows:
            raise ValueError(f"No usable rows found in {self.metadata_jsonl} for split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        receptor_sequence = str(row[self.receptor_field])
        peptide_sequence = str(row[self.peptide_field])
        receptor_key = "|".join(
            [
                str(row.get("source_file", "")),
                str(row.get("receptor_chain_id", "")),
                str(row.get("track_b_patch_residue_ids", "")),
            ]
        )
        return {
            "sample_id": str(row.get("candidate_id", index)),
            "pdb_id": str(row.get("pdb_id", "")),
            "split": str(row.get("split", "")),
            "receptor_key": receptor_key,
            "peptide_key": peptide_sequence,
            "receptor_tokens": encode_sequence(receptor_sequence),
            "peptide_tokens": encode_sequence(peptide_sequence),
            "peptide_length": int(row.get("peptide_length", len(peptide_sequence))),
            "avg_contact_count": float(row.get("avg_contact_count", 0.0)),
            "contact_coverage": float(row.get("contact_coverage", 0.0)),
        }


def collate_pepclip(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    receptor_tokens = pad_sequence(
        [item["receptor_tokens"] for item in batch],
        batch_first=True,
        padding_value=PAD_TOKEN,
    )
    peptide_tokens = pad_sequence(
        [item["peptide_tokens"] for item in batch],
        batch_first=True,
        padding_value=PAD_TOKEN,
    )
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "pdb_id": [item["pdb_id"] for item in batch],
        "split": [item["split"] for item in batch],
        "receptor_key": [item["receptor_key"] for item in batch],
        "peptide_key": [item["peptide_key"] for item in batch],
        "receptor_tokens": receptor_tokens,
        "peptide_tokens": peptide_tokens,
        "peptide_length": torch.tensor([item["peptide_length"] for item in batch], dtype=torch.long),
        "avg_contact_count": torch.tensor([item["avg_contact_count"] for item in batch], dtype=torch.float),
        "contact_coverage": torch.tensor([item["contact_coverage"] for item in batch], dtype=torch.float),
    }


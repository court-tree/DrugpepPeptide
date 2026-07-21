from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
PAD_TOKEN = 0
UNK_TOKEN = 1
AA_TO_ID = {aa: i + 2 for i, aa in enumerate(AMINO_ACIDS)}

ELEMENTS = [
    "C",
    "N",
    "O",
    "S",
    "P",
    "SE",
    "MG",
    "ZN",
    "CA",
    "FE",
    "CL",
    "NA",
    "K",
]
ELEMENT_PAD_TOKEN = 0
ELEMENT_UNK_TOKEN = 1
ELEMENT_TO_ID = {element: i + 2 for i, element in enumerate(ELEMENTS)}

ATOM_NAMES = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "CD",
    "CD1",
    "CD2",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "CZ",
    "CZ2",
    "CZ3",
    "CH2",
    "ND1",
    "ND2",
    "NE",
    "NE1",
    "NE2",
    "NH1",
    "NH2",
    "NZ",
    "OD1",
    "OD2",
    "OE1",
    "OE2",
    "OG",
    "OG1",
    "OH",
    "SD",
    "SG",
    "OXT",
]
RESIDUE_NAMES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "ASX",
    "GLX",
    "SEC",
    "PYL",
]
ATOM_NAME_PAD_TOKEN = 0
ATOM_NAME_UNK_TOKEN = 1
ATOM_NAME_TO_ID = {name: i + 2 for i, name in enumerate(ATOM_NAMES)}
RESIDUE_NAME_PAD_TOKEN = 0
RESIDUE_NAME_UNK_TOKEN = 1
RESIDUE_NAME_TO_ID = {name: i + 2 for i, name in enumerate(RESIDUE_NAMES)}


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


class TrackAReader:
    def __init__(self, path: str | Path, data_format: str = "jsonl") -> None:
        self.path = Path(path)
        self.data_format = data_format

    def rows(self) -> Iterable[Dict[str, Any]]:
        if self.data_format == "jsonl":
            yield from read_jsonl(self.path)
            return
        if self.data_format == "lmdb":
            yield from self._read_lmdb()
            return
        raise ValueError(f"Unsupported data_format={self.data_format!r}; expected jsonl or lmdb")

    def _read_lmdb(self) -> Iterable[Dict[str, Any]]:
        try:
            import lmdb
        except ImportError as exc:
            raise ImportError("Reading Step8 LMDB requires the 'lmdb' Python package") from exc

        env = lmdb.open(
            str(self.path),
            readonly=True,
            lock=False,
            subdir=True,
            readahead=False,
            max_readers=1,
        )
        try:
            with env.begin() as txn:
                raw_keys = txn.get(b"__keys__")
                if raw_keys is None:
                    raise ValueError(f"LMDB at {self.path} is missing __keys__")
                keys = json.loads(raw_keys.decode("utf-8"))
                for key in keys:
                    raw_value = txn.get(str(key).encode("utf-8"))
                    if raw_value is None:
                        raise KeyError(f"LMDB at {self.path} is missing key {key!r}")
                    yield json.loads(raw_value.decode("utf-8"))
        finally:
            env.close()


class TrackBReader(TrackAReader):
    pass


class PepCLIPDataset(Dataset):
    def __init__(
        self,
        track_a_path: str | Path,
        split: str | None = None,
        data_format: str = "jsonl",
        receptor_field: str = "receptor_patch_sequence",
        peptide_field: str = "peptide_sequence",
    ) -> None:
        self.track_a_path = Path(track_a_path)
        self.data_format = data_format
        self.split = split
        self.receptor_field = receptor_field
        self.peptide_field = peptide_field
        self.rows: List[Dict[str, Any]] = []

        for row in TrackAReader(self.track_a_path, data_format=data_format).rows():
            if split is not None and row.get("split") != split:
                continue
            receptor_sequence = str(row.get(receptor_field) or row.get("receptor_sequence", ""))
            peptide_sequence = str(row.get(peptide_field, ""))
            if receptor_sequence and peptide_sequence:
                self.rows.append(row)

        if not self.rows:
            raise ValueError(f"No usable rows found in {self.track_a_path} for split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        receptor_sequence = str(row.get(self.receptor_field) or row.get("receptor_sequence", ""))
        peptide_sequence = str(row[self.peptide_field])
        patch_indices = row.get("receptor_patch_seq_indices", row.get("track_b_patch_residue_ids", ""))
        patch_ids = row.get("receptor_patch_residue_ids", row.get("track_b_patch_residue_ids", ""))
        receptor_key = "|".join(
            [
                str(row.get("source_file", "")),
                str(row.get("receptor_chain_id", "")),
                str(patch_indices or patch_ids),
            ]
        )
        sample_id = row.get("sample_id", row.get("candidate_id", index))
        return {
            "sample_id": str(sample_id),
            "pdb_id": str(row.get("pdb_id", "")),
            "split": str(row.get("split", "")),
            "receptor_key": receptor_key,
            "peptide_key": peptide_sequence,
            "conformer_cluster_id": str(row.get("conformer_cluster_id", "")),
            "peptide_sequence_id": str(row.get("peptide_sequence_id", peptide_sequence)),
            "peptide_homology_80_id": str(row.get("peptide_homology_80_id", "")),
            "receptor_family_30_id": str(row.get("receptor_family_30_id", "")),
            "receptor_interface_key": str(row.get("receptor_interface_key", receptor_key)),
            "receptor_sequence": receptor_sequence,
            "peptide_sequence": peptide_sequence,
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
        "conformer_cluster_id": [item["conformer_cluster_id"] for item in batch],
        "peptide_sequence_id": [item["peptide_sequence_id"] for item in batch],
        "peptide_homology_80_id": [item["peptide_homology_80_id"] for item in batch],
        "receptor_family_30_id": [item["receptor_family_30_id"] for item in batch],
        "receptor_interface_key": [item["receptor_interface_key"] for item in batch],
        "receptor_sequence": [item["receptor_sequence"] for item in batch],
        "peptide_sequence": [item["peptide_sequence"] for item in batch],
        "receptor_tokens": receptor_tokens,
        "peptide_tokens": peptide_tokens,
        "peptide_length": torch.tensor([item["peptide_length"] for item in batch], dtype=torch.long),
        "avg_contact_count": torch.tensor([item["avg_contact_count"] for item in batch], dtype=torch.float),
        "contact_coverage": torch.tensor([item["contact_coverage"] for item in batch], dtype=torch.float),
    }


def encode_element(element: str) -> int:
    return ELEMENT_TO_ID.get(str(element).strip().upper(), ELEMENT_UNK_TOKEN)


def encode_atom_name(atom_name: str) -> int:
    return ATOM_NAME_TO_ID.get(str(atom_name).strip().upper(), ATOM_NAME_UNK_TOKEN)


def encode_residue_name(residue_name: str) -> int:
    return RESIDUE_NAME_TO_ID.get(str(residue_name).strip().upper(), RESIDUE_NAME_UNK_TOKEN)


def atom_tensors(atoms: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if not atoms:
        return {
            "coords": torch.zeros((1, 3), dtype=torch.float32),
            "elements": torch.tensor([ELEMENT_UNK_TOKEN], dtype=torch.long),
            "atom_names": torch.tensor([ATOM_NAME_UNK_TOKEN], dtype=torch.long),
            "residue_names": torch.tensor([RESIDUE_NAME_UNK_TOKEN], dtype=torch.long),
        }
    coords = torch.tensor(
        [[float(atom["x"]), float(atom["y"]), float(atom["z"])] for atom in atoms],
        dtype=torch.float32,
    )
    elements = torch.tensor([encode_element(str(atom.get("element", ""))) for atom in atoms], dtype=torch.long)
    atom_names = torch.tensor([encode_atom_name(str(atom.get("atom_name", ""))) for atom in atoms], dtype=torch.long)
    residue_names = torch.tensor(
        [encode_residue_name(str(atom.get("residue_name", ""))) for atom in atoms],
        dtype=torch.long,
    )
    return {
        "coords": coords,
        "elements": elements,
        "atom_names": atom_names,
        "residue_names": residue_names,
    }


class PepCLIP3DDataset(Dataset):
    def __init__(
        self,
        track_b_path: str | Path,
        split: str | None = None,
        data_format: str = "jsonl",
        max_receptor_atoms: int | None = None,
        max_peptide_atoms: int | None = None,
    ) -> None:
        self.track_b_path = Path(track_b_path)
        self.data_format = data_format
        self.split = split
        self.max_receptor_atoms = max_receptor_atoms
        self.max_peptide_atoms = max_peptide_atoms
        self.rows: List[Dict[str, Any]] = []

        for row in TrackBReader(self.track_b_path, data_format=data_format).rows():
            if split is not None and row.get("split") != split:
                continue
            if (row.get("patch_atoms") or row.get("receptor_atoms")) and row.get("peptide_atoms"):
                self.rows.append(row)

        if not self.rows:
            raise ValueError(f"No usable 3D rows found in {self.track_b_path} for split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        receptor_raw_atoms = row.get("patch_atoms") or row["receptor_atoms"]
        peptide_raw_atoms = row["peptide_atoms"]
        if self.max_receptor_atoms is not None:
            receptor_raw_atoms = receptor_raw_atoms[: self.max_receptor_atoms]
        if self.max_peptide_atoms is not None:
            peptide_raw_atoms = peptide_raw_atoms[: self.max_peptide_atoms]
        receptor_atoms = atom_tensors(receptor_raw_atoms)
        peptide_atoms = atom_tensors(peptide_raw_atoms)
        patch_residue_ids = row.get("patch_residue_ids", [])
        peptide_residue_ids = row.get("peptide_residue_ids", [])
        receptor_key = row.get("receptor_key") or "|".join(
            [
                str(row.get("source_file", "")),
                str(row.get("receptor_chain_id", "")),
                str(patch_residue_ids),
            ]
        )
        peptide_key = row.get("peptide_key") or "|".join(
            [
                str(row.get("source_file", "")),
                str(row.get("peptide_source_chain_id", "")),
                str(peptide_residue_ids),
            ]
        )
        sample_id = row.get("sample_id", row.get("candidate_id", index))
        return {
            "sample_id": str(sample_id),
            "pdb_id": str(row.get("pdb_id", "")),
            "split": str(row.get("split", "")),
            "receptor_key": receptor_key,
            "peptide_key": peptide_key,
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
            "conformer_cluster_id": str(row.get("conformer_cluster_id", "")),
            "peptide_homology_80_id": str(row.get("peptide_homology_80_id", "")),
            "receptor_family_30_id": str(row.get("receptor_family_30_id", "")),
            "receptor_interface_key": str(row.get("receptor_interface_key", receptor_key)),
            "peptide_sequence_id": str(row.get("peptide_sequence_id", row.get("peptide_key", ""))),
        }


class PepCLIP3DConformerDataset(Dataset):
    def __init__(
        self,
        base: PepCLIP3DDataset,
        conformer_views_path: str | Path,
        conformer_evidence_path: str | Path,
        max_auxiliary_conformers_per_sample: int = 1,
    ) -> None:
        if max_auxiliary_conformers_per_sample < 1:
            raise ValueError("max_auxiliary_conformers_per_sample must be >= 1")
        self.base = base
        self.rows = base.rows
        self.views = {str(row["sample_id"]): row for row in read_jsonl(conformer_views_path)}
        self.evidence = {str(row["conformer_id"]): row for row in read_jsonl(conformer_evidence_path)}
        self.max_auxiliary_conformers_per_sample = max_auxiliary_conformers_per_sample
        missing = [
            str(row.get("sample_id", row.get("candidate_id", index)))
            for index, row in enumerate(base.rows)
            if str(row.get("sample_id", row.get("candidate_id", index))) not in self.views
        ]
        if missing:
            raise ValueError(f"Conformer views missing {len(missing)} base samples; first={missing[:3]}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.base[index]
        view = self.views[item["sample_id"]]
        primary = self.evidence[str(view["primary_bound_conformer_id"])]
        auxiliary_ids = list(view.get("auxiliary_conformer_ids", []))
        enabled = bool(view.get("auxiliary_consistency_enabled") and auxiliary_ids)
        if enabled:
            sample_count = min(self.max_auxiliary_conformers_per_sample, len(auxiliary_ids))
            sampled_auxiliary_ids = random.sample(auxiliary_ids, k=sample_count)
        else:
            sampled_auxiliary_ids = []
        auxiliary = self.evidence[sampled_auxiliary_ids[0]] if sampled_auxiliary_ids else primary
        primary_atoms = atom_tensors(primary["backbone_atoms"])
        auxiliary_atoms = atom_tensors(auxiliary["backbone_atoms"])
        auxiliary_multi = [atom_tensors(self.evidence[conformer_id]["backbone_atoms"]) for conformer_id in sampled_auxiliary_ids]
        item.update(
            {
                "view_primary_coords": primary_atoms["coords"],
                "view_primary_elements": primary_atoms["elements"],
                "view_primary_atom_names": primary_atoms["atom_names"],
                "view_primary_residue_names": primary_atoms["residue_names"],
                "view_auxiliary_coords": auxiliary_atoms["coords"],
                "view_auxiliary_elements": auxiliary_atoms["elements"],
                "view_auxiliary_atom_names": auxiliary_atoms["atom_names"],
                "view_auxiliary_residue_names": auxiliary_atoms["residue_names"],
                "view_auxiliary_multi": auxiliary_multi,
                "view_auxiliary_sampled_count": len(sampled_auxiliary_ids),
                "view_auxiliary_sampled_ids": sampled_auxiliary_ids,
                "max_auxiliary_conformers_per_sample": self.max_auxiliary_conformers_per_sample,
                "consistency_enabled": enabled,
            }
        )
        return item


def pad_atom_clouds(
    coords: List[torch.Tensor],
    elements: List[torch.Tensor],
    atom_names: List[torch.Tensor],
    residue_names: List[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    batch_size = len(coords)
    max_atoms = max(item.shape[0] for item in coords)
    padded_coords = torch.zeros((batch_size, max_atoms, 3), dtype=torch.float32)
    padded_elements = torch.full((batch_size, max_atoms), ELEMENT_PAD_TOKEN, dtype=torch.long)
    padded_atom_names = torch.full((batch_size, max_atoms), ATOM_NAME_PAD_TOKEN, dtype=torch.long)
    padded_residue_names = torch.full((batch_size, max_atoms), RESIDUE_NAME_PAD_TOKEN, dtype=torch.long)
    mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    for i, (coord, element, atom_name, residue_name) in enumerate(zip(coords, elements, atom_names, residue_names)):
        n_atoms = coord.shape[0]
        padded_coords[i, :n_atoms] = coord
        padded_elements[i, :n_atoms] = element
        padded_atom_names[i, :n_atoms] = atom_name
        padded_residue_names[i, :n_atoms] = residue_name
        mask[i, :n_atoms] = True
    return {
        "coords": padded_coords,
        "elements": padded_elements,
        "atom_names": padded_atom_names,
        "residue_names": padded_residue_names,
        "mask": mask,
    }


def collate_pepclip_3d(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    receptor = pad_atom_clouds(
        [item["receptor_coords"] for item in batch],
        [item["receptor_elements"] for item in batch],
        [item["receptor_atom_names"] for item in batch],
        [item["receptor_residue_names"] for item in batch],
    )
    peptide = pad_atom_clouds(
        [item["peptide_coords"] for item in batch],
        [item["peptide_elements"] for item in batch],
        [item["peptide_atom_names"] for item in batch],
        [item["peptide_residue_names"] for item in batch],
    )
    output = {
        "sample_id": [item["sample_id"] for item in batch],
        "pdb_id": [item["pdb_id"] for item in batch],
        "split": [item["split"] for item in batch],
        "receptor_key": [item["receptor_key"] for item in batch],
        "peptide_key": [item["peptide_key"] for item in batch],
        "receptor_coords": receptor["coords"],
        "receptor_elements": receptor["elements"],
        "receptor_atom_names": receptor["atom_names"],
        "receptor_residue_names": receptor["residue_names"],
        "receptor_mask": receptor["mask"],
        "peptide_coords": peptide["coords"],
        "peptide_elements": peptide["elements"],
        "peptide_atom_names": peptide["atom_names"],
        "peptide_residue_names": peptide["residue_names"],
        "peptide_mask": peptide["mask"],
        "num_receptor_atoms": torch.tensor([item["num_receptor_atoms"] for item in batch], dtype=torch.long),
        "num_peptide_atoms": torch.tensor([item["num_peptide_atoms"] for item in batch], dtype=torch.long),
        "conformer_cluster_id": [item.get("conformer_cluster_id", "") for item in batch],
        "peptide_sequence_id": [item.get("peptide_sequence_id", "") for item in batch],
        "peptide_homology_80_id": [item.get("peptide_homology_80_id", "") for item in batch],
        "receptor_family_30_id": [item.get("receptor_family_30_id", "") for item in batch],
        "receptor_interface_key": [item.get("receptor_interface_key", "") for item in batch],
    }
    if "view_primary_coords" in batch[0]:
        primary = pad_atom_clouds(
            [item["view_primary_coords"] for item in batch],
            [item["view_primary_elements"] for item in batch],
            [item["view_primary_atom_names"] for item in batch],
            [item["view_primary_residue_names"] for item in batch],
        )
        auxiliary = pad_atom_clouds(
            [item["view_auxiliary_coords"] for item in batch],
            [item["view_auxiliary_elements"] for item in batch],
            [item["view_auxiliary_atom_names"] for item in batch],
            [item["view_auxiliary_residue_names"] for item in batch],
        )
        for prefix, values in (("view_primary", primary), ("view_auxiliary", auxiliary)):
            output[f"{prefix}_coords"] = values["coords"]
            output[f"{prefix}_elements"] = values["elements"]
            output[f"{prefix}_atom_names"] = values["atom_names"]
            output[f"{prefix}_residue_names"] = values["residue_names"]
            output[f"{prefix}_mask"] = values["mask"]
        output["consistency_enabled"] = torch.tensor(
            [bool(item["consistency_enabled"]) for item in batch],
            dtype=torch.bool,
        )
        multi_coords = []
        multi_elements = []
        multi_atom_names = []
        multi_residue_names = []
        owner_index = []
        for batch_index, item in enumerate(batch):
            for auxiliary_item in item.get("view_auxiliary_multi", []):
                multi_coords.append(auxiliary_item["coords"])
                multi_elements.append(auxiliary_item["elements"])
                multi_atom_names.append(auxiliary_item["atom_names"])
                multi_residue_names.append(auxiliary_item["residue_names"])
                owner_index.append(batch_index)
        if multi_coords:
            multi_auxiliary = pad_atom_clouds(
                multi_coords,
                multi_elements,
                multi_atom_names,
                multi_residue_names,
            )
            output["view_auxiliary_multi_coords"] = multi_auxiliary["coords"]
            output["view_auxiliary_multi_elements"] = multi_auxiliary["elements"]
            output["view_auxiliary_multi_atom_names"] = multi_auxiliary["atom_names"]
            output["view_auxiliary_multi_residue_names"] = multi_auxiliary["residue_names"]
            output["view_auxiliary_multi_mask"] = multi_auxiliary["mask"]
            output["view_auxiliary_owner_index"] = torch.tensor(owner_index, dtype=torch.long)
            output["view_auxiliary_sampled_count"] = torch.tensor(
                [int(item.get("view_auxiliary_sampled_count", 0)) for item in batch],
                dtype=torch.long,
            )
    return output

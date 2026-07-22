"""Relation-balanced Phase-3 Dataset backed by explicit data-version contracts."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Literal

from torch.utils.data import Dataset

from phase3.drugclip.data_contract import (
    DataContract,
    assert_path_matches_manifest,
    load_data_contract,
)
from phase3.drugclip.io_utils import read_jsonl
from phase3.drugclip.random_conformer_v3 import clash15_details
from phase3.drugclip.random_conformers import AA1_TO_3


RELATION_SCHEMA = "drugclip-random-augmentation-pairs-v2"
DATABASE_CONTRACT = "drugclip-exact-peptide-random-conformer-v2"
CACHE_SCHEMA = "drugclip-random-conformer-cache-v1"
EXPECTED_CONFORMERS = 10
DatasetMode = Literal["train_random", "fixed"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _rng(global_seed: int, epoch: int, label: str, identity: str = "") -> random.Random:
    material = f"{global_seed}|{epoch}|{label}|{identity}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(seed)


def _validate_backbone_format(
    conformer: dict[str, Any],
    peptide_sequence: str,
    *,
    generator_id: str,
    require_clash15: bool,
) -> None:
    if str(conformer.get("generator_id")) != generator_id:
        raise ValueError("cache_conformer_generator_mismatch")
    if str(conformer.get("peptide_sequence")) != peptide_sequence:
        raise ValueError("cache_conformer_sequence_mismatch")
    atoms = conformer.get("backbone_atoms")
    if not isinstance(atoms, list) or len(atoms) != 3 * len(peptide_sequence):
        raise ValueError("cache_backbone_atom_count_mismatch")
    for residue_index, amino_acid in enumerate(peptide_sequence, start=1):
        triplet = atoms[(residue_index - 1) * 3 : residue_index * 3]
        if [str(atom.get("atom_name")) for atom in triplet] != ["N", "CA", "C"]:
            raise ValueError("cache_backbone_atom_order_mismatch")
        for atom in triplet:
            if str(atom.get("residue_id")) != f"P:{residue_index}":
                raise ValueError("cache_backbone_residue_id_mismatch")
            if str(atom.get("residue_name")) != AA1_TO_3[amino_acid]:
                raise ValueError("cache_backbone_residue_name_mismatch")
            if any(not math.isfinite(float(atom[axis])) for axis in ("x", "y", "z")):
                raise ValueError("cache_nonfinite_coordinate")
    if require_clash15 and clash15_details(atoms)["has_clash"]:
        raise ValueError("cache_clash15_violation")


def _validate_cache_row(
    row: dict[str, Any],
    seen_conformer_ids: set[str],
    contract: DataContract,
) -> None:
    cache_id = str(row.get("cache_id") or "")
    peptide_sequence = str(row.get("peptide_sequence") or "")
    if not cache_id:
        raise ValueError("cache_id_missing")
    if row.get("schema_version") != contract.cache_schema:
        raise ValueError(f"unsupported_cache_schema:{row.get('schema_version')}")
    if row.get("generator_id") != contract.generator_id:
        raise ValueError(f"unsupported_cache_generator:{row.get('generator_id')}")
    if int(row.get("max_conformers_requested", -1)) != EXPECTED_CONFORMERS:
        raise ValueError("cache_max_conformers_requested_mismatch")
    conformers = row.get("conformers")
    if not isinstance(conformers, list) or len(conformers) != EXPECTED_CONFORMERS:
        raise ValueError("cache_requires_exactly_10_conformers")
    indices = [int(item.get("conformer_index", -1)) for item in conformers]
    if sorted(indices) != list(range(EXPECTED_CONFORMERS)):
        raise ValueError("cache_conformer_indices_must_be_0_to_9")
    local_ids: set[str] = set()
    for conformer in conformers:
        conformer_id = str(conformer.get("conformer_id") or "")
        if not conformer_id or conformer_id in local_ids or conformer_id in seen_conformer_ids:
            raise ValueError("cache_duplicate_conformer_id")
        local_ids.add(conformer_id)
        seen_conformer_ids.add(conformer_id)
        _validate_backbone_format(
            conformer,
            peptide_sequence,
            generator_id=contract.generator_id,
            require_clash15=contract.data_version == "v3",
        )


def materialize_random_conformer(
    row: dict[str, Any],
    conformer: dict[str, Any],
    biological_pair_id: str,
) -> dict[str, Any]:
    pair = row["pair"]
    interface = row["interface"]
    peptide = str(pair["peptide_sequence"])
    return {
        "sample_id": pair["pair_id"],
        "real_pair_id": pair["pair_id"],
        "biological_pair_id": biological_pair_id,
        "interface_pair_id": pair["pair_id"],
        "conformer_index": int(conformer["conformer_index"]),
        "split": row["split"],
        "receptor_id": pair["receptor_id"],
        "receptor_sequence": pair["receptor_sequence"],
        "receptor_patch_sequence": interface["receptor_patch_sequence"],
        "peptide_sequence": peptide,
        "target_peptide_sequence": peptide,
        "random_conformer_cache_id": row["random_conformer_cache_id"],
        "conformer_cluster_id": conformer["conformer_id"],
        "conformer_source_kind": "random_sequence_augmentation",
        "is_true_bound": False,
        "known_positive_group": row["known_positive_group"],
        "receptor_key": interface["receptor_interface_key"],
        "peptide_key": peptide,
        "patch_atoms": interface["receptor_atoms"],
        "receptor_atoms": interface["receptor_atoms"],
        "peptide_atoms": conformer["backbone_atoms"],
    }


class Phase3RandomConformerDataset(Dataset):
    """One equally weighted interface--peptide positive per Dataset index.

    ``biological_pair_id`` remains attached for audit and analysis only.  It
    never controls Dataset length or selects an interface for training.
    """

    def __init__(
        self,
        random_pairs_jsonl: str | Path,
        random_conformer_cache_jsonl: str | Path,
        biological_pairs_jsonl: str | Path,
        pair_splits_jsonl: str | Path,
        split: str,
        mode: DatasetMode = "train_random",
        global_seed: int = 0,
        fixed_conformer_index: int | None = None,
        data_version: str = "v2",
        dataset_root: str | Path | None = None,
        expected_manifest_sha256: str | None = None,
    ) -> None:
        if mode not in {"train_random", "fixed"}:
            raise ValueError(f"unsupported_dataset_mode:{mode}")
        if mode == "fixed" and fixed_conformer_index is None:
            raise ValueError("fixed mode requires fixed_conformer_index")
        if fixed_conformer_index is not None and not 0 <= fixed_conformer_index < EXPECTED_CONFORMERS:
            raise ValueError("fixed_conformer_index must be in [0, 9]")

        self.random_pairs_jsonl = Path(random_pairs_jsonl).resolve()
        self.random_conformer_cache_jsonl = Path(random_conformer_cache_jsonl).resolve()
        self.biological_pairs_jsonl = Path(biological_pairs_jsonl).resolve()
        self.pair_splits_jsonl = Path(pair_splits_jsonl).resolve()
        self.split = str(split)
        self.mode = mode
        self.global_seed = int(global_seed)
        self.fixed_conformer_index = fixed_conformer_index
        self.epoch = 0
        self.data_contract = load_data_contract(
            data_version=data_version,
            dataset_root=dataset_root,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        assert_path_matches_manifest(
            self.data_contract,
            self.random_pairs_jsonl,
            "04_training_input/random_conformer_pairs.jsonl",
        )
        assert_path_matches_manifest(
            self.data_contract,
            self.random_conformer_cache_jsonl,
            "03_random_conformer_cache/random_conformer_cache.jsonl",
        )
        assert_path_matches_manifest(
            self.data_contract,
            self.biological_pairs_jsonl,
            "dependencies/biological_pairs.jsonl",
        )
        assert_path_matches_manifest(
            self.data_contract,
            self.pair_splits_jsonl,
            "02_leakage_safe_split/pair_splits.jsonl",
        )

        biological_key_to_id: dict[tuple[str, str], str] = {}
        for row in read_jsonl(self.biological_pairs_jsonl):
            key = (str(row["biological_receptor_id"]), str(row["peptide_sequence"]))
            if key in biological_key_to_id:
                raise ValueError(f"ambiguous_biological_relation_mapping:{key}")
            biological_key_to_id[key] = str(row["pair_id"])

        split_by_interface_pair: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(self.pair_splits_jsonl):
            pair_id = str(row["pair_id"])
            if pair_id in split_by_interface_pair:
                raise ValueError(f"duplicate_interface_pair_metadata:{pair_id}")
            split_by_interface_pair[pair_id] = row

        cache_by_id: dict[str, dict[str, Any]] = {}
        cache_by_split_sequence: dict[tuple[str, str], str] = {}
        seen_conformer_ids: set[str] = set()
        for row in read_jsonl(self.random_conformer_cache_jsonl):
            _validate_cache_row(row, seen_conformer_ids, self.data_contract)
            cache_id = str(row["cache_id"])
            split_sequence = (str(row["split"]), str(row["peptide_sequence"]))
            if cache_id in cache_by_id:
                raise ValueError(f"duplicate_cache_id:{cache_id}")
            if split_sequence in cache_by_split_sequence:
                raise ValueError(f"duplicate_split_peptide_cache:{split_sequence}")
            cache_by_id[cache_id] = row
            cache_by_split_sequence[split_sequence] = cache_id

        relation_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        interface_pair_rows: dict[str, dict[str, Any]] = {}
        interface_row_count = 0
        for row in read_jsonl(self.random_pairs_jsonl):
            if row.get("split") != self.split:
                continue
            if row.get("schema_version") != self.data_contract.relation_schema:
                raise ValueError(
                    f"unsupported_random_pair_schema:{row.get('schema_version')}; "
                    f"expected {self.data_contract.relation_schema}; v1 is superseded"
                )
            if row.get("database_contract") != self.data_contract.database_contract:
                raise ValueError("unsupported_database_contract")
            pair = row.get("pair", {})
            interface = row.get("interface", {})
            interface_pair_id = str(pair.get("pair_id") or "")
            metadata = split_by_interface_pair.get(interface_pair_id)
            if metadata is None:
                raise ValueError(f"missing_interface_pair_metadata:{interface_pair_id}")
            if str(metadata.get("split")) != self.split:
                raise ValueError(f"interface_pair_split_mismatch:{interface_pair_id}")
            peptide_sequence = str(pair.get("peptide_sequence") or "")
            if str(metadata.get("peptide_sequence")) != peptide_sequence:
                raise ValueError(f"interface_pair_peptide_mismatch:{interface_pair_id}")
            interface_id = str(pair.get("receptor_id") or "")
            if (
                str(metadata.get("receptor_interface_id")) != interface_id
                or str(interface.get("receptor_interface_id")) != interface_id
            ):
                raise ValueError(f"interface_identity_mismatch:{interface_pair_id}")
            biological_key = (
                str(metadata.get("biological_receptor_id") or ""),
                peptide_sequence,
            )
            biological_pair_id = biological_key_to_id.get(biological_key)
            if biological_pair_id is None:
                raise ValueError(f"missing_biological_relation_mapping:{biological_key}")

            cache_id = str(row.get("random_conformer_cache_id") or "")
            cache = cache_by_id.get(cache_id)
            if cache is None:
                raise ValueError(f"missing_random_conformer_cache:{interface_pair_id}")
            if str(cache.get("split")) != self.split:
                raise ValueError(f"cache_split_mismatch:{interface_pair_id}")
            if str(cache.get("peptide_sequence")) != peptide_sequence:
                raise ValueError(f"cache_peptide_sequence_mismatch:{interface_pair_id}")
            indexed_row = {**row, "_cache": cache, "_biological_pair_id": biological_pair_id}
            if interface_pair_id in interface_pair_rows:
                raise ValueError(f"duplicate_interface_pair_id:{interface_pair_id}")
            interface_pair_rows[interface_pair_id] = indexed_row
            relation_rows[biological_pair_id].append(indexed_row)
            interface_row_count += 1

        if not relation_rows:
            raise ValueError(
                f"No {self.data_contract.data_version} biological relations found for split={self.split!r}"
            )
        self.relation_rows = {
            relation_id: sorted(rows, key=lambda item: str(item["pair"]["pair_id"]))
            for relation_id, rows in sorted(relation_rows.items())
        }
        self.relation_ids = list(self.relation_rows)
        self.interface_pair_rows = dict(sorted(interface_pair_rows.items()))
        self.interface_pair_ids = list(self.interface_pair_rows)
        self.interface_row_count = interface_row_count
        self._plan: list[dict[str, Any]] = []
        self._mapping_summary = {
            "schema_version": "phase3-interface-pair-mapping-summary-v2",
            "data_version": self.data_contract.data_version,
            "dataset_version": self.data_contract.dataset_version,
            "manifest_sha256": self.data_contract.manifest_sha256,
            "relation_schema": self.data_contract.relation_schema,
            "database_contract": self.data_contract.database_contract,
            "cache_schema": self.data_contract.cache_schema,
            "generator_id": self.data_contract.generator_id,
            "qc_id": self.data_contract.qc_id,
            "split": self.split,
            "random_pairs_jsonl": str(self.random_pairs_jsonl),
            "random_conformer_cache_jsonl": str(self.random_conformer_cache_jsonl),
            "biological_pairs_jsonl": str(self.biological_pairs_jsonl),
            "biological_pairs_sha256": sha256_file(self.biological_pairs_jsonl),
            "pair_splits_jsonl": str(self.pair_splits_jsonl),
            "interface_rows": self.interface_row_count,
            "interface_pairs": len(self.interface_pair_ids),
            "biological_relations": len(self.relation_ids),
            "interfaces_per_relation": dict(
                (str(count), frequency)
                for count, frequency in sorted(
                    Counter(len(rows) for rows in self.relation_rows.values()).items()
                )
            ),
            "missing_mappings": 0,
            "ambiguous_mappings": 0,
        }
        self.set_epoch(0)

    def mapping_summary(self) -> dict[str, Any]:
        return dict(self._mapping_summary)

    def write_mapping_summary(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self._mapping_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _build_plan(self, epoch: int) -> list[dict[str, Any]]:
        interface_pair_ids = list(self.interface_pair_ids)
        if self.mode == "train_random":
            _rng(self.global_seed, epoch, "interface_pair_order").shuffle(interface_pair_ids)
        plan: list[dict[str, Any]] = []
        for interface_pair_id in interface_pair_ids:
            row = self.interface_pair_rows[interface_pair_id]
            biological_pair_id = str(row["_biological_pair_id"])
            if self.mode == "train_random":
                conformer_index = _rng(
                    self.global_seed, epoch, "conformer", interface_pair_id
                ).randrange(EXPECTED_CONFORMERS)
            else:
                conformer_index = int(self.fixed_conformer_index or 0)
            plan.append(
                {
                    "biological_pair_id": biological_pair_id,
                    "interface_pair_id": interface_pair_id,
                    "receptor_interface_id": str(row["pair"]["receptor_id"]),
                    "conformer_index": conformer_index,
                    "peptide_sequence": str(row["pair"]["peptide_sequence"]),
                }
            )
        return plan

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        effective_epoch = self.epoch if self.mode == "train_random" else 0
        self._plan = self._build_plan(effective_epoch)

    def epoch_plan(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._plan]

    def peptide_sequence_for_index(self, index: int) -> str:
        return str(self._plan[index]["peptide_sequence"])

    def __len__(self) -> int:
        return len(self.interface_pair_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        selected = self._plan[index]
        interface_pair_id = str(selected["interface_pair_id"])
        row = self.interface_pair_rows[interface_pair_id]
        conformer_index = int(selected["conformer_index"])
        conformer_by_index = {
            int(item["conformer_index"]): item for item in row["_cache"]["conformers"]
        }
        return materialize_random_conformer(
            row,
            conformer_by_index[conformer_index],
            biological_pair_id=str(selected["biological_pair_id"]),
        )


class InterfacePairSubsetDataset(Dataset):
    """An interface-pair-ID view preserving one item per selected pair."""

    def __init__(self, base: Phase3RandomConformerDataset, interface_pair_ids: list[str]) -> None:
        unique = [str(item) for item in interface_pair_ids]
        if len(unique) != len(set(unique)):
            raise ValueError("interface-pair subset contains duplicate interface_pair_id")
        unknown = sorted(set(unique) - set(base.interface_pair_ids))
        if unknown:
            raise ValueError(f"interface-pair subset contains unknown IDs: {unknown[:3]}")
        if not unique:
            raise ValueError("interface-pair subset must be non-empty")
        self.base = base
        self.interface_pair_ids = unique
        self.split = base.split
        self.mode = base.mode
        self.global_seed = base.global_seed
        self.epoch = 0
        self.interface_row_count = len(unique)
        self._base_indices: list[int] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.base.set_epoch(self.epoch)
        base_index_by_interface_pair = {
            str(plan["interface_pair_id"]): index
            for index, plan in enumerate(self.base.epoch_plan())
        }
        self._base_indices = [base_index_by_interface_pair[interface_pair_id] for interface_pair_id in self.interface_pair_ids]

    def epoch_plan(self) -> list[dict[str, Any]]:
        return [self.base.epoch_plan()[index] for index in self._base_indices]

    def peptide_sequence_for_index(self, index: int) -> str:
        return self.base.peptide_sequence_for_index(self._base_indices[index])

    def __len__(self) -> int:
        return len(self.interface_pair_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.base[self._base_indices[index]]

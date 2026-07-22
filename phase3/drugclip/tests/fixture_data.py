from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from phase3.drugclip.io_utils import write_jsonl
from phase3.drugclip.data_contract import sha256_file
from phase3.drugclip.random_conformer_v3 import GENERATOR_ID as V3_GENERATOR_ID
from phase3.drugclip.random_conformers import AA1_TO_3
from phase3.drugclip.random_conformers import generate_conformer


DEFAULT_RELATIONS = [
    ("bio:1", "br:1", "ACDEFGHI", 2),
    ("bio:2", "br:2", "ACDEFGHI", 1),
    ("bio:3", "br:3", "KLMNPQRS", 1),
    ("bio:4", "br:4", "TVWYACDE", 1),
]


def build_fixture(
    root: Path,
    relations: list[tuple[str, str, str, int]] | None = None,
    split: str = "train",
) -> dict[str, Path]:
    relations = relations or list(DEFAULT_RELATIONS)
    biological_rows = []
    split_rows = []
    pair_rows = []
    caches = []
    interfaces_by_peptide: defaultdict[str, list[str]] = defaultdict(list)
    relation_specs: list[dict[str, Any]] = []

    for biological_pair_id, biological_receptor_id, peptide, interface_count in relations:
        biological_rows.append(
            {
                "pair_id": biological_pair_id,
                "biological_receptor_id": biological_receptor_id,
                "peptide_sequence": peptide,
            }
        )
        interfaces = [f"iface:{biological_pair_id}:{index}" for index in range(interface_count)]
        interfaces_by_peptide[peptide].extend(interfaces)
        relation_specs.append(
            {
                "biological_pair_id": biological_pair_id,
                "biological_receptor_id": biological_receptor_id,
                "peptide": peptide,
                "interfaces": interfaces,
            }
        )

    unique_peptides = sorted({item[2] for item in relations})
    cache_ids = {}
    for cache_index, peptide in enumerate(unique_peptides):
        cache_id = f"cache:{cache_index}"
        cache_ids[peptide] = cache_id
        caches.append(
            {
                "cache_id": cache_id,
                "split": split,
                "peptide_sequence": peptide,
                "generator_id": "internal-coordinate-rama-v1",
                "max_conformers_requested": 10,
                "schema_version": "drugclip-random-conformer-cache-v1",
                "conformers": [generate_conformer(peptide, split, index) for index in range(10)],
            }
        )

    receptor_atoms = generate_conformer("ACDEFGHI", split, 0)["backbone_atoms"]
    for spec in relation_specs:
        peptide = spec["peptide"]
        for interface_index, interface_id in enumerate(spec["interfaces"]):
            interface_pair_id = f"pair:{spec['biological_pair_id']}:{interface_index}"
            split_rows.append(
                {
                    "pair_id": interface_pair_id,
                    "biological_receptor_id": spec["biological_receptor_id"],
                    "receptor_interface_id": interface_id,
                    "peptide_sequence": peptide,
                    "split": split,
                }
            )
            pair_rows.append(
                {
                    "schema_version": "drugclip-random-augmentation-pairs-v2",
                    "database_contract": "drugclip-exact-peptide-random-conformer-v2",
                    "split": split,
                    "pair": {
                        "pair_id": interface_pair_id,
                        "receptor_id": interface_id,
                        "receptor_sequence": "ACDEFGHI",
                        "peptide_sequence": peptide,
                    },
                    "interface": {
                        "receptor_interface_id": interface_id,
                        "receptor_interface_key": interface_id,
                        "receptor_patch_sequence": "ACDEFGHI",
                        "receptor_atoms": receptor_atoms,
                    },
                    "random_conformer_cache_id": cache_ids[peptide],
                    "known_positive_group": {
                        "receptor_peptides": [peptide],
                        "peptide_receptors": sorted(interfaces_by_peptide[peptide]),
                    },
                }
            )

    paths = {
        "biological": root / "biological.jsonl",
        "splits": root / "pair_splits.jsonl",
        "pairs": root / "pairs.jsonl",
        "cache": root / "cache.jsonl",
    }
    write_jsonl(paths["biological"], biological_rows)
    write_jsonl(paths["splits"], split_rows)
    write_jsonl(paths["pairs"], pair_rows)
    write_jsonl(paths["cache"], caches)
    return paths


def build_versioned_fixture(
    root: Path,
    data_version: str,
    relations: list[tuple[str, str, str, int]] | None = None,
    split: str = "train",
) -> dict[str, Path]:
    if data_version == "v2":
        return build_fixture(root, relations=relations, split=split)
    if data_version != "v3":
        raise ValueError(f"unsupported fixture data_version:{data_version}")

    pairs = build_fixture(root / "_source_v2", relations=relations, split=split)
    v3 = root / "random_conformer_v3"
    paths = {
        "biological": v3 / "dependencies" / "biological_pairs.jsonl",
        "splits": v3 / "02_leakage_safe_split" / "pair_splits.jsonl",
        "pairs": v3 / "04_training_input" / "random_conformer_pairs.jsonl",
        "cache": v3 / "03_random_conformer_cache" / "random_conformer_cache.jsonl",
        "known_positive": v3 / "01_interface_pairs" / "known_positive_groups.json",
        "receptor_interfaces": v3 / "01_interface_pairs" / "receptor_interfaces.jsonl",
        "manifest": v3 / "DATA_MANIFEST.json",
        "root": v3,
    }
    biological = list(_read_jsonl(pairs["biological"]))
    split_rows = list(_read_jsonl(pairs["splits"]))
    pair_rows = list(_read_jsonl(pairs["pairs"]))
    cache_rows = list(_read_jsonl(pairs["cache"]))

    for row in pair_rows:
        row["schema_version"] = "drugclip-random-augmentation-pairs-v3"
        row["database_contract"] = "drugclip-exact-peptide-random-conformer-v3"
    for row in cache_rows:
        row["schema_version"] = "drugclip-random-conformer-cache-v2"
        row["generator_id"] = V3_GENERATOR_ID
        for conformer in row["conformers"]:
            conformer["generator_id"] = V3_GENERATOR_ID
            conformer["conformer_id"] = str(conformer["conformer_id"]).replace("rand:", "randv3:", 1)
            conformer["attempt_index"] = 0
            conformer["base_v2_seed"] = conformer["seed"]
            conformer["backbone_atoms"] = _straight_backbone(row["peptide_sequence"], offset=float(conformer["conformer_index"]))

    write_jsonl(paths["biological"], biological)
    write_jsonl(paths["splits"], split_rows)
    write_jsonl(paths["pairs"], pair_rows)
    write_jsonl(paths["cache"], cache_rows)
    paths["known_positive"].parent.mkdir(parents=True, exist_ok=True)
    paths["known_positive"].write_text('{"schema_version":"known-positive-groups-json"}\n', encoding="utf-8")
    write_jsonl(paths["receptor_interfaces"], [{"receptor_atoms": pair_rows[0]["interface"]["receptor_atoms"]}])

    formal_files = []
    roles = {
        "04_training_input/random_conformer_pairs.jsonl": (paths["pairs"], True, "drugclip-random-augmentation-pairs-v3"),
        "03_random_conformer_cache/random_conformer_cache.jsonl": (paths["cache"], True, "drugclip-random-conformer-cache-v2"),
        "dependencies/biological_pairs.jsonl": (paths["biological"], True, "biological-pairs-jsonl"),
        "02_leakage_safe_split/pair_splits.jsonl": (paths["splits"], True, "drugclip-interface-positive-v2"),
        "01_interface_pairs/known_positive_groups.json": (paths["known_positive"], True, "known-positive-groups-json"),
        "01_interface_pairs/receptor_interfaces.jsonl": (paths["receptor_interfaces"], True, "receptor-interface-jsonl"),
    }
    for relative, (path, training_required, schema) in roles.items():
        formal_files.append(
            {
                "relative_path": relative,
                "training_required": training_required,
                "validation_required": True,
                "audit_only": False,
                "schema": schema,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "records_or_lines": _count_lines(path),
                "role": "fixture",
            }
        )
    manifest = {
        "manifest_schema": "pepclip-data-manifest-v2",
        "dataset_name": "fixture",
        "dataset_version": "random_conformer_v3",
        "parent_dataset": "random_conformer_v2",
        "relation_schema": "drugclip-random-augmentation-pairs-v3",
        "cache_schema": "drugclip-random-conformer-cache-v2",
        "generator_id": V3_GENERATOR_ID,
        "clash_rule": {
            "atoms": ["N", "CA", "C"],
            "minimum_residue_index_gap": 2,
            "reject_if_distance_angstrom_less_than": 1.5,
        },
        "counts": {
            "caches": len(cache_rows),
            "conformers": 10 * len(cache_rows),
            "conformers_per_cache": 10,
            "pairs": len(pair_rows),
        },
        "formal_files": formal_files,
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _straight_backbone(peptide: str, offset: float = 0.0) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for residue_index, aa in enumerate(peptide, start=1):
        base = float(residue_index - 1) * 5.0
        atoms.extend(
            [
                {
                    "residue_id": f"P:{residue_index}",
                    "residue_name": AA1_TO_3[aa],
                    "atom_name": "N",
                    "element": "N",
                    "x": base,
                    "y": offset,
                    "z": 0.0,
                },
                {
                    "residue_id": f"P:{residue_index}",
                    "residue_name": AA1_TO_3[aa],
                    "atom_name": "CA",
                    "element": "C",
                    "x": base + 1.5,
                    "y": offset,
                    "z": 0.0,
                },
                {
                    "residue_id": f"P:{residue_index}",
                    "residue_name": AA1_TO_3[aa],
                    "atom_name": "C",
                    "element": "C",
                    "x": base + 3.0,
                    "y": offset,
                    "z": 0.0,
                },
            ]
        )
    return atoms

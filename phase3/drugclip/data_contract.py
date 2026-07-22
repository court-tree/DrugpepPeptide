"""Explicit data-version contracts for Phase-3 DrugCLIP datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

from phase3.drugclip.random_conformers import GENERATOR_ID as V2_GENERATOR_ID
from phase3.drugclip.random_conformers import SCHEMA_VERSION as V2_CACHE_SCHEMA
from phase3.drugclip.random_conformer_v3 import (
    CACHE_SCHEMA as V3_CACHE_SCHEMA,
    DATABASE_CONTRACT as V3_DATABASE_CONTRACT,
    DATASET_VERSION as V3_DATASET_VERSION,
    GENERATOR_ID as V3_GENERATOR_ID,
    RELATION_SCHEMA as V3_RELATION_SCHEMA,
)


DataVersion = Literal["v2", "v3"]

V2_DATASET_VERSION = "random_conformer_v2"
V2_RELATION_SCHEMA = "drugclip-random-augmentation-pairs-v2"
V2_DATABASE_CONTRACT = "drugclip-exact-peptide-random-conformer-v2"
V2_QC_ID = "no-clash15-parent-v2"
V3_QC_ID = "clash15-nca-c-gap2-distance-lt1.5-v1"

TRAINING_REQUIRED_FILES = {
    "04_training_input/random_conformer_pairs.jsonl",
    "03_random_conformer_cache/random_conformer_cache.jsonl",
    "dependencies/biological_pairs.jsonl",
    "02_leakage_safe_split/pair_splits.jsonl",
    "01_interface_pairs/known_positive_groups.json",
    "01_interface_pairs/receptor_interfaces.jsonl",
}


@dataclass(frozen=True)
class DataContract:
    data_version: DataVersion
    dataset_version: str
    dataset_root: Path | None
    manifest_path: Path | None
    manifest_sha256: str | None
    relation_schema: str
    database_contract: str
    cache_schema: str
    generator_id: str
    qc_id: str
    file_sha256: dict[str, str]
    manifest: dict[str, Any] | None = None


def normalize_data_version(value: str) -> DataVersion:
    normalized = str(value).lower()
    if normalized not in {"v2", "v3"}:
        raise ValueError(f"unsupported_data_version:{value}")
    return normalized  # type: ignore[return-value]


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _manifest_file_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = manifest.get("formal_files")
    if not isinstance(files, list):
        raise ValueError("manifest_formal_files_missing")
    indexed: dict[str, dict[str, Any]] = {}
    for row in files:
        relative = str(row.get("relative_path") or "").replace("\\", "/")
        if not relative:
            raise ValueError("manifest_file_relative_path_missing")
        if relative in indexed:
            raise ValueError(f"manifest_duplicate_file:{relative}")
        indexed[relative] = row
    return indexed


def _verify_manifest_hashes(root: Path, manifest: dict[str, Any], file_index: dict[str, dict[str, Any]]) -> None:
    required = set(TRAINING_REQUIRED_FILES)
    missing = sorted(required - set(file_index))
    if missing:
        raise ValueError(f"manifest_missing_training_required_files:{missing}")
    for relative, row in file_index.items():
        if not bool(row.get("training_required")) and relative not in required:
            continue
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"manifest_required_file_missing:{path}")
        expected = str(row.get("sha256") or "").upper()
        if len(expected) != 64:
            raise ValueError(f"manifest_file_sha256_missing:{relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"manifest_file_sha256_mismatch:{relative}")


def load_data_contract(
    *,
    data_version: str = "v2",
    dataset_root: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> DataContract:
    version = normalize_data_version(data_version)
    if version == "v2":
        if dataset_root is not None:
            root = Path(dataset_root).resolve()
            manifest = root / "DATA_MANIFEST.json"
            if manifest.exists():
                loaded = json.loads(manifest.read_text(encoding="utf-8"))
                if loaded.get("dataset_version") != V2_DATASET_VERSION:
                    raise ValueError("v2_dataset_rejects_non_v2_manifest")
        return DataContract(
            data_version="v2",
            dataset_version=V2_DATASET_VERSION,
            dataset_root=Path(dataset_root).resolve() if dataset_root is not None else None,
            manifest_path=None,
            manifest_sha256=None,
            relation_schema=V2_RELATION_SCHEMA,
            database_contract=V2_DATABASE_CONTRACT,
            cache_schema=V2_CACHE_SCHEMA,
            generator_id=V2_GENERATOR_ID,
            qc_id=V2_QC_ID,
            file_sha256={},
            manifest=None,
        )

    if dataset_root is None:
        raise ValueError("v3_requires_dataset_root")
    root = Path(dataset_root).resolve()
    manifest_path = root / "DATA_MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"v3_manifest_missing:{manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256.upper():
        raise ValueError("manifest_sha256_mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_schema") != "pepclip-data-manifest-v2":
        raise ValueError("unsupported_manifest_schema")
    if manifest.get("dataset_version") != V3_DATASET_VERSION:
        raise ValueError("v3_dataset_version_mismatch")
    if manifest.get("relation_schema") != V3_RELATION_SCHEMA:
        raise ValueError("v3_relation_schema_mismatch")
    if manifest.get("cache_schema") != V3_CACHE_SCHEMA:
        raise ValueError("v3_cache_schema_mismatch")
    if manifest.get("generator_id") != V3_GENERATOR_ID:
        raise ValueError("v3_generator_id_mismatch")
    clash_rule = manifest.get("clash_rule")
    if not isinstance(clash_rule, dict):
        raise ValueError("v3_clash_rule_missing")
    if (
        clash_rule.get("atoms") != ["N", "CA", "C"]
        or int(clash_rule.get("minimum_residue_index_gap", -1)) != 2
        or float(clash_rule.get("reject_if_distance_angstrom_less_than", -1.0)) != 1.5
    ):
        raise ValueError("v3_clash15_qc_contract_mismatch")
    file_index = _manifest_file_index(manifest)
    _verify_manifest_hashes(root, manifest, file_index)
    file_sha256 = {
        relative: str(row["sha256"]).upper()
        for relative, row in file_index.items()
        if "sha256" in row
    }
    return DataContract(
        data_version="v3",
        dataset_version=V3_DATASET_VERSION,
        dataset_root=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        relation_schema=V3_RELATION_SCHEMA,
        database_contract=V3_DATABASE_CONTRACT,
        cache_schema=V3_CACHE_SCHEMA,
        generator_id=V3_GENERATOR_ID,
        qc_id=V3_QC_ID,
        file_sha256=file_sha256,
        manifest=manifest,
    )


def assert_path_matches_manifest(contract: DataContract, path: str | Path, relative: str) -> None:
    if contract.data_version != "v3":
        return
    if contract.dataset_root is None:
        raise ValueError("v3_contract_lacks_dataset_root")
    expected = (contract.dataset_root / relative).resolve()
    actual = Path(path).resolve()
    if actual != expected:
        raise ValueError(f"v3_path_not_manifest_formal_file:{relative}")

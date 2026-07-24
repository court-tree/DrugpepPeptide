"""Read-only validator for bounded full-heavy sequence caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from phase3.drugclip.build_bounded_full_heavy_cache import (
    EXPECTED_FORMAL_CONFORMER_COUNT,
    EXPECTED_FORMAL_SEQUENCE_COUNT,
    _check_no_stale_temporary_files,
    _load_json,
    _plan_sequence_records,
    _verify_canonical_record,
    load_descriptor_contract,
    production_payload_validator,
    select_smoke_sequences,
    sequence_file_key,
    validate_existing_sequence_file,
)
from phase3.drugclip.full_heavy_adaptation_contract import (
    CACHE_MANIFEST_SCHEMA,
    CONFORMERS_PER_SEQUENCE,
    canonical_json_sha256,
    sequence_sha256,
    sha256_file,
)


VALIDATION_SCHEMA = "phase3-v2-bounded-full-heavy-cache-validation-v1"


def validate_cache_read_only(
    *,
    descriptor_contract: dict[str, Any],
    cache_root: Path,
    smoke: bool = False,
    payload_validator=production_payload_validator,
) -> dict[str, Any]:
    cache_root = Path(cache_root).resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(f"cache_root_missing:{cache_root}")
    _check_no_stale_temporary_files(cache_root)
    contract = _load_json(cache_root / "cache_contract.json")
    _verify_canonical_record(
        contract,
        "contract_canonical_sha256",
        "cache_contract_canonical_sha256_mismatch",
    )
    if (
        contract.get("plan_descriptor_file_sha256")
        != descriptor_contract["file_sha256"]
        or contract.get("plan_descriptor_canonical_sha256")
        != descriptor_contract["canonical_sha256"]
    ):
        raise ValueError("cache_plan_descriptor_binding_mismatch")
    if any(
        str(value).lower().find("safe265") >= 0
        or str(value).lower().find("safe373") >= 0
        for value in (cache_root, contract)
    ):
        raise ValueError("evaluation_cache_path_or_contract_forbidden")
    all_records = _plan_sequence_records(descriptor_contract)
    if smoke:
        expected_records = select_smoke_sequences(descriptor_contract)
        expected_sequences = sorted(
            row["peptide_sequence"] for row in expected_records
        )
        manifest_path = cache_root / "smoke_manifest.json"
        if (cache_root / "cache_manifest.json").exists():
            raise ValueError("smoke_must_not_have_formal_cache_manifest")
        if not manifest_path.is_file():
            raise ValueError("smoke_manifest_missing")
        manifest = _load_json(manifest_path)
        _verify_canonical_record(
            manifest,
            "smoke_manifest_canonical_sha256",
            "smoke_manifest_canonical_sha256_mismatch",
        )
        if (
            manifest.get("formal_cache") is not False
            or manifest.get("not_valid_for_training") is not True
            or int(manifest.get("sequence_count", -1)) != 5
        ):
            raise ValueError("smoke_manifest_boundary_mismatch")
    else:
        expected_sequences = descriptor_contract["required_sequences"]
        manifest_path = cache_root / "cache_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("formal_cache_incomplete_manifest_missing")
        manifest = _load_json(manifest_path)
        if manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA:
            raise ValueError("formal_cache_manifest_schema_mismatch")
        _verify_canonical_record(
            manifest,
            "manifest_canonical_sha256",
            "formal_cache_manifest_canonical_sha256_mismatch",
        )
        if (
            manifest.get("status") != "MATERIALIZED"
            or manifest.get("purpose") != "bounded_train_valid_only"
            or int(manifest.get("sequence_count", -1))
            != EXPECTED_FORMAL_SEQUENCE_COUNT
            or int(manifest.get("conformer_count", -1))
            != EXPECTED_FORMAL_CONFORMER_COUNT
        ):
            raise ValueError("formal_cache_manifest_count_or_status_mismatch")
    if contract.get("required_peptide_sequences") != expected_sequences:
        raise ValueError("cache_contract_required_sequence_mismatch")
    if contract.get("required_peptide_sequences_sha256") != sequence_sha256(
        expected_sequences
    ):
        raise ValueError("cache_contract_required_sequence_sha256_mismatch")

    sequences_dir = cache_root / "sequences"
    actual_files = sorted(sequences_dir.glob("*.json"))
    expected_names = {
        f"{sequence_file_key(sequence)}.json" for sequence in expected_sequences
    }
    actual_names = {path.name for path in actual_files}
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise ValueError(
            f"cache_sequence_file_set_mismatch:missing={missing}:extra={extra}"
        )
    rows = []
    atom_identity_hashes = {}
    coordinate_count = 0
    for sequence in expected_sequences:
        path = sequences_dir / f"{sequence_file_key(sequence)}.json"
        payload, audit = validate_existing_sequence_file(
            path,
            sequence=sequence,
            cache_contract=contract,
            payload_validator=payload_validator,
        )
        record = all_records[sequence]
        if (
            int(payload["atom_count"])
            != int(record["theoretical_heavy_atom_count"])
        ):
            raise ValueError(f"cache_theoretical_atom_count_mismatch:{sequence}")
        coordinate_count += len(payload["conformers"])
        atom_identity_hashes[sequence] = payload["atom_identity_sha256"]
        rows.append(
            {
                "peptide_sequence": sequence,
                "sequence_file_sha256": sha256_file(path),
                "payload_canonical_sha256": payload[
                    "payload_canonical_sha256"
                ],
                "atom_identity_sha256": payload["atom_identity_sha256"],
                "coordinate_sha256": [
                    value["coordinate_sha256"]
                    for value in payload["conformers"]
                ],
                "accepted_attempt_indices": payload[
                    "accepted_attempt_indices"
                ],
                "split_roles": record["split_roles"],
                "validation_status": audit.get("status", "PASS"),
            }
        )
    if coordinate_count != len(expected_sequences) * CONFORMERS_PER_SEQUENCE:
        raise ValueError("cache_aggregate_conformer_count_mismatch")
    if not smoke:
        if manifest.get("required_peptide_sequences") != expected_sequences:
            raise ValueError("manifest_required_sequence_mismatch")
        if manifest.get("required_peptide_sequences_sha256") != (
            sequence_sha256(expected_sequences)
        ):
            raise ValueError("manifest_required_sequence_sha256_mismatch")
        expected_file_hashes = {
            row["peptide_sequence"]: row["sequence_file_sha256"]
            for row in rows
        }
        if manifest.get("sequence_file_sha256") != expected_file_hashes:
            raise ValueError("manifest_sequence_file_sha256_mismatch")
        expected_aggregate = canonical_json_sha256(
            [
                [row["peptide_sequence"], row["sequence_file_sha256"]]
                for row in rows
            ]
        )
        if manifest.get("sequence_file_sha256_aggregate") != expected_aggregate:
            raise ValueError("manifest_sequence_file_aggregate_mismatch")
        index_path = cache_root / str(manifest["index_path"])
        if sha256_file(index_path) != manifest.get("index_sha256"):
            raise ValueError("manifest_cache_index_sha256_mismatch")
        index_rows = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [row["peptide_sequence"] for row in index_rows] != expected_sequences:
            raise ValueError("cache_index_sequence_order_mismatch")
    return {
        "schema_version": VALIDATION_SCHEMA,
        "classification": (
            "CACHE_MATERIALIZER_SMOKE_PASS"
            if smoke
            else "BOUNDED_FULL_HEAVY_CACHE_VALIDATION_PASS"
        ),
        "formal_cache": not smoke,
        "not_valid_for_training": smoke,
        "sequence_count": len(rows),
        "conformer_count": coordinate_count,
        "required_sequence_sha256": sequence_sha256(expected_sequences),
        "sequence_file_sha256_aggregate": canonical_json_sha256(
            [
                [row["peptide_sequence"], row["sequence_file_sha256"]]
                for row in rows
            ]
        ),
        "target_bound_generation_inputs_used": False,
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only validation of a bounded full-heavy cache."
    )
    parser.add_argument("--plan-descriptor", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--smoke-five", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    descriptor = load_descriptor_contract(Path(args.plan_descriptor))
    report = validate_cache_read_only(
        descriptor_contract=descriptor,
        cache_root=Path(args.cache_root),
        smoke=args.smoke_five,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

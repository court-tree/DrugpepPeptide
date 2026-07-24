from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from phase3.drugclip.build_bounded_full_heavy_plan import (
    canonical_json_bytes,
)
from phase3.drugclip.full_heavy_adaptation_contract import (
    ATOM_CAP_EXCLUSIVE,
    CONFORMERS_PER_SEQUENCE,
    ELIGIBILITY_REGISTRY_SCHEMA,
    MAX_TRAIN_PAIRS,
    MAX_VALID_PAIRS,
    PLAN_SCHEMA,
    PLAN_SELECTION_ALGORITHM_VERSION,
    SAFE373_PLAN_CANONICAL_SHA256,
    bounded_plan_selection_key,
    canonical_json_sha256,
    sequence_sha256,
    sha256_file,
    validate_bounded_plan_descriptor,
)
from phase3.drugclip.train import _validate_full_heavy_cli_contract


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _plan(
    split: str,
    pair_ids: list[str],
    sequence: str,
    atom_count: int,
) -> dict:
    return {
        "schema_version": PLAN_SCHEMA,
        "split": split,
        "selection_algorithm_version": PLAN_SELECTION_ALGORITHM_VERSION,
        "pair_count": len(pair_ids),
        "interface_pair_ids": pair_ids,
        "interface_pair_ids_sha256": sequence_sha256(pair_ids),
        "unique_peptide_sequence_count": 1,
        "unique_peptide_sequences": [sequence],
        "unique_peptide_sequences_sha256": sequence_sha256([sequence]),
        "sequence_records": [{
            "peptide_sequence": sequence,
            "selected_pair_count": len(pair_ids),
            "theoretical_heavy_atom_count": atom_count,
        }],
        "conformers_per_sequence": CONFORMERS_PER_SEQUENCE,
        "future_required_conformer_count": CONFORMERS_PER_SEQUENCE,
    }


def _descriptor_fixture(root: Path) -> tuple[Path, dict]:
    train_all = [f"train:{index:05d}" for index in range(MAX_TRAIN_PAIRS)]
    valid_all = [f"valid:{index:05d}" for index in range(MAX_VALID_PAIRS)]
    train_plan_ids = sorted(
        train_all,
        key=lambda pair_id: (
            bounded_plan_selection_key("train", pair_id),
            pair_id,
        ),
    )
    valid_plan_ids = sorted(
        valid_all,
        key=lambda pair_id: (
            bounded_plan_selection_key("valid", pair_id),
            pair_id,
        ),
    )
    registry = [
        {
            "schema_version": ELIGIBILITY_REGISTRY_SCHEMA,
            "peptide_sequence": "A",
            "split": "train",
            "interface_pair_ids": sorted(train_all),
            "structure_instance_classifications": [
                "ordinary_linear_standard"
            ],
            "chemistry_classification": "ordinary_linear_standard",
            "theoretical_heavy_atom_count": 6,
            "torsion_prior_covered": True,
            "eligible": True,
        },
        {
            "schema_version": ELIGIBILITY_REGISTRY_SCHEMA,
            "peptide_sequence": "G",
            "split": "valid",
            "interface_pair_ids": sorted(valid_all),
            "structure_instance_classifications": [
                "ordinary_linear_standard"
            ],
            "chemistry_classification": "ordinary_linear_standard",
            "theoretical_heavy_atom_count": 5,
            "torsion_prior_covered": True,
            "eligible": True,
        },
    ]
    registry_path = root / "registry.jsonl"
    _write_jsonl(registry_path, registry)
    report_path = root / "report.json"
    _write_json(
        report_path,
        {
            "classification": "CORE_LINEAR_SUBSET_SUFFICIENT",
            "registry": {"sequence_count": 2},
        },
    )
    train_source = root / "train.jsonl"
    valid_source = root / "valid.jsonl"
    biological_source = root / "biological.jsonl"
    train_source.write_text("{}\n", encoding="utf-8")
    valid_source.write_text("{}\n", encoding="utf-8")
    biological_source.write_text("{}\n", encoding="utf-8")
    safe_path = root / "safe373.json"
    _write_json(
        safe_path,
        {
            "plan_canonical_sha256": SAFE373_PLAN_CANONICAL_SHA256,
            "safe_query_interface_pair_ids": [],
            "safe_peptide_candidate_ids": [],
        },
    )
    plans = {
        "train": _plan("train", train_plan_ids, "A", 6),
        "valid": _plan("valid", valid_plan_ids, "G", 5),
    }
    for split in ("train", "valid"):
        _write_json(root / f"{split}_plan.json", plans[split])
    required_sequences = ["A", "G"]
    descriptor_core = {
        "schema_version": PLAN_SCHEMA,
        "frozen_inputs": {
            "eligibility_registry": {
                "path": registry_path.name,
                "file_sha256": sha256_file(registry_path),
                "canonical_sha256": canonical_json_sha256(registry),
                "schema_version": ELIGIBILITY_REGISTRY_SCHEMA,
                "sequence_count": 2,
            },
            "full_split_audit_report": {
                "path": report_path.name,
                "file_sha256": sha256_file(report_path),
            },
            "formal_split_sources": {
                "train": {
                    "path": train_source.name,
                    "file_sha256": sha256_file(train_source),
                    "pair_count": len(train_all),
                },
                "valid": {
                    "path": valid_source.name,
                    "file_sha256": sha256_file(valid_source),
                    "pair_count": len(valid_all),
                },
                "biological_relations": {
                    "path": biological_source.name,
                    "file_sha256": sha256_file(biological_source),
                },
            },
        },
        "eligibility_contract": {
            "rule_version": ELIGIBILITY_REGISTRY_SCHEMA,
            "required_chemistry_classification": (
                "ordinary_linear_standard"
            ),
            "theoretical_heavy_atom_count_operator": "<",
            "theoretical_heavy_atom_count_limit": ATOM_CAP_EXCLUSIVE,
            "torsion_prior_coverage_required": True,
            "sequence_level_all_structure_instances_required": True,
        },
        "selection_contract": {
            "algorithm_version": PLAN_SELECTION_ALGORITHM_VERSION,
            "namespace": PLAN_SCHEMA,
            "primary_key": (
                'SHA256("phase3-v2-bounded-full-heavy-plan-v1" + "\\0" '
                '+ split + "\\0" + interface_pair_id)'
            ),
            "tie_breaker": "interface_pair_id",
            "target_train_pair_count": MAX_TRAIN_PAIRS,
            "target_valid_pair_count": MAX_VALID_PAIRS,
        },
        "plan_files": {
            split: {
                "path": f"{split}_plan.json",
                "file_sha256": sha256_file(root / f"{split}_plan.json"),
            }
            for split in ("train", "valid")
        },
        "plans": plans,
        "safe373_evaluation_exclusion": {
            "plan_path": safe_path.name,
            "plan_file_sha256": sha256_file(safe_path),
            "plan_canonical_sha256": SAFE373_PLAN_CANONICAL_SHA256,
            "train_overlap_required_zero": {
                "query_pair": True,
                "peptide_sequence": True,
                "biological_relation": True,
            },
            "valid_overlap_report": {
                "query_pair_ids": [],
                "query_pair_ids_sha256": sequence_sha256([]),
                "peptide_sequences": [],
                "peptide_sequences_sha256": sequence_sha256([]),
                "biological_relation_ids": [],
                "biological_relation_ids_sha256": sequence_sha256([]),
            },
        },
        "future_cache_requirement": {
            "generation_status": "NOT_BUILT",
            "cache_status": "NOT_BUILT",
            "cache_manifest_path": None,
            "cache_manifest_sha256": None,
            "conformers_per_sequence": CONFORMERS_PER_SEQUENCE,
            "required_peptide_sequences": required_sequences,
            "required_peptide_sequences_sha256": sequence_sha256(
                required_sequences
            ),
            "future_required_conformer_count": 2 * CONFORMERS_PER_SEQUENCE,
            "safe373_evaluation_cache_reuse_forbidden": True,
        },
    }
    descriptor = {
        **descriptor_core,
        "descriptor_canonical_sha256": canonical_json_sha256(
            descriptor_core
        ),
    }
    descriptor_path = root / "descriptor.json"
    _write_json(descriptor_path, descriptor)
    inputs = {
        "train_interface_pair_ids": train_all,
        "valid_interface_pair_ids": valid_all,
        "train_sequence_by_pair": dict.fromkeys(train_all, "A"),
        "valid_sequence_by_pair": dict.fromkeys(valid_all, "G"),
        "train_relation_by_pair": {
            pair_id: f"train-relation:{index}"
            for index, pair_id in enumerate(train_all)
        },
        "valid_relation_by_pair": {
            pair_id: f"valid-relation:{index}"
            for index, pair_id in enumerate(valid_all)
        },
    }
    return descriptor_path, inputs


def _rewrite_descriptor(path: Path, value: dict) -> None:
    core = {
        key: item
        for key, item in value.items()
        if key != "descriptor_canonical_sha256"
    }
    value["descriptor_canonical_sha256"] = canonical_json_sha256(core)
    _write_json(path, value)


class BoundedFullHeavyPlanTests(unittest.TestCase):
    def test_selection_key_matches_exact_contract(self) -> None:
        expected = hashlib.sha256(
            (
                PLAN_SCHEMA
                + "\0train\0interface_pair:test"
            ).encode("utf-8")
        ).hexdigest().upper()
        self.assertEqual(
            bounded_plan_selection_key("train", "interface_pair:test"),
            expected,
        )

    def test_descriptor_is_byte_deterministic_and_exact_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, inputs = _descriptor_fixture(Path(temporary))
            first = path.read_bytes()
            value = json.loads(first)
            _rewrite_descriptor(path, value)
            self.assertEqual(first, path.read_bytes())
            result = validate_bounded_plan_descriptor(path, **inputs)
            self.assertEqual(
                len(result["train_interface_pair_ids"]),
                MAX_TRAIN_PAIRS,
            )
            self.assertEqual(
                len(result["valid_interface_pair_ids"]),
                MAX_VALID_PAIRS,
            )

    def test_ineligible_sequence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, inputs = _descriptor_fixture(root)
            registry_path = root / "registry.jsonl"
            rows = [
                json.loads(line)
                for line in registry_path.read_text().splitlines()
            ]
            rows[0]["chemistry_classification"] = "known_disulfide"
            rows[0]["structure_instance_classifications"] = [
                "known_disulfide"
            ]
            rows[0]["eligible"] = False
            _write_jsonl(registry_path, rows)
            descriptor = json.loads(path.read_text())
            contract = descriptor["frozen_inputs"]["eligibility_registry"]
            contract["file_sha256"] = sha256_file(registry_path)
            contract["canonical_sha256"] = canonical_json_sha256(rows)
            _rewrite_descriptor(path, descriptor)
            with self.assertRaisesRegex(
                ValueError, "eligible_plan_capacity"
            ):
                validate_bounded_plan_descriptor(path, **inputs)

    def test_atom_count_192_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, inputs = _descriptor_fixture(root)
            registry_path = root / "registry.jsonl"
            rows = [
                json.loads(line)
                for line in registry_path.read_text().splitlines()
            ]
            rows[0]["theoretical_heavy_atom_count"] = 192
            rows[0]["eligible"] = False
            _write_jsonl(registry_path, rows)
            descriptor = json.loads(path.read_text())
            contract = descriptor["frozen_inputs"]["eligibility_registry"]
            contract["file_sha256"] = sha256_file(registry_path)
            contract["canonical_sha256"] = canonical_json_sha256(rows)
            _rewrite_descriptor(path, descriptor)
            with self.assertRaisesRegex(
                ValueError, "eligible_plan_capacity"
            ):
                validate_bounded_plan_descriptor(path, **inputs)

    def test_wrong_split_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, inputs = _descriptor_fixture(root)
            descriptor = json.loads(path.read_text())
            plan = copy.deepcopy(descriptor["plans"]["train"])
            plan["interface_pair_ids"][0] = "valid:00000"
            plan["interface_pair_ids_sha256"] = sequence_sha256(
                plan["interface_pair_ids"]
            )
            plan_path = root / "train_plan.json"
            _write_json(plan_path, plan)
            descriptor["plans"]["train"] = plan
            descriptor["plan_files"]["train"]["file_sha256"] = sha256_file(
                plan_path
            )
            _rewrite_descriptor(path, descriptor)
            with self.assertRaisesRegex(ValueError, "wrong_split"):
                validate_bounded_plan_descriptor(path, **inputs)

    def test_pair_order_or_sha_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, inputs = _descriptor_fixture(root)
            descriptor = json.loads(path.read_text())
            plan = copy.deepcopy(descriptor["plans"]["train"])
            plan["interface_pair_ids"] = list(
                reversed(plan["interface_pair_ids"])
            )
            plan_path = root / "train_plan.json"
            _write_json(plan_path, plan)
            descriptor["plans"]["train"] = plan
            descriptor["plan_files"]["train"]["file_sha256"] = sha256_file(
                plan_path
            )
            _rewrite_descriptor(path, descriptor)
            with self.assertRaisesRegex(ValueError, "ordered_sha256"):
                validate_bounded_plan_descriptor(path, **inputs)

    def test_registry_and_report_sha_changes_are_rejected(self) -> None:
        for filename, regex in (
            ("registry.jsonl", "registry_file_sha256"),
            ("report.json", "audit_report_sha256"),
        ):
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    path, inputs = _descriptor_fixture(root)
                    with (root / filename).open("a", encoding="utf-8") as handle:
                        handle.write(" ")
                    with self.assertRaisesRegex(ValueError, regex):
                        validate_bounded_plan_descriptor(path, **inputs)

    def test_train_valid_sequence_or_relation_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, inputs = _descriptor_fixture(Path(temporary))
            bad = dict(inputs)
            bad["valid_sequence_by_pair"] = dict.fromkeys(
                inputs["valid_interface_pair_ids"], "A"
            )
            with self.assertRaisesRegex(ValueError, "sequence_leakage"):
                validate_bounded_plan_descriptor(path, **bad)
            bad = dict(inputs)
            first_train_relation = next(
                iter(inputs["train_relation_by_pair"].values())
            )
            bad["valid_relation_by_pair"] = dict(
                inputs["valid_relation_by_pair"]
            )
            first_valid = inputs["valid_interface_pair_ids"][0]
            bad["valid_relation_by_pair"][first_valid] = first_train_relation
            with self.assertRaisesRegex(ValueError, "relation_leakage"):
                validate_bounded_plan_descriptor(path, **bad)

    def test_train_safe373_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, inputs = _descriptor_fixture(root)
            descriptor = json.loads(path.read_text())
            train_pair = descriptor["plans"]["train"][
                "interface_pair_ids"
            ][0]
            safe_path = root / "safe373.json"
            _write_json(
                safe_path,
                {
                    "plan_canonical_sha256": SAFE373_PLAN_CANONICAL_SHA256,
                    "safe_query_interface_pair_ids": [train_pair],
                    "safe_peptide_candidate_ids": ["A"],
                },
            )
            descriptor["safe373_evaluation_exclusion"][
                "plan_file_sha256"
            ] = sha256_file(safe_path)
            _rewrite_descriptor(path, descriptor)
            with self.assertRaisesRegex(ValueError, "train_safe373"):
                validate_bounded_plan_descriptor(path, **inputs)

    def test_descriptor_only_runner_fails_before_cache(self) -> None:
        args = argparse.Namespace(
            full_heavy_adaptation_manifest=None,
            full_heavy_plan_descriptor="descriptor.json",
            full_heavy_cache_manifest=None,
        )
        with self.assertRaisesRegex(ValueError, "cache_not_built"):
            _validate_full_heavy_cli_contract(args)

    def test_legacy_runner_mode_remains_unrequested(self) -> None:
        args = argparse.Namespace(
            full_heavy_adaptation_manifest=None,
            full_heavy_plan_descriptor=None,
            full_heavy_cache_manifest=None,
        )
        self.assertFalse(_validate_full_heavy_cli_contract(args))
        source = (
            Path(__file__).resolve().parents[1] / "train.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn(
            "sorted(train_base.interface_pair_ids)[:4096]", source
        )
        self.assertIn(
            "train_dataset = _interface_pair_subset(", source
        )


if __name__ == "__main__":
    unittest.main()

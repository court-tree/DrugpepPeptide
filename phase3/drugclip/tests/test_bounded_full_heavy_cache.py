from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from phase3.drugclip.build_bounded_full_heavy_cache import (
    CACHE_CONTRACT_SCHEMA,
    ConformerCoverageError,
    _canonical_record,
    _load_json,
    atomic_write_json,
    canonical_json_sha256,
    coordinate_sha256,
    load_descriptor_contract,
    materialize_cache,
    select_smoke_sequences,
    sequence_file_key,
)
from phase3.drugclip.full_heavy_adaptation_contract import (
    CACHE_MANIFEST_SCHEMA,
    sequence_sha256,
    sha256_file,
)
from phase3.drugclip.validate_bounded_full_heavy_cache import (
    validate_cache_read_only,
)


SEQUENCES = ["AAAAA", "GGGGG", "APAAA", "AAAAAA", "VVVVV"]
TOOL = {
    "commit_sha": "FASPR_COMMIT",
    "binary_sha256": "FASPR_BINARY",
    "rotamer_library_sha256": "FASPR_LIBRARY",
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _descriptor(root: Path, sequences: list[str] | None = None) -> dict:
    sequences = sorted(sequences or SEQUENCES)
    train_sequences = sequences[:3]
    valid_sequences = sequences[3:]
    train_path = root / "train_plan.json"
    valid_path = root / "valid_plan.json"
    for path, split, values in (
        (train_path, "train", train_sequences),
        (valid_path, "valid", valid_sequences),
    ):
        _write_json(
            path,
            {
                "split": split,
                "sequence_records": [
                    {
                        "peptide_sequence": sequence,
                        "theoretical_heavy_atom_count": 1,
                        "selected_pair_count": 1,
                    }
                    for sequence in values
                ],
            },
        )
    descriptor_core = {
        "schema_version": "phase3-v2-bounded-full-heavy-plan-v1",
        "plan_files": {
            "train": {
                "path": train_path.name,
                "file_sha256": sha256_file(train_path),
            },
            "valid": {
                "path": valid_path.name,
                "file_sha256": sha256_file(valid_path),
            },
        },
        "future_cache_requirement": {
            "required_peptide_sequences": sequences,
            "required_peptide_sequences_sha256": sequence_sha256(sequences),
            "conformers_per_sequence": 10,
            "future_required_conformer_count": len(sequences) * 10,
            "generation_status": "NOT_BUILT",
            "cache_status": "NOT_BUILT",
        },
    }
    descriptor = {
        **descriptor_core,
        "descriptor_canonical_sha256": canonical_json_sha256(descriptor_core),
    }
    descriptor_path = root / "descriptor.json"
    _write_json(descriptor_path, descriptor)
    return load_descriptor_contract(
        descriptor_path,
        expected_file_sha256=sha256_file(descriptor_path),
        expected_canonical_sha256=descriptor[
            "descriptor_canonical_sha256"
        ],
        enforce_formal_counts=False,
    )


def _payload(sequence: str) -> dict:
    identities = [
        {
            "atom_index": 0,
            "atom_name": "CA",
            "element": "C",
            "residue_index": 1,
            "residue_name": "ALA",
        }
    ]
    conformers = []
    for index in range(10):
        coordinates = [[float(index), 0.0, 0.0]]
        conformers.append(
            {
                "conformer_index": index,
                "attempt_index": 0,
                "coordinates": coordinates,
                "coordinate_sha256": coordinate_sha256(coordinates),
            }
        )
    return {
        "peptide_sequence": sequence,
        "generator_id": "fixture",
        "generator_version": {"fixture": True},
        "atom_count": 1,
        "atom_identity": identities,
        "atom_identity_sha256": canonical_json_sha256(identities),
        "conformer_count": 10,
        "conformers": conformers,
        "accepted_attempt_indices": [0] * 10,
        "total_generation_seconds": 1.0,
    }


def _validator(payload: dict) -> dict:
    if payload.get("target_bound_generation_inputs_used") is not False:
        raise ValueError("forbidden_generation_input")
    return {"status": "PASS"}


def _run(
    descriptor: dict,
    root: Path,
    *,
    resume: bool = False,
    generator=None,
    formal: bool = False,
    stop_after: int | None = None,
    prior_manifest_sha: str = "PRIOR_MANIFEST_FILE",
    prior_jsonl_sha: str = "PRIOR_JSONL",
    tool: dict | None = None,
) -> dict:
    generator = generator or (lambda sequence, _work: _payload(sequence))
    selected = [
        {
            **row,
            "selection_reason": ["fixture"],
        }
        for row in (
            {
                "peptide_sequence": sequence,
                "theoretical_heavy_atom_count": 1,
                "split_roles": (
                    ["train"] if sequence in sorted(SEQUENCES)[:3] else ["valid"]
                ),
            }
            for sequence in sorted(SEQUENCES)
        )
    ]
    return materialize_cache(
        descriptor_contract=descriptor,
        cache_root=root,
        prior_manifest_file_sha256=prior_manifest_sha,
        prior_jsonl_file_sha256=prior_jsonl_sha,
        tool=tool or TOOL,
        sequence_generator=generator,
        payload_validator=_validator,
        formal_cache=formal,
        selected_records=None if formal else selected,
        resume=resume,
        stop_after_new_sequences=stop_after,
        expected_formal_sequence_count=len(SEQUENCES),
    )


def _sequence_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted((root / "sequences").glob("*.json"))
    }


class BoundedFullHeavyCacheTests(unittest.TestCase):
    def test_two_identical_fixture_runs_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            _run(descriptor, root / "one")
            _run(descriptor, root / "two")
            self.assertEqual(
                _sequence_bytes(root / "one"), _sequence_bytes(root / "two")
            )
            self.assertEqual(
                (root / "one" / "smoke_manifest.json").read_bytes(),
                (root / "two" / "smoke_manifest.json").read_bytes(),
            )

    def test_interruption_resume_matches_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            interrupted = _run(
                descriptor, root / "resume", stop_after=2
            )
            self.assertFalse(interrupted["cache_manifest_created"])
            _run(descriptor, root / "resume", resume=True)
            _run(descriptor, root / "one_shot")
            self.assertEqual(
                _sequence_bytes(root / "resume"),
                _sequence_bytes(root / "one_shot"),
            )
            self.assertEqual(
                (root / "resume" / "smoke_manifest.json").read_bytes(),
                (root / "one_shot" / "smoke_manifest.json").read_bytes(),
            )

    def test_resume_does_not_regenerate_completed_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            calls: list[str] = []

            def generator(sequence: str, _work: Path) -> dict:
                calls.append(sequence)
                return _payload(sequence)

            _run(descriptor, root / "cache", generator=generator, stop_after=2)
            first = list(calls)
            _run(
                descriptor,
                root / "cache",
                generator=generator,
                resume=True,
            )
            self.assertEqual(calls[:2], first)
            self.assertEqual(len(calls), 5)

    def test_descriptor_sha_change_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            _run(descriptor, root / "cache", stop_after=1)
            changed = dict(descriptor)
            changed["file_sha256"] = "CHANGED"
            with self.assertRaisesRegex(ValueError, "contract_mismatch"):
                _run(changed, root / "cache", resume=True)

    def test_tool_sha_change_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            _run(descriptor, root / "cache", stop_after=1)
            changed = dict(TOOL)
            changed["binary_sha256"] = "CHANGED"
            with self.assertRaisesRegex(ValueError, "contract_mismatch"):
                _run(
                    descriptor, root / "cache", resume=True, tool=changed
                )

    def test_prior_change_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            _run(descriptor, root / "cache", stop_after=1)
            with self.assertRaisesRegex(ValueError, "contract_mismatch"):
                _run(
                    descriptor,
                    root / "cache",
                    resume=True,
                    prior_jsonl_sha="CHANGED",
                )

    def test_qc_change_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            _run(descriptor, root / "cache", stop_after=1)
            path = root / "cache" / "cache_contract.json"
            contract = _load_json(path)
            contract["nonlocal_heavy_atom_clash_threshold_angstrom"] = 0.5
            contract = _canonical_record(contract, "contract_canonical_sha256")
            _write_json(path, contract)
            with self.assertRaisesRegex(ValueError, "contract_mismatch"):
                _run(descriptor, root / "cache", resume=True)

    def test_generator_change_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            _run(descriptor, root / "cache", stop_after=1)
            path = root / "cache" / "cache_contract.json"
            contract = _load_json(path)
            contract["generator_version"] = "CHANGED"
            contract = _canonical_record(contract, "contract_canonical_sha256")
            _write_json(path, contract)
            with self.assertRaisesRegex(ValueError, "contract_mismatch"):
                _run(descriptor, root / "cache", resume=True)

    def test_corrupted_sequence_file_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            _run(descriptor, root / "cache", stop_after=1)
            path = next((root / "cache" / "sequences").glob("*.json"))
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                _run(descriptor, root / "cache", resume=True)

    def test_byte_changed_semantically_equal_file_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            _run(descriptor, root / "cache", stop_after=1)
            path = next((root / "cache" / "sequences").glob("*.json"))
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "file_sha256_changed"):
                _run(descriptor, root / "cache", resume=True)

    def test_stale_temporary_file_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            _run(descriptor, root / "cache", stop_after=1)
            (root / "cache" / "sequences" / ".orphan.json.tmp").write_text(
                "partial", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "stale_temporary"):
                _run(descriptor, root / "cache", resume=True)

    def test_slot_exhaustion_preserves_failure_and_no_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")

            def generator(sequence: str, _work: Path) -> dict:
                raise ConformerCoverageError(
                    "slot_exhausted",
                    details={
                        "logical_conformer_index": 0,
                        "attempts": 25,
                    },
                )

            with self.assertRaises(ConformerCoverageError):
                _run(descriptor, root / "cache", generator=generator)
            self.assertFalse((root / "cache" / "cache_manifest.json").exists())
            self.assertEqual(
                len(list((root / "cache" / "failures").glob("*.json"))), 1
            )
            with self.assertRaisesRegex(RuntimeError, "slot_exhaustion"):
                _run(
                    descriptor,
                    root / "cache",
                    generator=lambda sequence, _work: _payload(sequence),
                    resume=True,
                )

    def test_incomplete_cache_validator_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            _run(descriptor, root / "cache", stop_after=2)
            with self.assertRaisesRegex(ValueError, "smoke_manifest_missing"):
                validate_cache_read_only(
                    descriptor_contract=descriptor,
                    cache_root=root / "cache",
                    smoke=True,
                    payload_validator=_validator,
                )

    def test_formal_manifest_only_after_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            _run(
                descriptor,
                root / "cache",
                formal=True,
                stop_after=2,
            )
            self.assertFalse((root / "cache" / "cache_manifest.json").exists())
            _run(descriptor, root / "cache", formal=True, resume=True)
            manifest = _load_json(root / "cache" / "cache_manifest.json")
            self.assertEqual(manifest["schema_version"], CACHE_MANIFEST_SCHEMA)
            self.assertEqual(manifest["sequence_count"], 5)

    def test_missing_or_extra_sequence_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            _run(descriptor, root / "cache")
            path = next((root / "cache" / "sequences").glob("*.json"))
            path.rename(path.with_name("EXTRA.json"))
            with self.assertRaisesRegex(ValueError, "file_set_mismatch"):
                validate_cache_read_only(
                    descriptor_contract=descriptor,
                    cache_root=root / "cache",
                    smoke=True,
                    payload_validator=_validator,
                )

    def test_coordinate_and_atom_hash_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            _run(descriptor, root / "cache")
            path = next((root / "cache" / "sequences").glob("*.json"))
            payload = _load_json(path)
            payload["atom_identity_sha256"] = "TAMPERED"
            payload = _canonical_record(payload, "payload_canonical_sha256")
            _write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "atom_identity"):
                validate_cache_read_only(
                    descriptor_contract=descriptor,
                    cache_root=root / "cache",
                    smoke=True,
                    payload_validator=_validator,
                )

    def test_coordinate_hash_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            _run(descriptor, root / "cache")
            path = next((root / "cache" / "sequences").glob("*.json"))
            payload = _load_json(path)
            payload["conformers"][0]["coordinate_sha256"] = "TAMPERED"
            payload = _canonical_record(payload, "payload_canonical_sha256")
            _write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "coordinate_sha256"):
                validate_cache_read_only(
                    descriptor_contract=descriptor,
                    cache_root=root / "cache",
                    smoke=True,
                    payload_validator=_validator,
                )

    def test_safe373_cache_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="safe373_") as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            with self.assertRaisesRegex(ValueError, "evaluation_cache_path"):
                _run(descriptor, root / "cache")

    def test_generation_input_receptor_or_evidence_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")

            def generator(sequence: str, _work: Path) -> dict:
                value = _payload(sequence)
                value["target_bound_generation_inputs_used"] = True
                value["evidence"] = "forbidden"
                return value

            with self.assertRaisesRegex(
                ValueError, "forbidden_generation_input"
            ):
                _run(descriptor, root / "cache", generator=generator)
            self.assertFalse((root / "cache" / "cache_manifest.json").exists())

    def test_smoke_manifest_boundary_and_read_only_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = _descriptor(root / "plan")
            result = _run(descriptor, root / "cache")
            self.assertEqual(result["classification"], "CACHE_MATERIALIZER_SMOKE_PASS")
            report = validate_cache_read_only(
                descriptor_contract=descriptor,
                cache_root=root / "cache",
                smoke=True,
                payload_validator=_validator,
            )
            self.assertEqual(report["sequence_count"], 5)
            self.assertTrue(report["not_valid_for_training"])
            contract = _load_json(root / "cache" / "cache_contract.json")
            self.assertEqual(contract["schema_version"], CACHE_CONTRACT_SCHEMA)
            self.assertFalse(contract["formal_cache"])


if __name__ == "__main__":
    unittest.main()

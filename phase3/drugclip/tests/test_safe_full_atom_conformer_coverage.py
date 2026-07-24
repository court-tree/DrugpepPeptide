import argparse
import json
from pathlib import Path
import tempfile
import unittest

from phase3.drugclip import build_safe_full_atom_conformer_coverage as build
from phase3.drugclip import validate_safe_full_atom_conformer_coverage as validate


def _row(sequence, pair_id, classification, heavy_atoms=42, reason=None):
    return {
        "peptide_sequence": sequence,
        "interface_pair_id": pair_id,
        "chemistry_classification": classification,
        "theoretical_heavy_atom_count": heavy_atoms,
        "exclusion_reason": reason,
    }


def _deterministic_payload(elapsed=1.0):
    geometry = {
        "status": "PASS",
        "topology_contract": build.CANONICAL_TOPOLOGY_CONTRACT,
        "coordinate_chirality_match": True,
        "chirality_audit": {"status": "PASS"},
        "minimum_nonlocal_heavy_atom_distance_angstrom": 0.8,
    }
    return {
        "peptide_sequence": "AI",
        "atom_count": 17,
        "atom_identity_sha256": "ATOM",
        "canonical_coordinate_set_sha256": "SET",
        "accepted_attempt_indices": [0],
        "rejection_log_semantic_sha256": "REJECT",
        "attempt_audit": [
            {
                "logical_conformer_index": 0,
                "attempt_index": 0,
                "accepted": True,
                "rejection_reason": None,
                "elapsed_seconds": elapsed,
            }
        ],
        "conformers": [
            {
                "conformer_index": 0,
                "attempt_index": 0,
                "coordinate_sha256": "COORD",
                "faspr_output_sha256": "FASPR",
                "geometry_audit": geometry,
                "train_only_backbone_audit": {
                    "backbone_coordinate_sha256": "BACKBONE"
                },
            }
        ],
    }


class SafeFullAtomCoverageTests(unittest.TestCase):
    def test_candidate_contract_uses_strict_sequence_precedence(self):
        rows = [
            _row("AAAA", "q1", "ordinary_linear_standard"),
            _row("AAAA", "q2", "receptor_covalent", reason="bond"),
            _row("GGGG", "q3", "ordinary_linear_standard", heavy_atoms=17),
        ]
        contract = build.derive_candidate_contract(
            rows, enforce_expected_counts=False
        )
        by_sequence = {
            row["peptide_sequence"]: row for row in contract["candidates"]
        }
        self.assertEqual(
            by_sequence["AAAA"]["chemistry_classification"],
            "receptor_covalent",
        )
        self.assertEqual(contract["safe_sequence_count"], 1)
        self.assertEqual(contract["safe_query_count"], 1)
        self.assertTrue(contract["safe_query_targets_all_present"])

    def test_candidate_contract_rejects_inconsistent_heavy_atom_count(self):
        rows = [
            _row("AAAA", "q1", "ordinary_linear_standard", 20),
            _row("AAAA", "q2", "ordinary_linear_standard", 21),
        ]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            build.derive_candidate_contract(
                rows, enforce_expected_counts=False
            )

    def test_atomic_json_does_not_overwrite_completed_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            build.atomic_write_json(path, {"status": "PASS"})
            with self.assertRaises(FileExistsError):
                build.atomic_write_json(path, {"status": "FAIL"})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["status"], "PASS"
            )
            self.assertFalse((path.parent / f".{path.name}.tmp").exists())

    def test_atomic_json_progress_replace_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "progress.json"
            build.atomic_write_json(path, {"count": 1})
            build.atomic_write_json(path, {"count": 2}, replace_existing=True)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["count"], 2
            )

    def test_cli_requires_all_explicit_input_paths(self):
        for module in (build, validate):
            parser = module.build_parser()
            with self.assertRaises(SystemExit):
                parser.parse_args([])
            parsed = parser.parse_args(
                [
                    "--chemistry-audit", "chem.jsonl",
                    "--prior-manifest", "manifest.json",
                    "--prior-jsonl", "prior.jsonl",
                    "--faspr-executable", "FASPR",
                    "--faspr-rotamer-library", "dun2010bbdep.bin",
                    "--output-dir", "out",
                ]
            )
            self.assertEqual(parsed.output_dir, "out")

    def test_cache_key_depends_only_on_sequence(self):
        self.assertEqual(
            build.sequence_cache_key("SAVTTVVN"),
            build.sequence_cache_key("SAVTTVVN"),
        )
        self.assertNotEqual(
            build.sequence_cache_key("SAVTTVVN"),
            build.sequence_cache_key("TLAPADGPTTDEVTLQV"),
        )

    def test_validation_enforces_strict_atom_cap(self):
        payload = {"atom_count": 192, "peptide_sequence": "A"}
        with self.assertRaisesRegex(ValueError, "strictly_below_192"):
            original = validate.validate_panel_payload
            validate.validate_panel_payload = lambda value: {"status": "PASS"}
            try:
                validate._validate_payload(payload)
            finally:
                validate.validate_panel_payload = original

    def test_manifest_canonical_hash_detects_mutation(self):
        core = {"status": "PASS", "count": 265}
        manifest = {
            **core,
            "manifest_canonical_sha256": build.canonical_json_sha256(core),
        }
        validate._verify_canonical_manifest(manifest)
        manifest["count"] = 264
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate._verify_canonical_manifest(manifest)

    def test_attempt_audit_records_canonical_topology_and_atom_count(self):
        payload = _deterministic_payload()
        build.enrich_attempt_audit(payload)
        attempt = payload["attempt_audit"][0]
        self.assertEqual(attempt["atom_count"], 17)
        self.assertEqual(
            attempt["canonical_topology_qc"]["topology_contract"],
            build.CANONICAL_TOPOLOGY_CONTRACT,
        )

    def test_deterministic_sequence_record_excludes_elapsed_time(self):
        first = _deterministic_payload(elapsed=1.0)
        second = _deterministic_payload(elapsed=99.0)
        self.assertEqual(
            build.deterministic_sequence_record(first),
            build.deterministic_sequence_record(second),
        )

    def test_deterministic_global_manifest_is_path_and_time_independent(self):
        record = build.deterministic_sequence_record(
            _deterministic_payload()
        )
        arguments = {
            "prior_manifest_sha256": "PRIOR",
            "prior_jsonl_sha256": "JSONL",
            "faspr_source_commit": "COMMIT",
            "faspr_binary_sha256": "BINARY",
            "faspr_rotamer_library_sha256": "LIBRARY",
            "input_candidate_list_sha256": "CANDIDATES",
            "chemistry_audit_sha256": "CHEMISTRY",
        }
        first = build.deterministic_generation_manifest_core(
            [record], **arguments
        )
        second = build.deterministic_generation_manifest_core(
            [record], **arguments
        )
        self.assertEqual(
            build.canonical_json_sha256(first),
            build.canonical_json_sha256(second),
        )

    def test_validation_rejects_noncanonical_topology_contract(self):
        payload = _deterministic_payload()
        payload["conformers"][0]["geometry_audit"][
            "topology_contract"
        ] = "rdkit-mol-from-sequence"
        original = validate.validate_panel_payload
        validate.validate_panel_payload = lambda value: {"status": "PASS"}
        try:
            with self.assertRaisesRegex(ValueError, "topology_contract"):
                validate._validate_payload(payload)
        finally:
            validate.validate_panel_payload = original


if __name__ == "__main__":
    unittest.main()

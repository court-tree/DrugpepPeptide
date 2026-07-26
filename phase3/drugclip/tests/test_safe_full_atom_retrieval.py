import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from phase3.drugclip import evaluate_safe_full_atom_retrieval as evaluator


def _synthetic_plan_and_chemistry():
    safe_sequences = [f"SAFE{index:03d}" for index in range(265)]
    rejected_sequences = [f"REJECT{index:03d}" for index in range(105)]
    sequences = (
        safe_sequences
        + safe_sequences[:108]
        + rejected_sequences
        + rejected_sequences[:34]
    )
    plan = []
    chemistry = []
    for index, sequence in enumerate(sequences):
        pair_id = f"pair:{index:03d}"
        plan.append({
            "interface_pair_id": pair_id,
            "biological_pair_id": f"bio:{index:03d}",
            "peptide_sequence": sequence,
            "receptor_interface_id": f"iface:{index:03d}",
            "conformer_index": 0,
        })
        safe = sequence.startswith("SAFE")
        chemistry.append({
            "interface_pair_id": pair_id,
            "peptide_sequence": sequence,
            "chemistry_classification": (
                "ordinary_linear_standard"
                if safe
                else "modified_or_nonstandard"
            ),
            "theoretical_heavy_atom_count": 42,
            "exclusion_reason": None if safe else "synthetic",
        })
    return plan, chemistry


class SafeFullAtomRetrievalTests(unittest.TestCase):
    def test_safe_plan_preserves_fixed_relative_order_and_counts(self):
        plan, chemistry = _synthetic_plan_and_chemistry()
        record, safe_plan = evaluator.build_safe373_plan(
            plan,
            chemistry,
            fixed_plan_file_sha256="FILE",
            fixed_plan_canonical_sha256="CANONICAL",
            chemistry_audit_sha256="CHEMISTRY",
        )
        self.assertEqual(len(safe_plan), 373)
        self.assertEqual(record["counts"], {
            "queries": 373,
            "peptide_candidates": 265,
            "receptor_candidates": 512,
        })
        expected = [
            row["interface_pair_id"]
            for row in plan
            if row["peptide_sequence"].startswith("SAFE")
        ]
        self.assertEqual(
            record["safe_query_interface_pair_ids"], expected
        )

    def test_safe_plan_hash_detects_mutation(self):
        plan, chemistry = _synthetic_plan_and_chemistry()
        record, _ = evaluator.build_safe373_plan(
            plan,
            chemistry,
            fixed_plan_file_sha256="FILE",
            fixed_plan_canonical_sha256="CANONICAL",
            chemistry_audit_sha256="CHEMISTRY",
        )
        evaluator._verify_canonical_object(
            record, "plan_canonical_sha256", "plan"
        )
        record["counts"]["queries"] = 372
        with self.assertRaisesRegex(ValueError, "canonical"):
            evaluator._verify_canonical_object(
                record, "plan_canonical_sha256", "plan"
            )

    def test_full_atom_conversion_is_finite_and_has_no_unk(self):
        atoms = evaluator.full_atom_atoms(
            [{
                "atom_name": "CA",
                "element": "C",
                "residue_index": 1,
                "residue_name": "ALA",
            }],
            [[1.0, 2.0, 3.0]],
        )
        self.assertEqual(atoms[0]["residue_id"], "1")
        self.assertEqual(atoms[0]["x"], 1.0)

    def test_full_atom_conversion_rejects_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "UNK"):
            evaluator.full_atom_atoms(
                [{
                    "atom_name": "FAKE",
                    "element": "C",
                    "residue_index": 1,
                    "residue_name": "ALA",
                }],
                [[0.0, 0.0, 0.0]],
            )

    def test_candidate_representative_uses_first_sampler_pair(self):
        items = [
            {
                "interface_pair_id": "p1",
                "peptide_sequence": "AAA",
                "receptor_id": "r1",
            },
            {
                "interface_pair_id": "p2",
                "peptide_sequence": "AAA",
                "receptor_id": "r2",
            },
        ]
        ids, representatives = evaluator.candidate_contract(
            items, ["p2", "p1"], "r2p"
        )
        self.assertEqual(ids, ["AAA"])
        self.assertEqual(representatives["AAA"], "p2")

    def test_rank_scores_recomputes_in_bank_known_positive_mask(self):
        item = {
            "interface_pair_id": "p1",
            "peptide_sequence": "AAA",
            "receptor_id": "r1",
            "known_positive_group": {
                "receptor_peptides": ["AAA", "BBB", "OUT"],
                "peptide_receptors": ["r1"],
            },
        }
        records, metrics = evaluator.rank_scores(
            torch.tensor([[0.1, 0.9, 0.8]]),
            [item],
            ["AAA", "BBB", "CCC"],
            "r2p",
        )
        self.assertEqual(records[0]["rank"], 2)
        self.assertEqual(records[0]["known_positive_candidates_excluded"], 1)
        self.assertEqual(metrics["target_missing"], 0)

    def test_metric_summary_includes_dispersion_and_worst_rank(self):
        metrics = evaluator.metric_summary([1, 3, 5])
        self.assertAlmostEqual(metrics["mean_rank"], 3.0)
        self.assertAlmostEqual(
            metrics["rank_standard_deviation"], np.std([1, 3, 5])
        )
        self.assertEqual(metrics["worst_rank"], 5.0)

    def test_paired_bootstrap_is_paired_and_reports_zero_crossing(self):
        indices = np.tile(np.arange(3), (10, 1))
        result = evaluator.paired_bootstrap(
            [1, 2, 3], [1, 2, 3], indices, 17
        )
        for row in result["later_minus_earlier"].values():
            self.assertEqual(row["point_estimate"], 0.0)
            self.assertTrue(row["crosses_zero"])

    def test_preregistered_comparisons_cover_required_questions(self):
        labels = {
            row["label"] for row in evaluator.preregistered_comparisons()
        }
        self.assertTrue({
            "D0_minus_C0",
            "Dmean10_minus_Cmean10",
            "Dmean10_minus_one_d_only",
            "Dmean10_minus_A",
            "Dmean10_minus_D0",
            "phase3_epoch0_minus_phase2_Dmean10",
        } <= labels)

    def test_dynamic_model_label_enters_models_and_bootstrap_comparisons(self):
        label = "phase3_v2_step032"
        entries = evaluator.checkpoint_model_entries(
            Path("phase2.pt"),
            {"model_label": label, "checkpoint": Path("step_032.pt")},
        )
        self.assertEqual([name for name, _ in entries], [
            "phase2_baseline", label,
        ])
        comparisons = evaluator.preregistered_comparisons(label)
        self.assertTrue(any(
            row["later_model"] == label
            and row["label"] == f"{label}_minus_phase2_Dmean10"
            for row in comparisons
        ))
        serialized = json.dumps(comparisons, sort_keys=True)
        self.assertNotIn("phase3_v1_epoch0", serialized)
        self.assertNotIn("phase3_epoch0_minus_phase2", serialized)

    def test_legacy_v1_default_checkpoint_contract_is_unchanged(self):
        cli = evaluator.build_parser().parse_args([
            "--pilot-output", "pilot",
            "--dataset-root", "dataset",
            "--chemistry-audit", "chemistry.jsonl",
            "--safe265-cache-dir", "cache",
            "--candidate-evidence-jsonl", "candidate.jsonl",
            "--expanded-evidence-jsonl", "expanded.jsonl",
            "--mmcif-root", "mmcif",
            "--qbiolip-root", "qbiolip",
            "--biolip-root", "biolip",
            "--phase2-checkpoint", "phase2.pt",
            "--source-model-configs", "configs.json",
            "--output-dir", "output",
        ])
        contract = evaluator.resolve_phase3_checkpoint_contract(
            cli, Path(evaluator.__file__).resolve().parents[2]
        )
        self.assertEqual(contract["mode"], "legacy_phase3_v1_epoch0")
        self.assertEqual(contract["model_label"], "phase3_v1_epoch0")
        self.assertEqual(
            contract["actual_sha256"],
            evaluator.FIXED512_EXPECTED["phase3_checkpoint_sha256"],
        )
        labels = {
            row["label"] for row in evaluator.preregistered_comparisons()
        }
        self.assertIn("phase3_epoch0_minus_phase2_Dmean10", labels)

    def test_explicit_checkpoint_sha_and_label_contract_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "step_032.pt"
            checkpoint.write_bytes(b"future-v2-checkpoint-fixture")
            expected = evaluator._file_hash(checkpoint)
            cli = unittest.mock.Mock(
                phase3_checkpoint=str(checkpoint),
                phase3_checkpoint_sha256=expected.lower(),
                phase3_model_label="phase3_v2_step032",
            )
            contract = evaluator.resolve_phase3_checkpoint_contract(
                cli, Path(temporary)
            )
        self.assertEqual(contract["mode"], "explicit")
        self.assertEqual(contract["actual_sha256"], expected)
        self.assertEqual(contract["expected_sha256"], expected)
        self.assertEqual(contract["model_label"], "phase3_v2_step032")

    def test_explicit_checkpoint_sha_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "step_032.pt"
            checkpoint.write_bytes(b"fixture")
            cli = unittest.mock.Mock(
                phase3_checkpoint=str(checkpoint),
                phase3_checkpoint_sha256="0" * 64,
                phase3_model_label="phase3_v2_step032",
            )
            with self.assertRaisesRegex(
                ValueError, "phase3_checkpoint_sha256_mismatch"
            ):
                evaluator.resolve_phase3_checkpoint_contract(
                    cli, Path(temporary)
                )

    def test_partial_explicit_checkpoint_contract_is_rejected(self):
        combinations = [
            ("step.pt", None, None),
            (None, "0" * 64, None),
            (None, None, "phase3_v2_step032"),
            ("step.pt", "0" * 64, None),
            ("step.pt", None, "phase3_v2_step032"),
            (None, "0" * 64, "phase3_v2_step032"),
        ]
        for checkpoint, sha256, label in combinations:
            with self.subTest(
                checkpoint=checkpoint, sha256=sha256, label=label
            ):
                cli = unittest.mock.Mock(
                    phase3_checkpoint=checkpoint,
                    phase3_checkpoint_sha256=sha256,
                    phase3_model_label=label,
                )
                with self.assertRaisesRegex(
                    ValueError, "requires_checkpoint_sha256_and_label"
                ):
                    evaluator.resolve_phase3_checkpoint_contract(
                        cli, Path(".")
                    )

    def test_invalid_or_conflicting_explicit_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "step.pt"
            checkpoint.write_bytes(b"fixture")
            expected = evaluator._file_hash(checkpoint)
            for label in ("", " phase3_v2_step032", "phase2_baseline"):
                with self.subTest(label=label):
                    cli = unittest.mock.Mock(
                        phase3_checkpoint=str(checkpoint),
                        phase3_checkpoint_sha256=expected,
                        phase3_model_label=label,
                    )
                    with self.assertRaises(ValueError):
                        evaluator.resolve_phase3_checkpoint_contract(
                            cli, Path(temporary)
                        )
        with self.assertRaisesRegex(
            ValueError, "duplicate_evaluation_model_label"
        ):
            evaluator.checkpoint_model_entries(
                Path("phase2.pt"),
                {
                    "model_label": "phase2_baseline",
                    "checkpoint": Path("phase3.pt"),
                },
            )

    def test_dynamic_label_loads_explicit_checkpoint_state(self):
        sentinel = object()
        with patch.object(
            evaluator,
            "_load_input_domain_model",
            return_value=sentinel,
        ) as loader:
            result = evaluator._load_model(
                "phase3_v2_step032",
                Path("step_032.pt"),
                Path("phase2.pt"),
                {},
                unittest.mock.Mock(),
                Path("."),
                torch.device("cpu"),
            )
        self.assertIs(result, sentinel)
        self.assertEqual(
            loader.call_args.args[0], evaluator.LEGACY_PHASE3_MODEL_LABEL
        )
        self.assertEqual(loader.call_args.args[1], Path("step_032.pt"))

    def test_score_mean_requires_ten_matrices(self):
        matrices = [torch.full((2, 3), float(index)) for index in range(10)]
        mean = evaluator.arithmetic_mean_score(matrices)
        self.assertEqual(list(mean.shape), [2, 3])
        self.assertTrue(torch.equal(mean, torch.full((2, 3), 4.5)))

    def test_cli_requires_explicit_contract_paths(self):
        parser = evaluator.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        parsed = parser.parse_args([
            "--pilot-output", "pilot",
            "--dataset-root", "dataset",
            "--chemistry-audit", "chemistry.jsonl",
            "--safe265-cache-dir", "cache",
            "--candidate-evidence-jsonl", "candidate.jsonl",
            "--expanded-evidence-jsonl", "expanded.jsonl",
            "--mmcif-root", "mmcif",
            "--qbiolip-root", "qbiolip",
            "--biolip-root", "biolip",
            "--phase2-checkpoint", "phase2.pt",
            "--source-model-configs", "configs.json",
            "--output-dir", "output",
            "--preflight-only",
        ])
        self.assertTrue(parsed.preflight_only)

    def test_module_has_no_training_calls(self):
        source = Path(evaluator.__file__).read_text(encoding="utf-8")
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.optim", source)


if __name__ == "__main__":
    unittest.main()

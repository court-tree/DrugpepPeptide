from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from phase3.drugclip.evaluate_full_retrieval import (
    _checkpoint_specs,
    _rank_direction,
    _validate_checkpoint_data_contract,
    parse_args,
)


class FullCandidateRetrievalTests(unittest.TestCase):
    def test_explicit_checkpoint_and_model_label_are_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = _checkpoint_specs("step_032.pt", "step_032", root, root / "phase2.pt", root)
            self.assertEqual(specs, [("phase2_baseline", root / "phase2.pt"), ("step_032", (root / "step_032.pt").resolve())])
        with patch("sys.argv", ["evaluate", "--checkpoint", "chosen.pt", "--model-label", "chosen"]):
            args = parse_args()
        self.assertEqual(args.checkpoint, "chosen.pt")
        self.assertEqual(args.model_label, "chosen")

    def test_evaluator_rejects_v2_checkpoint_for_v3_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.pt"
            torch.save({"run_config": {"data_version": "v2"}}, path)
            with self.assertRaisesRegex(ValueError, "data_version"):
                _validate_checkpoint_data_contract(path, {"data_version": "v3"})

    def test_exact_known_positives_are_excluded_but_target_is_retained(self) -> None:
        rows = {
            "pair:a": {
                "interface_pair_id": "pair:a", "receptor_interface_id": "iface:a", "peptide_sequence": "AAA",
                "known_positive_group": {"receptor_peptides": ["AAA", "BBB"], "peptide_receptors": ["iface:a"]},
                "receptor_embedding": torch.tensor([1.0, 0.0]), "peptide_embedding": torch.tensor([1.0, 0.0]),
            },
            "pair:b": {
                "interface_pair_id": "pair:b", "receptor_interface_id": "iface:b", "peptide_sequence": "BBB",
                "known_positive_group": {"receptor_peptides": ["BBB"], "peptide_receptors": ["iface:b"]},
                "receptor_embedding": torch.tensor([0.0, 1.0]), "peptide_embedding": torch.tensor([0.9, 0.0]),
            },
            "pair:c": {
                "interface_pair_id": "pair:c", "receptor_interface_id": "iface:c", "peptide_sequence": "CCC",
                "known_positive_group": {"receptor_peptides": ["CCC"], "peptide_receptors": ["iface:c"]},
                "receptor_embedding": torch.tensor([0.0, 1.0]), "peptide_embedding": torch.tensor([0.8, 0.0]),
            },
        }
        plan = [{"interface_pair_id": "pair:a"}]
        records, metrics = _rank_direction(rows, plan, ["AAA", "BBB", "CCC"], torch.stack([rows["pair:a"]["peptide_embedding"], rows["pair:b"]["peptide_embedding"], rows["pair:c"]["peptide_embedding"]]), "r2p", 1.0)
        self.assertEqual(records[0]["rank"], 1)
        self.assertEqual(records[0]["candidate_count"], 2)
        self.assertEqual(records[0]["known_positive_candidates_excluded"], 1)
        self.assertEqual(metrics["known_positive_exclusion_total"], 1)

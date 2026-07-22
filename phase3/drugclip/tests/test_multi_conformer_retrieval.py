from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from phase3.drugclip.evaluate_multi_conformer_retrieval import (
    _bootstrap_difference, _mean_score, _set_fixed_conformer, parse_args,
)


class _Base:
    fixed_conformer_index = 0


class _Dataset:
    def __init__(self) -> None:
        self.base = _Base()
        self._plan = []
    def set_epoch(self, epoch: int) -> None:
        self._plan = [{"interface_pair_id": "pair:1", "receptor_interface_id": "iface:1", "peptide_sequence": "AAA", "biological_pair_id": "bio:1", "conformer_index": self.base.fixed_conformer_index}]
    def epoch_plan(self):
        return self._plan


class MultiConformerRetrievalTests(unittest.TestCase):
    def test_explicit_checkpoint_and_model_label_are_parsed(self) -> None:
        with patch("sys.argv", ["evaluate", "--checkpoint", "step_032.pt", "--model-label", "step_032"]):
            args = parse_args()
        self.assertEqual(args.checkpoint, "step_032.pt")
        self.assertEqual(args.model_label, "step_032")

    def test_every_plan_row_switches_to_same_conformer(self) -> None:
        dataset = _Dataset()
        reference = [{"interface_pair_id": "pair:1", "receptor_interface_id": "iface:1", "peptide_sequence": "AAA", "biological_pair_id": "bio:1", "conformer_index": 0}]
        plan = _set_fixed_conformer(dataset, 7, reference)
        self.assertEqual(plan[0]["conformer_index"], 7)

    def test_target_only_conformer_change_is_rejected(self) -> None:
        dataset = _Dataset()
        reference = [{"interface_pair_id": "pair:1", "receptor_interface_id": "iface:1", "peptide_sequence": "AAA", "biological_pair_id": "bio:1", "conformer_index": 0}]
        dataset.set_epoch = lambda epoch: setattr(dataset, "_plan", [{**reference[0], "conformer_index": 0}])
        with self.assertRaisesRegex(RuntimeError, "not every candidate/query"):
            _set_fixed_conformer(dataset, 7, reference)

    def test_mean_score_requires_all_ten_and_uses_arithmetic_mean(self) -> None:
        matrices = [torch.full((2, 3), float(index)) for index in range(10)]
        self.assertTrue(torch.equal(_mean_score(matrices), torch.full((2, 3), 4.5)))
        self.assertFalse(torch.equal(_mean_score(matrices), torch.full((2, 3), 9.0)))
        with self.assertRaisesRegex(ValueError, "exactly ten"):
            _mean_score(matrices[:-1])

    def test_paired_bootstrap_is_reproducible(self) -> None:
        baseline = [
            {"interface_pair_id": "a", "ranks_by_conformer": [2] * 10, "mean_rank": 2.0, "worst_rank": 2, "rank_std": 0.0},
            {"interface_pair_id": "b", "ranks_by_conformer": [4] * 10, "mean_rank": 4.0, "worst_rank": 4, "rank_std": 0.0},
        ]
        improved = [
            {"interface_pair_id": "a", "ranks_by_conformer": [1] * 10, "mean_rank": 1.0, "worst_rank": 1, "rank_std": 0.0},
            {"interface_pair_id": "b", "ranks_by_conformer": [3] * 10, "mean_rank": 3.0, "worst_rank": 3, "rank_std": 0.0},
        ]
        self.assertEqual(_bootstrap_difference(baseline, improved, 2000, 17), _bootstrap_difference(baseline, improved, 2000, 17))

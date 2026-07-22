from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from phase3.drugclip.losses import masked_bidirectional_known_positive_loss


def loss_for(
    logits: torch.Tensor,
    groups: list[dict],
    receptors: list[str] | None = None,
    peptides: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    size = logits.shape[0]
    return masked_bidirectional_known_positive_loss(
        logits,
        logits.t().contiguous(),
        receptors or [f"iface:{index}" for index in range(size)],
        peptides or [f"P{index}" for index in range(size)],
        groups,
    )


class KnownPositiveLossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.tensor(
            [[3.0, 1.0, -2.0], [0.5, 2.0, 0.2], [-1.0, 0.3, 1.5]],
            requires_grad=True,
        )
        self.receptors = ["iface:I1", "iface:I2", "iface:I3"]
        self.peptides = ["P1", "P2", "P3"]
        self.empty = [{}, {}, {}]

    def test_no_extra_known_positive_matches_standard_bidirectional_clip(self):
        result = loss_for(self.logits, self.empty, self.receptors, self.peptides)
        targets = torch.arange(3)
        expected_r2p = F.cross_entropy(self.logits, targets)
        expected_p2r = F.cross_entropy(self.logits.t(), targets)
        self.assertTrue(torch.allclose(result["loss_receptor_to_peptide"], expected_r2p))
        self.assertTrue(torch.allclose(result["loss_peptide_to_receptor"], expected_p2r))
        self.assertTrue(torch.allclose(result["loss_total"], expected_r2p + expected_p2r))
        self.assertFalse(result["receptor_to_peptide_mask"].any())
        self.assertFalse(result["peptide_to_receptor_mask"].any())

    def test_receptor_to_peptide_extra_known_positive_is_masked_not_positive(self):
        groups = [{"receptor_peptides": ["P1", "P2"]}, {}, {}]
        result = loss_for(self.logits, groups, self.receptors, self.peptides)
        self.assertTrue(result["receptor_to_peptide_mask"][0, 1])
        self.assertFalse(result["receptor_to_peptide_mask"][0, 0])
        self.assertFalse(result["peptide_to_receptor_mask"].any())

    def test_peptide_to_receptor_uses_exact_interface_id(self):
        groups = [{"peptide_receptors": ["iface:I1", "iface:I2"]}, {}, {}]
        result = loss_for(self.logits, groups, self.receptors, self.peptides)
        self.assertTrue(result["peptide_to_receptor_mask"][0, 1])
        self.assertFalse(result["peptide_to_receptor_mask"][0, 0])
        wrong_semantic_id = [{"peptide_receptors": ["interface_pair:I2"]}, {}, {}]
        wrong_result = loss_for(self.logits, wrong_semantic_id, self.receptors, self.peptides)
        self.assertFalse(wrong_result["peptide_to_receptor_mask"].any())

    def test_both_directions_can_mask_independently(self):
        groups = [
            {"receptor_peptides": ["P1", "P2"], "peptide_receptors": ["iface:I1", "iface:I3"]},
            {},
            {},
        ]
        result = loss_for(self.logits, groups, self.receptors, self.peptides)
        self.assertTrue(result["receptor_to_peptide_mask"][0, 1])
        self.assertTrue(result["peptide_to_receptor_mask"][0, 2])

    def test_similar_but_not_exact_peptide_does_not_mask(self):
        groups = [{"receptor_peptides": ["P1_similar"]}, {}, {}]
        result = loss_for(self.logits, groups, self.receptors, self.peptides)
        self.assertFalse(result["receptor_to_peptide_mask"].any())

    def test_diagonal_is_never_masked_even_if_list_contains_it(self):
        groups = [
            {"receptor_peptides": ["P1"], "peptide_receptors": ["iface:I1"]},
            {},
            {},
        ]
        result = loss_for(self.logits, groups, self.receptors, self.peptides)
        self.assertFalse(torch.diagonal(result["receptor_to_peptide_mask"]).any())
        self.assertFalse(torch.diagonal(result["peptide_to_receptor_mask"]).any())

    def test_masked_loss_and_both_direction_gradients_are_finite(self):
        groups = [{"receptor_peptides": ["P2"], "peptide_receptors": ["iface:I2"]}, {}, {}]
        logits = self.logits.detach().clone().requires_grad_(True)
        result = loss_for(logits, groups, self.receptors, self.peptides)
        result["loss_total"].backward()
        self.assertTrue(torch.isfinite(result["loss_total"]))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.norm()), 0.0)

    def test_each_direction_individually_has_a_finite_nonzero_gradient(self):
        groups = [{"receptor_peptides": ["P2"], "peptide_receptors": ["iface:I2"]}, {}, {}]
        logits = self.logits.detach().clone().requires_grad_(True)
        result = loss_for(logits, groups, self.receptors, self.peptides)
        result["loss_receptor_to_peptide"].backward(retain_graph=True)
        r2p_gradient = logits.grad.detach().clone()
        logits.grad.zero_()
        result["loss_peptide_to_receptor"].backward()
        p2r_gradient = logits.grad.detach().clone()
        self.assertTrue(torch.isfinite(r2p_gradient).all())
        self.assertTrue(torch.isfinite(p2r_gradient).all())
        self.assertGreater(float(r2p_gradient.norm()), 0.0)
        self.assertGreater(float(p2r_gradient.norm()), 0.0)

    def test_changing_masked_candidate_does_not_change_its_query_row_loss(self):
        groups = [{"receptor_peptides": ["P2"]}, {}, {}]
        first = loss_for(self.logits, groups, self.receptors, self.peptides)["loss_receptor_to_peptide"]
        changed = self.logits.detach().clone()
        changed[0, 1] = 1.0e4
        second = loss_for(changed, groups, self.receptors, self.peptides)["loss_receptor_to_peptide"]
        self.assertTrue(torch.allclose(first, second))

    def test_changing_unmasked_negative_changes_query_loss(self):
        groups = [{"receptor_peptides": ["P2"]}, {}, {}]
        first = loss_for(self.logits, groups, self.receptors, self.peptides)["loss_receptor_to_peptide"]
        changed = self.logits.detach().clone()
        changed[0, 2] = 1.0e4
        second = loss_for(changed, groups, self.receptors, self.peptides)["loss_receptor_to_peptide"]
        self.assertFalse(torch.allclose(first, second))

    def test_each_masked_row_retains_its_diagonal_target(self):
        groups = [
            {"receptor_peptides": ["P1", "P2", "P3"], "peptide_receptors": self.receptors},
            {"receptor_peptides": ["P1", "P2", "P3"], "peptide_receptors": self.receptors},
            {"receptor_peptides": ["P1", "P2", "P3"], "peptide_receptors": self.receptors},
        ]
        result = loss_for(self.logits, groups, self.receptors, self.peptides)
        for row in range(3):
            self.assertFalse(result["receptor_to_peptide_mask"][row, row])
            self.assertFalse(result["peptide_to_receptor_mask"][row, row])
            self.assertEqual(int((~result["receptor_to_peptide_mask"][row]).sum()), 1)
            self.assertEqual(int((~result["peptide_to_receptor_mask"][row]).sum()), 1)


if __name__ == "__main__":
    unittest.main()

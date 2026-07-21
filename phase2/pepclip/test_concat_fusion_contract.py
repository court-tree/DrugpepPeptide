from __future__ import annotations

import unittest

import torch
from torch import nn

from phase2.pepclip.data import collate_pepclip, collate_pepclip_3d
from phase2.pepclip.losses import grouped_duplicate_mask, symmetric_in_batch_softmax_loss
from phase2.pepclip.model_3d import PepCLIP3DModel
from phase2.pepclip.train_concat_fusion import PepCLIPConcatFusionModel


class GroupedDuplicateMaskTests(unittest.TestCase):
    def test_multiple_identity_groups_are_merged(self) -> None:
        mask = grouped_duplicate_mask(
            [["a", "a", "b"], ["x", "y", "y"]], torch.device("cpu")
        )
        expected = torch.tensor(
            [[False, True, False], [True, False, True], [False, True, False]]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_empty_keys_do_not_create_duplicates(self) -> None:
        mask = grouped_duplicate_mask([["", "", "x"]], torch.device("cpu"))
        self.assertFalse(mask.any())

    def test_symmetric_loss_accepts_grouped_identity_keys(self) -> None:
        logits = torch.tensor(
            [[3.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 3.0]],
            requires_grad=True,
        )
        loss = symmetric_in_batch_softmax_loss(
            logits,
            logits.t(),
            receptor_key_groups=[["r1", "r1", "r2"]],
            peptide_key_groups=[["p1", "p2", "p2"]],
        )
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))


class DataAndModelContractTests(unittest.TestCase):
    def test_relation_metadata_survives_1d_and_3d_collation(self) -> None:
        metadata = {
            "conformer_cluster_id": "cluster:1",
            "peptide_sequence_id": "pep:1",
            "peptide_homology_80_id": "pep80:1",
            "receptor_family_30_id": "rfam30:1",
            "receptor_interface_key": "interface:1",
        }
        one_d = collate_pepclip(
            [{
                "sample_id": "sample:1", "pdb_id": "1abc", "split": "train",
                "receptor_key": "receptor:1", "peptide_key": "ACD",
                "receptor_sequence": "ACDE", "peptide_sequence": "ACD",
                "receptor_tokens": torch.tensor([1, 2, 3, 4]),
                "peptide_tokens": torch.tensor([1, 2, 3]),
                "peptide_length": 3, "avg_contact_count": 2.0, "contact_coverage": 1.0,
                **metadata,
            }]
        )
        three_d = collate_pepclip_3d(
            [{
                "sample_id": "sample:1", "pdb_id": "1abc", "split": "train",
                "receptor_key": "receptor:1", "peptide_key": "ACD",
                "receptor_coords": torch.tensor([[0.0, 0.0, 0.0]]),
                "receptor_elements": torch.tensor([1]),
                "receptor_atom_names": torch.tensor([1]),
                "receptor_residue_names": torch.tensor([1]),
                "peptide_coords": torch.tensor([[1.0, 0.0, 0.0]]),
                "peptide_elements": torch.tensor([1]),
                "peptide_atom_names": torch.tensor([1]),
                "peptide_residue_names": torch.tensor([1]),
                "num_receptor_atoms": 1, "num_peptide_atoms": 1,
                **metadata,
            }]
        )
        for key, value in metadata.items():
            self.assertEqual(one_d[key], [value])
            self.assertEqual(three_d[key], [value])

    def test_egnn_model_uses_explicit_num_neighbors(self) -> None:
        model = PepCLIP3DModel(
            num_elements=4,
            num_atom_names=4,
            num_residue_names=4,
            encoder_type="egnn",
            element_dim=8,
            hidden_dim=16,
            output_dim=8,
            dropout=0.0,
            num_layers=1,
            num_rbf=4,
            num_neighbors=3,
        )
        self.assertEqual(model.receptor_encoder.num_neighbors, 3)
        self.assertEqual(model.peptide_encoder.num_neighbors, 3)


class _Stub1DTower(nn.Module):
    def encode_receptor(self, receptor_tokens, receptor_sequences):
        return torch.ones((receptor_tokens.shape[0], 3), device=receptor_tokens.device)

    def encode_peptide(self, peptide_tokens, peptide_sequences):
        return torch.full((peptide_tokens.shape[0], 3), 2.0, device=peptide_tokens.device)


class _Stub3DTower(nn.Module):
    def encode_receptor(self, receptor_coords, **kwargs):
        return torch.ones((receptor_coords.shape[0], 2), device=receptor_coords.device)

    def encode_peptide(self, peptide_coords, **kwargs):
        return torch.full((peptide_coords.shape[0], 2), 2.0, device=peptide_coords.device)


class ConcatFusionForwardTests(unittest.TestCase):
    def test_minimal_forward_embedding_shapes(self) -> None:
        model = PepCLIPConcatFusionModel(
            model_1d=_Stub1DTower(),
            model_3d=_Stub3DTower(),
            concat_dim=5,
            hidden_dim=8,
            output_dim=4,
            dropout=0.0,
            temperature=0.1,
        )
        batch = {
            "one_d": {
                "receptor_tokens": torch.ones((2, 3), dtype=torch.long),
                "peptide_tokens": torch.ones((2, 2), dtype=torch.long),
                "receptor_sequence": ["AAA", "BBB"],
                "peptide_sequence": ["AA", "BB"],
            },
            "three_d": {
                "receptor_coords": torch.zeros((2, 1, 3)),
                "receptor_elements": torch.ones((2, 1), dtype=torch.long),
                "receptor_mask": torch.ones((2, 1), dtype=torch.bool),
                "receptor_atom_names": torch.ones((2, 1), dtype=torch.long),
                "receptor_residue_names": torch.ones((2, 1), dtype=torch.long),
                "peptide_coords": torch.zeros((2, 1, 3)),
                "peptide_elements": torch.ones((2, 1), dtype=torch.long),
                "peptide_mask": torch.ones((2, 1), dtype=torch.bool),
                "peptide_atom_names": torch.ones((2, 1), dtype=torch.long),
                "peptide_residue_names": torch.ones((2, 1), dtype=torch.long),
            },
        }
        output = model(batch)
        self.assertEqual(tuple(output["receptor_emb"].shape), (2, 4))
        self.assertEqual(tuple(output["peptide_emb"].shape), (2, 4))
        self.assertEqual(tuple(output["logits_per_receptor"].shape), (2, 2))
        self.assertEqual(tuple(output["logits_per_peptide"].shape), (2, 2))
        self.assertTrue(torch.isfinite(output["logits_per_receptor"]).all())


if __name__ == "__main__":
    unittest.main()

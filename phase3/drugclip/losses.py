"""Numerically safe, diagonal-target known-positive contrastive loss."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def build_known_positive_masks(
    receptor_interface_ids: list[str],
    peptide_sequences: list[str],
    known_positive_groups: list[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask only off-diagonal, exact known positives from negative competition.

    ``receptor_peptides`` contains peptide sequences and is used for
    receptor-to-peptide rows.  ``peptide_receptors`` contains formal
    ``iface:...`` receptor-interface IDs and is used for peptide-to-receptor
    rows.  The diagonal remains the sole positive target in both directions.
    """
    batch_size = len(receptor_interface_ids)
    if not (
        len(peptide_sequences) == len(known_positive_groups) == batch_size
    ):
        raise ValueError("known-positive metadata length must match batch size")

    receptor_to_peptide = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
    peptide_to_receptor = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
    for row_index, group in enumerate(known_positive_groups):
        receptor_peptides = {str(item) for item in group.get("receptor_peptides", [])}
        peptide_receptors = {str(item) for item in group.get("peptide_receptors", [])}
        for candidate_index in range(batch_size):
            if candidate_index == row_index:
                continue
            if peptide_sequences[candidate_index] in receptor_peptides:
                receptor_to_peptide[row_index, candidate_index] = True
            if receptor_interface_ids[candidate_index] in peptide_receptors:
                peptide_to_receptor[row_index, candidate_index] = True

    diagonal = torch.eye(batch_size, dtype=torch.bool, device=device)
    if (receptor_to_peptide & diagonal).any() or (peptide_to_receptor & diagonal).any():
        raise RuntimeError("known-positive mask must never remove a diagonal target")
    return receptor_to_peptide, peptide_to_receptor


def masked_bidirectional_known_positive_loss(
    logits_receptor_to_peptide: torch.Tensor,
    logits_peptide_to_receptor: torch.Tensor,
    receptor_interface_ids: list[str],
    peptide_sequences: list[str],
    known_positive_groups: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Return sum of directional CE losses and their exact negative masks."""
    if logits_receptor_to_peptide.ndim != 2 or logits_receptor_to_peptide.shape[0] != logits_receptor_to_peptide.shape[1]:
        raise ValueError("receptor-to-peptide logits must be a square matrix")
    if logits_peptide_to_receptor.shape != logits_receptor_to_peptide.shape:
        raise ValueError("directional logits must have identical square shapes")
    if not torch.isfinite(logits_receptor_to_peptide).all() or not torch.isfinite(logits_peptide_to_receptor).all():
        raise ValueError("unmasked logits must be finite")

    batch_size = logits_receptor_to_peptide.shape[0]
    receptor_to_peptide_mask, peptide_to_receptor_mask = build_known_positive_masks(
        receptor_interface_ids,
        peptide_sequences,
        known_positive_groups,
        logits_receptor_to_peptide.device,
    )
    target = torch.arange(batch_size, device=logits_receptor_to_peptide.device)
    # -inf removes a candidate from logsumexp / softmax normalization exactly.
    masked_r2p = logits_receptor_to_peptide.float().masked_fill(receptor_to_peptide_mask, -torch.inf)
    masked_p2r = logits_peptide_to_receptor.float().masked_fill(peptide_to_receptor_mask, -torch.inf)
    loss_r2p = F.cross_entropy(masked_r2p, target)
    loss_p2r = F.cross_entropy(masked_p2r, target)
    loss_total = loss_r2p + loss_p2r
    if not torch.isfinite(loss_total):
        raise FloatingPointError("known-positive contrastive loss is non-finite")
    return {
        "loss_receptor_to_peptide": loss_r2p,
        "loss_peptide_to_receptor": loss_p2r,
        "loss_total": loss_total,
        "receptor_to_peptide_mask": receptor_to_peptide_mask,
        "peptide_to_receptor_mask": peptide_to_receptor_mask,
    }

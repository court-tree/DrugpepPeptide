from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F


def duplicate_mask(keys: List[str], device: torch.device) -> torch.Tensor:
    n = len(keys)
    mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    for i, left in enumerate(keys):
        for j, right in enumerate(keys):
            if i != j and left and right and left == right:
                mask[i, j] = True
    return mask


def grouped_duplicate_mask(key_groups: Sequence[Sequence[str]], device: torch.device) -> torch.Tensor:
    if not key_groups:
        return torch.zeros((0, 0), dtype=torch.bool, device=device)
    n = len(key_groups[0])
    mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    for keys in key_groups:
        if len(keys) != n:
            raise ValueError("All duplicate-mask key groups must have batch length")
        mask |= duplicate_mask(list(keys), device)
    return mask


def symmetric_in_batch_softmax_loss(
    logits_per_receptor: torch.Tensor,
    logits_per_peptide: torch.Tensor,
    receptor_keys: List[str] | None = None,
    peptide_keys: List[str] | None = None,
    receptor_key_groups: Sequence[Sequence[str]] | None = None,
    peptide_key_groups: Sequence[Sequence[str]] | None = None,
) -> torch.Tensor:
    n = logits_per_receptor.size(0)
    target = torch.arange(n, device=logits_per_receptor.device)

    receptor_groups = []
    peptide_groups = []
    if receptor_keys is not None:
        receptor_groups.append(receptor_keys)
    if receptor_key_groups:
        receptor_groups.extend(receptor_key_groups)
    if peptide_keys is not None:
        peptide_groups.append(peptide_keys)
    if peptide_key_groups:
        peptide_groups.extend(peptide_key_groups)

    if receptor_groups:
        logits_per_peptide = logits_per_peptide.masked_fill(
            grouped_duplicate_mask(receptor_groups, logits_per_peptide.device),
            -1e6,
        )
    if peptide_groups:
        logits_per_receptor = logits_per_receptor.masked_fill(
            grouped_duplicate_mask(peptide_groups, logits_per_receptor.device),
            -1e6,
        )

    receptor_loss = F.cross_entropy(logits_per_receptor.float(), target)
    peptide_loss = F.cross_entropy(logits_per_peptide.float(), target)
    return 0.5 * receptor_loss + 0.5 * peptide_loss


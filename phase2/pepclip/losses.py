from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def duplicate_mask(keys: List[str], device: torch.device) -> torch.Tensor:
    n = len(keys)
    mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    for i, left in enumerate(keys):
        for j, right in enumerate(keys):
            if i != j and left == right:
                mask[i, j] = True
    return mask


def symmetric_in_batch_softmax_loss(
    logits_per_receptor: torch.Tensor,
    logits_per_peptide: torch.Tensor,
    receptor_keys: List[str] | None = None,
    peptide_keys: List[str] | None = None,
) -> torch.Tensor:
    n = logits_per_receptor.size(0)
    target = torch.arange(n, device=logits_per_receptor.device)

    if receptor_keys is not None:
        logits_per_peptide = logits_per_peptide.masked_fill(
            duplicate_mask(receptor_keys, logits_per_peptide.device),
            -1e6,
        )
    if peptide_keys is not None:
        logits_per_receptor = logits_per_receptor.masked_fill(
            duplicate_mask(peptide_keys, logits_per_receptor.device),
            -1e6,
        )

    receptor_loss = F.cross_entropy(logits_per_receptor.float(), target)
    peptide_loss = F.cross_entropy(logits_per_peptide.float(), target)
    return 0.5 * receptor_loss + 0.5 * peptide_loss


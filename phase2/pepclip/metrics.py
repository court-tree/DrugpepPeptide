from __future__ import annotations

from typing import Dict, Iterable

import torch


@torch.no_grad()
def retrieval_metrics(logits: torch.Tensor, ks: Iterable[int] = (1, 5, 10)) -> Dict[str, float]:
    n = logits.size(0)
    target = torch.arange(n, device=logits.device)
    order = logits.argsort(dim=1, descending=True)
    ranks = (order == target[:, None]).nonzero()[:, 1] + 1
    out = {
        "median_rank": float(ranks.float().median().item()),
        "mean_rank": float(ranks.float().mean().item()),
        "mrr": float((1.0 / ranks.float()).mean().item()),
    }
    for k in ks:
        capped_k = min(k, n)
        out[f"recall_at_{k}"] = float((ranks <= capped_k).float().mean().item())
    return out


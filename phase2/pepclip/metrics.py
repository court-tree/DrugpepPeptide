from __future__ import annotations

from typing import Dict, Iterable

import torch


@torch.no_grad()
def retrieval_metrics(logits: torch.Tensor, ks: Iterable[int] = (1, 5, 10)) -> Dict[str, float]:
    n = logits.size(0)
    target = torch.arange(n, device=logits.device)
    target_scores = logits[target, target].unsqueeze(1)
    ranks = (logits > target_scores).sum(dim=1) + 1
    return rank_metrics(ranks, n, ks)


@torch.no_grad()
def retrieval_metrics_from_embeddings(
    query_emb: torch.Tensor,
    target_emb: torch.Tensor,
    ks: Iterable[int] = (1, 5, 10),
    chunk_size: int = 1024,
) -> Dict[str, float]:
    if query_emb.size(0) != target_emb.size(0):
        raise ValueError("query_emb and target_emb must have the same number of rows")
    n = query_emb.size(0)
    ranks = []
    target_scores = (query_emb * target_emb).sum(dim=1)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        scores = query_emb[start:end] @ target_emb.t()
        cur_target_scores = target_scores[start:end].unsqueeze(1)
        ranks.append((scores > cur_target_scores).sum(dim=1) + 1)
    ranks_tensor = torch.cat(ranks, dim=0)
    return rank_metrics(ranks_tensor, n, ks)


def rank_metrics(ranks: torch.Tensor, n: int, ks: Iterable[int]) -> Dict[str, float]:
    out = {
        "median_rank": float(ranks.float().median().item()),
        "mean_rank": float(ranks.float().mean().item()),
        "mrr": float((1.0 / ranks.float()).mean().item()),
    }
    for k in ks:
        capped_k = min(k, n)
        out[f"recall_at_{k}"] = float((ranks <= capped_k).float().mean().item())
    return out

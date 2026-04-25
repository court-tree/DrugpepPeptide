from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from .data import PAD_TOKEN


class MeanPoolSequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.project = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(PAD_TOKEN).unsqueeze(-1)
        embedded = self.embedding(tokens)
        summed = (embedded * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp_min(1)
        pooled = summed / lengths
        return F.normalize(self.project(pooled), dim=-1)


class PepCLIPModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
        init_temperature: float = 1.0 / 14.0,
    ) -> None:
        super().__init__()
        self.receptor_encoder = MeanPoolSequenceEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.peptide_encoder = MeanPoolSequenceEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    def encode_receptor(self, receptor_tokens: torch.Tensor) -> torch.Tensor:
        return self.receptor_encoder(receptor_tokens)

    def encode_peptide(self, peptide_tokens: torch.Tensor) -> torch.Tensor:
        return self.peptide_encoder(peptide_tokens)

    def forward(self, receptor_tokens: torch.Tensor, peptide_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        receptor_emb = self.encode_receptor(receptor_tokens)
        peptide_emb = self.encode_peptide(peptide_tokens)
        logits_per_receptor = receptor_emb @ peptide_emb.t()
        logits_per_receptor = logits_per_receptor * self.logit_scale.exp().clamp(max=100.0)
        return {
            "receptor_emb": receptor_emb,
            "peptide_emb": peptide_emb,
            "logits_per_receptor": logits_per_receptor,
            "logits_per_peptide": logits_per_receptor.t(),
            "logit_scale": self.logit_scale.exp().detach(),
        }


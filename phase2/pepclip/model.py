from __future__ import annotations

import math
from typing import Dict, List

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


class HFProteinEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int = 128,
        freeze_backbone: bool = True,
        max_length: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HFProteinEncoder requires transformers. Install it or use --encoder_type mean_pool."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        self.max_length = max_length
        self.freeze_backbone = freeze_backbone

        hidden_size = int(self.backbone.config.hidden_size)
        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, sequences: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        special_tokens_mask = encoded.pop("special_tokens_mask", None)
        attention_mask = encoded["attention_mask"].bool()
        if special_tokens_mask is not None:
            attention_mask = attention_mask & special_tokens_mask.eq(0)

        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone(**encoded)
        else:
            outputs = self.backbone(**encoded)

        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(self.project(pooled), dim=-1)


class PepCLIPModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_type: str = "mean_pool",
        hf_model_name_or_path: str | None = None,
        freeze_hf_backbone: bool = True,
        hf_max_length: int = 512,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
        init_temperature: float = 1.0 / 14.0,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "mean_pool":
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
        elif encoder_type == "hf_esm":
            if not hf_model_name_or_path:
                raise ValueError("--hf_model_name_or_path is required when encoder_type='hf_esm'")
            self.receptor_encoder = HFProteinEncoder(
                model_name_or_path=hf_model_name_or_path,
                output_dim=output_dim,
                freeze_backbone=freeze_hf_backbone,
                max_length=hf_max_length,
                dropout=dropout,
            )
            self.peptide_encoder = HFProteinEncoder(
                model_name_or_path=hf_model_name_or_path,
                output_dim=output_dim,
                freeze_backbone=freeze_hf_backbone,
                max_length=hf_max_length,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported encoder_type={encoder_type!r}")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    def encode_receptor(
        self,
        receptor_tokens: torch.Tensor | None = None,
        receptor_sequences: List[str] | None = None,
    ) -> torch.Tensor:
        if self.encoder_type == "hf_esm":
            if receptor_sequences is None:
                raise ValueError("receptor_sequences is required for hf_esm")
            return self.receptor_encoder(receptor_sequences)
        if receptor_tokens is None:
            raise ValueError("receptor_tokens is required for mean_pool")
        return self.receptor_encoder(receptor_tokens)

    def encode_peptide(
        self,
        peptide_tokens: torch.Tensor | None = None,
        peptide_sequences: List[str] | None = None,
    ) -> torch.Tensor:
        if self.encoder_type == "hf_esm":
            if peptide_sequences is None:
                raise ValueError("peptide_sequences is required for hf_esm")
            return self.peptide_encoder(peptide_sequences)
        if peptide_tokens is None:
            raise ValueError("peptide_tokens is required for mean_pool")
        return self.peptide_encoder(peptide_tokens)

    def forward(
        self,
        receptor_tokens: torch.Tensor | None = None,
        peptide_tokens: torch.Tensor | None = None,
        receptor_sequences: List[str] | None = None,
        peptide_sequences: List[str] | None = None,
    ) -> Dict[str, torch.Tensor]:
        receptor_emb = self.encode_receptor(receptor_tokens, receptor_sequences)
        peptide_emb = self.encode_peptide(peptide_tokens, peptide_sequences)
        logits_per_receptor = receptor_emb @ peptide_emb.t()
        logits_per_receptor = logits_per_receptor * self.logit_scale.exp().clamp(max=100.0)
        return {
            "receptor_emb": receptor_emb,
            "peptide_emb": peptide_emb,
            "logits_per_receptor": logits_per_receptor,
            "logits_per_peptide": logits_per_receptor.t(),
            "logit_scale": self.logit_scale.exp().detach(),
        }

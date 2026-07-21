from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from .data import ATOM_NAME_PAD_TOKEN, ELEMENT_PAD_TOKEN, RESIDUE_NAME_PAD_TOKEN


class RadialAtomCloudEncoder(nn.Module):
    """Small rotation/translation-invariant atom-cloud encoder for 3D smoke baselines."""

    def __init__(
        self,
        num_elements: int,
        element_dim: int = 32,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
        coord_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.coord_scale = coord_scale
        self.element_embedding = nn.Embedding(num_elements, element_dim, padding_idx=ELEMENT_PAD_TOKEN)
        self.atom_mlp = nn.Sequential(
            nn.Linear(element_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        coords: torch.Tensor,
        elements: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()
        lengths = mask_f.sum(dim=1).clamp_min(1.0)
        center = (coords * mask_f).sum(dim=1, keepdim=True) / lengths.unsqueeze(1)
        centered = (coords - center) / self.coord_scale
        radius = centered.norm(dim=-1, keepdim=True)
        features = torch.cat([self.element_embedding(elements), radius, radius.square(), mask_f], dim=-1)
        atom_hidden = self.atom_mlp(features) * mask_f
        mean_pool = atom_hidden.sum(dim=1) / lengths
        max_pool = atom_hidden.masked_fill(~mask.unsqueeze(-1), -1.0e4).max(dim=1).values
        atom_count = torch.log1p(lengths.squeeze(-1)).unsqueeze(-1) / 10.0
        pooled = torch.cat([mean_pool, max_pool, atom_count], dim=-1)
        return F.normalize(self.project(pooled), dim=-1)


class DistanceRBF(nn.Module):
    def __init__(
        self,
        num_kernels: int = 32,
        cutoff: float = 20.0,
    ) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_kernels)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / max(float(centers[1] - centers[0]) ** 2, 1.0e-6)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff.square())


class UniMolStyleAtomEncoder(nn.Module):
    """Lightweight Uni-Mol-style atom transformer with distance-aware pair features."""

    def __init__(
        self,
        num_elements: int,
        num_atom_names: int,
        num_residue_names: int,
        element_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        output_dim: int = 128,
        dropout: float = 0.1,
        num_rbf: int = 32,
        distance_cutoff: float = 20.0,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, element_dim))
        self.element_embedding = nn.Embedding(num_elements, element_dim, padding_idx=ELEMENT_PAD_TOKEN)
        self.atom_name_embedding = nn.Embedding(num_atom_names, element_dim, padding_idx=ATOM_NAME_PAD_TOKEN)
        self.residue_name_embedding = nn.Embedding(
            num_residue_names,
            element_dim,
            padding_idx=RESIDUE_NAME_PAD_TOKEN,
        )
        self.input_norm = nn.LayerNorm(element_dim)
        self.rbf = DistanceRBF(num_kernels=num_rbf, cutoff=distance_cutoff)
        self.distance_project = nn.Linear(num_rbf, element_dim)
        self.attn_bias_project = nn.Linear(num_rbf, num_heads)
        self.layers = nn.ModuleList(
            [DistanceBiasedTransformerLayer(element_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(element_dim)
        self.project = nn.Sequential(
            nn.Linear(element_dim * 2, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, output_dim),
        )

    def forward(
        self,
        coords: torch.Tensor,
        elements: torch.Tensor,
        atom_names: torch.Tensor | None,
        residue_names: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = coords.size(0)
        atom_hidden = self.element_embedding(elements)
        if atom_names is not None:
            atom_hidden = atom_hidden + self.atom_name_embedding(atom_names)
        if residue_names is not None:
            atom_hidden = atom_hidden + self.residue_name_embedding(residue_names)
        atom_hidden = self.input_norm(atom_hidden)

        mask_f = mask.unsqueeze(-1).float()
        lengths = mask_f.sum(dim=1).clamp_min(1.0)
        center = (coords * mask_f).sum(dim=1, keepdim=True) / lengths.unsqueeze(1)
        centered = coords - center

        distances = torch.cdist(centered, centered)
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        rbf = self.rbf(distances)
        distance_context = self.distance_project(rbf)
        distance_context = (distance_context * pair_mask.unsqueeze(-1).float()).sum(dim=2)
        distance_context = distance_context / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        atom_hidden = atom_hidden + distance_context

        cls = self.cls_token.expand(batch_size, -1, -1)
        hidden = torch.cat([cls, atom_hidden], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.bool, device=mask.device),
                ~mask,
            ],
            dim=1,
        )
        attn_bias_atoms = self.attn_bias_project(rbf).permute(0, 3, 1, 2)
        attn_bias = torch.zeros(
            (batch_size, attn_bias_atoms.size(1), hidden.size(1), hidden.size(1)),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        attn_bias[:, :, 1:, 1:] = attn_bias_atoms
        invalid_pair = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=mask.device),
                mask,
            ],
            dim=1,
        )
        valid_pair = invalid_pair.unsqueeze(1) & invalid_pair.unsqueeze(2)
        attn_bias = attn_bias.masked_fill(~valid_pair.unsqueeze(1), -1.0e4)

        for layer in self.layers:
            hidden = layer(hidden, padding_mask=padding_mask, attn_bias=attn_bias)
        hidden = self.final_norm(hidden)
        cls_hidden = hidden[:, 0]
        atom_output = hidden[:, 1:] * mask_f
        mean_hidden = atom_output.sum(dim=1) / lengths
        pooled = torch.cat([cls_hidden, mean_hidden], dim=-1)
        return F.normalize(self.project(pooled), dim=-1)


class DistanceBiasedTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        attn_mask = attn_bias.reshape(batch_size * self.num_heads, seq_len, seq_len)
        normed = self.norm1(hidden)
        attended, _ = self.attn(
            normed,
            normed,
            normed,
            key_padding_mask=padding_mask.float().masked_fill(padding_mask, -1.0e4),
            attn_mask=attn_mask,
            need_weights=False,
        )
        hidden = hidden + self.dropout1(attended)
        hidden = hidden + self.dropout2(self.ffn(self.norm2(hidden)))
        return hidden


class EGNNLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float,
        coord_update_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.coord_update_scale = coord_update_scale
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_dist: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, hidden_dim = h.shape
        gather_index = edge_index.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
        h_j = torch.gather(h.unsqueeze(1).expand(-1, num_nodes, -1, -1), 2, gather_index)
        h_i = h.unsqueeze(2).expand_as(h_j)
        edge_input = torch.cat([h_i, h_j, edge_rbf, edge_dist.unsqueeze(-1)], dim=-1)
        messages = self.edge_mlp(edge_input) * edge_mask.unsqueeze(-1).float()
        denom = edge_mask.sum(dim=2, keepdim=True).clamp_min(1).float()
        aggregated = messages.sum(dim=2) / denom
        h = self.norm(h + self.node_mlp(torch.cat([h, aggregated], dim=-1)))

        gather_coord_index = edge_index.unsqueeze(-1).expand(-1, -1, -1, 3)
        coord_j = torch.gather(coords.unsqueeze(1).expand(-1, num_nodes, -1, -1), 2, gather_coord_index)
        coord_i = coords.unsqueeze(2)
        coord_diff = coord_i - coord_j
        coord_weight = self.coord_mlp(messages) * edge_mask.unsqueeze(-1).float()
        coord_delta = (coord_diff * coord_weight).sum(dim=2) / denom
        coords = coords + self.coord_update_scale * coord_delta
        return h, coords


class EGNNAtomEncoder(nn.Module):
    def __init__(
        self,
        num_elements: int,
        num_atom_names: int,
        num_residue_names: int,
        element_dim: int = 128,
        hidden_dim: int = 512,
        output_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 4,
        num_rbf: int = 32,
        distance_cutoff: float = 20.0,
        num_neighbors: int = 32,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.element_embedding = nn.Embedding(num_elements, element_dim, padding_idx=ELEMENT_PAD_TOKEN)
        self.atom_name_embedding = nn.Embedding(num_atom_names, element_dim, padding_idx=ATOM_NAME_PAD_TOKEN)
        self.residue_name_embedding = nn.Embedding(
            num_residue_names,
            element_dim,
            padding_idx=RESIDUE_NAME_PAD_TOKEN,
        )
        self.input_project = nn.Sequential(
            nn.LayerNorm(element_dim),
            nn.Linear(element_dim, hidden_dim),
            nn.SiLU(),
        )
        self.rbf = DistanceRBF(num_kernels=num_rbf, cutoff=distance_cutoff)
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim=hidden_dim, edge_dim=num_rbf, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.project = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def build_edges(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = coords.shape
        k = min(self.num_neighbors, max(num_nodes - 1, 1))
        distances = torch.cdist(coords, coords)
        valid_pair = mask.unsqueeze(1) & mask.unsqueeze(2)
        eye = torch.eye(num_nodes, dtype=torch.bool, device=coords.device).unsqueeze(0)
        valid_pair = valid_pair & ~eye
        masked_distances = distances.masked_fill(~valid_pair, 1.0e6)
        edge_dist, edge_index = masked_distances.topk(k=k, dim=-1, largest=False)
        edge_mask = edge_dist < 1.0e5
        edge_rbf = self.rbf(edge_dist.clamp_max(1.0e3))
        return edge_index, edge_rbf, edge_dist, edge_mask

    def forward(
        self,
        coords: torch.Tensor,
        elements: torch.Tensor,
        atom_names: torch.Tensor | None,
        residue_names: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()
        lengths = mask_f.sum(dim=1).clamp_min(1.0)
        center = (coords * mask_f).sum(dim=1, keepdim=True) / lengths.unsqueeze(1)
        coords = coords - center

        h = self.element_embedding(elements)
        if atom_names is not None:
            h = h + self.atom_name_embedding(atom_names)
        if residue_names is not None:
            h = h + self.residue_name_embedding(residue_names)
        h = self.input_project(h) * mask_f

        edge_index, edge_rbf, edge_dist, edge_mask = self.build_edges(coords, mask)
        for layer in self.layers:
            h, coords = layer(h, coords, edge_index, edge_rbf, edge_dist, edge_mask)
            h = h * mask_f

        h = self.final_norm(h) * mask_f
        mean_pool = h.sum(dim=1) / lengths
        max_pool = h.masked_fill(~mask.unsqueeze(-1), -1.0e4).max(dim=1).values
        atom_count = torch.log1p(lengths.squeeze(-1)).unsqueeze(-1) / 10.0
        pooled = torch.cat([mean_pool, max_pool, atom_count], dim=-1)
        return F.normalize(self.project(pooled), dim=-1)


class PepCLIP3DModel(nn.Module):
    def __init__(
        self,
        num_elements: int,
        num_atom_names: int | None = None,
        num_residue_names: int | None = None,
        encoder_type: str = "radial",
        element_dim: int = 32,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
        coord_scale: float = 10.0,
        num_layers: int = 4,
        num_heads: int = 8,
        num_rbf: int = 32,
        distance_cutoff: float = 20.0,
        num_neighbors: int = 32,
        init_temperature: float = 1.0 / 14.0,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "radial":
            self.receptor_encoder = RadialAtomCloudEncoder(
                num_elements=num_elements,
                element_dim=element_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                coord_scale=coord_scale,
            )
            self.peptide_encoder = RadialAtomCloudEncoder(
                num_elements=num_elements,
                element_dim=element_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                coord_scale=coord_scale,
            )
        elif encoder_type == "unimol_style":
            if num_atom_names is None or num_residue_names is None:
                raise ValueError("num_atom_names and num_residue_names are required for unimol_style")
            self.receptor_encoder = UniMolStyleAtomEncoder(
                num_elements=num_elements,
                num_atom_names=num_atom_names,
                num_residue_names=num_residue_names,
                element_dim=element_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ffn_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                num_rbf=num_rbf,
                distance_cutoff=distance_cutoff,
            )
            self.peptide_encoder = UniMolStyleAtomEncoder(
                num_elements=num_elements,
                num_atom_names=num_atom_names,
                num_residue_names=num_residue_names,
                element_dim=element_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ffn_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                num_rbf=num_rbf,
                distance_cutoff=distance_cutoff,
            )
        elif encoder_type == "egnn":
            if num_atom_names is None or num_residue_names is None:
                raise ValueError("num_atom_names and num_residue_names are required for egnn")
            self.receptor_encoder = EGNNAtomEncoder(
                num_elements=num_elements,
                num_atom_names=num_atom_names,
                num_residue_names=num_residue_names,
                element_dim=element_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                num_layers=num_layers,
                num_rbf=num_rbf,
                distance_cutoff=distance_cutoff,
                num_neighbors=num_neighbors,
            )
            self.peptide_encoder = EGNNAtomEncoder(
                num_elements=num_elements,
                num_atom_names=num_atom_names,
                num_residue_names=num_residue_names,
                element_dim=element_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                num_layers=num_layers,
                num_rbf=num_rbf,
                distance_cutoff=distance_cutoff,
                num_neighbors=num_neighbors,
            )
        else:
            raise ValueError(f"Unsupported 3D encoder_type={encoder_type!r}")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    def encode_receptor(
        self,
        receptor_coords: torch.Tensor,
        receptor_elements: torch.Tensor,
        receptor_mask: torch.Tensor,
        receptor_atom_names: torch.Tensor | None = None,
        receptor_residue_names: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.encoder_type in {"unimol_style", "egnn"}:
            return self.receptor_encoder(
                receptor_coords,
                receptor_elements,
                receptor_atom_names,
                receptor_residue_names,
                receptor_mask,
            )
        return self.receptor_encoder(receptor_coords, receptor_elements, receptor_mask)

    def encode_peptide(
        self,
        peptide_coords: torch.Tensor,
        peptide_elements: torch.Tensor,
        peptide_mask: torch.Tensor,
        peptide_atom_names: torch.Tensor | None = None,
        peptide_residue_names: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.encoder_type in {"unimol_style", "egnn"}:
            return self.peptide_encoder(
                peptide_coords,
                peptide_elements,
                peptide_atom_names,
                peptide_residue_names,
                peptide_mask,
            )
        return self.peptide_encoder(peptide_coords, peptide_elements, peptide_mask)

    def forward(
        self,
        receptor_coords: torch.Tensor,
        receptor_elements: torch.Tensor,
        receptor_mask: torch.Tensor,
        peptide_coords: torch.Tensor,
        peptide_elements: torch.Tensor,
        peptide_mask: torch.Tensor,
        receptor_atom_names: torch.Tensor | None = None,
        receptor_residue_names: torch.Tensor | None = None,
        peptide_atom_names: torch.Tensor | None = None,
        peptide_residue_names: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        receptor_emb = self.encode_receptor(
            receptor_coords,
            receptor_elements,
            receptor_mask,
            receptor_atom_names,
            receptor_residue_names,
        )
        peptide_emb = self.encode_peptide(
            peptide_coords,
            peptide_elements,
            peptide_mask,
            peptide_atom_names,
            peptide_residue_names,
        )
        logits_per_receptor = receptor_emb @ peptide_emb.t()
        logits_per_receptor = logits_per_receptor * self.logit_scale.exp().clamp(max=100.0)
        return {
            "receptor_emb": receptor_emb,
            "peptide_emb": peptide_emb,
            "logits_per_receptor": logits_per_receptor,
            "logits_per_peptide": logits_per_receptor.t(),
            "logit_scale": self.logit_scale.exp().detach(),
        }

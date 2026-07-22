"""Formal Phase-2 fusion forward adapter for Phase-3 batches."""

from __future__ import annotations

from typing import Any

import torch

from phase2.pepclip.train_concat_fusion import (
    PepCLIPConcatFusionModel,
    move_1d_batch,
    move_3d_batch,
)
from phase3.drugclip.losses import masked_bidirectional_known_positive_loss


def forward_phase2_fusion_batch(
    model: PepCLIPConcatFusionModel,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run unchanged Phase-2 towers/fusion and attach formal Phase-3 metadata."""
    one_d = batch["one_d"]
    peptide_sequences = [str(item) for item in one_d["peptide_sequence"]]
    if len(peptide_sequences) != len(set(peptide_sequences)):
        raise ValueError("Phase-3 fusion batch contains duplicate peptide_sequence")
    moved = {
        "one_d": move_1d_batch(one_d, device),
        "three_d": move_3d_batch(batch["three_d"], device),
    }
    outputs = model(moved)
    receptor_embedding = outputs["receptor_emb"]
    peptide_embedding = outputs["peptide_emb"]
    similarity_matrix = outputs["logits_per_receptor"]
    batch_size = len(peptide_sequences)
    if receptor_embedding.shape != (batch_size, 512) or peptide_embedding.shape != (batch_size, 512):
        raise ValueError("formal Phase-2 fusion checkpoint must emit 512-dimensional embeddings")
    if similarity_matrix.shape != (batch_size, batch_size):
        raise ValueError("Phase-2 similarity matrix must be B x B")
    if not all(torch.isfinite(value).all() for value in (receptor_embedding, peptide_embedding, similarity_matrix)):
        raise FloatingPointError("Phase-2 fusion forward emitted non-finite values")
    return {
        "receptor_embedding": receptor_embedding,
        "peptide_embedding": peptide_embedding,
        "similarity_matrix": similarity_matrix,
        "similarity_matrix_transpose": outputs["logits_per_peptide"],
        "temperature": model.temperature,
        "batch_biological_pair_id": list(one_d["biological_pair_id"]),
        "batch_interface_pair_id": list(one_d["interface_pair_id"]),
        "batch_receptor_interface_id": list(one_d["receptor_interface_id"]),
        "batch_peptide_sequence": peptide_sequences,
        "known_positive_group": list(one_d["known_positive_group"]),
        "moved_batch": moved,
    }


def forward_and_known_positive_loss(
    model: PepCLIPConcatFusionModel,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Formal forward plus diagonal-target bidirectional known-positive loss."""
    result = forward_phase2_fusion_batch(model, batch, device)
    loss = masked_bidirectional_known_positive_loss(
        result["similarity_matrix"],
        result["similarity_matrix_transpose"],
        result["batch_receptor_interface_id"],
        result["batch_peptide_sequence"],
        result["known_positive_group"],
    )
    return {**result, **loss}

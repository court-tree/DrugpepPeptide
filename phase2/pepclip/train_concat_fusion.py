from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shlex
import sys
from typing import Any, Dict, Iterable, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import (
    AA_TO_ID,
    ATOM_NAME_TO_ID,
    ELEMENT_TO_ID,
    RESIDUE_NAME_TO_ID,
    PepCLIP3DDataset,
    PepCLIPDataset,
    collate_pepclip,
    collate_pepclip_3d,
)
from .losses import symmetric_in_batch_softmax_loss
from .metrics import rank_metrics
from .model import PepCLIPModel
from .model_3d import PepCLIP3DModel


REFERENCE_RESULTS = {
    "1D-only": {"recall_at_10": 0.47284},
    "3D-only": {"recall_at_10": 0.46704},
    "old concat": {"recall_at_10": 0.68244},
    "score fusion 0.40:0.60": {"recall_at_10": 0.69166},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen-tower 1D+3D concat fusion head for PepCLIP."
    )
    parser.add_argument("--train_track_a", required=True)
    parser.add_argument("--valid_track_a", required=True)
    parser.add_argument("--train_track_b", required=True)
    parser.add_argument("--valid_track_b", required=True)
    parser.add_argument("--data_format", choices=["jsonl", "lmdb"], default="lmdb")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--checkpoint_1d", required=True)
    parser.add_argument("--checkpoint_3d", required=True)
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help=(
            "Resume fusion-head training from checkpoint_best.pt or checkpoint_last.pt. "
            "--epochs is the total target epoch. The fusion heads, optimizer state, "
            "history, and best Recall@10 are restored."
        ),
    )
    parser.add_argument(
        "--reset_optimizer_on_resume",
        action="store_true",
        help="Restore fusion heads/history but initialize a fresh optimizer using --lr.",
    )
    parser.add_argument(
        "--config_1d",
        default=None,
        help="Optional 1D run_config.json. Defaults to checkpoint parent/run_config.json.",
    )
    parser.add_argument(
        "--config_3d",
        default=None,
        help="Optional 3D run_config.json. Defaults to checkpoint parent/run_config.json.",
    )
    parser.add_argument(
        "--hf_model_name_or_path_1d",
        default=None,
        help="Optional override for the 1D HF/ESM model path stored in the source run_config.json.",
    )
    parser.add_argument(
        "--allow_partial_tower_checkpoint_load",
        action="store_true",
        help=(
            "Allow source 1D/3D tower checkpoints to load with missing or unexpected keys. "
            "Use this only for audited compatibility runs where the source checkpoint predates "
            "the current fusion entry point."
        ),
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_chunk_size", type=int, default=512)
    parser.add_argument("--retrieval_topk", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help=(
            "Stop after this many consecutive epochs without validation improvement. "
            "Set 0 to disable early stopping."
        ),
    )
    parser.add_argument(
        "--early_stopping_metric",
        choices=["recall_at_10", "mrr", "recall_at_5", "recall_at_1"],
        default="recall_at_10",
        help="Validation metric used for best checkpointing and early stopping.",
    )
    parser.add_argument(
        "--tower_lr",
        type=float,
        default=1e-6,
        help="Learning rate for partially unfrozen 1D/3D tower parameters.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--fusion_hidden_dim", type=int, default=512)
    parser.add_argument("--fusion_output_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0 / 14.0)
    parser.add_argument(
        "--unfreeze_1d_last_n_layers",
        type=int,
        default=0,
        help=(
            "Unfreeze the last N ESM backbone blocks in both receptor and peptide 1D encoders. "
            "Their final normalization and projection heads are also unfrozen."
        ),
    )
    parser.add_argument(
        "--unfreeze_3d_last_n_layers",
        type=int,
        default=0,
        help=(
            "Unfreeze the last N EGNN/Uni-Mol-style blocks in both receptor and peptide "
            "3D encoders. Their final normalization and projection heads are also unfrozen."
        ),
    )
    parser.add_argument(
        "--disable_duplicate_mask",
        action="store_true",
        help="Do not mask duplicate receptor/peptide keys inside in-batch contrastive loss.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train_conformer_views", default=None)
    parser.add_argument("--conformer_evidence", default=None)
    parser.add_argument("--conformer_consistency_weight", type=float, default=0.1)
    parser.add_argument("--max_auxiliary_conformers_per_sample", type=int, default=1)
    return parser.parse_args()


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def default_config_path(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).resolve().parent / "run_config.json"


def checkpoint_has_tower_state(checkpoint_path: str | Path, device: torch.device) -> bool:
    state = torch.load(checkpoint_path, map_location=device)
    return bool(state.get("tower_state_dict"))


def write_run_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    command = shlex.join([sys.executable, "-B", "-m", "phase2.pepclip.train_concat_fusion", *sys.argv[1:]])
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "reference_results.json").write_text(
        json.dumps(REFERENCE_RESULTS, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def set_frozen(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def set_trainable(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def unfreeze_1d_last_layers(model: PepCLIPModel, num_layers: int) -> Dict[str, int]:
    if num_layers < 0:
        raise ValueError("--unfreeze_1d_last_n_layers must be non-negative")
    if num_layers == 0:
        return {"requested_layers": 0, "available_layers": 0, "trainable_parameters": 0}
    if model.encoder_type != "hf_esm":
        raise ValueError("Partial 1D unfreezing currently requires encoder_type='hf_esm'")

    available_layers: int | None = None
    for encoder in (model.receptor_encoder, model.peptide_encoder):
        backbone_encoder = getattr(encoder.backbone, "encoder", None)
        layers = getattr(backbone_encoder, "layer", None)
        if layers is None:
            raise ValueError("Could not locate ESM backbone.encoder.layer for partial unfreezing")
        if num_layers > len(layers):
            raise ValueError(
                f"Requested {num_layers} 1D layers, but the ESM backbone only has {len(layers)}"
            )
        available_layers = len(layers)
        encoder.freeze_backbone = False
        for layer in layers[-num_layers:]:
            set_trainable(layer)
        final_norm = getattr(backbone_encoder, "emb_layer_norm_after", None)
        if final_norm is not None:
            set_trainable(final_norm)
        set_trainable(encoder.project)

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "requested_layers": num_layers,
        "available_layers": int(available_layers or 0),
        "trainable_parameters": trainable,
    }


def unfreeze_3d_last_layers(model: PepCLIP3DModel, num_layers: int) -> Dict[str, int]:
    if num_layers < 0:
        raise ValueError("--unfreeze_3d_last_n_layers must be non-negative")
    if num_layers == 0:
        return {"requested_layers": 0, "available_layers": 0, "trainable_parameters": 0}
    if model.encoder_type not in {"egnn", "unimol_style"}:
        raise ValueError(
            "Partial 3D unfreezing currently requires encoder_type='egnn' or 'unimol_style'"
        )

    available_layers: int | None = None
    for encoder in (model.receptor_encoder, model.peptide_encoder):
        layers = getattr(encoder, "layers", None)
        if layers is None:
            raise ValueError("Could not locate 3D encoder.layers for partial unfreezing")
        if num_layers > len(layers):
            raise ValueError(
                f"Requested {num_layers} 3D layers, but the encoder only has {len(layers)}"
            )
        available_layers = len(layers)
        for layer in layers[-num_layers:]:
            set_trainable(layer)
        set_trainable(encoder.final_norm)
        set_trainable(encoder.project)

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "requested_layers": num_layers,
        "available_layers": int(available_layers or 0),
        "trainable_parameters": trainable,
    }


def load_model_state(
    module: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    label: str,
    allow_partial: bool = False,
) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    raw_state = state.get("model_state_dict", state)
    missing, unexpected = module.load_state_dict(raw_state, strict=False)
    print(
        json.dumps(
            {
                "event": f"loaded_{label}_checkpoint",
                "checkpoint": str(checkpoint_path),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
                "allow_partial": allow_partial,
                "missing_key_examples": list(missing[:20]),
                "unexpected_key_examples": list(unexpected[:20]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if (missing or unexpected) and not allow_partial:
        raise ValueError(
            f"{label} checkpoint did not load cleanly: missing={len(missing)} unexpected={len(unexpected)}"
        )


def build_1d_model(
    config: Dict[str, Any],
    checkpoint_path: str | Path,
    device: torch.device,
    allow_partial_checkpoint_load: bool = False,
) -> PepCLIPModel:
    model = PepCLIPModel(
        vocab_size=max(AA_TO_ID.values()) + 1,
        encoder_type=config.get("encoder_type", "mean_pool"),
        hf_model_name_or_path=config.get("hf_model_name_or_path"),
        freeze_hf_backbone=bool(config.get("freeze_hf_backbone", True)),
        hf_max_length=int(config.get("hf_max_length", 512)),
        embed_dim=int(config.get("embed_dim", 128)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        output_dim=int(config.get("output_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    load_model_state(model, checkpoint_path, device, "1d", allow_partial_checkpoint_load)
    set_frozen(model)
    return model


def build_3d_model(
    config: Dict[str, Any],
    checkpoint_path: str | Path,
    device: torch.device,
    allow_partial_checkpoint_load: bool = False,
) -> PepCLIP3DModel:
    model = PepCLIP3DModel(
        num_elements=max(ELEMENT_TO_ID.values()) + 1,
        num_atom_names=max(ATOM_NAME_TO_ID.values()) + 1,
        num_residue_names=max(RESIDUE_NAME_TO_ID.values()) + 1,
        encoder_type=config.get("encoder_type", "radial"),
        element_dim=int(config.get("element_dim", 32)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        output_dim=int(config.get("output_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
        coord_scale=float(config.get("coord_scale", 10.0)),
        num_layers=int(config.get("num_layers", 4)),
        num_heads=int(config.get("num_heads", 8)),
        num_rbf=int(config.get("num_rbf", 32)),
        distance_cutoff=float(config.get("distance_cutoff", 20.0)),
        num_neighbors=int(config.get("num_neighbors", 32)),
    ).to(device)
    load_model_state(model, checkpoint_path, device, "3d", allow_partial_checkpoint_load)
    set_frozen(model)
    return model


class PairedTrackABDataset(Dataset):
    def __init__(
        self,
        track_a_path: str | Path,
        track_b_path: str | Path,
        data_format: str,
        receptor_field: str = "receptor_patch_sequence",
        max_receptor_atoms: int | None = None,
        max_peptide_atoms: int | None = None,
        conformer_views_path: str | Path | None = None,
        conformer_evidence_path: str | Path | None = None,
        max_auxiliary_conformers_per_sample: int = 1,
    ) -> None:
        self.track_a = PepCLIPDataset(
            track_a_path,
            split=None,
            data_format=data_format,
            receptor_field=receptor_field,
        )
        self.track_b = PepCLIP3DDataset(
            track_b_path,
            split=None,
            data_format=data_format,
            max_receptor_atoms=max_receptor_atoms,
            max_peptide_atoms=max_peptide_atoms,
        )
        if bool(conformer_views_path) != bool(conformer_evidence_path):
            raise ValueError("conformer views and evidence must be provided together")
        if conformer_views_path:
            try:
                from .data import PepCLIP3DConformerDataset
            except ImportError as exc:
                raise ImportError(
                    "Conformer consistency training requires a data.py version that "
                    "provides PepCLIP3DConformerDataset."
                ) from exc
            self.track_b = PepCLIP3DConformerDataset(
                self.track_b,
                conformer_views_path,
                conformer_evidence_path,
                max_auxiliary_conformers_per_sample=max_auxiliary_conformers_per_sample,
            )
        if len(self.track_a) != len(self.track_b):
            raise ValueError(f"Track A/B row count mismatch: {len(self.track_a)} vs {len(self.track_b)}")
        self.alignment_errors = self._validate_alignment()

    def _validate_alignment(self) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        for index in range(len(self.track_a)):
            a_sample = str(self.track_a.rows[index].get("sample_id", self.track_a.rows[index].get("candidate_id", index)))
            b_sample = str(self.track_b.rows[index].get("sample_id", self.track_b.rows[index].get("candidate_id", index)))
            if a_sample != b_sample:
                errors.append({"index": index, "track_a_sample_id": a_sample, "track_b_sample_id": b_sample})
                if len(errors) >= 20:
                    break
        if errors:
            raise ValueError(f"Track A/B sample_id alignment failed. First errors: {errors[:3]}")
        return errors

    def __len__(self) -> int:
        return len(self.track_a)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item_1d = self.track_a[index]
        item_3d = self.track_b[index]
        if item_1d["sample_id"] != item_3d["sample_id"]:
            raise ValueError(
                f"sample_id mismatch at index={index}: {item_1d['sample_id']} vs {item_3d['sample_id']}"
            )
        return {"one_d": item_1d, "three_d": item_3d}


def collate_fusion(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    one_d = collate_pepclip([item["one_d"] for item in batch])
    three_d = collate_pepclip_3d([item["three_d"] for item in batch])
    if one_d["sample_id"] != three_d["sample_id"]:
        raise ValueError("Batch Track A/B sample_id order mismatch")
    return {"one_d": one_d, "three_d": three_d}


def move_1d_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    batch = dict(batch)
    batch["receptor_tokens"] = batch["receptor_tokens"].to(device)
    batch["peptide_tokens"] = batch["peptide_tokens"].to(device)
    return batch


def move_3d_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    batch = dict(batch)
    tensor_keys = [
        "receptor_coords",
        "receptor_elements",
        "receptor_atom_names",
        "receptor_residue_names",
        "receptor_mask",
        "peptide_coords",
        "peptide_elements",
        "peptide_atom_names",
        "peptide_residue_names",
        "peptide_mask",
        "view_primary_coords",
        "view_primary_elements",
        "view_primary_atom_names",
        "view_primary_residue_names",
        "view_primary_mask",
        "view_auxiliary_coords",
        "view_auxiliary_elements",
        "view_auxiliary_atom_names",
        "view_auxiliary_residue_names",
        "view_auxiliary_mask",
        "view_auxiliary_multi_coords",
        "view_auxiliary_multi_elements",
        "view_auxiliary_multi_atom_names",
        "view_auxiliary_multi_residue_names",
        "view_auxiliary_multi_mask",
        "view_auxiliary_owner_index",
        "view_auxiliary_sampled_count",
        "consistency_enabled",
    ]
    batch = dict(batch)
    for key in tensor_keys:
        if key in batch:
            batch[key] = batch[key].to(device)
    return batch


class FusionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class PepCLIPConcatFusionModel(nn.Module):
    def __init__(
        self,
        model_1d: PepCLIPModel,
        model_3d: PepCLIP3DModel,
        concat_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.model_1d = model_1d
        self.model_3d = model_3d
        self.receptor_fusion = FusionHead(concat_dim, hidden_dim, output_dim, dropout)
        self.peptide_fusion = FusionHead(concat_dim, hidden_dim, output_dim, dropout)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("--temperature must be positive")

    @staticmethod
    def _has_trainable_parameters(module: nn.Module) -> bool:
        return any(param.requires_grad for param in module.parameters())

    def encode_fused(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        one_d = batch["one_d"]
        three_d = batch["three_d"]
        enable_1d_grad = torch.is_grad_enabled() and self._has_trainable_parameters(self.model_1d)
        enable_3d_grad = torch.is_grad_enabled() and self._has_trainable_parameters(self.model_3d)
        with torch.set_grad_enabled(enable_1d_grad):
            receptor_1d = self.model_1d.encode_receptor(
                receptor_tokens=one_d["receptor_tokens"],
                receptor_sequences=one_d["receptor_sequence"],
            )
            peptide_1d = self.model_1d.encode_peptide(
                peptide_tokens=one_d["peptide_tokens"],
                peptide_sequences=one_d["peptide_sequence"],
            )
        with torch.set_grad_enabled(enable_3d_grad):
            receptor_3d = self.model_3d.encode_receptor(
                receptor_coords=three_d["receptor_coords"],
                receptor_elements=three_d["receptor_elements"],
                receptor_mask=three_d["receptor_mask"],
                receptor_atom_names=three_d["receptor_atom_names"],
                receptor_residue_names=three_d["receptor_residue_names"],
            )
            peptide_3d = self.model_3d.encode_peptide(
                peptide_coords=three_d["peptide_coords"],
                peptide_elements=three_d["peptide_elements"],
                peptide_mask=three_d["peptide_mask"],
                peptide_atom_names=three_d["peptide_atom_names"],
                peptide_residue_names=three_d["peptide_residue_names"],
            )
        receptor_concat = torch.cat([receptor_1d, receptor_3d], dim=-1)
        peptide_concat = torch.cat([peptide_1d, peptide_3d], dim=-1)
        receptor_fused = self.receptor_fusion(receptor_concat)
        peptide_fused = self.peptide_fusion(peptide_concat)
        logits = receptor_fused @ peptide_fused.t() / self.temperature
        return {
            "receptor_emb": receptor_fused,
            "peptide_emb": peptide_fused,
            "logits_per_receptor": logits,
            "logits_per_peptide": logits.t(),
        }

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.encode_fused(batch)


def logit_diagnostics(logits: torch.Tensor) -> Dict[str, float]:
    detached = logits.detach()
    eye = torch.eye(detached.size(0), dtype=torch.bool, device=detached.device)
    offdiag = detached[~eye]
    return {
        "logits_std": float(detached.std().item()),
        "diag_mean": float(detached.diag().mean().item()),
        "offdiag_mean": float(offdiag.mean().item()) if offdiag.numel() else 0.0,
    }


def average_diagnostics(rows: List[Dict[str, float]], weights: List[int]) -> Dict[str, float]:
    if not rows:
        return {"logits_std": 0.0, "diag_mean": 0.0, "offdiag_mean": 0.0}
    total = max(sum(weights), 1)
    return {
        key: float(sum(row[key] * weight for row, weight in zip(rows, weights)) / total)
        for key in ("logits_std", "diag_mean", "offdiag_mean")
    }


def run_epoch(
    model: PepCLIPConcatFusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    eval_chunk_size: int,
    disable_duplicate_mask: bool,
    conformer_consistency_weight: float = 0.0,
) -> Dict[str, float]:
    model.train(train)
    model.model_1d.eval()
    model.model_3d.eval()
    total_loss = 0.0
    total_items = 0
    receptor_embs: List[torch.Tensor] = []
    peptide_embs: List[torch.Tensor] = []
    diag_rows: List[Dict[str, float]] = []
    diag_weights: List[int] = []

    for batch in loader:
        batch = {
            "one_d": move_1d_batch(batch["one_d"], device),
            "three_d": move_3d_batch(batch["three_d"], device),
        }
        with torch.set_grad_enabled(train):
            outputs = model(batch)
            one_d = batch["one_d"]
            receptor_keys = None if disable_duplicate_mask else one_d["receptor_key"]
            peptide_keys = None if disable_duplicate_mask else one_d["peptide_key"]
            loss = symmetric_in_batch_softmax_loss(
                outputs["logits_per_receptor"],
                outputs["logits_per_peptide"],
                receptor_keys=receptor_keys,
                peptide_keys=peptide_keys,
                receptor_key_groups=None
                if disable_duplicate_mask
                else [one_d["receptor_interface_key"], one_d["receptor_family_30_id"]],
                peptide_key_groups=None
                if disable_duplicate_mask
                else [
                    one_d["peptide_sequence_id"],
                    one_d["peptide_homology_80_id"],
                    one_d["conformer_cluster_id"],
                ],
            )
            if conformer_consistency_weight > 0 and "consistency_enabled" in batch["three_d"]:
                enabled = batch["three_d"]["consistency_enabled"]
                if bool(enabled.any()):
                    with torch.no_grad():
                        peptide_1d = model.model_1d.encode_peptide(
                            peptide_tokens=one_d["peptide_tokens"],
                            peptide_sequences=one_d["peptide_sequence"],
                        )
                        primary_3d = model.model_3d.encode_peptide(
                            batch["three_d"]["view_primary_coords"],
                            batch["three_d"]["view_primary_elements"],
                            batch["three_d"]["view_primary_mask"],
                            batch["three_d"]["view_primary_atom_names"],
                            batch["three_d"]["view_primary_residue_names"],
                        )
                    primary_fused = model.peptide_fusion(torch.cat([peptide_1d, primary_3d], dim=-1))
                    if "view_auxiliary_multi_coords" in batch["three_d"]:
                        owner_index = batch["three_d"]["view_auxiliary_owner_index"]
                        with torch.no_grad():
                            auxiliary_3d = model.model_3d.encode_peptide(
                                batch["three_d"]["view_auxiliary_multi_coords"],
                                batch["three_d"]["view_auxiliary_multi_elements"],
                                batch["three_d"]["view_auxiliary_multi_mask"],
                                batch["three_d"]["view_auxiliary_multi_atom_names"],
                                batch["three_d"]["view_auxiliary_multi_residue_names"],
                            )
                        auxiliary_fused = model.peptide_fusion(torch.cat([peptide_1d[owner_index], auxiliary_3d], dim=-1))
                        consistency_loss = (
                            1.0 - (primary_fused[owner_index] * auxiliary_fused).sum(dim=-1)
                        ).mean()
                    else:
                        with torch.no_grad():
                            auxiliary_3d = model.model_3d.encode_peptide(
                                batch["three_d"]["view_auxiliary_coords"],
                                batch["three_d"]["view_auxiliary_elements"],
                                batch["three_d"]["view_auxiliary_mask"],
                                batch["three_d"]["view_auxiliary_atom_names"],
                                batch["three_d"]["view_auxiliary_residue_names"],
                            )
                        auxiliary_fused = model.peptide_fusion(torch.cat([peptide_1d, auxiliary_3d], dim=-1))
                        consistency_loss = (
                            1.0 - (primary_fused[enabled] * auxiliary_fused[enabled]).sum(dim=-1)
                        ).mean()
                    loss = loss + conformer_consistency_weight * consistency_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0,
                )
                optimizer.step()

        batch_size = len(batch["one_d"]["sample_id"])
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        diag_rows.append(logit_diagnostics(outputs["logits_per_receptor"]))
        diag_weights.append(batch_size)
        if not train:
            receptor_embs.append(outputs["receptor_emb"].detach().cpu())
            peptide_embs.append(outputs["peptide_emb"].detach().cpu())

    result: Dict[str, float] = {"loss": total_loss / max(total_items, 1)}
    result.update(average_diagnostics(diag_rows, diag_weights))
    if not train and receptor_embs:
        receptor_emb = torch.cat(receptor_embs, dim=0)
        peptide_emb = torch.cat(peptide_embs, dim=0)
        result.update(evaluate_embeddings(receptor_emb, peptide_emb, eval_chunk_size=eval_chunk_size)["metrics"])
    return result


@torch.no_grad()
def evaluate_embeddings(
    receptor_emb: torch.Tensor,
    peptide_emb: torch.Tensor,
    eval_chunk_size: int,
    topk: int = 10,
) -> Dict[str, Any]:
    if receptor_emb.size(0) != peptide_emb.size(0):
        raise ValueError("receptor_emb and peptide_emb must have same row count")
    n = receptor_emb.size(0)
    target_scores = (receptor_emb * peptide_emb).sum(dim=1)
    ranks: List[torch.Tensor] = []
    top_indices: List[torch.Tensor] = []
    top_scores: List[torch.Tensor] = []
    for start in range(0, n, eval_chunk_size):
        end = min(start + eval_chunk_size, n)
        scores = receptor_emb[start:end] @ peptide_emb.t()
        cur_target_scores = target_scores[start:end].unsqueeze(1)
        ranks.append((scores > cur_target_scores).sum(dim=1) + 1)
        cur_topk = min(topk, n)
        values, indices = torch.topk(scores, k=cur_topk, dim=1)
        top_scores.append(values.cpu())
        top_indices.append(indices.cpu())
    rank_tensor = torch.cat(ranks, dim=0)
    return {
        "metrics": rank_metrics(rank_tensor, n, ks=(1, 5, 10)),
        "ranks": rank_tensor.cpu(),
        "top_indices": torch.cat(top_indices, dim=0),
        "top_scores": torch.cat(top_scores, dim=0),
    }


@torch.no_grad()
def collect_embeddings(
    model: PepCLIPConcatFusionModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, List[Dict[str, str]]]:
    model.eval()
    receptor_embs: List[torch.Tensor] = []
    peptide_embs: List[torch.Tensor] = []
    metadata: List[Dict[str, str]] = []
    for batch in loader:
        moved = {
            "one_d": move_1d_batch(batch["one_d"], device),
            "three_d": move_3d_batch(batch["three_d"], device),
        }
        outputs = model(moved)
        receptor_embs.append(outputs["receptor_emb"].detach().cpu())
        peptide_embs.append(outputs["peptide_emb"].detach().cpu())
        one_d = batch["one_d"]
        for i, sample_id in enumerate(one_d["sample_id"]):
            metadata.append(
                {
                    "sample_id": str(sample_id),
                    "pdb_id": str(one_d["pdb_id"][i]),
                    "receptor_key": str(one_d["receptor_key"][i]),
                    "peptide_key": str(one_d["peptide_key"][i]),
                    "peptide_sequence": str(one_d["peptide_sequence"][i]),
                }
            )
    return torch.cat(receptor_embs, dim=0), torch.cat(peptide_embs, dim=0), metadata


def write_monitor_retrieval(
    output_path: Path,
    metadata: List[Dict[str, str]],
    eval_result: Dict[str, Any],
) -> None:
    ranks = eval_result["ranks"].tolist()
    top_indices = eval_result["top_indices"].tolist()
    top_scores = eval_result["top_scores"].tolist()
    with output_path.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(metadata):
            hits = []
            for score, index in zip(top_scores[i], top_indices[i]):
                target = metadata[index]
                hits.append(
                    {
                        "rank_index": int(index),
                        "score": float(score),
                        "sample_id": target["sample_id"],
                        "peptide_key": target["peptide_key"],
                        "peptide_sequence": target.get("peptide_sequence", ""),
                        "is_true_pair": int(index) == i,
                    }
                )
            handle.write(
                json.dumps(
                    {
                        "query_index": i,
                        "query_sample_id": row["sample_id"],
                        "query_receptor_key": row["receptor_key"],
                        "true_peptide_key": row["peptide_key"],
                        "rank": int(ranks[i]),
                        "top_hits": hits,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_alignment_report(output_dir: Path, train_dataset: PairedTrackABDataset, valid_dataset: PairedTrackABDataset) -> None:
    report = {
        "train": {
            "track_a": str(train_dataset.track_a.track_a_path),
            "track_b": str(train_dataset.track_b.track_b_path),
            "rows": len(train_dataset),
            "sample_id_aligned": True,
        },
        "valid": {
            "track_a": str(valid_dataset.track_a.track_a_path),
            "track_b": str(valid_dataset.track_b.track_b_path),
            "rows": len(valid_dataset),
            "sample_id_aligned": True,
        },
    }
    (output_dir / "alignment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_checkpoint(
    path: Path,
    model: PepCLIPConcatFusionModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    valid_metrics: Dict[str, float],
    best_recall: float,
    history: List[Dict[str, Any]],
) -> None:
    tower_trainable = any(
        param.requires_grad
        for module in (model.model_1d, model.model_3d)
        for param in module.parameters()
    )
    tower_state = None
    if tower_trainable:
        tower_state = {
            "model_1d": model.model_1d.state_dict(),
            "model_3d": model.model_3d.state_dict(),
        }
    torch.save(
        {
            "fusion_state_dict": {
                "receptor_fusion": model.receptor_fusion.state_dict(),
                "peptide_fusion": model.peptide_fusion.state_dict(),
            },
            "tower_state_dict": tower_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "valid": valid_metrics,
            "best_recall": best_recall,
            "history": history,
        },
        path,
    )


def load_resume_checkpoint(
    model: PepCLIPConcatFusionModel,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | Path,
    device: torch.device,
    lr: float,
    tower_lr: float,
    reset_optimizer: bool,
) -> tuple[int, float, int, List[Dict[str, Any]]]:
    state = torch.load(checkpoint_path, map_location=device)
    fusion_state = state.get("fusion_state_dict")
    if not isinstance(fusion_state, dict):
        raise ValueError(f"Fusion checkpoint has no fusion_state_dict: {checkpoint_path}")

    model.receptor_fusion.load_state_dict(fusion_state["receptor_fusion"], strict=True)
    model.peptide_fusion.load_state_dict(fusion_state["peptide_fusion"], strict=True)
    tower_state = state.get("tower_state_dict")
    if tower_state:
        model.model_1d.load_state_dict(tower_state["model_1d"], strict=True)
        model.model_3d.load_state_dict(tower_state["model_3d"], strict=True)

    optimizer_restored = False
    if not reset_optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            group_name = param_group.get("group_name", "fusion")
            param_group["lr"] = tower_lr if group_name == "tower" else lr
        optimizer_restored = True

    completed_epoch = int(state.get("epoch", 0))
    history = list(state.get("history", []))
    if not history:
        history_path = Path(checkpoint_path).parent / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))

    best_recall = float(state.get("best_recall", -1.0))
    best_epoch = 0
    for row in history:
        valid = row.get("valid", {})
        recall = float(valid.get("recall_at_10", -1.0))
        if recall >= best_recall:
            best_epoch = int(row.get("epoch", 0))
    if best_epoch == 0 and history:
        best_row = max(
            history,
            key=lambda row: float(row.get("valid", {}).get("recall_at_10", -1.0)),
        )
        best_epoch = int(best_row.get("epoch", 0))
        best_recall = float(best_row.get("valid", {}).get("recall_at_10", best_recall))

    start_epoch = completed_epoch + 1
    print(
        json.dumps(
            {
                "event": "resumed_fusion_checkpoint",
                "checkpoint": str(checkpoint_path),
                "completed_epoch": completed_epoch,
                "start_epoch": start_epoch,
                "best_epoch": best_epoch,
                "best_recall_at_10": best_recall,
                "optimizer_restored": optimizer_restored,
                "fusion_lr": lr,
                "tower_lr": tower_lr,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return start_epoch, best_recall, best_epoch, history


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(output_dir, args)
    device = torch.device(args.device)

    config_1d_path = Path(args.config_1d) if args.config_1d else default_config_path(args.checkpoint_1d)
    config_3d_path = Path(args.config_3d) if args.config_3d else default_config_path(args.checkpoint_3d)
    if not config_1d_path.exists():
        raise FileNotFoundError(f"1D config not found: {config_1d_path}")
    if not config_3d_path.exists():
        raise FileNotFoundError(f"3D config not found: {config_3d_path}")
    config_1d = read_json(config_1d_path)
    config_3d = read_json(config_3d_path)
    if args.hf_model_name_or_path_1d:
        config_1d["hf_model_name_or_path"] = args.hf_model_name_or_path_1d
    (output_dir / "source_model_configs.json").write_text(
        json.dumps(
            {
                "config_1d_path": str(config_1d_path),
                "config_3d_path": str(config_3d_path),
                "config_1d": config_1d,
                "config_3d": config_3d,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    model_1d = build_1d_model(
        config_1d,
        args.checkpoint_1d,
        device,
        allow_partial_checkpoint_load=args.allow_partial_tower_checkpoint_load,
    )
    model_3d = build_3d_model(
        config_3d,
        args.checkpoint_3d,
        device,
        allow_partial_checkpoint_load=args.allow_partial_tower_checkpoint_load,
    )

    train_dataset = PairedTrackABDataset(
        args.train_track_a,
        args.train_track_b,
        data_format=args.data_format,
        receptor_field=config_1d.get("receptor_field", "receptor_patch_sequence"),
        max_receptor_atoms=config_3d.get("max_receptor_atoms"),
        max_peptide_atoms=config_3d.get("max_peptide_atoms"),
        conformer_views_path=args.train_conformer_views,
        conformer_evidence_path=args.conformer_evidence,
        max_auxiliary_conformers_per_sample=args.max_auxiliary_conformers_per_sample,
    )
    valid_dataset = PairedTrackABDataset(
        args.valid_track_a,
        args.valid_track_b,
        data_format=args.data_format,
        receptor_field=config_1d.get("receptor_field", "receptor_patch_sequence"),
        max_receptor_atoms=config_3d.get("max_receptor_atoms"),
        max_peptide_atoms=config_3d.get("max_peptide_atoms"),
    )
    write_alignment_report(output_dir, train_dataset, valid_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fusion,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fusion,
    )

    dim_1d = int(config_1d.get("output_dim", 128))
    dim_3d = int(config_3d.get("output_dim", 128))
    model = PepCLIPConcatFusionModel(
        model_1d=model_1d,
        model_3d=model_3d,
        concat_dim=dim_1d + dim_3d,
        hidden_dim=args.fusion_hidden_dim,
        output_dim=args.fusion_output_dim,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)

    unfreeze_1d_report = unfreeze_1d_last_layers(
        model.model_1d,
        args.unfreeze_1d_last_n_layers,
    )
    unfreeze_3d_report = unfreeze_3d_last_layers(
        model.model_3d,
        args.unfreeze_3d_last_n_layers,
    )
    fusion_params = [
        param
        for module in (model.receptor_fusion, model.peptide_fusion)
        for param in module.parameters()
        if param.requires_grad
    ]
    tower_params = [
        param
        for module in (model.model_1d, model.model_3d)
        for param in module.parameters()
        if param.requires_grad
    ]
    trainable_params = fusion_params + tower_params
    parameter_groups = [
        {
            "params": fusion_params,
            "lr": args.lr,
            "group_name": "fusion",
        }
    ]
    if tower_params:
        parameter_groups.append(
            {
                "params": tower_params,
                "lr": args.tower_lr,
                "group_name": "tower",
            }
        )
    print(
        json.dumps(
            {
                "event": "trainable_parameters",
                "total": sum(param.numel() for param in model.parameters()),
                "trainable": sum(param.numel() for param in trainable_params),
                "fusion_trainable": sum(param.numel() for param in fusion_params),
                "tower_trainable": sum(param.numel() for param in tower_params),
                "frozen_towers": not bool(tower_params),
                "unfreeze_1d": unfreeze_1d_report,
                "unfreeze_3d": unfreeze_3d_report,
                "concat_dim": dim_1d + dim_3d,
                "fusion_hidden_dim": args.fusion_hidden_dim,
                "fusion_output_dim": args.fusion_output_dim,
                "temperature": args.temperature,
                "fusion_lr": args.lr,
                "tower_lr": args.tower_lr if tower_params else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)

    history: List[Dict[str, Any]] = []
    best_recall = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    start_epoch = 1
    if args.resume_checkpoint:
        resume_has_tower_state = checkpoint_has_tower_state(args.resume_checkpoint, device)
        if tower_params and not resume_has_tower_state and not args.reset_optimizer_on_resume:
            raise ValueError(
                "Starting partial tower unfreezing from a frozen-tower checkpoint changes "
                "optimizer parameter groups. "
                "Add --reset_optimizer_on_resume when starting the joint fine-tuning stage."
            )
        start_epoch, best_recall, best_epoch, history = load_resume_checkpoint(
            model,
            optimizer,
            args.resume_checkpoint,
            device,
            lr=args.lr,
            tower_lr=args.tower_lr,
            reset_optimizer=args.reset_optimizer_on_resume,
        )
    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint already completed epoch {start_epoch - 1}, "
            f"but --epochs is only {args.epochs}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            eval_chunk_size=args.eval_chunk_size,
            disable_duplicate_mask=args.disable_duplicate_mask,
            conformer_consistency_weight=args.conformer_consistency_weight,
        )
        valid_metrics = run_epoch(
            model,
            valid_loader,
            optimizer,
            device,
            train=False,
            eval_chunk_size=args.eval_chunk_size,
            disable_duplicate_mask=args.disable_duplicate_mask,
            conformer_consistency_weight=0.0,
        )
        row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        recall = float(valid_metrics.get(args.early_stopping_metric, 0.0))
        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "checkpoint_best.pt",
                model,
                optimizer,
                args,
                epoch,
                valid_metrics,
                best_recall,
                history,
            )
            receptor_emb, peptide_emb, metadata = collect_embeddings(model, valid_loader, device)
            eval_result = evaluate_embeddings(
                receptor_emb,
                peptide_emb,
                eval_chunk_size=args.eval_chunk_size,
                topk=args.retrieval_topk,
            )
            write_monitor_retrieval(output_dir / "monitor_retrieval_best.jsonl", metadata, eval_result)
            torch.save(
                {
                    "receptor_embeddings": receptor_emb,
                    "peptide_embeddings": peptide_emb,
                    "metadata": metadata,
                    "metrics": eval_result["metrics"],
                    "epoch": epoch,
                },
                output_dir / "monitor_embeddings_best.pt",
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            output_dir / "checkpoint_last.pt",
            model,
            optimizer,
            args,
            epoch,
            valid_metrics,
            best_recall,
            history,
        )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "best_epoch": best_epoch,
                    "best_recall_at_10": best_recall,
                    "best_metric": args.early_stopping_metric,
                    "early_stopping_patience": args.early_stopping_patience,
                    "epochs_without_improvement": epochs_without_improvement,
                    "early_stopped": early_stopped,
                    "latest": row,
                    "reference_results": REFERENCE_RESULTS,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            early_stopped = True
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "best_epoch": best_epoch,
                        "best_recall_at_10": best_recall,
                        "best_metric": args.early_stopping_metric,
                        "early_stopping_patience": args.early_stopping_patience,
                        "epochs_without_improvement": epochs_without_improvement,
                        "early_stopped": early_stopped,
                        "latest": row,
                        "reference_results": REFERENCE_RESULTS,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "event": "early_stopping",
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "metric": args.early_stopping_metric,
                        "best_value": best_recall,
                        "patience": args.early_stopping_patience,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break


if __name__ == "__main__":
    main()

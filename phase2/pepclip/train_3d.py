from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .data import ATOM_NAME_TO_ID, ELEMENT_TO_ID, RESIDUE_NAME_TO_ID, PepCLIP3DDataset, collate_pepclip_3d
from .losses import symmetric_in_batch_softmax_loss
from .metrics import retrieval_metrics_from_embeddings
from .model_3d import PepCLIP3DModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal PepCLIP 3D-only skeleton from Step8 Track B.")
    parser.add_argument("--train_track_b", required=True)
    parser.add_argument("--valid_track_b", required=True)
    parser.add_argument("--data_format", choices=["jsonl", "lmdb"], default="lmdb")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_split", default=None)
    parser.add_argument("--valid_split", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_chunk_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--encoder_type", choices=["radial", "unimol_style"], default="radial")
    parser.add_argument("--element_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--coord_scale", type=float, default=10.0)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_rbf", type=int, default=32)
    parser.add_argument("--distance_cutoff", type=float, default=20.0)
    parser.add_argument("--max_receptor_atoms", type=int, default=None)
    parser.add_argument("--max_peptide_atoms", type=int, default=None)
    parser.add_argument(
        "--freeze_peptide_encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="DrugCLIP-style option: freeze the peptide/molecule 3D encoder.",
    )
    parser.add_argument(
        "--freeze_receptor_encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Usually false for DrugCLIP-style training because the pocket/receptor encoder is learned.",
    )
    parser.add_argument(
        "--peptide_encoder_checkpoint",
        default=None,
        help="Optional checkpoint used to initialize the peptide encoder before freezing.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def move_batch(batch: Dict, device: torch.device) -> Dict:
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
    ]
    for key in tensor_keys:
        batch[key] = batch[key].to(device)
    return batch


def forward_model(model: PepCLIP3DModel, batch: Dict) -> Dict[str, torch.Tensor]:
    return model(
        receptor_coords=batch["receptor_coords"],
        receptor_elements=batch["receptor_elements"],
        receptor_mask=batch["receptor_mask"],
        receptor_atom_names=batch["receptor_atom_names"],
        receptor_residue_names=batch["receptor_residue_names"],
        peptide_coords=batch["peptide_coords"],
        peptide_elements=batch["peptide_elements"],
        peptide_mask=batch["peptide_mask"],
        peptide_atom_names=batch["peptide_atom_names"],
        peptide_residue_names=batch["peptide_residue_names"],
    )


def load_peptide_encoder_checkpoint(model: PepCLIP3DModel, checkpoint_path: str, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    raw_state = state.get("model_state_dict", state)
    peptide_state = {}
    for key, value in raw_state.items():
        if key.startswith("peptide_encoder."):
            peptide_state[key.removeprefix("peptide_encoder.")] = value
        elif not key.startswith("receptor_encoder.") and not key.startswith("logit_scale"):
            peptide_state[key] = value
    missing, unexpected = model.peptide_encoder.load_state_dict(peptide_state, strict=False)
    print(
        json.dumps(
            {
                "event": "loaded_peptide_encoder_checkpoint",
                "checkpoint": checkpoint_path,
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            },
            ensure_ascii=False,
        )
    )


def set_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def run_epoch(
    model: PepCLIP3DModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    eval_chunk_size: int = 1024,
    freeze_peptide_encoder: bool = False,
    freeze_receptor_encoder: bool = False,
) -> Dict[str, float]:
    model.train(train)
    if freeze_peptide_encoder:
        model.peptide_encoder.eval()
    if freeze_receptor_encoder:
        model.receptor_encoder.eval()
    total_loss = 0.0
    total_items = 0
    receptor_embs = []
    peptide_embs = []

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            outputs = forward_model(model, batch)
            loss = symmetric_in_batch_softmax_loss(
                outputs["logits_per_receptor"],
                outputs["logits_per_peptide"],
                receptor_keys=batch["receptor_key"],
                peptide_keys=batch["peptide_key"],
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        batch_size = batch["receptor_coords"].size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        if not train:
            receptor_embs.append(outputs["receptor_emb"].detach().cpu())
            peptide_embs.append(outputs["peptide_emb"].detach().cpu())

    result = {"loss": total_loss / max(total_items, 1)}
    if not train and receptor_embs:
        receptor_emb = torch.cat(receptor_embs, dim=0)
        peptide_emb = torch.cat(peptide_embs, dim=0)
        result.update(
            retrieval_metrics_from_embeddings(
                receptor_emb,
                peptide_emb,
                chunk_size=eval_chunk_size,
            )
        )
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_dataset = PepCLIP3DDataset(
        args.train_track_b,
        split=args.train_split,
        data_format=args.data_format,
        max_receptor_atoms=args.max_receptor_atoms,
        max_peptide_atoms=args.max_peptide_atoms,
    )
    valid_dataset = PepCLIP3DDataset(
        args.valid_track_b,
        split=args.valid_split,
        data_format=args.data_format,
        max_receptor_atoms=args.max_receptor_atoms,
        max_peptide_atoms=args.max_peptide_atoms,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pepclip_3d,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pepclip_3d,
    )

    model = PepCLIP3DModel(
        num_elements=max(ELEMENT_TO_ID.values()) + 1,
        num_atom_names=max(ATOM_NAME_TO_ID.values()) + 1,
        num_residue_names=max(RESIDUE_NAME_TO_ID.values()) + 1,
        encoder_type=args.encoder_type,
        element_dim=args.element_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        coord_scale=args.coord_scale,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_rbf=args.num_rbf,
        distance_cutoff=args.distance_cutoff,
    ).to(device)

    if args.peptide_encoder_checkpoint:
        load_peptide_encoder_checkpoint(model, args.peptide_encoder_checkpoint, device)
    if args.freeze_peptide_encoder:
        set_trainable(model.peptide_encoder, False)
        model.peptide_encoder.eval()
    if args.freeze_receptor_encoder:
        set_trainable(model.receptor_encoder, False)
        model.receptor_encoder.eval()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain after applying freeze options")
    print(
        json.dumps(
            {
                "event": "trainable_parameters",
                "total": sum(param.numel() for param in model.parameters()),
                "trainable": sum(param.numel() for param in trainable_params),
                "freeze_peptide_encoder": args.freeze_peptide_encoder,
                "freeze_receptor_encoder": args.freeze_receptor_encoder,
            },
            ensure_ascii=False,
        )
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_recall = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            freeze_peptide_encoder=args.freeze_peptide_encoder,
            freeze_receptor_encoder=args.freeze_receptor_encoder,
        )
        valid_metrics = run_epoch(
            model,
            valid_loader,
            optimizer,
            device,
            train=False,
            eval_chunk_size=args.eval_chunk_size,
            freeze_peptide_encoder=args.freeze_peptide_encoder,
            freeze_receptor_encoder=args.freeze_receptor_encoder,
        )
        row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        recall = valid_metrics.get("recall_at_10", 0.0)
        if recall > best_recall:
            best_recall = recall
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "valid": valid_metrics,
                },
                output_dir / "checkpoint_best.pt",
            )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .data import AA_TO_ID, PepCLIPDataset, collate_pepclip
from .losses import symmetric_in_batch_softmax_loss
from .metrics import retrieval_metrics
from .model import PepCLIPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal PepCLIP 1D skeleton.")
    parser.add_argument("--metadata_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_split", default="main_train")
    parser.add_argument("--valid_split", default="monitor")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def move_batch(batch: Dict, device: torch.device) -> Dict:
    batch = dict(batch)
    batch["receptor_tokens"] = batch["receptor_tokens"].to(device)
    batch["peptide_tokens"] = batch["peptide_tokens"].to(device)
    return batch


def run_epoch(model, loader, optimizer, device: torch.device, train: bool) -> Dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_items = 0
    receptor_embs = []
    peptide_embs = []

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            outputs = model(batch["receptor_tokens"], batch["peptide_tokens"])
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

        batch_size = batch["receptor_tokens"].size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        if not train:
            receptor_embs.append(outputs["receptor_emb"].detach().cpu())
            peptide_embs.append(outputs["peptide_emb"].detach().cpu())

    result = {"loss": total_loss / max(total_items, 1)}
    if not train and receptor_embs:
        receptor_emb = torch.cat(receptor_embs, dim=0)
        peptide_emb = torch.cat(peptide_embs, dim=0)
        result.update(retrieval_metrics(receptor_emb @ peptide_emb.t()))
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_dataset = PepCLIPDataset(args.metadata_jsonl, split=args.train_split)
    valid_dataset = PepCLIPDataset(args.metadata_jsonl, split=args.valid_split)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pepclip,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pepclip,
    )

    model = PepCLIPModel(
        vocab_size=max(AA_TO_ID.values()) + 1,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_recall = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        valid_metrics = run_epoch(model, valid_loader, optimizer, device, train=False)
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

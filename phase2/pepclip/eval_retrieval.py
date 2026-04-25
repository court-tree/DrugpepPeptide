from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import AA_TO_ID, PepCLIPDataset, collate_pepclip
from .metrics import retrieval_metrics
from .model import PepCLIPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PepCLIP retrieval on one split.")
    parser.add_argument("--metadata_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="monitor")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device)
    model_args = state.get("args", {})
    model = PepCLIPModel(
        vocab_size=max(AA_TO_ID.values()) + 1,
        embed_dim=int(model_args.get("embed_dim", 128)),
        hidden_dim=int(model_args.get("hidden_dim", 256)),
        output_dim=int(model_args.get("output_dim", 128)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    dataset = PepCLIPDataset(args.metadata_jsonl, split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pepclip)

    receptor_embs = []
    peptide_embs = []
    for batch in loader:
        receptor_tokens = batch["receptor_tokens"].to(device)
        peptide_tokens = batch["peptide_tokens"].to(device)
        receptor_embs.append(model.encode_receptor(receptor_tokens).cpu())
        peptide_embs.append(model.encode_peptide(peptide_tokens).cpu())

    receptor_emb = torch.cat(receptor_embs, dim=0)
    peptide_emb = torch.cat(peptide_embs, dim=0)
    logits = receptor_emb @ peptide_emb.t()
    metrics = retrieval_metrics(logits)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


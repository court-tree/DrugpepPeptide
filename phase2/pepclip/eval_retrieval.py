from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import AA_TO_ID, PepCLIPDataset, collate_pepclip
from .metrics import retrieval_metrics_from_embeddings
from .model import PepCLIPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PepCLIP retrieval on one split.")
    parser.add_argument("--track_a")
    parser.add_argument("--data_format", choices=["jsonl", "lmdb"], default="jsonl")
    parser.add_argument("--metadata_jsonl", help="Legacy Step7 final_metadata.jsonl input")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="monitor")
    parser.add_argument("--receptor_field", default="receptor_patch_sequence")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_chunk_size", type=int, default=1024)
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
        encoder_type=str(model_args.get("encoder_type", "mean_pool")),
        hf_model_name_or_path=model_args.get("hf_model_name_or_path"),
        freeze_hf_backbone=bool(model_args.get("freeze_hf_backbone", True)),
        hf_max_length=int(model_args.get("hf_max_length", 512)),
        embed_dim=int(model_args.get("embed_dim", 128)),
        hidden_dim=int(model_args.get("hidden_dim", 256)),
        output_dim=int(model_args.get("output_dim", 128)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    if args.metadata_jsonl:
        track_a = args.metadata_jsonl
        split = args.split
        data_format = "jsonl"
        receptor_field = "receptor_sequence" if args.receptor_field == "receptor_patch_sequence" else args.receptor_field
    else:
        if not args.track_a:
            raise ValueError("Provide --track_a, or use legacy --metadata_jsonl")
        track_a = args.track_a
        split = None
        data_format = args.data_format
        receptor_field = args.receptor_field

    dataset = PepCLIPDataset(
        track_a,
        split=split,
        data_format=data_format,
        receptor_field=receptor_field,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pepclip)

    receptor_embs = []
    peptide_embs = []
    for batch in loader:
        receptor_tokens = batch["receptor_tokens"].to(device)
        peptide_tokens = batch["peptide_tokens"].to(device)
        receptor_embs.append(
            model.encode_receptor(
                receptor_tokens=receptor_tokens,
                receptor_sequences=batch["receptor_sequence"],
            ).cpu()
        )
        peptide_embs.append(
            model.encode_peptide(
                peptide_tokens=peptide_tokens,
                peptide_sequences=batch["peptide_sequence"],
            ).cpu()
        )

    receptor_emb = torch.cat(receptor_embs, dim=0)
    peptide_emb = torch.cat(peptide_embs, dim=0)
    metrics = retrieval_metrics_from_embeddings(
        receptor_emb,
        peptide_emb,
        chunk_size=args.eval_chunk_size,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import AA_TO_ID, PepCLIPDataset, collate_pepclip
from .model import PepCLIPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PepCLIP receptor and peptide embeddings.")
    parser.add_argument("--track_a", required=True)
    parser.add_argument("--data_format", choices=["jsonl", "lmdb"], default="lmdb")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default=None, help="Optional split filter for legacy mixed JSONL inputs")
    parser.add_argument("--receptor_field", default="receptor_patch_sequence")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(checkpoint: str, device: torch.device) -> PepCLIPModel:
    state = torch.load(checkpoint, map_location=device)
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
    return model


def metadata_rows(batch: Dict) -> List[Dict[str, str]]:
    rows = []
    for i, sample_id in enumerate(batch["sample_id"]):
        rows.append(
            {
                "sample_id": str(sample_id),
                "pdb_id": str(batch["pdb_id"][i]),
                "split": str(batch["split"][i]),
                "receptor_key": str(batch["receptor_key"][i]),
                "peptide_key": str(batch["peptide_key"][i]),
                "receptor_sequence": str(batch["receptor_sequence"][i]),
                "peptide_sequence": str(batch["peptide_sequence"][i]),
            }
        )
    return rows


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model = build_model(args.checkpoint, device)
    dataset = PepCLIPDataset(
        args.track_a,
        split=args.split,
        data_format=args.data_format,
        receptor_field=args.receptor_field,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pepclip,
    )

    receptor_embs = []
    peptide_embs = []
    rows = []
    for batch in loader:
        receptor_tokens = batch["receptor_tokens"].to(device)
        peptide_tokens = batch["peptide_tokens"].to(device)
        receptor_embs.append(
            model.encode_receptor(
                receptor_tokens=receptor_tokens,
                receptor_sequences=batch["receptor_sequence"],
            ).cpu().numpy().astype(np.float32)
        )
        peptide_embs.append(
            model.encode_peptide(
                peptide_tokens=peptide_tokens,
                peptide_sequences=batch["peptide_sequence"],
            ).cpu().numpy().astype(np.float32)
        )
        rows.extend(metadata_rows(batch))

    receptor_emb = np.concatenate(receptor_embs, axis=0)
    peptide_emb = np.concatenate(peptide_embs, axis=0)
    np.save(output_dir / "receptor_embeddings.npy", receptor_emb)
    np.save(output_dir / "peptide_embeddings.npy", peptide_emb)

    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "num_samples": int(receptor_emb.shape[0]),
        "embedding_dim": int(receptor_emb.shape[1]),
        "track_a": str(args.track_a),
        "checkpoint": str(args.checkpoint),
        "receptor_embeddings": str(output_dir / "receptor_embeddings.npy"),
        "peptide_embeddings": str(output_dir / "peptide_embeddings.npy"),
        "metadata": str(output_dir / "metadata.jsonl"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


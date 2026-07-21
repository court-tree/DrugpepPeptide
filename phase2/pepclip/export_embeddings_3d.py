from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ATOM_NAME_TO_ID, ELEMENT_TO_ID, RESIDUE_NAME_TO_ID, PepCLIP3DDataset, collate_pepclip_3d
from .model_3d import PepCLIP3DModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PepCLIP 3D receptor and peptide embeddings.")
    parser.add_argument("--track_b", required=True)
    parser.add_argument("--data_format", choices=["jsonl", "lmdb"], default="lmdb")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_receptor_atoms", type=int, default=None)
    parser.add_argument("--max_peptide_atoms", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(checkpoint: str, device: torch.device) -> Tuple[PepCLIP3DModel, Dict[str, Any]]:
    state = torch.load(checkpoint, map_location=device)
    model_args = state.get("args", {})
    model = PepCLIP3DModel(
        num_elements=max(ELEMENT_TO_ID.values()) + 1,
        num_atom_names=max(ATOM_NAME_TO_ID.values()) + 1,
        num_residue_names=max(RESIDUE_NAME_TO_ID.values()) + 1,
        encoder_type=str(model_args.get("encoder_type", "radial")),
        element_dim=int(model_args.get("element_dim", 32)),
        hidden_dim=int(model_args.get("hidden_dim", 256)),
        output_dim=int(model_args.get("output_dim", 128)),
        dropout=0.0,
        coord_scale=float(model_args.get("coord_scale", 10.0)),
        num_layers=int(model_args.get("num_layers", 4)),
        num_heads=int(model_args.get("num_heads", 8)),
        num_rbf=int(model_args.get("num_rbf", 32)),
        distance_cutoff=float(model_args.get("distance_cutoff", 20.0)),
        num_neighbors=int(model_args.get("num_neighbors", 32)),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, model_args


def metadata_rows(batch: Dict) -> List[Dict[str, str | int]]:
    rows = []
    for i, sample_id in enumerate(batch["sample_id"]):
        rows.append(
            {
                "sample_id": str(sample_id),
                "pdb_id": str(batch["pdb_id"][i]),
                "split": str(batch["split"][i]),
                "receptor_key": str(batch["receptor_key"][i]),
                "peptide_key": str(batch["peptide_key"][i]),
                "num_receptor_atoms": int(batch["num_receptor_atoms"][i]),
                "num_peptide_atoms": int(batch["num_peptide_atoms"][i]),
            }
        )
    return rows


def move_batch(batch: Dict, device: torch.device) -> Dict:
    batch = dict(batch)
    for key in [
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
    ]:
        batch[key] = batch[key].to(device)
    return batch


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, model_args = build_model(args.checkpoint, device)
    dataset = PepCLIP3DDataset(
        args.track_b,
        split=args.split,
        data_format=args.data_format,
        max_receptor_atoms=args.max_receptor_atoms if args.max_receptor_atoms is not None else model_args.get("max_receptor_atoms"),
        max_peptide_atoms=args.max_peptide_atoms if args.max_peptide_atoms is not None else model_args.get("max_peptide_atoms"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pepclip_3d,
    )

    receptor_embs = []
    peptide_embs = []
    rows = []
    for batch in loader:
        rows.extend(metadata_rows(batch))
        batch = move_batch(batch, device)
        receptor_embs.append(
            model.encode_receptor(
                batch["receptor_coords"],
                batch["receptor_elements"],
                batch["receptor_mask"],
                batch["receptor_atom_names"],
                batch["receptor_residue_names"],
            ).cpu().numpy().astype(np.float32)
        )
        peptide_embs.append(
            model.encode_peptide(
                batch["peptide_coords"],
                batch["peptide_elements"],
                batch["peptide_mask"],
                batch["peptide_atom_names"],
                batch["peptide_residue_names"],
            ).cpu().numpy().astype(np.float32)
        )

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
        "track_b": str(args.track_b),
        "checkpoint": str(args.checkpoint),
        "receptor_embeddings": str(output_dir / "receptor_embeddings.npy"),
        "peptide_embeddings": str(output_dir / "peptide_embeddings.npy"),
        "metadata": str(output_dir / "metadata.jsonl"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

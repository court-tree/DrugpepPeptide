from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .data import ATOM_NAME_TO_ID, ELEMENT_TO_ID, RESIDUE_NAME_TO_ID, PepCLIP3DDataset, collate_pepclip_3d
from .metrics import retrieval_metrics_from_embeddings
from .model_3d import PepCLIP3DModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PepCLIP 3D retrieval on one Track B split.")
    parser.add_argument("--track_b", required=True)
    parser.add_argument("--data_format", choices=["jsonl", "lmdb"], default="lmdb")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_chunk_size", type=int, default=1024)
    parser.add_argument("--max_receptor_atoms", type=int, default=None)
    parser.add_argument("--max_peptide_atoms", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def move_batch(batch, device: torch.device):
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
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device)
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

    dataset = PepCLIP3DDataset(
        args.track_b,
        split=args.split,
        data_format=args.data_format,
        max_receptor_atoms=args.max_receptor_atoms if args.max_receptor_atoms is not None else model_args.get("max_receptor_atoms"),
        max_peptide_atoms=args.max_peptide_atoms if args.max_peptide_atoms is not None else model_args.get("max_peptide_atoms"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pepclip_3d)

    receptor_embs = []
    peptide_embs = []
    for batch in loader:
        batch = move_batch(batch, device)
        receptor_embs.append(
            model.encode_receptor(
                batch["receptor_coords"],
                batch["receptor_elements"],
                batch["receptor_mask"],
                batch["receptor_atom_names"],
                batch["receptor_residue_names"],
            ).cpu()
        )
        peptide_embs.append(
            model.encode_peptide(
                batch["peptide_coords"],
                batch["peptide_elements"],
                batch["peptide_mask"],
                batch["peptide_atom_names"],
                batch["peptide_residue_names"],
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

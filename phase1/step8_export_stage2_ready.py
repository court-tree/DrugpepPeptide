from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common import (
    chain_residues,
    chain_sequence,
    find_chain_by_name,
    get_model,
    iter_jsonl,
    load_structure,
    residue_id_string,
    write_json,
)


def atom_record(atom, residue_id: str, residue_index: int, residue_name: str) -> Dict[str, Any]:
    return {
        "atom_name": atom.name.strip(),
        "element": atom.element.name.strip().upper(),
        "residue_id": residue_id,
        "residue_index": residue_index,
        "residue_name": residue_name,
        "x": float(atom.pos.x),
        "y": float(atom.pos.y),
        "z": float(atom.pos.z),
    }


def residue_atom_records(residue, residue_index: int) -> List[Dict[str, Any]]:
    residue_id = residue_id_string(residue)
    records: List[Dict[str, Any]] = []
    for atom in residue:
        if atom.element.name == "H":
            continue
        records.append(atom_record(atom, residue_id, residue_index, residue.name, ))
    return records


def build_chain_index(chain_items) -> Dict[str, int]:
    return {
        residue_id_string(item.residue): int(item.seq_index)
        for item in chain_items
    }


def find_residues_by_ids(chain_items, residue_ids: Sequence[str]):
    residue_set = set(residue_ids)
    return [item for item in chain_items if residue_id_string(item.residue) in residue_set]


def make_track_a_row(row: Dict[str, Any], receptor_items, patch_items) -> Dict[str, Any]:
    receptor_sequence = chain_sequence(receptor_items)
    patch_residue_ids = [residue_id_string(item.residue) for item in patch_items]
    patch_seq_indices = [int(item.seq_index) for item in patch_items]
    patch_sequence = "".join(
        receptor_sequence[idx] if 0 <= idx < len(receptor_sequence) else "X"
        for idx in patch_seq_indices
    )
    return {
        "sample_id": row["candidate_id"],
        "split": row["split"],
        "pdb_id": row["pdb_id"],
        "source_file": row["source_file"],
        "receptor_chain_id": row["receptor_chain_id"],
        "peptide_source_chain_id": row["peptide_source_chain_id"],
        "receptor_sequence": receptor_sequence,
        "receptor_patch_residue_ids": patch_residue_ids,
        "receptor_patch_seq_indices": patch_seq_indices,
        "receptor_patch_sequence": patch_sequence,
        "peptide_sequence": row["track_a_peptide_sequence"],
        "peptide_length": row["peptide_length"],
        "avg_contact_count": row["avg_contact_count"],
        "contact_coverage": row["contact_coverage"],
        "has_n_cap_proxy": row.get("has_n_cap_proxy", False),
        "has_c_cap_proxy": row.get("has_c_cap_proxy", False),
    }


def make_track_b_row(row: Dict[str, Any], peptide_items, patch_items) -> Dict[str, Any]:
    peptide_atoms: List[Dict[str, Any]] = []
    for item in peptide_items:
        peptide_atoms.extend(residue_atom_records(item.residue, int(item.seq_index)))

    patch_atoms: List[Dict[str, Any]] = []
    for item in patch_items:
        patch_atoms.extend(residue_atom_records(item.residue, int(item.seq_index)))

    return {
        "sample_id": row["candidate_id"],
        "split": row["split"],
        "pdb_id": row["pdb_id"],
        "source_file": row["source_file"],
        "receptor_chain_id": row["receptor_chain_id"],
        "peptide_source_chain_id": row["peptide_source_chain_id"],
        "peptide_residue_ids": list(row["track_b_peptide_residue_ids"]),
        "patch_residue_ids": list(row["track_b_patch_residue_ids"]),
        "peptide_atoms": peptide_atoms,
        "patch_atoms": patch_atoms,
        "patch_cutoff": row["patch_cutoff"],
        "has_n_cap_proxy": row.get("has_n_cap_proxy", False),
        "has_c_cap_proxy": row.get("has_c_cap_proxy", False),
    }


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def ensure_empty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def split_output_paths(out_dir: Path) -> Dict[str, Path]:
    return {
        "track_a_main": out_dir / "track_a_main_train.jsonl",
        "track_a_monitor": out_dir / "track_a_monitor.jsonl",
        "track_b_main": out_dir / "track_b_main_train.jsonl",
        "track_b_monitor": out_dir / "track_b_monitor.jsonl",
    }


def choose_paths(split_name: str, paths: Dict[str, Path]) -> Tuple[Path, Path]:
    if split_name == "monitor":
        return paths["track_a_monitor"], paths["track_b_monitor"]
    return paths["track_a_main"], paths["track_b_main"]


def process_rows(rows: Iterable[Dict[str, Any]], pdb_dir: Path, out_dir: Path, progress_every: int) -> Dict[str, Any]:
    start = time.time()
    paths = split_output_paths(out_dir)
    for path in paths.values():
        ensure_empty(path)

    current_source: Optional[str] = None
    structure = None
    model = None
    receptor_cache: Dict[str, Any] = {}
    peptide_cache: Dict[str, Any] = {}

    processed = 0
    split_counts = {"main_train": 0, "monitor": 0}

    for row in rows:
        source_file = row["source_file"]
        if source_file != current_source:
            structure = load_structure(pdb_dir / source_file)
            model = get_model(structure)
            receptor_cache = {}
            peptide_cache = {}
            current_source = source_file

        assert model is not None

        receptor_chain_id = row["receptor_chain_id"]
        peptide_chain_id = row["peptide_source_chain_id"]

        if receptor_chain_id not in receptor_cache:
            receptor_chain = find_chain_by_name(model, receptor_chain_id)
            receptor_cache[receptor_chain_id] = chain_residues(receptor_chain)
        if peptide_chain_id not in peptide_cache:
            peptide_chain = find_chain_by_name(model, peptide_chain_id)
            peptide_cache[peptide_chain_id] = chain_residues(peptide_chain)

        receptor_items = receptor_cache[receptor_chain_id]
        peptide_items_all = peptide_cache[peptide_chain_id]

        left_idx = int(row["final_left_index"])
        right_idx = int(row["final_right_index"])
        peptide_items = peptide_items_all[left_idx:right_idx + 1]
        patch_items = find_residues_by_ids(receptor_items, row["track_b_patch_residue_ids"])

        track_a_row = make_track_a_row(row, receptor_items, patch_items)
        track_b_row = make_track_b_row(row, peptide_items, patch_items)

        path_a, path_b = choose_paths(str(row["split"]), paths)
        append_jsonl(path_a, track_a_row)
        append_jsonl(path_b, track_b_row)

        split_name = str(row["split"])
        split_counts[split_name] = split_counts.get(split_name, 0) + 1
        processed += 1

        if progress_every > 0 and processed % progress_every == 0:
            elapsed = time.time() - start
            speed = processed / elapsed if elapsed > 0 else 0.0
            print(
                f"[STEP8] exported {processed} rows | main={split_counts.get('main_train', 0)} | "
                f"monitor={split_counts.get('monitor', 0)} | elapsed={elapsed/60:.1f} min | speed={speed:.1f} rows/s",
                flush=True,
            )

    summary = {
        "input_rows": processed,
        "split_counts": split_counts,
        "track_a_main_jsonl": str(paths["track_a_main"]),
        "track_a_monitor_jsonl": str(paths["track_a_monitor"]),
        "track_b_main_jsonl": str(paths["track_b_main"]),
        "track_b_monitor_jsonl": str(paths["track_b_monitor"]),
        "elapsed_sec": round(time.time() - start, 3),
        "export_format": "stage2_ready_jsonl_dual_track",
        "track_a_contents": "receptor full sequence + patch residue ids/indices + peptide sequence",
        "track_b_contents": "patch heavy atoms + peptide heavy atoms",
    }
    write_json(out_dir / "step8_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase1 Step8: export stage2-ready dual-track dataset")
    parser.add_argument("--input_jsonl", required=True, help="Phase1 final_metadata.jsonl")
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--progress_every", type=int, default=5000)
    args = parser.parse_args()

    rows = iter_jsonl(Path(args.input_jsonl))
    process_rows(
        rows=rows,
        pdb_dir=Path(args.pdb_dir),
        out_dir=Path(args.output_dir),
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()

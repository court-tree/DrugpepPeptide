from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from common import (
    chain_residues,
    find_chain_by_name,
    get_atom_by_name,
    get_model,
    load_structure,
    residue_id_string,
    window_bounds_match_candidate,
)


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "MSE", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}


def get_atom_coord(residue: gemmi.Residue, atom_name: str) -> Optional[np.ndarray]:
    atom = get_atom_by_name(residue, atom_name)
    if atom is None:
        return None
    return np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)


def check_peptide_bond(left_res: gemmi.Residue, right_res: gemmi.Residue, min_dist: float = 1.2, max_dist: float = 1.8) -> bool:
    c_pos = get_atom_coord(left_res, "C")
    n_pos = get_atom_coord(right_res, "N")
    if c_pos is None or n_pos is None:
        return False
    dist = float(np.linalg.norm(c_pos - n_pos))
    return min_dist <= dist <= max_dist


def residue_heavy_atom_coords(residue: gemmi.Residue) -> np.ndarray:
    coords = []
    for atom in residue:
        if atom.element.name == "H":
            continue
        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def peptide_window_coords(peptide_residues: Sequence[gemmi.Residue]) -> np.ndarray:
    blocks = [residue_heavy_atom_coords(res) for res in peptide_residues if len(residue_heavy_atom_coords(res)) > 0]
    if not blocks:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(blocks, axis=0)


def build_proxy_cap_meta(all_pep_res, left_idx: int, right_idx: int) -> Dict[str, Any]:
    has_n_cap = False
    has_c_cap = False
    if left_idx > 0:
        has_n_cap = check_peptide_bond(all_pep_res[left_idx - 1].residue, all_pep_res[left_idx].residue)
    if right_idx + 1 < len(all_pep_res):
        has_c_cap = check_peptide_bond(all_pep_res[right_idx].residue, all_pep_res[right_idx + 1].residue)
    return {
        "has_n_cap_proxy": bool(has_n_cap),
        "has_c_cap_proxy": bool(has_c_cap),
        "has_natural_left_terminus": bool(left_idx == 0),
        "has_natural_right_terminus": bool(right_idx + 1 >= len(all_pep_res)),
    }


def collect_patch_residues(
    receptor_residues,
    peptide_residues: Sequence[gemmi.Residue],
    patch_cutoff: float,
) -> List[gemmi.Residue]:
    pep_coords = peptide_window_coords(peptide_residues)
    if len(pep_coords) == 0:
        return []

    receptor_blocks = []
    atom_to_res_idx = []
    for idx, item in enumerate(receptor_residues):
        coords = residue_heavy_atom_coords(item.residue)
        if len(coords) == 0:
            continue
        receptor_blocks.append(coords)
        atom_to_res_idx.extend([idx] * len(coords))

    if not receptor_blocks:
        return []

    receptor_coords = np.concatenate(receptor_blocks, axis=0)
    tree = cKDTree(receptor_coords)
    neighbors = tree.query_ball_point(pep_coords, r=patch_cutoff)
    residue_indices = sorted(
        {
            atom_to_res_idx[atom_idx]
            for atom_hits in neighbors
            for atom_idx in atom_hits
        }
    )
    return [receptor_residues[idx].residue for idx in residue_indices]


def residue_one_letter(residue: gemmi.Residue) -> str:
    try:
        info = gemmi.find_tabulated_residue(residue.name)
        code = info.one_letter_code
        if code and code != "?":
            return code
    except Exception:
        pass
    return "X"


def process_row(row: Dict[str, Any], pdb_dir: Path, patch_cutoff: float) -> Dict[str, Any]:
    structure = load_structure(pdb_dir / row["source_file"])
    model = get_model(structure)
    receptor_chain = find_chain_by_name(model, row["receptor_chain_id"])
    peptide_chain = find_chain_by_name(model, row["peptide_source_chain_id"])

    all_receptor_res = chain_residues(receptor_chain)
    all_peptide_res = chain_residues(peptide_chain)

    left_idx = int(row["final_left_index"])
    right_idx = int(row["final_right_index"])
    peptide_window_items = all_peptide_res[left_idx:right_idx + 1]
    if not window_bounds_match_candidate(row, peptide_window_items):
        raise ValueError("Candidate bounds do not match peptide window")

    peptide_residues = [item.residue for item in peptide_window_items]
    patch_residues = collect_patch_residues(all_receptor_res, peptide_residues, patch_cutoff)
    peptide_sequence = "".join(residue_one_letter(res) for res in peptide_residues)

    final_row = dict(row)
    final_row.update(
        {
            "track_a_receptor_chain_id": row["receptor_chain_id"],
            "track_a_peptide_sequence": peptide_sequence,
            "track_b_patch_residue_ids": [residue_id_string(res) for res in patch_residues],
            "track_b_patch_num_residues": len(patch_residues),
            "track_b_peptide_residue_ids": [residue_id_string(res) for res in peptide_residues],
            "patch_cutoff": patch_cutoff,
        }
    )
    final_row.update(build_proxy_cap_meta(all_peptide_res, left_idx, right_idx))
    return final_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase1 Step7: finalize simplified dataset metadata")
    parser.add_argument("--main_jsonl", required=True)
    parser.add_argument("--monitor_jsonl", required=True)
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--patch_cutoff", type=float, default=6.0)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for path_str in (args.main_jsonl, args.monitor_jsonl):
        with Path(path_str).open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))

    start = time.time()
    output_rows = [process_row(row, Path(args.pdb_dir), args.patch_cutoff) for row in rows]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "patch_cutoff": args.patch_cutoff,
        "elapsed_sec": round(time.time() - start, 3),
    }
    (output_path.parent / "step7_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gemmi
import numpy as np
from scipy.spatial import cKDTree


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "MSE", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
}

AA_NORMALIZE = {
    "MSE": "MET",
}


@dataclass
class Step3Config:
    # geometry
    anchor_cutoff: float = 5.0
    contact_cutoff: float = 6.0
    nms_min_seq_gap: int = 3
    max_anchors_per_task: int = 5
    min_len: int = 8
    max_len: int = 20

    # anti-explosion filters
    min_contact_residues_6a: int = 4
    min_contact_ratio_6a: float = 0.50

    # cap per anchor after local ranking
    max_windows_per_anchor: int = 24

    # task-level metadata
    assembly_id: str = "native_file"
    method_name: str = "anchor_contact_filtered_window_pool_v3"


_WORKER_CFG: Optional[Step3Config] = None


# =========================================================
# Basic structure utils
# =========================================================
def is_protein_residue(residue: gemmi.Residue) -> bool:
    return residue.name in STANDARD_AA


def normalize_residue_name(resname: str) -> str:
    return AA_NORMALIZE.get(resname, resname)


def load_structure(path: Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    return st


def get_first_model(st: gemmi.Structure) -> gemmi.Model:
    if len(st) == 0:
        raise ValueError(f"Empty structure: {st.name}")
    return st[0]


def heavy_atoms(residue: gemmi.Residue) -> List[gemmi.Atom]:
    return [a for a in residue if a.element.name != "H"]


def residue_heavy_atom_coords(residue: gemmi.Residue) -> np.ndarray:
    coords = [[a.pos.x, a.pos.y, a.pos.z] for a in heavy_atoms(residue)]
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def find_chain_by_name(model: gemmi.Model, chain_name: str) -> gemmi.Chain:
    target = str(chain_name).strip()
    for chain in model:
        if str(chain.name).strip() == target:
            return chain
    available = [str(c.name) for c in model]
    raise KeyError(f"Chain not found: {chain_name}; available_chains={available}")


def chain_residues_with_alignment(chain: gemmi.Chain) -> List[Tuple[gemmi.Residue, np.ndarray]]:
    out: List[Tuple[gemmi.Residue, np.ndarray]] = []
    for residue in chain:
        if not is_protein_residue(residue):
            continue
        # 保留空坐标残基占位，避免与后续 postscore 的索引坐标系错位
        out.append((residue, residue_heavy_atom_coords(residue)))
    return out


def residue_seqid_num(residue: gemmi.Residue) -> int:
    if residue.seqid.num is None:
        raise ValueError(f"Residue missing seqid.num: {residue.name}")
    return int(residue.seqid.num)


def residue_seqid_icode(residue: gemmi.Residue) -> str:
    return str(residue.seqid.icode).strip()


def make_source_file(path: Path) -> str:
    return path.name


# =========================================================
# Contact helpers
# =========================================================
def build_receptor_kdtree(
    receptor_residues: Sequence[Tuple[gemmi.Residue, np.ndarray]]
) -> Tuple[Optional[cKDTree], np.ndarray]:
    coords_list: List[np.ndarray] = []
    atom_to_res_idx: List[int] = []
    for res_idx, (_, coords) in enumerate(receptor_residues):
        if len(coords) == 0:
            continue
        coords_list.append(coords)
        atom_to_res_idx.extend([res_idx] * len(coords))

    if not coords_list:
        return None, np.zeros((0,), dtype=np.int32)

    all_coords = np.concatenate(coords_list, axis=0)
    return cKDTree(all_coords), np.asarray(atom_to_res_idx, dtype=np.int32)


def peptide_min_distance_to_receptor(
    ligand_residues: Sequence[Tuple[gemmi.Residue, np.ndarray]],
    receptor_tree: cKDTree,
    atom_to_res_idx: np.ndarray,
    contact_cutoff: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for lig_idx, (_, coords) in enumerate(ligand_residues):
        if len(coords) == 0:
            out.append({
                "lig_idx": lig_idx,
                "min_dist": math.inf,
                "nearest_rec_res_idx": -1,
                "n_contact_rec_residues": 0,
            })
            continue

        dists, nn_idx = receptor_tree.query(coords, k=1)
        min_pos = int(np.argmin(dists))
        min_dist = float(dists[min_pos])
        nearest_atom_idx = int(nn_idx[min_pos])
        nearest_rec_res_idx = int(atom_to_res_idx[nearest_atom_idx])

        neighbors = receptor_tree.query_ball_point(coords, r=contact_cutoff)
        touched_res = set()
        for atom_hits in neighbors:
            for atom_idx in atom_hits:
                touched_res.add(int(atom_to_res_idx[atom_idx]))

        out.append({
            "lig_idx": lig_idx,
            "min_dist": min_dist,
            "nearest_rec_res_idx": nearest_rec_res_idx,
            "n_contact_rec_residues": len(touched_res),
        })

    return out


def pick_anchor_indices(
    ligand_stats: Sequence[Dict[str, Any]],
    cfg: Step3Config,
) -> List[int]:
    candidates = [x for x in ligand_stats if x["min_dist"] < cfg.anchor_cutoff]
    candidates.sort(key=lambda x: (x["min_dist"], -x["n_contact_rec_residues"], x["lig_idx"]))

    selected: List[int] = []
    for item in candidates:
        idx = int(item["lig_idx"])
        if all(abs(idx - prev) >= cfg.nms_min_seq_gap for prev in selected):
            selected.append(idx)
        if len(selected) >= cfg.max_anchors_per_task:
            break
    return selected


def residue_contact_flags(
    ligand_residues: Sequence[Tuple[gemmi.Residue, np.ndarray]],
    receptor_tree: cKDTree,
    cutoff: float,
) -> List[bool]:
    flags: List[bool] = []
    for _, coords in ligand_residues:
        if len(coords) == 0:
            flags.append(False)
            continue
        dists, _ = receptor_tree.query(coords, k=1)
        flags.append(float(np.min(dists)) < cutoff)
    return flags


def prefix_sum_bool(flags: Sequence[bool]) -> np.ndarray:
    arr = np.asarray([1 if x else 0 for x in flags], dtype=np.int32)
    return np.concatenate([[0], np.cumsum(arr)])


def contact_count_in_window(prefix: np.ndarray, left_idx: int, right_idx: int) -> int:
    return int(prefix[right_idx + 1] - prefix[left_idx])


def get_atom_by_name(residue: gemmi.Residue, atom_name: str) -> Optional[gemmi.Atom]:
    for atom in residue:
        if atom.name.strip() == atom_name:
            return atom
    return None


def peptide_bond_continuous(
    left_res: gemmi.Residue,
    right_res: gemmi.Residue,
    min_dist: float = 1.2,
    max_dist: float = 1.6,
) -> bool:
    c_atom = get_atom_by_name(left_res, "C")
    n_atom = get_atom_by_name(right_res, "N")
    if c_atom is None or n_atom is None:
        return False
    dx = c_atom.pos.x - n_atom.pos.x
    dy = c_atom.pos.y - n_atom.pos.y
    dz = c_atom.pos.z - n_atom.pos.z
    dist = float((dx * dx + dy * dy + dz * dz) ** 0.5)
    return min_dist <= dist <= max_dist


def window_is_backbone_continuous(residues: Sequence[gemmi.Residue]) -> bool:
    if len(residues) <= 1:
        return True
    for i in range(len(residues) - 1):
        if not peptide_bond_continuous(residues[i], residues[i + 1]):
            return False
    return True


# =========================================================
# Candidate generation per task
# =========================================================
def validate_window_by_contact(
    left_idx: int,
    right_idx: int,
    contact_prefix_6a: np.ndarray,
    cfg: Step3Config,
) -> Tuple[bool, int, float]:
    length = right_idx - left_idx + 1
    n_contact = contact_count_in_window(contact_prefix_6a, left_idx, right_idx)
    ratio = n_contact / length if length > 0 else 0.0
    keep = (n_contact >= cfg.min_contact_residues_6a) and (ratio >= cfg.min_contact_ratio_6a)
    return keep, n_contact, ratio


def enumerate_windows_for_anchor(
    ligand_residues: Sequence[Tuple[gemmi.Residue, np.ndarray]],
    anchor_idx: int,
    contact_prefix_6a: np.ndarray,
    anchor_rank: int,
    total_anchors: int,
    ligand_stats: Sequence[Dict[str, Any]],
    task_meta: Dict[str, Any],
    cfg: Step3Config,
) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []

    n = len(ligand_residues)
    window_counter = 0
    raw_enumerated = 0
    passed_contact_filter = 0

    for length in range(cfg.min_len, cfg.max_len + 1):
        min_left = max(0, anchor_idx - length + 1)
        max_left = min(anchor_idx, n - length)
        for left_idx in range(min_left, max_left + 1):
            right_idx = left_idx + length - 1
            if not (left_idx <= anchor_idx <= right_idx):
                continue

            raw_enumerated += 1

            peptide_residues = [res for res, _ in ligand_residues[left_idx:right_idx + 1]]
            if not window_is_backbone_continuous(peptide_residues):
                continue

            keep, n_contact_res_6a, contact_ratio_6a = validate_window_by_contact(
                left_idx, right_idx, contact_prefix_6a, cfg
            )
            if not keep:
                continue

            passed_contact_filter += 1

            start_res = peptide_residues[0]
            end_res = peptide_residues[-1]

            center = 0.5 * (left_idx + right_idx)
            anchor_offset_to_center = float(anchor_idx - center)

            window_counter += 1
            windows.append({
                "candidate_id": str(uuid.uuid4()),
                "parent_task_id": task_meta["parent_task_id"],
                "pdb_id": task_meta["pdb_id"],
                "source_file": task_meta["source_file"],
                "assembly_id": task_meta["assembly_id"],
                "chain_pair_id": task_meta["chain_pair_id"],
                "direction": task_meta["direction"],
                "receptor_chain_id": task_meta["receptor_chain_id"],
                "peptide_source_chain_id": task_meta["peptide_source_chain_id"],
                "method": cfg.method_name,

                "anchor_min_distance": float(ligand_stats[anchor_idx]["min_dist"]),
                "anchor_peptide_res_index": int(anchor_idx),
                "anchor_receptor_res_index": int(ligand_stats[anchor_idx]["nearest_rec_res_idx"]),

                "grown_left_index": int(left_idx),
                "grown_right_index": int(right_idx),
                "final_left_index": int(left_idx),
                "final_right_index": int(right_idx),
                "hit_left_growth_cap": bool(left_idx == 0),
                "hit_right_growth_cap": bool(right_idx == n - 1),

                "peptide_length": int(length),
                "selected_window_len": int(length),
                "peptide_start_resseq": int(residue_seqid_num(start_res)),
                "peptide_start_icode": residue_seqid_icode(start_res),
                "peptide_start_resname": normalize_residue_name(start_res.name),
                "peptide_end_resseq": int(residue_seqid_num(end_res)),
                "peptide_end_icode": residue_seqid_icode(end_res),
                "peptide_end_resname": normalize_residue_name(end_res.name),

                "method_a_anchor_offset_to_center": anchor_offset_to_center,
                "method_a_window_index": int(window_counter),
                "method_a_total_windows_for_anchor": 0,  # fill later
                "method_a_anchor_rank": int(anchor_rank),
                "method_a_total_anchors_for_task": int(total_anchors),
                "method_a_anchor_nms_group": int(anchor_rank),

                # step3-side screening metadata
                "step3_n_contact_residues_6A": int(n_contact_res_6a),
                "step3_contact_ratio_6A": float(contact_ratio_6a),
                "step3_raw_windows_for_anchor": 0,                  # fill later
                "step3_passed_contact_filter_for_anchor": 0,        # fill later
            })

    total_windows = len(windows)

    if cfg.max_windows_per_anchor > 0 and total_windows > cfg.max_windows_per_anchor:
        windows.sort(
            key=lambda x: (
                -x["step3_contact_ratio_6A"],
                -x["step3_n_contact_residues_6A"],
                abs(x["method_a_anchor_offset_to_center"]),
                -min(x["peptide_length"], 14),
            )
        )
        windows = windows[: cfg.max_windows_per_anchor]
        total_windows = len(windows)

    for i, item in enumerate(windows, start=1):
        item["method_a_window_index"] = i
        item["method_a_total_windows_for_anchor"] = total_windows
        item["step3_raw_windows_for_anchor"] = raw_enumerated
        item["step3_passed_contact_filter_for_anchor"] = passed_contact_filter

    return windows


def make_parent_task_id(pdb_id: str, chain_pair_id: str, direction: str) -> str:
    base = f"{pdb_id}::{chain_pair_id}::{direction}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))


def generate_candidates_for_task(
    cif_path: Path,
    pdb_id: str,
    receptor_chain_id: str,
    peptide_chain_id: str,
    cfg: Step3Config,
) -> List[Dict[str, Any]]:
    st = load_structure(cif_path)
    model = get_first_model(st)

    receptor_chain = find_chain_by_name(model, receptor_chain_id)
    peptide_chain = find_chain_by_name(model, peptide_chain_id)

    receptor_residues = chain_residues_with_alignment(receptor_chain)
    ligand_residues = chain_residues_with_alignment(peptide_chain)

    if len(receptor_residues) == 0 or len(ligand_residues) == 0:
        return []

    receptor_tree, atom_to_res_idx = build_receptor_kdtree(receptor_residues)
    if receptor_tree is None:
        return []

    ligand_stats = peptide_min_distance_to_receptor(
        ligand_residues,
        receptor_tree,
        atom_to_res_idx,
        cfg.contact_cutoff,
    )
    anchor_indices = pick_anchor_indices(ligand_stats, cfg)
    if not anchor_indices:
        return []

    contact_flags_6a = residue_contact_flags(ligand_residues, receptor_tree, cfg.contact_cutoff)
    contact_prefix_6a = prefix_sum_bool(contact_flags_6a)

    chain_pair_id = f"{receptor_chain_id}__{peptide_chain_id}"
    direction = f"{peptide_chain_id}_as_peptide__{receptor_chain_id}_as_receptor"
    parent_task_id = make_parent_task_id(pdb_id, chain_pair_id, direction)

    task_meta = {
        "parent_task_id": parent_task_id,
        "pdb_id": pdb_id,
        "source_file": make_source_file(cif_path),
        "assembly_id": cfg.assembly_id,
        "chain_pair_id": chain_pair_id,
        "direction": direction,
        "receptor_chain_id": receptor_chain_id,
        "peptide_source_chain_id": peptide_chain_id,
    }

    all_windows: List[Dict[str, Any]] = []
    total_anchors = len(anchor_indices)
    for anchor_rank, anchor_idx in enumerate(anchor_indices, start=1):
        windows = enumerate_windows_for_anchor(
            ligand_residues=ligand_residues,
            anchor_idx=anchor_idx,
            contact_prefix_6a=contact_prefix_6a,
            anchor_rank=anchor_rank,
            total_anchors=total_anchors,
            ligand_stats=ligand_stats,
            task_meta=task_meta,
            cfg=cfg,
        )
        all_windows.extend(windows)

    return all_windows


# =========================================================
# Multiprocessing Engine
# =========================================================
def init_worker(cfg: Step3Config) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def worker(payload: Tuple[Dict[str, Any], str]) -> Dict[str, Any]:
    task, cif_dir_str = payload
    cif_dir = Path(cif_dir_str)

    try:
        if _WORKER_CFG is None:
            raise RuntimeError("Worker config is not initialized.")

        cif_path = cif_dir / task["source_file"]
        candidates = generate_candidates_for_task(
            cif_path=cif_path,
            pdb_id=task["pdb_id"],
            receptor_chain_id=task["receptor_chain_id"],
            peptide_chain_id=task["peptide_source_chain_id"],
            cfg=_WORKER_CFG,
        )
        return {
            "ok": True,
            "task": task,
            "candidates": candidates,
            "error": None
        }
    except Exception as e:
        return {
            "ok": False,
            "task": task,
            "candidates": [],
            "error": {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(limit=3),
            }
        }


# =========================================================
# Task loading
# =========================================================
def load_tasks_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            tasks.append(json.loads(s))
    return tasks


# =========================================================
# Main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step-3 v3: anchor-contact-filtered candidate generation"
    )
    parser.add_argument("--cif_dir", type=str, required=True, help="Directory containing structure files (.cif)")
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--error_jsonl", type=str, default="")
    parser.add_argument(
        "--task_jsonl",
        type=str,
        required=True,
        help="Explicit task list JSONL; each row needs pdb_id, source_file, receptor_chain_id, peptide_source_chain_id",
    )
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--chunksize", type=int, default=20)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--anchor_cutoff", type=float, default=5.0)
    parser.add_argument("--contact_cutoff", type=float, default=6.0)
    parser.add_argument("--nms_min_seq_gap", type=int, default=3)
    parser.add_argument("--max_anchors_per_task", type=int, default=5)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--min_contact_residues_6a", type=int, default=4)
    parser.add_argument("--min_contact_ratio_6a", type=float, default=0.50)
    parser.add_argument("--max_windows_per_anchor", type=int, default=24)
    parser.add_argument("--limit_tasks", type=int, default=0)
    args = parser.parse_args()

    cfg = Step3Config(
        anchor_cutoff=args.anchor_cutoff,
        contact_cutoff=args.contact_cutoff,
        nms_min_seq_gap=args.nms_min_seq_gap,
        max_anchors_per_task=args.max_anchors_per_task,
        min_len=args.min_len,
        max_len=args.max_len,
        min_contact_residues_6a=args.min_contact_residues_6a,
        min_contact_ratio_6a=args.min_contact_ratio_6a,
        max_windows_per_anchor=args.max_windows_per_anchor,
    )

    cif_dir = Path(args.cif_dir)
    output_jsonl = Path(args.output_jsonl)
    error_jsonl = Path(args.error_jsonl) if args.error_jsonl else None
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if error_jsonl is not None:
        error_jsonl.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks_from_jsonl(Path(args.task_jsonl))

    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]

    pid = os.getpid()

    print("=" * 80, flush=True)
    print("[START] Step-3 v3 anchor-contact-filtered (Parallel Edition)", flush=True)
    print(f"[START] PID                          = {pid}", flush=True)
    print(f"[START] cif_dir                      = {cif_dir}", flush=True)
    print(f"[START] output_jsonl                 = {output_jsonl}", flush=True)
    print(f"[START] error_jsonl                  = {error_jsonl}", flush=True)
    print(f"[START] task_jsonl                   = {args.task_jsonl}", flush=True)
    print(f"[START] num_tasks                    = {len(tasks)}", flush=True)
    print(f"[START] workers                      = {args.workers}", flush=True)
    print(f"[START] chunksize                    = {args.chunksize}", flush=True)
    print(f"[START] anchor_cutoff                = {cfg.anchor_cutoff}", flush=True)
    print(f"[START] contact_cutoff               = {cfg.contact_cutoff}", flush=True)
    print(f"[START] nms_min_seq_gap              = {cfg.nms_min_seq_gap}", flush=True)
    print(f"[START] max_anchors_per_task         = {cfg.max_anchors_per_task}", flush=True)
    print(f"[START] min_len                      = {cfg.min_len}", flush=True)
    print(f"[START] max_len                      = {cfg.max_len}", flush=True)
    print(f"[START] min_contact_residues_6a      = {cfg.min_contact_residues_6a}", flush=True)
    print(f"[START] min_contact_ratio_6a         = {cfg.min_contact_ratio_6a}", flush=True)
    print(f"[START] max_windows_per_anchor       = {cfg.max_windows_per_anchor}", flush=True)
    print("=" * 80, flush=True)

    total_tasks = 0
    total_candidates = 0
    error_count = 0
    start_time = time.time()

    with output_jsonl.open("w", encoding="utf-8") as fout:
        ferr = error_jsonl.open("w", encoding="utf-8") if error_jsonl is not None else None
        try:
            payloads = ((task, str(cif_dir)) for task in tasks)

            with mp.Pool(
                processes=args.workers,
                initializer=init_worker,
                initargs=(cfg,)
            ) as pool:
                for result in pool.imap_unordered(worker, payloads, chunksize=args.chunksize):
                    total_tasks += 1

                    if result["ok"]:
                        cands = result["candidates"]
                        for item in cands:
                            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                        total_candidates += len(cands)
                    else:
                        error_count += 1
                        if ferr is not None:
                            err_data = {
                                "task": result["task"],
                                **result["error"]
                            }
                            ferr.write(json.dumps(err_data, ensure_ascii=False) + "\n")

                    if total_tasks % args.progress_every == 0:
                        elapsed = time.time() - start_time
                        speed = total_tasks / elapsed if elapsed > 0 else 0.0
                        print(
                            f"[PROGRESS] tasks={total_tasks}/{len(tasks)} | candidates={total_candidates} | "
                            f"errors={error_count} | elapsed={elapsed/60:.1f} min | speed={speed:.2f} tasks/s",
                            flush=True,
                        )
        finally:
            if ferr is not None:
                ferr.close()

    elapsed = time.time() - start_time
    print("=" * 80, flush=True)
    print("[DONE] Step-3 v3 finished.", flush=True)
    print(f"[DONE] Processed tasks      : {total_tasks}", flush=True)
    print(f"[DONE] Total candidates     : {total_candidates}", flush=True)
    print(f"[DONE] Error tasks          : {error_count}", flush=True)
    print(f"[DONE] Elapsed time         : {elapsed/60:.2f} min", flush=True)
    print(f"[DONE] Output JSONL         : {output_jsonl}", flush=True)
    if error_jsonl is not None:
        print(f"[DONE] Error JSONL          : {error_jsonl}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
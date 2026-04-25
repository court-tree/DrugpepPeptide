from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import (
    build_receptor_tree,
    chain_residues,
    compute_window_contact_stats,
    find_chain_by_name,
    get_model,
    load_structure,
    residue_id_string,
    residue_seqid_icode,
    residue_seqid_num,
    window_is_continuous,
    write_json,
)


class Step3Config:
    def __init__(
        self,
        anchor_cutoff: float = 6.0,
        contact_cutoff: float = 6.0,
        min_len: int = 8,
        max_len: int = 20,
        anchor_local_radius: int = 3,
        anchor_nms_gap: int = 2,
        min_anchor_contact_count: int = 2,
        max_anchors_per_task: int = 3,
        max_candidates_per_task: int = 16,
        max_candidates_len_8_10: int = 6,
        max_candidates_len_11_14: int = 6,
        max_candidates_len_15_20: int = 4,
    ) -> None:
        self.anchor_cutoff = anchor_cutoff
        self.contact_cutoff = contact_cutoff
        self.min_len = min_len
        self.max_len = max_len
        self.anchor_local_radius = anchor_local_radius
        self.anchor_nms_gap = anchor_nms_gap
        self.min_anchor_contact_count = min_anchor_contact_count
        self.max_anchors_per_task = max_anchors_per_task
        self.max_candidates_per_task = max_candidates_per_task
        self.max_candidates_len_8_10 = max_candidates_len_8_10
        self.max_candidates_len_11_14 = max_candidates_len_11_14
        self.max_candidates_len_15_20 = max_candidates_len_15_20


_WORKER_CFG: Optional[Step3Config] = None


def init_worker(cfg: Step3Config) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def collect_anchor_seed_indices(task: Dict[str, Any], peptide_residues, receptor_tree, cutoff: float) -> List[int]:
    residue_id_to_idx = {
        residue_id_string(item.residue): idx
        for idx, item in enumerate(peptide_residues)
    }

    source_contact_residue_ids = task.get("source_contact_residue_ids") or []
    task_seed_indices = [
        residue_id_to_idx[res_id]
        for res_id in source_contact_residue_ids
        if res_id in residue_id_to_idx
    ]
    if task_seed_indices:
        return sorted(set(task_seed_indices))

    anchors: List[int] = []
    for idx, item in enumerate(peptide_residues):
        if len(item.coords) == 0:
            continue
        dists, _ = receptor_tree.query(item.coords, k=1)
        if float(dists.min()) <= cutoff:
            anchors.append(idx)
    return anchors


def residue_direct_contact_stats(peptide_item, receptor_tree, atom_to_res_idx, cutoff: float) -> Dict[str, Any]:
    if len(peptide_item.coords) == 0:
        return {
            "direct_contact_count": 0,
            "direct_min_distance": None,
        }

    dists, _ = receptor_tree.query(peptide_item.coords, k=1)
    direct_min_distance = float(dists.min())
    neighbors = receptor_tree.query_ball_point(peptide_item.coords, r=cutoff)
    receptor_residue_ids = {
        int(atom_to_res_idx[atom_idx])
        for atom_neighbors in neighbors
        for atom_idx in atom_neighbors
    }
    return {
        "direct_contact_count": len(receptor_residue_ids),
        "direct_min_distance": direct_min_distance,
    }


def rank_anchor_indices(peptide_residues, receptor_tree, atom_to_res_idx, seed_indices: List[int], cfg: Step3Config) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    n = len(peptide_residues)
    for anchor_idx in seed_indices:
        direct_stats = residue_direct_contact_stats(
            peptide_residues[anchor_idx],
            receptor_tree,
            atom_to_res_idx,
            cfg.contact_cutoff,
        )
        if int(direct_stats["direct_contact_count"]) < cfg.min_anchor_contact_count:
            continue

        left_idx = max(0, anchor_idx - cfg.anchor_local_radius)
        right_idx = min(n - 1, anchor_idx + cfg.anchor_local_radius)
        window = peptide_residues[left_idx:right_idx + 1]
        if not window or not window_is_continuous(window):
            continue
        stats = compute_window_contact_stats(window, receptor_tree, atom_to_res_idx, cfg.contact_cutoff)
        ranked.append(
            {
                "anchor_idx": anchor_idx,
                "local_left_index": left_idx,
                "local_right_index": right_idx,
                "local_avg_contact_count": stats["avg_contact_count"],
                "local_contact_coverage": stats["contact_coverage"],
                "local_longest_contact_run": stats["longest_contact_run"],
                "anchor_direct_contact_count": direct_stats["direct_contact_count"],
                "anchor_direct_min_distance": direct_stats["direct_min_distance"],
            }
        )

    # A hotspot anchor should be strong at the anchor residue itself, not only
    # embedded in a good surrounding window. We therefore rank first by direct
    # receptor-residue contacts, then use the local window average as tie-breaker.
    ranked.sort(
        key=lambda x: (
            int(x["anchor_direct_contact_count"]),
            float(x["local_avg_contact_count"]),
        ),
        reverse=True,
    )
    return ranked


def select_anchor_indices(ranked_anchors: List[Dict[str, Any]], cfg: Step3Config) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for anchor in ranked_anchors:
        anchor_idx = int(anchor["anchor_idx"])
        if any(abs(anchor_idx - int(prev["anchor_idx"])) <= cfg.anchor_nms_gap for prev in selected):
            continue
        selected.append(anchor)
        if len(selected) >= cfg.max_anchors_per_task:
            break
    return selected


def better_candidate(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return float(a["avg_contact_count"]) > float(b["avg_contact_count"])


def candidate_length_band(peptide_length: int) -> str:
    if peptide_length <= 10:
        return "8_10"
    if peptide_length <= 14:
        return "11_14"
    return "15_20"


def select_candidates_by_length_band(
    candidates: List[Dict[str, Any]],
    cfg: Step3Config,
) -> List[Dict[str, Any]]:
    """Keep high-contact candidates while reserving slots for longer windows.

    Pure top-N by avg_contact_count tends to favor short windows, because short
    peptides can achieve high averages with a compact contact core. We first
    allocate a small quota to each length band, then backfill any unused slots
    with the best remaining candidates. This keeps total output bounded while
    giving 15-20 aa windows a chance to enter Step4.
    """
    quotas = {
        "8_10": cfg.max_candidates_len_8_10,
        "11_14": cfg.max_candidates_len_11_14,
        "15_20": cfg.max_candidates_len_15_20,
    }
    by_band: Dict[str, List[Dict[str, Any]]] = {band: [] for band in quotas}
    remaining: List[Dict[str, Any]] = []

    for candidate in candidates:
        band = candidate_length_band(int(candidate["peptide_length"]))
        candidate["step3_length_band"] = band
        if band in by_band:
            by_band[band].append(candidate)
        else:
            remaining.append(candidate)

    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    for band in ("8_10", "11_14", "15_20"):
        band_candidates = sorted(by_band[band], key=lambda x: float(x["avg_contact_count"]), reverse=True)
        for candidate in band_candidates[:max(0, quotas[band])]:
            if candidate["candidate_id"] in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate["candidate_id"])

    # If a band has fewer candidates than its quota, do not waste those slots.
    # Backfill by avg_contact_count from all unselected candidates, preserving
    # the same total max_candidates_per_task budget.
    backfill_pool = [
        candidate
        for candidate in sorted(candidates + remaining, key=lambda x: float(x["avg_contact_count"]), reverse=True)
        if candidate["candidate_id"] not in selected_ids
    ]
    for candidate in backfill_pool:
        if len(selected) >= cfg.max_candidates_per_task:
            break
        selected.append(candidate)
        selected_ids.add(candidate["candidate_id"])

    selected.sort(key=lambda x: float(x["avg_contact_count"]), reverse=True)
    return selected[:cfg.max_candidates_per_task]


def enumerate_windows(task: Dict[str, Any], pdb_dir: Path, cfg: Step3Config) -> List[Dict[str, Any]]:
    structure = load_structure(pdb_dir / task["source_file"])
    model = get_model(structure)
    receptor_chain = find_chain_by_name(model, task["receptor_chain_id"])
    peptide_chain = find_chain_by_name(model, task["peptide_source_chain_id"])
    receptor_residues = chain_residues(receptor_chain)
    peptide_residues = chain_residues(peptide_chain)
    receptor_tree, atom_to_res_idx = build_receptor_tree(receptor_residues)
    if receptor_tree is None:
        return []

    seed_indices = collect_anchor_seed_indices(task, peptide_residues, receptor_tree, cfg.anchor_cutoff)
    ranked_anchors = rank_anchor_indices(peptide_residues, receptor_tree, atom_to_res_idx, seed_indices, cfg)
    selected_anchors = select_anchor_indices(ranked_anchors, cfg)
    candidates_by_bounds: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for anchor_info in selected_anchors:
        anchor_idx = int(anchor_info["anchor_idx"])
        n = len(peptide_residues)
        for length in range(cfg.min_len, cfg.max_len + 1):
            min_left = max(0, anchor_idx - length + 1)
            max_left = min(anchor_idx, n - length)
            for left_idx in range(min_left, max_left + 1):
                right_idx = left_idx + length - 1
                window = peptide_residues[left_idx:right_idx + 1]
                if not window_is_continuous(window):
                    continue
                stats = compute_window_contact_stats(window, receptor_tree, atom_to_res_idx, cfg.contact_cutoff)
                start_res = window[0].residue
                end_res = window[-1].residue
                candidate = {
                    "candidate_id": str(uuid.uuid4()),
                    "parent_task_id": task["task_id"],
                    "pdb_id": task["pdb_id"],
                    "source_file": task["source_file"],
                    "assembly_id": task.get("assembly_id", "native_file"),
                    "chain_pair_id": task["chain_pair_id"],
                    "direction": task["direction"],
                    "receptor_chain_id": task["receptor_chain_id"],
                    "peptide_source_chain_id": task["peptide_source_chain_id"],
                    "method": "window_avg_contact_phase1_v2",
                    "anchor_peptide_res_index": anchor_idx,
                    "anchor_receptor_res_index": -1,
                    "final_left_index": left_idx,
                    "final_right_index": right_idx,
                    "peptide_length": len(window),
                    "peptide_start_resseq": residue_seqid_num(start_res),
                    "peptide_start_icode": residue_seqid_icode(start_res),
                    "peptide_end_resseq": residue_seqid_num(end_res),
                    "peptide_end_icode": residue_seqid_icode(end_res),
                    "peptide_residue_ids": [residue_id_string(x.residue) for x in window],
                    "avg_contact_count": stats["avg_contact_count"],
                    "total_contact_count": stats["total_contact_count"],
                    "contact_coverage": stats["contact_coverage"],
                    "longest_contact_run": stats["longest_contact_run"],
                    "anchor_local_avg_contact_count": anchor_info["local_avg_contact_count"],
                    "anchor_local_contact_coverage": anchor_info["local_contact_coverage"],
                    "anchor_local_longest_contact_run": anchor_info["local_longest_contact_run"],
                    "anchor_direct_contact_count": anchor_info["anchor_direct_contact_count"],
                    "anchor_direct_min_distance": anchor_info["anchor_direct_min_distance"],
                }
                bounds_key = (left_idx, right_idx)
                prev = candidates_by_bounds.get(bounds_key)
                if prev is None or better_candidate(candidate, prev):
                    candidates_by_bounds[bounds_key] = candidate

    candidates = list(candidates_by_bounds.values())
    candidates.sort(key=lambda x: float(x["avg_contact_count"]), reverse=True)
    return select_candidates_by_length_band(candidates, cfg)


def worker(payload: Tuple[Dict[str, Any], str]) -> Dict[str, Any]:
    task, pdb_dir_str = payload
    try:
        if _WORKER_CFG is None:
            raise RuntimeError("Worker config is not initialized.")
        rows = enumerate_windows(task, Path(pdb_dir_str), _WORKER_CFG)
        return {"ok": True, "task_id": task.get("task_id", ""), "rows": rows, "error": None}
    except Exception as e:
        return {
            "ok": False,
            "task_id": task.get("task_id", ""),
            "rows": [],
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(limit=3),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase1 Step3: local-window candidate generation")
    parser.add_argument("--tasks_jsonl", required=True)
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--error_jsonl", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--anchor_cutoff", type=float, default=6.0)
    parser.add_argument("--contact_cutoff", type=float, default=6.0)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--anchor_local_radius", type=int, default=3)
    parser.add_argument("--anchor_nms_gap", type=int, default=2)
    parser.add_argument("--min_anchor_contact_count", type=int, default=2)
    parser.add_argument("--max_anchors_per_task", type=int, default=3)
    parser.add_argument("--max_candidates_per_task", type=int, default=16)
    parser.add_argument("--max_candidates_len_8_10", type=int, default=6)
    parser.add_argument("--max_candidates_len_11_14", type=int, default=6)
    parser.add_argument("--max_candidates_len_15_20", type=int, default=4)
    args = parser.parse_args()

    cfg = Step3Config(
        anchor_cutoff=args.anchor_cutoff,
        contact_cutoff=args.contact_cutoff,
        min_len=args.min_len,
        max_len=args.max_len,
        anchor_local_radius=args.anchor_local_radius,
        anchor_nms_gap=args.anchor_nms_gap,
        min_anchor_contact_count=args.min_anchor_contact_count,
        max_anchors_per_task=args.max_anchors_per_task,
        max_candidates_per_task=args.max_candidates_per_task,
        max_candidates_len_8_10=args.max_candidates_len_8_10,
        max_candidates_len_11_14=args.max_candidates_len_11_14,
        max_candidates_len_15_20=args.max_candidates_len_15_20,
    )

    tasks: List[Dict[str, Any]] = []
    with Path(args.tasks_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                tasks.append(json.loads(s))

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = Path(args.error_jsonl) if args.error_jsonl else None
    if error_path is not None:
        error_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    n_written = 0
    n_errors = 0
    task_candidate_counts: List[int] = []
    f_err = error_path.open("w", encoding="utf-8") if error_path is not None else None
    try:
        with output_path.open("w", encoding="utf-8") as f_out:
            if args.workers <= 1:
                init_worker(cfg)
                for task in tasks:
                    result = worker((task, args.pdb_dir))
                    if result["ok"]:
                        task_candidate_counts.append(len(result["rows"]))
                        for row in result["rows"]:
                            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                            n_written += 1
                    else:
                        n_errors += 1
                        if f_err is not None:
                            f_err.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(cfg,)) as pool:
                    payloads = ((task, args.pdb_dir) for task in tasks)
                    for result in pool.imap_unordered(worker, payloads, chunksize=8):
                        if result["ok"]:
                            task_candidate_counts.append(len(result["rows"]))
                            for row in result["rows"]:
                                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                                n_written += 1
                        else:
                            n_errors += 1
                            if f_err is not None:
                                f_err.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        if f_err is not None:
            f_err.close()

    write_json(
        output_path.with_name("step3_summary.json"),
        {
            "tasks": len(tasks),
            "candidates_written": n_written,
            "errors": n_errors,
            "elapsed_sec": round(time.time() - start, 3),
            "algorithm": "local_window_avg_contact_anchor_ranked_length_band_retention",
            "length_band_retention": {
                "8_10": cfg.max_candidates_len_8_10,
                "11_14": cfg.max_candidates_len_11_14,
                "15_20": cfg.max_candidates_len_15_20,
                "max_candidates_per_task": cfg.max_candidates_per_task,
                "backfill_by": "avg_contact_count",
            },
            "tasks_with_zero_candidates": sum(1 for x in task_candidate_counts if x == 0),
            "avg_candidates_per_task": (sum(task_candidate_counts) / len(task_candidate_counts)) if task_candidate_counts else 0.0,
            "max_candidates_single_task": max(task_candidate_counts) if task_candidate_counts else 0,
        },
    )


if __name__ == "__main__":
    main()

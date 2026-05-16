from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        max_anchors_per_task: int = 6,
        max_candidates_per_task: int = 24,
        min_longest_contact_run: int = 4,
        quality_top_k: int = 16,
        length_band_mid_min_keep: int = 4,
        length_band_long_min_keep: int = 4,
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
        self.min_longest_contact_run = min_longest_contact_run
        self.quality_top_k = quality_top_k
        self.length_band_mid_min_keep = length_band_mid_min_keep
        self.length_band_long_min_keep = length_band_long_min_keep


LengthBand = Tuple[str, int, int]


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

    # Rank anchors by local window contact quality first, then by the anchor
    # residue's direct receptor-residue contacts.
    ranked.sort(
        key=lambda x: (
            float(x["local_avg_contact_count"]),
            int(x["anchor_direct_contact_count"]),
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


def length_band_name(peptide_length: int, bands: Sequence[LengthBand]) -> Optional[str]:
    for name, min_len, max_len in bands:
        if min_len <= peptide_length <= max_len:
            return name
    return None


def length_bands_from_config(cfg: Step3Config) -> List[LengthBand]:
    return [("short_8_10", 8, 10), ("mid_11_14", 11, 14), ("long_15_20", 15, 20)]


def length_band_min_keep(cfg: Step3Config) -> Dict[str, int]:
    return {
        "mid_11_14": cfg.length_band_mid_min_keep,
        "long_15_20": cfg.length_band_long_min_keep,
    }


def select_candidates_with_length_band_retention(
    candidates: List[Dict[str, Any]],
    cfg: Step3Config,
) -> List[Dict[str, Any]]:
    sorted_candidates = sorted(
        candidates,
        key=lambda x: float(x["avg_contact_count"]),
        reverse=True,
    )
    selected: List[Dict[str, Any]] = []
    selected_ids = set()

    for row in sorted_candidates[:cfg.quality_top_k]:
        if len(selected) >= cfg.max_candidates_per_task:
            break
        row["step3_selection_stage"] = "quality_top_k"
        selected.append(row)
        selected_ids.add(row["candidate_id"])

    for band_name, min_keep in length_band_min_keep(cfg).items():
        if min_keep <= 0 or len(selected) >= cfg.max_candidates_per_task:
            continue
        current_count = sum(1 for x in selected if x.get("length_band") == band_name)
        needed = max(0, min_keep - current_count)
        if needed == 0:
            continue
        bucket = [
            x for x in sorted_candidates
            if x["candidate_id"] not in selected_ids
            and x.get("length_band") == band_name
        ]
        for row in bucket[:needed]:
            row["step3_selection_stage"] = "length_band_min_keep"
            selected.append(row)
            selected_ids.add(row["candidate_id"])
            if len(selected) >= cfg.max_candidates_per_task:
                break

    for row in sorted_candidates:
        if len(selected) >= cfg.max_candidates_per_task:
            break
        if row["candidate_id"] in selected_ids:
            continue
        row["step3_selection_stage"] = "avg_contact_backfill"
        selected.append(row)
        selected_ids.add(row["candidate_id"])

    selected.sort(key=lambda x: float(x["avg_contact_count"]), reverse=True)
    return selected


def centered_window_bound_candidates(anchor_idx: int, length: int, n_residues: int) -> List[Tuple[int, int]]:
    """Generate a few near-symmetric windows by expanding around the anchor.

    The old logic enumerated every window containing the anchor. That made the
    candidate set dense and shifted the method closer to "anchor-conditioned
    exhaustive sliding windows". The updated logic is stricter: for each target
    length we start from the anchor as the center and only consider the few
    windows that are closest to symmetric expansion around that anchor.
    """
    if length <= 0 or n_residues <= 0 or anchor_idx < 0 or anchor_idx >= n_residues:
        return []

    # For odd lengths there is one perfectly centered pattern.
    # For even lengths there are two near-centered patterns: left-heavier or
    # right-heavier by one residue. Near chain ends, windows are shifted back
    # into valid bounds.
    left_options = {max(0, (length - 1) // 2), max(0, length // 2)}
    candidates: List[Tuple[int, int]] = []
    seen = set()

    for left_span in sorted(left_options):
        right_span = length - 1 - left_span
        left_idx = anchor_idx - left_span
        right_idx = anchor_idx + right_span

        if left_idx < 0:
            right_idx += -left_idx
            left_idx = 0
        if right_idx >= n_residues:
            left_idx -= right_idx - (n_residues - 1)
            right_idx = n_residues - 1

        left_idx = max(0, left_idx)
        right_idx = min(n_residues - 1, right_idx)

        if right_idx - left_idx + 1 != length:
            continue

        bounds = (left_idx, right_idx)
        if bounds not in seen:
            seen.add(bounds)
            candidates.append(bounds)

    return candidates


def best_centered_window_for_length(
    anchor_idx: int,
    length: int,
    peptide_residues,
    receptor_tree,
    atom_to_res_idx,
    cfg: Step3Config,
) -> Optional[Dict[str, Any]]:
    """Pick the best near-centered window for one anchor/length pair.

    Even after moving to center-out expansion, even lengths may allow two
    almost-symmetric windows. We score that tiny candidate set and keep only the
    better one, instead of exhaustively enumerating every anchor-containing
    window across the whole chain.
    """
    n = len(peptide_residues)
    best: Optional[Dict[str, Any]] = None

    for left_idx, right_idx in centered_window_bound_candidates(anchor_idx, length, n):
        window = peptide_residues[left_idx:right_idx + 1]
        if not window_is_continuous(window):
            continue
        stats = compute_window_contact_stats(window, receptor_tree, atom_to_res_idx, cfg.contact_cutoff)
        candidate = {
            "final_left_index": left_idx,
            "final_right_index": right_idx,
            "peptide_length": len(window),
            "avg_contact_count": stats["avg_contact_count"],
            "total_contact_count": stats["total_contact_count"],
            "contact_coverage": stats["contact_coverage"],
            "longest_contact_run": stats["longest_contact_run"],
            "peptide_residue_ids": [residue_id_string(x.residue) for x in window],
            "start_res": window[0].residue,
            "end_res": window[-1].residue,
        }
        if best is None or better_candidate(candidate, best):
            best = candidate
    return best


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
        for length in range(cfg.min_len, cfg.max_len + 1):
            window_candidate = best_centered_window_for_length(
                anchor_idx=anchor_idx,
                length=length,
                peptide_residues=peptide_residues,
                receptor_tree=receptor_tree,
                atom_to_res_idx=atom_to_res_idx,
                cfg=cfg,
            )
            if window_candidate is None:
                continue

            start_res = window_candidate["start_res"]
            end_res = window_candidate["end_res"]
            left_idx = int(window_candidate["final_left_index"])
            right_idx = int(window_candidate["final_right_index"])
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
                "method": "center_out_hotspot_growth_phase1_v1",
                "anchor_peptide_res_index": anchor_idx,
                "anchor_receptor_res_index": -1,
                "final_left_index": left_idx,
                "final_right_index": right_idx,
                "peptide_length": int(window_candidate["peptide_length"]),
                "length_band": length_band_name(
                    int(window_candidate["peptide_length"]),
                    length_bands_from_config(cfg),
                ),
                "peptide_start_resseq": residue_seqid_num(start_res),
                "peptide_start_icode": residue_seqid_icode(start_res),
                "peptide_end_resseq": residue_seqid_num(end_res),
                "peptide_end_icode": residue_seqid_icode(end_res),
                "peptide_residue_ids": list(window_candidate["peptide_residue_ids"]),
                "avg_contact_count": window_candidate["avg_contact_count"],
                "total_contact_count": window_candidate["total_contact_count"],
                "contact_coverage": window_candidate["contact_coverage"],
                "longest_contact_run": window_candidate["longest_contact_run"],
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
    return select_candidates_with_length_band_retention(candidates, cfg)


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
    parser.add_argument("--max_anchors_per_task", type=int, default=6)
    parser.add_argument("--max_candidates_per_task", type=int, default=24)
    parser.add_argument("--min_longest_contact_run", type=int, default=4)
    parser.add_argument("--quality_top_k", type=int, default=16)
    parser.add_argument("--length_band_mid_min_keep", type=int, default=4)
    parser.add_argument("--length_band_long_min_keep", type=int, default=4)
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
        min_longest_contact_run=args.min_longest_contact_run,
        quality_top_k=args.quality_top_k,
        length_band_mid_min_keep=args.length_band_mid_min_keep,
        length_band_long_min_keep=args.length_band_long_min_keep,
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
            "algorithm": "center_out_hotspot_growth_topk_avg_contact",
            "score_basis": "avg_contact_count",
            "candidate_selection": "quality_top_k_then_length_band_min_keep_then_avg_contact_backfill",
            "quality_top_k": cfg.quality_top_k,
            "length_bands": [
                {"name": name, "min_len": min_len, "max_len": max_len}
                for name, min_len, max_len in length_bands_from_config(cfg)
            ],
            "length_band_min_keep": length_band_min_keep(cfg),
            "step3_window_gate": "none",
            "min_longest_contact_run": cfg.min_longest_contact_run,
            "min_longest_contact_run_applied": False,
            "tasks_with_zero_candidates": sum(1 for x in task_candidate_counts if x == 0),
            "avg_candidates_per_task": (sum(task_candidate_counts) / len(task_candidate_counts)) if task_candidate_counts else 0.0,
            "max_candidates_single_task": max(task_candidate_counts) if task_candidate_counts else 0,
        },
    )


if __name__ == "__main__":
    main()

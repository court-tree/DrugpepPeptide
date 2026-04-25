from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from common import (
    build_receptor_tree,
    chain_residues,
    compute_window_contact_stats,
    find_chain_by_name,
    get_model,
    load_structure,
    window_bounds_match_candidate,
    window_is_continuous,
    write_json,
)


class Step4Config:
    def __init__(
        self,
        contact_cutoff: float = 6.0,
        min_avg_contact_count: float = 3.5,
        min_contact_coverage_short: float = 0.5,
        min_contact_coverage_mid: float = 0.4,
        min_contact_coverage_long: float = 0.3,
    ) -> None:
        self.contact_cutoff = contact_cutoff
        self.min_avg_contact_count = min_avg_contact_count
        self.min_contact_coverage_short = min_contact_coverage_short
        self.min_contact_coverage_mid = min_contact_coverage_mid
        self.min_contact_coverage_long = min_contact_coverage_long


def min_contact_coverage_for_length(peptide_length: int, cfg: Step4Config) -> float:
    """Use a lighter coverage floor for longer peptides.

    A long peptide can contain a compact binding core plus flexible/flanking
    residues. Using one global coverage cutoff would penalize those cases
    twice: contact density is already averaged over length, and coverage is
    also diluted by length.
    """
    if peptide_length <= 10:
        return cfg.min_contact_coverage_short
    if peptide_length <= 14:
        return cfg.min_contact_coverage_mid
    return cfg.min_contact_coverage_long


_WORKER_CFG: Optional[Step4Config] = None


def init_worker(cfg: Step4Config) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def annotate_candidate(candidate: Dict[str, Any], pdb_dir: Path, cfg: Step4Config) -> Dict[str, Any]:
    structure = load_structure(pdb_dir / candidate["source_file"])
    model = get_model(structure)
    receptor_chain = find_chain_by_name(model, candidate["receptor_chain_id"])
    peptide_chain = find_chain_by_name(model, candidate["peptide_source_chain_id"])

    receptor_residues = chain_residues(receptor_chain)
    peptide_residues = chain_residues(peptide_chain)
    receptor_tree, atom_to_res_idx = build_receptor_tree(receptor_residues)
    if receptor_tree is None:
        raise ValueError("Empty receptor heavy-atom coordinates")

    left_idx = int(candidate["final_left_index"])
    right_idx = int(candidate["final_right_index"])
    peptide_window = peptide_residues[left_idx:right_idx + 1]
    if not window_bounds_match_candidate(candidate, peptide_window):
        raise ValueError("Window bounds do not match candidate residue ids")
    if not window_is_continuous(peptide_window):
        raise ValueError("Window backbone is not continuous")

    stats = compute_window_contact_stats(peptide_window, receptor_tree, atom_to_res_idx, cfg.contact_cutoff)
    peptide_length = len(peptide_window)
    min_contact_coverage = min_contact_coverage_for_length(peptide_length, cfg)
    physically_reasonable = (
        stats["avg_contact_count"] >= cfg.min_avg_contact_count
        and stats["contact_coverage"] >= min_contact_coverage
    )

    if stats["avg_contact_count"] < cfg.min_avg_contact_count:
        drop_reason = "low_avg_contact_count"
    elif stats["contact_coverage"] < min_contact_coverage:
        drop_reason = "low_contact_coverage"
    else:
        drop_reason = None

    out = dict(candidate)
    out.update(
        {
            "avg_contact_count": stats["avg_contact_count"],
            "total_contact_count": stats["total_contact_count"],
            "contact_coverage": stats["contact_coverage"],
            "longest_contact_run": stats["longest_contact_run"],
            "step4_min_contact_coverage_for_length": min_contact_coverage,
            "step4_passes_physical_sanity": physically_reasonable,
            "step4_drop_reason": drop_reason,
        }
    )
    return out


def worker(payload: Tuple[Dict[str, Any], str]) -> Dict[str, Any]:
    candidate, pdb_dir_str = payload
    try:
        if _WORKER_CFG is None:
            raise RuntimeError("Worker config is not initialized.")
        row = annotate_candidate(candidate, Path(pdb_dir_str), _WORKER_CFG)
        return {"ok": True, "row": row, "error": None}
    except Exception as e:
        return {
            "ok": False,
            "row": None,
            "error": {
                "candidate_id": candidate.get("candidate_id", ""),
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(limit=3),
            },
        }


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase1 Step4: per-candidate physical sanity filtering")
    parser.add_argument("--candidate_jsonl", required=True)
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--error_jsonl", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--contact_cutoff", type=float, default=6.0)
    parser.add_argument("--min_avg_contact_count", type=float, default=3.5)
    parser.add_argument(
        "--min_contact_coverage",
        type=float,
        default=None,
        help="Legacy uniform coverage floor. If set, applies the same cutoff to all lengths.",
    )
    parser.add_argument("--min_contact_coverage_short", type=float, default=0.5)
    parser.add_argument("--min_contact_coverage_mid", type=float, default=0.4)
    parser.add_argument("--min_contact_coverage_long", type=float, default=0.3)
    args = parser.parse_args()

    min_contact_coverage_short = args.min_contact_coverage_short
    min_contact_coverage_mid = args.min_contact_coverage_mid
    min_contact_coverage_long = args.min_contact_coverage_long
    if args.min_contact_coverage is not None:
        min_contact_coverage_short = args.min_contact_coverage
        min_contact_coverage_mid = args.min_contact_coverage
        min_contact_coverage_long = args.min_contact_coverage

    cfg = Step4Config(
        contact_cutoff=args.contact_cutoff,
        min_avg_contact_count=args.min_avg_contact_count,
        min_contact_coverage_short=min_contact_coverage_short,
        min_contact_coverage_mid=min_contact_coverage_mid,
        min_contact_coverage_long=min_contact_coverage_long,
    )

    start = time.time()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    error_path: Optional[Path] = None
    if args.error_jsonl:
        error_path = Path(args.error_jsonl)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text("", encoding="utf-8")

    annotated_count = 0
    passed_count = 0
    error_count = 0
    dropped_physical_count = 0

    def handle_result(result: Dict[str, Any]) -> None:
        nonlocal annotated_count, passed_count, error_count, dropped_physical_count
        if result["ok"]:
            row = result["row"]
            if row is None:
                error_count += 1
                return
            annotated_count += 1
            if row["step4_passes_physical_sanity"]:
                passed_count += 1
                append_jsonl(output_path, row)
            else:
                dropped_physical_count += 1
        else:
            error_count += 1
            if error_path is not None:
                append_jsonl(error_path, result["error"])

    candidate_iter = iter_jsonl(Path(args.candidate_jsonl))
    if args.workers <= 1:
        init_worker(cfg)
        for candidate in candidate_iter:
            handle_result(worker((candidate, args.pdb_dir)))
    else:
        with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(cfg,)) as pool:
            payloads = ((candidate, args.pdb_dir) for candidate in candidate_iter)
            for result in pool.imap_unordered(worker, payloads, chunksize=8):
                handle_result(result)

    input_count = annotated_count + error_count

    write_json(
        output_path.with_name("step4_summary.json"),
        {
            "input_candidates": input_count,
            "annotated_candidates": annotated_count,
            "passed_physical_sanity": passed_count,
            "kept_after_physical_filter": passed_count,
            "dropped_by_physical_filter": dropped_physical_count,
            "errors": error_count,
            "elapsed_sec": round(time.time() - start, 3),
            "physical_filter_basis": "avg_contact_count_and_length_adaptive_contact_coverage",
            "min_avg_contact_count": cfg.min_avg_contact_count,
            "length_adaptive_contact_coverage": {
                "8_10": cfg.min_contact_coverage_short,
                "11_14": cfg.min_contact_coverage_mid,
                "15_20": cfg.min_contact_coverage_long,
            },
            "legacy_uniform_min_contact_coverage": args.min_contact_coverage,
            "dedup_applied": False,
        },
    )


if __name__ == "__main__":
    main()

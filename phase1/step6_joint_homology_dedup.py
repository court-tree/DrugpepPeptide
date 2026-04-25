from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from common import (
    chain_residues,
    chain_sequence,
    find_chain_by_name,
    get_model,
    load_structure,
    sequence_identity_same_length,
    write_json,
)


def attach_sequences(rows: List[Dict[str, Any]], pdb_dir: Path) -> List[Dict[str, Any]]:
    cache: Dict[str, Dict[str, str]] = {}
    out: List[Dict[str, Any]] = []
    total = len(rows)
    for i, row in enumerate(rows, 1):
        key = f"{row['source_file']}|{row['receptor_chain_id']}|{row['peptide_source_chain_id']}"
        cached = cache.get(key)
        if cached is None:
            structure = load_structure(pdb_dir / row["source_file"])
            model = get_model(structure)
            receptor_chain = find_chain_by_name(model, row["receptor_chain_id"])
            peptide_chain = find_chain_by_name(model, row["peptide_source_chain_id"])
            cached = {
                "receptor_sequence": chain_sequence(chain_residues(receptor_chain)),
                "full_peptide_source_sequence": chain_sequence(chain_residues(peptide_chain)),
            }
            cache[key] = cached

        left_idx = int(row["final_left_index"])
        right_idx = int(row["final_right_index"])
        peptide_seq = cached["full_peptide_source_sequence"][left_idx:right_idx + 1]

        enriched = dict(row)
        enriched["receptor_sequence"] = cached["receptor_sequence"]
        enriched["peptide_sequence"] = peptide_seq
        out.append(enriched)

        if i % 10000 == 0 or i == total:
            print(
                f"[STEP6] attach_sequences {i}/{total} | chain_cache={len(cache)}",
                flush=True,
            )
    return out


def length_compatible(seq_a: str, seq_b: str, min_coverage: float) -> bool:
    if not seq_a or not seq_b:
        return False
    short = min(len(seq_a), len(seq_b))
    long = max(len(seq_a), len(seq_b))
    return (short / long) >= min_coverage


def peptide_signature_keys(seq: str) -> Set[str]:
    if not seq:
        return set()
    # A high-identity short peptide pair should share at least one short k-mer;
    # this index is only a candidate generator, final identity is checked exactly.
    k = 3 if len(seq) <= 10 else 4
    if len(seq) <= k:
        return {seq}
    return {seq[i:i + k] for i in range(0, len(seq) - k + 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase1 Step6: joint receptor+peptide homology dedup")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--main_output_jsonl", required=True)
    parser.add_argument("--monitor_output_jsonl", required=True)
    parser.add_argument("--dropped_output_jsonl", required=True)
    parser.add_argument("--receptor_identity_threshold", type=float, default=0.85)
    parser.add_argument("--peptide_identity_threshold", type=float, default=0.85)
    parser.add_argument(
        "--peptide_min_coverage",
        type=float,
        default=0.70,
        help="Minimum short/long peptide length coverage before two peptides can compete as homologous duplicates.",
    )
    parser.add_argument("--monitor_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--progress_every", type=int, default=10000)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))

    start = time.time()
    print(f"[STEP6] loaded input_candidates={len(rows)}", flush=True)
    enriched = attach_sequences(rows, Path(args.pdb_dir))
    print(f"[STEP6] sorting enriched rows={len(enriched)}", flush=True)
    enriched.sort(
        key=lambda x: (
            # If two homologous peptide windows compete, prefer the longer
            # peptide first so a short window fully covered by a longer one is
            # dropped rather than becoming the representative.
            int(x.get("peptide_length", 0)),
            float(x.get("avg_contact_count", 0.0)),
            float(x.get("contact_coverage", 0.0)),
        ),
        reverse=True,
    )

    kept: List[Dict[str, Any]] = []
    peptide_index: Dict[str, List[int]] = defaultdict(list)
    dropped_count = 0
    checked_pairs = 0

    dropped_path = Path(args.dropped_output_jsonl)
    dropped_path.parent.mkdir(parents=True, exist_ok=True)
    dropped_path.write_text("", encoding="utf-8")

    total = len(enriched)
    for i, row in enumerate(enriched, 1):
        duplicate_of = None
        candidate_prev_indices: Set[int] = set()
        for key in peptide_signature_keys(row["peptide_sequence"]):
            candidate_prev_indices.update(peptide_index.get(key, []))

        for prev_idx in candidate_prev_indices:
            prev = kept[prev_idx]
            if not length_compatible(row["peptide_sequence"], prev["peptide_sequence"], args.peptide_min_coverage):
                continue
            if not length_compatible(row["receptor_sequence"], prev["receptor_sequence"], args.receptor_identity_threshold):
                continue

            receptor_identity = sequence_identity_same_length(row["receptor_sequence"], prev["receptor_sequence"])
            peptide_identity = sequence_identity_same_length(row["peptide_sequence"], prev["peptide_sequence"])
            checked_pairs += 1
            if receptor_identity >= args.receptor_identity_threshold and peptide_identity >= args.peptide_identity_threshold:
                duplicate_of = {
                    "candidate_id": prev["candidate_id"],
                    "receptor_identity": receptor_identity,
                    "peptide_identity": peptide_identity,
                }
                break

        if duplicate_of is None:
            kept_idx = len(kept)
            kept.append(row)
            for key in peptide_signature_keys(row["peptide_sequence"]):
                peptide_index[key].append(kept_idx)
        else:
            row = dict(row)
            row["drop_reason"] = "joint_receptor_peptide_homology"
            row["duplicate_of"] = duplicate_of
            dropped_count += 1
            with dropped_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if i % args.progress_every == 0 or i == total:
            elapsed = time.time() - start
            speed = i / elapsed if elapsed > 0 else 0.0
            print(
                f"[STEP6] dedup {i}/{total} | kept={len(kept)} | dropped={dropped_count} | "
                f"index_keys={len(peptide_index)} | checked_pairs={checked_pairs} | "
                f"elapsed={elapsed/60:.1f} min | speed={speed:.1f} rows/s",
                flush=True,
            )

    rng = random.Random(args.seed)
    print(f"[STEP6] split monitor_fraction={args.monitor_fraction}", flush=True)
    shuffled = list(kept)
    rng.shuffle(shuffled)
    monitor_size = int(round(len(shuffled) * args.monitor_fraction))
    monitor_ids = {row["candidate_id"] for row in shuffled[:monitor_size]}

    main_rows: List[Dict[str, Any]] = []
    monitor_rows: List[Dict[str, Any]] = []
    for row in kept:
        row = dict(row)
        if row["candidate_id"] in monitor_ids:
            row["split"] = "monitor"
            monitor_rows.append(row)
        else:
            row["split"] = "main_train"
            main_rows.append(row)

    for path_str, subset in [
        (args.main_output_jsonl, main_rows),
        (args.monitor_output_jsonl, monitor_rows),
    ]:
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_json(
        Path(args.main_output_jsonl).with_name("step6_summary.json"),
        {
            "input_candidates": len(rows),
            "kept_after_joint_dedup": len(kept),
            "dropped_candidates": dropped_count,
            "main_candidates": len(main_rows),
            "monitor_candidates": len(monitor_rows),
            "receptor_identity_threshold": args.receptor_identity_threshold,
            "peptide_identity_threshold": args.peptide_identity_threshold,
            "peptide_min_coverage": args.peptide_min_coverage,
            "representative_priority": "longer_peptide_then_avg_contact_count",
            "candidate_index": "peptide_kmer_prefilter_exact_identity_check",
            "monitor_fraction": args.monitor_fraction,
            "checked_pairs": checked_pairs,
            "elapsed_sec": round(time.time() - start, 3),
        },
    )
    print("[STEP6] done", flush=True)


if __name__ == "__main__":
    main()

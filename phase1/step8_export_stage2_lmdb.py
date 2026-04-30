from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import lmdb

from common import (
    chain_residues,
    find_chain_by_name,
    get_model,
    iter_jsonl,
    load_structure,
    write_json,
)
from step8_export_stage2_ready import (
    find_residues_by_ids,
    make_track_a_row,
    make_track_b_row,
)


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def encode_key(sample_id: str) -> bytes:
    return sample_id.encode("utf-8")


def encode_value(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def open_env(path: Path, map_size_bytes: int) -> lmdb.Environment:
    ensure_empty_dir(path)
    return lmdb.open(
        str(path),
        map_size=map_size_bytes,
        subdir=True,
        lock=True,
        readonly=False,
        meminit=False,
        map_async=True,
        readahead=False,
        writemap=False,
        max_dbs=1,
    )


def lmdb_output_paths(out_dir: Path) -> Dict[str, Path]:
    return {
        "track_a_main": out_dir / "track_a_main_train.lmdb",
        "track_a_monitor": out_dir / "track_a_monitor.lmdb",
        "track_b_main": out_dir / "track_b_main_train.lmdb",
        "track_b_monitor": out_dir / "track_b_monitor.lmdb",
    }


def choose_names(split_name: str) -> Tuple[str, str]:
    if split_name == "monitor":
        return "track_a_monitor", "track_b_monitor"
    return "track_a_main", "track_b_main"


def finalize_env(
    env: lmdb.Environment,
    keys: List[str],
    meta: Dict[str, Any],
) -> None:
    with env.begin(write=True) as txn:
        txn.put(b"__keys__", json.dumps(keys, ensure_ascii=False).encode("utf-8"))
        txn.put(b"__len__", str(len(keys)).encode("utf-8"))
        txn.put(b"__meta__", json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
    env.sync()
    env.close()


def process_rows(
    rows: Iterable[Dict[str, Any]],
    pdb_dir: Path,
    out_dir: Path,
    progress_every: int,
    map_size_gb: float,
) -> Dict[str, Any]:
    start = time.time()
    map_size_bytes = int(map_size_gb * (1024 ** 3))
    paths = lmdb_output_paths(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    envs = {name: open_env(path, map_size_bytes) for name, path in paths.items()}
    key_lists: Dict[str, List[str]] = {name: [] for name in paths}

    current_source: Optional[str] = None
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

        name_a, name_b = choose_names(str(row["split"]))
        sample_id = str(row["candidate_id"])
        key = encode_key(sample_id)

        with envs[name_a].begin(write=True) as txn_a:
            txn_a.put(key, encode_value(track_a_row))
        with envs[name_b].begin(write=True) as txn_b:
            txn_b.put(key, encode_value(track_b_row))

        key_lists[name_a].append(sample_id)
        key_lists[name_b].append(sample_id)

        split_name = str(row["split"])
        split_counts[split_name] = split_counts.get(split_name, 0) + 1
        processed += 1

        if progress_every > 0 and processed % progress_every == 0:
            elapsed = time.time() - start
            speed = processed / elapsed if elapsed > 0 else 0.0
            print(
                f"[STEP8-LMDB] exported {processed} rows | main={split_counts.get('main_train', 0)} | "
                f"monitor={split_counts.get('monitor', 0)} | elapsed={elapsed/60:.1f} min | speed={speed:.1f} rows/s",
                flush=True,
            )

    format_meta = {
        "export_format": "stage2_ready_lmdb_dual_track",
        "track_a_contents": "receptor full sequence + patch residue ids/indices + peptide sequence",
        "track_b_contents": "patch heavy atoms + peptide heavy atoms",
        "key_type": "sample_id",
        "value_encoding": "utf-8 json",
    }

    for name, env in envs.items():
        finalize_env(
            env,
            key_lists[name],
            {
                **format_meta,
                "split": "monitor" if "monitor" in name else "main_train",
                "track": "track_a" if "track_a" in name else "track_b",
                "num_entries": len(key_lists[name]),
            },
        )

    summary = {
        "input_rows": processed,
        "split_counts": split_counts,
        "track_a_main_lmdb": str(paths["track_a_main"]),
        "track_a_monitor_lmdb": str(paths["track_a_monitor"]),
        "track_b_main_lmdb": str(paths["track_b_main"]),
        "track_b_monitor_lmdb": str(paths["track_b_monitor"]),
        "map_size_gb_per_lmdb": map_size_gb,
        "elapsed_sec": round(time.time() - start, 3),
        **format_meta,
    }
    write_json(out_dir / "step8_lmdb_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase1 Step8: export stage2-ready dual-track LMDB dataset")
    parser.add_argument("--input_jsonl", required=True, help="Phase1 final_metadata.jsonl")
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--progress_every", type=int, default=5000)
    parser.add_argument("--map_size_gb", type=float, default=8.0)
    args = parser.parse_args()

    rows = iter_jsonl(Path(args.input_jsonl))
    process_rows(
        rows=rows,
        pdb_dir=Path(args.pdb_dir),
        out_dir=Path(args.output_dir),
        progress_every=args.progress_every,
        map_size_gb=args.map_size_gb,
    )


if __name__ == "__main__":
    main()

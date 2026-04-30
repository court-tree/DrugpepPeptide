from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import lmdb


def open_env(path: Path) -> lmdb.Environment:
    return lmdb.open(
        str(path),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        max_dbs=1,
    )


def decode_json(raw: bytes) -> Dict[str, Any]:
    return json.loads(raw.decode("utf-8"))


def load_keys(txn: lmdb.Transaction) -> List[str]:
    raw = txn.get(b"__keys__")
    if raw is None:
        raise ValueError("LMDB missing __keys__")
    keys = json.loads(raw.decode("utf-8"))
    if not isinstance(keys, list):
        raise ValueError("__keys__ is not a list")
    return [str(x) for x in keys]


def fetch_obj(txn: lmdb.Transaction, sample_id: str) -> Dict[str, Any]:
    raw = txn.get(sample_id.encode("utf-8"))
    if raw is None:
        raise ValueError(f"sample_id not found: {sample_id}")
    return decode_json(raw)


def check_key_alignment(keys_a: Sequence[str], keys_b: Sequence[str], label: str) -> Dict[str, Any]:
    set_a = set(keys_a)
    set_b = set(keys_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    return {
        "pair": label,
        "num_keys_a": len(keys_a),
        "num_keys_b": len(keys_b),
        "same_order": list(keys_a) == list(keys_b),
        "same_key_set": (not only_a and not only_b),
        "only_in_a_preview": only_a[:5],
        "only_in_b_preview": only_b[:5],
    }


def verify_pair(path_a: Path, path_b: Path, split_name: str, sample_check_limit: int) -> Dict[str, Any]:
    env_a = open_env(path_a)
    env_b = open_env(path_b)
    try:
        with env_a.begin() as txn_a, env_b.begin() as txn_b:
            keys_a = load_keys(txn_a)
            keys_b = load_keys(txn_b)
            alignment = check_key_alignment(keys_a, keys_b, split_name)

            if not alignment["same_key_set"]:
                raise ValueError(
                    f"{split_name} key sets differ: only_in_a={alignment['only_in_a_preview']} "
                    f"only_in_b={alignment['only_in_b_preview']}"
                )

            checked_samples = []
            for sample_id in list(keys_a)[:sample_check_limit]:
                row_a = fetch_obj(txn_a, sample_id)
                row_b = fetch_obj(txn_b, sample_id)

                if str(row_a.get("sample_id", "")) != sample_id:
                    raise ValueError(f"Track A sample_id mismatch for {sample_id}")
                if str(row_b.get("sample_id", "")) != sample_id:
                    raise ValueError(f"Track B sample_id mismatch for {sample_id}")
                if str(row_a.get("split", "")) != split_name:
                    raise ValueError(f"Track A split mismatch for {sample_id}: {row_a.get('split')}")
                if str(row_b.get("split", "")) != split_name:
                    raise ValueError(f"Track B split mismatch for {sample_id}: {row_b.get('split')}")

                checked_samples.append(
                    {
                        "sample_id": sample_id,
                        "track_a_peptide_length": row_a.get("peptide_length"),
                        "track_b_patch_residue_count": len(row_b.get("patch_residue_ids", [])),
                        "track_b_peptide_atom_count": len(row_b.get("peptide_atoms", [])),
                    }
                )

            return {
                **alignment,
                "checked_sample_rows": checked_samples,
            }
    finally:
        env_a.close()
        env_b.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check sample_id alignment between Track A and Track B LMDBs")
    parser.add_argument("--lmdb_dir", required=True, help="Directory containing the 4 LMDB outputs")
    parser.add_argument("--sample_check_limit", type=int, default=5, help="How many rows to inspect per split")
    args = parser.parse_args()

    lmdb_dir = Path(args.lmdb_dir)
    result_main = verify_pair(
        lmdb_dir / "track_a_main_train.lmdb",
        lmdb_dir / "track_b_main_train.lmdb",
        "main_train",
        args.sample_check_limit,
    )
    result_monitor = verify_pair(
        lmdb_dir / "track_a_monitor.lmdb",
        lmdb_dir / "track_b_monitor.lmdb",
        "monitor",
        args.sample_check_limit,
    )

    print(
        json.dumps(
            {
                "ok": True,
                "results": [result_main, result_monitor],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

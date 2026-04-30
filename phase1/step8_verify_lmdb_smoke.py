from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import lmdb


SPECIAL_KEYS = {b"__keys__", b"__len__", b"__meta__"}


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


def load_meta(txn: lmdb.Transaction) -> Dict[str, Any]:
    raw = txn.get(b"__meta__")
    if raw is None:
        raise ValueError("LMDB missing __meta__")
    return decode_json(raw)


def load_keys(txn: lmdb.Transaction) -> List[str]:
    raw = txn.get(b"__keys__")
    if raw is None:
        raise ValueError("LMDB missing __keys__")
    keys = json.loads(raw.decode("utf-8"))
    if not isinstance(keys, list):
        raise ValueError("__keys__ is not a list")
    return [str(x) for x in keys]


def load_declared_len(txn: lmdb.Transaction) -> int:
    raw = txn.get(b"__len__")
    if raw is None:
        raise ValueError("LMDB missing __len__")
    return int(raw.decode("utf-8"))


def count_data_keys(txn: lmdb.Transaction) -> int:
    n = 0
    cursor = txn.cursor()
    for key, _value in cursor:
        if key in SPECIAL_KEYS:
            continue
        n += 1
    return n


def inspect_sample(txn: lmdb.Transaction, sample_id: str) -> Dict[str, Any]:
    raw = txn.get(sample_id.encode("utf-8"))
    if raw is None:
        raise ValueError(f"sample_id not found: {sample_id}")
    obj = decode_json(raw)
    if str(obj.get("sample_id", "")) != sample_id:
        raise ValueError(f"sample_id mismatch inside value: key={sample_id} value={obj.get('sample_id')}")
    return obj


def summarize_sample(obj: Dict[str, Any], track_name: str) -> Dict[str, Any]:
    if track_name == "track_a":
        return {
            "sample_id": obj["sample_id"],
            "split": obj["split"],
            "peptide_length": obj.get("peptide_length"),
            "patch_residue_count": len(obj.get("receptor_patch_residue_ids", [])),
            "receptor_sequence_length": len(str(obj.get("receptor_sequence", ""))),
        }
    return {
        "sample_id": obj["sample_id"],
        "split": obj["split"],
        "peptide_atom_count": len(obj.get("peptide_atoms", [])),
        "patch_atom_count": len(obj.get("patch_atoms", [])),
        "patch_residue_count": len(obj.get("patch_residue_ids", [])),
    }


def verify_one_lmdb(path: Path, preview_count: int, full_count: bool) -> Dict[str, Any]:
    track_name = "track_a" if "track_a" in path.name else "track_b"
    env = open_env(path)
    try:
        with env.begin() as txn:
            meta = load_meta(txn)
            keys = load_keys(txn)
            declared_len = load_declared_len(txn)

            if declared_len != len(keys):
                raise ValueError(
                    f"declared __len__ != len(__keys__): {declared_len} vs {len(keys)}"
                )

            data_key_count = None
            if full_count:
                data_key_count = count_data_keys(txn)
                if declared_len != data_key_count:
                    raise ValueError(
                        f"declared __len__ != actual data key count: {declared_len} vs {data_key_count}"
                    )

            previews = []
            for sample_id in keys[:preview_count]:
                obj = inspect_sample(txn, sample_id)
                previews.append(summarize_sample(obj, track_name))

            return {
                "path": str(path),
                "track": track_name,
                "split": meta.get("split"),
                "num_entries": declared_len,
                "full_count_enabled": full_count,
                "actual_data_key_count": data_key_count,
                "preview_samples": previews,
            }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Stage2 LMDB exports")
    parser.add_argument("--lmdb_dir", required=True, help="Directory containing the 4 LMDB outputs")
    parser.add_argument("--preview_count", type=int, default=2, help="How many sample rows to preview per LMDB")
    parser.add_argument(
        "--full_count",
        action="store_true",
        help="Scan every key in each LMDB and verify actual data-key count; slower on large Track B LMDBs.",
    )
    args = parser.parse_args()

    lmdb_dir = Path(args.lmdb_dir)
    expected = [
        lmdb_dir / "track_a_main_train.lmdb",
        lmdb_dir / "track_a_monitor.lmdb",
        lmdb_dir / "track_b_main_train.lmdb",
        lmdb_dir / "track_b_monitor.lmdb",
    ]

    for path in expected:
        if not path.exists():
            raise FileNotFoundError(f"Missing LMDB path: {path}")

    results = [verify_one_lmdb(path, args.preview_count, args.full_count) for path in expected]
    print(json.dumps({"ok": True, "results": results}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Prepare Propedia records for the Phase-3 V1 source table.

The local Propedia archives may lack the ZIP central directory, so this adapter
extracts required PDB files by streaming local ZIP headers.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def canonical_peptide(sequence: str, min_len: int, max_len: int) -> bool:
    seq = sequence.strip().upper()
    return min_len <= len(seq) <= max_len and all(ch in CANONICAL_AA for ch in seq)


def read_complex_csv(path: Path, min_len: int, max_len: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdb_id = row.get("PDB", "").strip().lower()
            peptide_chain = row.get("Peptide Chain", "").strip()
            receptor_chain = row.get("Receptor Chain", "").strip()
            peptide_sequence = row.get("Peptide Sequence", "").strip().upper()
            if not pdb_id or not peptide_chain or not receptor_chain:
                continue
            if not canonical_peptide(peptide_sequence, min_len, max_len):
                continue
            rows.append(
                {
                    "raw": row,
                    "pdb_id": pdb_id,
                    "peptide_chain": peptide_chain,
                    "receptor_chain": receptor_chain,
                    "peptide_sequence": peptide_sequence,
                    "complex_zip_name": f"structures/complex/{pdb_id}_{peptide_chain}_{receptor_chain}.pdb",
                    "complex_file_name": f"{pdb_id}_{peptide_chain}_{receptor_chain}.pdb",
                }
            )
    return rows


def iter_local_zip_entries(zip_path: Path) -> Iterator[tuple[str, int, bytes]]:
    with zip_path.open("rb") as handle:
        while True:
            offset = handle.tell()
            header = handle.read(30)
            if not header:
                return
            if len(header) < 30 or header[:4] != b"PK\x03\x04":
                return
            (
                _sig,
                _version,
                flag,
                method,
                _mtime,
                _mdate,
                _crc,
                compressed_size,
                _uncompressed_size,
                name_len,
                extra_len,
            ) = struct.unpack("<IHHHHHIIIHH", header)
            name = handle.read(name_len).decode("utf-8", errors="replace")
            handle.read(extra_len)
            if flag & 0x08:
                raise ValueError(f"Unsupported data descriptor ZIP entry at offset {offset}: {name}")
            data = handle.read(compressed_size)
            yield name, method, data


def decompress_zip_entry(name: str, method: int, data: bytes) -> bytes:
    if method == 0:
        return data
    if method == 8:
        return zlib.decompress(data, -15)
    raise ValueError(f"Unsupported ZIP compression method {method} for {name}")


def extract_needed_complexes(zip_path: Path, output_dir: Path, needed: set[str]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, str] = {}
    remaining = set(needed)
    for name, method, data in iter_local_zip_entries(zip_path):
        if name not in remaining:
            continue
        try:
            payload = decompress_zip_entry(name, method, data)
        except zlib.error:
            remaining.remove(name)
            continue
        out_path = output_dir / Path(name).name
        out_path.write_bytes(payload)
        extracted[name] = str(out_path)
        remaining.remove(name)
        if not remaining:
            break
    return extracted


def build_records(
    propedia_root: Path,
    output_dir: Path,
    min_len: int,
    max_len: int,
    limit: Optional[int],
) -> dict[str, Any]:
    complex_csv = propedia_root / "complex.csv"
    complex_zip = propedia_root / "complex.zip"
    if not complex_csv.exists():
        raise FileNotFoundError(complex_csv)
    if not complex_zip.exists():
        raise FileNotFoundError(complex_zip)

    rows = read_complex_csv(complex_csv, min_len, max_len)
    if limit is not None:
        rows = rows[:limit]
    needed = {row["complex_zip_name"] for row in rows}
    structures_dir = output_dir / "structures"
    extracted = extract_needed_complexes(complex_zip, structures_dir, needed)

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "propedia_phase3_records.jsonl"
    written = 0
    missing = 0
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            structure_path = extracted.get(row["complex_zip_name"])
            if not structure_path:
                missing += 1
                continue
            raw = row["raw"]
            record: Dict[str, Any] = {
                "source_database": "Propedia",
                "source_entry_id": f"{row['pdb_id']}_{row['peptide_chain']}_{row['receptor_chain']}",
                "pdb_id": row["pdb_id"],
                "biological_assembly_id": "propedia_curated_complex",
                "assembly_confidence": "propedia_curated_complex",
                "source_confidence_tier": "tier_2_curated_positive",
                "complex_structure_file": structure_path,
                "receptor_chain_id": row["receptor_chain"],
                "peptide_chain_id": row["peptide_chain"],
                "propedia_peptide_sequence": row["peptide_sequence"],
                "propedia_resolution": raw.get("Resolution", ""),
                "propedia_sequence_cluster": raw.get("Sequence Cluster", ""),
                "propedia_interface_cluster": raw.get("Interface Cluster", ""),
                "propedia_binding_cluster": raw.get("Binding Cluster", ""),
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1

    summary = {
        "source_database": "Propedia",
        "source_policy_tier": "tier_2_curated_positive",
        "propedia_root": str(propedia_root),
        "output_dir": str(output_dir),
        "records_jsonl": str(records_path),
        "candidate_rows_after_v1_sequence_filter": len(rows),
        "needed_complex_files": len(needed),
        "extracted_complex_files": len(extracted),
        "written_records": written,
        "missing_complex_files": missing,
        "min_peptide_length": min_len,
        "max_peptide_length": max_len,
    }
    (output_dir / "propedia_prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--propedia_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_peptide_length", type=int, default=8)
    parser.add_argument("--max_peptide_length", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_records(
        propedia_root=Path(args.propedia_root),
        output_dir=Path(args.output_dir),
        min_len=args.min_peptide_length,
        max_len=args.max_peptide_length,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

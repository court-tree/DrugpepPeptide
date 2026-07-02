"""Build a receptor-family map for Phase-3 V1 using MMseqs2.

Input is a V1 `receptor_peptide_anchor.jsonl` file. Output is JSONL accepted by
`python -m phase3.v1 --receptor_family_map`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict


DEFAULT_MMSEQS = Path(r"E:\pep\phase3\tools\mmseqs2_local\mmseqs.sh")


def win_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = str(resolved)[3:].replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def read_receptor_sequences(anchor_jsonl: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    with anchor_jsonl.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("receptor_sequence_key")
            seq = row.get("receptor_sequence") or row.get("receptor_sequence_or_patch")
            if key and seq:
                seqs[str(key)] = str(seq)
    return seqs


def write_fasta(seqs: Dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for key, seq in sorted(seqs.items()):
            handle.write(f">{key}\n")
            for i in range(0, len(seq), 80):
                handle.write(seq[i : i + 80] + "\n")


def run_mmseqs(
    fasta: Path,
    output_prefix: Path,
    tmp_dir: Path,
    mmseqs_script: Path,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
) -> Path:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wsl.exe",
        "bash",
        "-lc",
        " ".join(
            [
                win_to_wsl(mmseqs_script),
                "easy-cluster",
                win_to_wsl(fasta),
                win_to_wsl(output_prefix),
                win_to_wsl(tmp_dir),
                "--min-seq-id",
                str(min_seq_id),
                "-c",
                str(coverage),
                "--cov-mode",
                str(cov_mode),
                "-v",
                "1",
            ]
        ),
    ]
    subprocess.run(cmd, check=True)
    return output_prefix.with_name(output_prefix.name + "_cluster.tsv")


def parse_cluster_tsv(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rep, member = line.rstrip("\n").split("\t")[:2]
            mapping[member] = f"mmseqs_rfam_{rep}"
    return mapping


def build_map(args: argparse.Namespace) -> Dict[str, Any]:
    anchor_jsonl = Path(args.anchor_jsonl)
    output_jsonl = Path(args.output_jsonl)
    work_dir = Path(args.work_dir)
    seqs = read_receptor_sequences(anchor_jsonl)
    if not seqs:
        raise ValueError("No receptor_sequence/receptor_sequence_key pairs found in anchor JSONL")
    fasta = work_dir / "receptor_sequences.fasta"
    write_fasta(seqs, fasta)
    cluster_tsv = run_mmseqs(
        fasta=fasta,
        output_prefix=work_dir / "mmseqs_receptor_family",
        tmp_dir=work_dir / "tmp",
        mmseqs_script=Path(args.mmseqs),
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
    )
    mapping = parse_cluster_tsv(cluster_tsv)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for key in sorted(seqs):
            family = mapping.get(key, f"mmseqs_rfam_{key}")
            row = {
                "receptor_sequence_key": key,
                "receptor_family_key": family,
                "method": "mmseqs_easy_cluster",
                "min_seq_id": args.min_seq_id,
                "coverage": args.coverage,
                "cov_mode": args.cov_mode,
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "anchor_jsonl": str(anchor_jsonl),
        "output_jsonl": str(output_jsonl),
        "work_dir": str(work_dir),
        "sequence_count": len(seqs),
        "family_count": len(set(mapping.get(key, f"mmseqs_rfam_{key}") for key in seqs)),
        "cluster_tsv": str(cluster_tsv),
        "min_seq_id": args.min_seq_id,
        "coverage": args.coverage,
        "cov_mode": args.cov_mode,
    }
    output_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--mmseqs", default=str(DEFAULT_MMSEQS))
    parser.add_argument("--min_seq_id", type=float, default=0.4)
    parser.add_argument("--coverage", type=float, default=0.6)
    parser.add_argument("--cov_mode", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    summary = build_map(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

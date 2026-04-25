from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import gemmi
import pandas as pd


STANDARD_AA = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def find_chain_by_name(model: gemmi.Model, chain_name: str) -> gemmi.Chain:
    available = [chain.name for chain in model]
    for chain in model:
        if chain.name == chain_name:
            return chain
    target = str(chain_name).strip()
    for chain in model:
        if str(chain.name).strip() == target:
            return chain
    raise KeyError(f"Chain not found: {chain_name}; available_chains={available}")


def extract_chain_sequence(cif_path: Path, chain_name: str) -> str:
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    if len(st) == 0:
        raise ValueError(f"Empty structure: {cif_path}")
    model = st[0]
    chain = find_chain_by_name(model, chain_name)

    seq_chars: List[str] = []
    for residue in chain:
        resname = residue.name.strip().upper()
        aa = STANDARD_AA.get(resname)
        if aa:
            seq_chars.append(aa)
    seq = "".join(seq_chars)
    if not seq:
        raise ValueError(f"No valid protein residues in {cif_path.name} chain {chain_name}")
    return seq


def build_receptor_table(df: pd.DataFrame, cif_dir: Path) -> pd.DataFrame:
    required = ["pdb_id", "source_file", "receptor_chain_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input JSONL: {missing}")

    dedup = (
        df[required]
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )
    rows: List[Dict] = []
    for row in dedup.to_dict(orient="records"):
        pdb_id = str(row["pdb_id"])
        source_file = str(row["source_file"])
        receptor_chain_id = str(row["receptor_chain_id"])
        receptor_key = f"{pdb_id}|{source_file}|{receptor_chain_id}"
        cif_path = cif_dir / source_file
        seq = extract_chain_sequence(cif_path, receptor_chain_id)
        rows.append(
            {
                "receptor_key": receptor_key,
                "pdb_id": pdb_id,
                "source_file": source_file,
                "receptor_chain_id": receptor_chain_id,
                "receptor_sequence": seq,
                "receptor_seq_len": len(seq),
            }
        )
    return pd.DataFrame(rows)


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(f">{row['receptor_key']}\n{row['receptor_sequence']}\n")


def maybe_windows_path(path: Path, mmseqs_bin: Path) -> str:
    """
    If we are calling a Windows mmseqs.exe from WSL, convert Linux paths to Windows paths.
    Otherwise keep native paths unchanged.
    """
    if mmseqs_bin.suffix.lower() != ".exe":
        return str(path)
    out = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
    return out


def make_mmseqs_temp_root(fasta_path: Path, mmseqs_bin: Path):
    """
    Windows mmseqs.exe behaves much better when all temp files live on a native
    Windows filesystem path (e.g. E:\\...) rather than a WSL /tmp UNC path.
    """
    parent = fasta_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if mmseqs_bin.suffix.lower() == ".exe":
        return tempfile.TemporaryDirectory(prefix="mmseqs_tmp_", dir=str(parent))
    return tempfile.TemporaryDirectory(prefix="pep_mmseqs_")


def cluster_exact(df_receptors: pd.DataFrame) -> pd.DataFrame:
    seq_to_rep: Dict[str, str] = {}
    cluster_rows: List[Dict] = []
    for row in df_receptors.to_dict(orient="records"):
        seq = row["receptor_sequence"]
        rep = seq_to_rep.setdefault(seq, row["receptor_key"])
        cluster_rows.append(
            {
                "receptor_key": row["receptor_key"],
                "cluster_rep": rep,
                "receptor_seq_cluster": f"exact:{rep}",
                "cluster_method": "exact_sequence",
            }
        )
    return pd.DataFrame(cluster_rows)


def run_mmseqs_cluster(
    fasta_path: Path,
    mmseqs_bin: Path,
    min_seq_id: float,
    coverage: float,
) -> pd.DataFrame:
    with make_mmseqs_temp_root(fasta_path, mmseqs_bin) as tmp_root:
        tmp_root_path = Path(tmp_root)
        db_path = tmp_root_path / "seqdb"
        clu_path = tmp_root_path / "clu"
        tsv_path = tmp_root_path / "clusters.tsv"
        work_tmp = tmp_root_path / "tmp"

        fasta_arg = maybe_windows_path(fasta_path, mmseqs_bin)
        db_arg = maybe_windows_path(db_path, mmseqs_bin)
        clu_arg = maybe_windows_path(clu_path, mmseqs_bin)
        tsv_arg = maybe_windows_path(tsv_path, mmseqs_bin)
        work_tmp_arg = maybe_windows_path(work_tmp, mmseqs_bin)
        mmseqs_arg = str(mmseqs_bin)

        cmds = [
            [mmseqs_arg, "createdb", fasta_arg, db_arg],
            [
                mmseqs_arg,
                "linclust",
                db_arg,
                clu_arg,
                work_tmp_arg,
                "--min-seq-id",
                str(min_seq_id),
                "-c",
                str(coverage),
                "--cov-mode",
                "0",
            ],
            [mmseqs_arg, "createtsv", db_arg, db_arg, clu_arg, tsv_arg],
        ]
        for cmd in cmds:
            subprocess.run(cmd, check=True)

        if not tsv_path.exists():
            raise FileNotFoundError(f"MMseqs cluster TSV not found: {tsv_path}")

        tsv = pd.read_csv(tsv_path, sep="\t", header=None, names=["cluster_rep", "receptor_key"])
        seq_id_pct = int(round(min_seq_id * 100))
        tsv["receptor_seq_cluster"] = f"mmseqs{seq_id_pct}:" + tsv["cluster_rep"].astype(str)
        tsv["cluster_method"] = "mmseqs_linclust"
        return tsv


def annotate_samples(df_samples: pd.DataFrame, df_receptors: pd.DataFrame, df_clusters: pd.DataFrame) -> pd.DataFrame:
    df = df_samples.copy()
    df["receptor_key"] = (
        df["pdb_id"].astype(str)
        + "|"
        + df["source_file"].astype(str)
        + "|"
        + df["receptor_chain_id"].astype(str)
    )

    merge_cols = ["receptor_key", "receptor_sequence", "receptor_seq_len"]
    df = df.merge(df_receptors[merge_cols], on="receptor_key", how="left")
    df = df.merge(
        df_clusters[["receptor_key", "cluster_rep", "receptor_seq_cluster", "cluster_method"]],
        on="receptor_key",
        how="left",
    )
    df["receptor_sequence_identity_cluster"] = df["receptor_seq_cluster"]
    return df


def summarize(
    df_receptors: pd.DataFrame,
    df_clusters: pd.DataFrame,
    df_annotated: pd.DataFrame,
    cluster_mode: str,
    min_seq_id: float,
    coverage: float,
    mmseqs_bin: str,
) -> Dict:
    vc = Counter(df_clusters["receptor_seq_cluster"].astype(str).tolist())
    top10 = dict(vc.most_common(10))
    if cluster_mode == "mmseqs":
        cluster_label_prefix = f"mmseqs{int(round(min_seq_id * 100))}"
    else:
        cluster_label_prefix = "exact"
    return {
        "step6x_version": "receptor_sequence_cluster_v1",
        "cluster_mode": cluster_mode,
        "cluster_label_prefix": cluster_label_prefix,
        "min_seq_id": float(min_seq_id),
        "coverage": float(coverage),
        "mmseqs_bin": str(mmseqs_bin or ""),
        "unique_receptors": int(len(df_receptors)),
        "annotated_rows": int(len(df_annotated)),
        "unique_clusters": int(df_clusters["receptor_seq_cluster"].nunique()),
        "largest_clusters_top10": top10,
        "sequence_length_summary": {
            "min": int(df_receptors["receptor_seq_len"].min()) if len(df_receptors) else 0,
            "median": float(df_receptors["receptor_seq_len"].median()) if len(df_receptors) else 0.0,
            "max": int(df_receptors["receptor_seq_len"].max()) if len(df_receptors) else 0,
            "mean": float(df_receptors["receptor_seq_len"].mean()) if len(df_receptors) else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build receptor sequence clusters and annotate samples")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--cif_dir", type=str, required=True)
    parser.add_argument("--out_receptors_csv", type=str, required=True)
    parser.add_argument("--out_fasta", type=str, required=True)
    parser.add_argument("--out_clusters_csv", type=str, required=True)
    parser.add_argument("--out_annotated_jsonl", type=str, required=True)
    parser.add_argument("--out_summary_json", type=str, required=True)
    parser.add_argument("--cluster_mode", type=str, default="mmseqs", choices=["mmseqs", "exact"])
    parser.add_argument("--mmseqs_bin", type=str, default="")
    parser.add_argument("--min_seq_id", type=float, default=0.70)
    parser.add_argument("--coverage", type=float, default=0.80)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    cif_dir = Path(args.cif_dir)
    out_receptors_csv = Path(args.out_receptors_csv)
    out_fasta = Path(args.out_fasta)
    out_clusters_csv = Path(args.out_clusters_csv)
    out_annotated_jsonl = Path(args.out_annotated_jsonl)
    out_summary_json = Path(args.out_summary_json)

    print("=" * 80, flush=True)
    print("[START] Step 6X: receptor sequence clustering", flush=True)
    print(f"[INPUT]  input_jsonl   = {input_path}", flush=True)
    print(f"[INPUT]  cif_dir       = {cif_dir}", flush=True)
    print(f"[PARAM]  cluster_mode  = {args.cluster_mode}", flush=True)
    if args.mmseqs_bin:
        print(f"[PARAM]  mmseqs_bin    = {args.mmseqs_bin}", flush=True)
    print("=" * 80, flush=True)

    df_samples = load_jsonl(input_path)
    df_receptors = build_receptor_table(df_samples, cif_dir)
    out_receptors_csv.parent.mkdir(parents=True, exist_ok=True)
    df_receptors.to_csv(out_receptors_csv, index=False)
    write_fasta(df_receptors, out_fasta)

    if args.cluster_mode == "mmseqs":
        if not args.mmseqs_bin:
            raise ValueError("--mmseqs_bin is required when cluster_mode=mmseqs")
        df_clusters = run_mmseqs_cluster(
            fasta_path=out_fasta,
            mmseqs_bin=Path(args.mmseqs_bin),
            min_seq_id=args.min_seq_id,
            coverage=args.coverage,
        )
    else:
        df_clusters = cluster_exact(df_receptors)

    out_clusters_csv.parent.mkdir(parents=True, exist_ok=True)
    df_clusters.to_csv(out_clusters_csv, index=False)

    df_annotated = annotate_samples(df_samples, df_receptors, df_clusters)
    save_jsonl(df_annotated, out_annotated_jsonl)

    summary = summarize(
        df_receptors,
        df_clusters,
        df_annotated,
        args.cluster_mode,
        args.min_seq_id,
        args.coverage,
        args.mmseqs_bin,
    )
    save_json(out_summary_json, summary)

    print(f"[DONE] unique_receptors = {summary['unique_receptors']}", flush=True)
    print(f"[DONE] unique_clusters  = {summary['unique_clusters']}", flush=True)
    print(f"[DONE] receptors_csv    = {out_receptors_csv}", flush=True)
    print(f"[DONE] fasta            = {out_fasta}", flush=True)
    print(f"[DONE] clusters_csv     = {out_clusters_csv}", flush=True)
    print(f"[DONE] annotated_jsonl  = {out_annotated_jsonl}", flush=True)
    print(f"[DONE] summary_json     = {out_summary_json}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()

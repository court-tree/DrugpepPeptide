from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================================================
# IO
# =========================================================
def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


# =========================================================
# Schema / compatibility
# =========================================================
def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "rBSA_proxy" not in df.columns:
        if "rBSA_raw" in df.columns:
            df["rBSA_proxy"] = df["rBSA_raw"]
        elif "rBSA" in df.columns:
            df["rBSA_proxy"] = df["rBSA"]
        else:
            raise ValueError("Missing rBSA_proxy / rBSA_raw / rBSA")

    if "pocket_num_residues" not in df.columns:
        if "pocket_size_6A" in df.columns:
            df["pocket_num_residues"] = df["pocket_size_6A"]
        elif "n_contact_residues_6A" in df.columns:
            df["pocket_num_residues"] = df["n_contact_residues_6A"]
        elif "n_contact_residues_step4" in df.columns:
            df["pocket_num_residues"] = df["n_contact_residues_step4"]
        else:
            raise ValueError("Missing pocket_num_residues / pocket_size_6A / n_contact_residues_6A")

    if "peptide_length" not in df.columns:
        raise ValueError("Missing peptide_length")

    if "candidate_id" not in df.columns:
        raise ValueError("Missing candidate_id")

    if "parent_task_id" not in df.columns:
        raise ValueError("Missing parent_task_id")

    return df


def first_present_value(row: pd.Series, keys: List[str]) -> str:
    for key in keys:
        if key in row.index:
            val = str(row.get(key, "")).strip()
            if val and val != "nan":
                return val
    return ""


def choose_group_key(row: pd.Series) -> str:
    """
    Prefer receptor family / sequence-cluster identifiers when available.
    Fallback order:
      1) receptor_family_id / pfam / receptor_seq_cluster
      2) pdb_id
      3) parent_task_id

    In the current Phase-1 pipeline, receptor_seq_cluster is expected to come
    from Step 6X MMseqs2 clustering. When cluster annotations are missing,
    falling back to pdb_id still gives Step 7 a meaningful high-frequency
    target capping behavior. Going finer than pdb_id tends to weaken capping.
    """
    family = first_present_value(
        row,
        [
            "receptor_family_id",
            "receptor_pfam",
            "receptor_pfam_id",
            "receptor_seq_cluster",
            "receptor_sequence_identity_cluster",
            "receptor_cluster_id",
        ],
    )
    if family:
        return f"family:{family}"

    pdb_id = first_present_value(row, ["pdb_id", "complex_pdb_id"])
    if pdb_id:
        return f"target:{pdb_id}"

    task = first_present_value(row, ["parent_task_id"])
    if task:
        return f"task:{task}"

    return "unknown"


# =========================================================
# Binning
# =========================================================
def make_bins_dict() -> Dict[str, list]:
    return {
        "patch": [-0.1, 25, 50, 100, np.inf],
        "peptide": [-0.1, 8, 14, np.inf],
        "rbsa": [-0.01, 0.35, 0.70, np.inf],
    }


def add_bins(df: pd.DataFrame, bins_dict: dict) -> pd.DataFrame:
    df = df.copy()
    df["patch_bin"] = pd.cut(
        df["pocket_num_residues"],
        bins=bins_dict["patch"],
        include_lowest=True,
        right=False,
    )
    df["peptide_bin"] = pd.cut(
        df["peptide_length"],
        bins=bins_dict["peptide"],
        include_lowest=True,
        right=False,
    )
    df["rbsa_bin"] = pd.cut(
        df["rBSA_proxy"],
        bins=bins_dict["rbsa"],
        include_lowest=True,
        right=False,
    )
    df["bin_key"] = (
        df["patch_bin"].astype(str)
        + " | "
        + df["peptide_bin"].astype(str)
        + " | "
        + df["rbsa_bin"].astype(str)
    )
    df["is_out_of_bins"] = df[["patch_bin", "peptide_bin", "rbsa_bin"]].isna().any(axis=1)
    return df


# =========================================================
# Sorting helpers
# =========================================================
def add_sort_keys(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(random_state)

    if "keep_prob" not in df.columns:
        df["keep_prob"] = 0.0
    if "score_final" not in df.columns:
        df["score_final"] = 0.0
    if "step5_selected_by_sampling" not in df.columns:
        df["step5_selected_by_sampling"] = False
    if "step5_selection_mode" not in df.columns:
        df["step5_selection_mode"] = ""

    df["sort_is_sampling"] = df["step5_selected_by_sampling"].astype(int)
    df["sort_random_tiebreak"] = rng.random(len(df))
    return df


# =========================================================
# Family capping
# =========================================================
def apply_family_capping(
    df: pd.DataFrame,
    max_per_group: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cap high-frequency groups.
    Priority order:
      1) step5_selected_by_sampling
      2) keep_prob
      3) score_final
      4) random tie-break
    """
    df = df.copy()
    if "group_key" not in df.columns:
        df["group_key"] = df.apply(choose_group_key, axis=1)

    df = add_sort_keys(df, random_state=random_state)

    kept_parts = []
    dropped_parts = []

    sort_cols = [
        "sort_is_sampling",
        "keep_prob",
        "score_final",
        "sort_random_tiebreak",
    ]
    ascending = [False, False, False, False]

    for _, sub in df.groupby("group_key", sort=False):
        sub = sub.copy()
        if len(sub) <= max_per_group:
            kept_parts.append(sub)
            continue

        sub = sub.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        kept = sub.iloc[:max_per_group].copy()
        dropped = sub.iloc[max_per_group:].copy()
        dropped["drop_reason"] = "family_capping"

        kept_parts.append(kept)
        dropped_parts.append(dropped)

    df_kept = pd.concat(kept_parts, ignore_index=True) if kept_parts else df.iloc[:0].copy()
    df_dropped = pd.concat(dropped_parts, ignore_index=True) if dropped_parts else df.iloc[:0].copy()

    return df_kept, df_dropped


# =========================================================
# Monitor split
# =========================================================
def stratified_monitor_split_by_group(
    df: pd.DataFrame,
    monitor_ratio: float,
    random_state: int,
    min_per_nonempty_bin: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group-level split to reduce leakage:
    - Work on unique group_key representatives per bin
    - Sample whole groups into monitor
    - Approximate monitor_ratio at group level
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    if "group_key" not in df.columns:
        raise ValueError("Missing group_key before monitor split")

    # one representative row per group for bin stratification
    reps = (
        df.sort_values(
            ["step5_selected_by_sampling", "keep_prob", "score_final"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        .groupby("group_key", as_index=False, sort=False)
        .head(1)
        .copy()
    )

    chosen_groups: List[str] = []

    for _, sub in reps.groupby("bin_key", sort=False):
        n = len(sub)
        if n == 0:
            continue
        target = max(min_per_nonempty_bin, int(round(n * monitor_ratio)))
        target = min(target, n)
        chosen = rng.choice(sub["group_key"].to_numpy(), size=target, replace=False)
        chosen_groups.extend(chosen.tolist())

    chosen_group_set = set(chosen_groups)

    df_monitor = df.loc[df["group_key"].isin(chosen_group_set)].copy()
    df_main = df.loc[~df["group_key"].isin(chosen_group_set)].copy()

    df_monitor["split"] = "monitor"
    df_main["split"] = "main_train"

    return df_main, df_monitor


# =========================================================
# Summary
# =========================================================
def summarize_numeric(series: pd.Series) -> Dict[str, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return {
            "count": 0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }

    return {
        "count": int(vals.shape[0]),
        "min": float(vals.min()),
        "p25": float(vals.quantile(0.25)),
        "median": float(vals.quantile(0.50)),
        "p75": float(vals.quantile(0.75)),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
    }


def top_n_counts(d: Dict[str, int], n: int = 20) -> Dict[str, int]:
    return dict(sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n])


def summarize_selection_modes(df: pd.DataFrame) -> Dict[str, int]:
    if "step5_selection_mode" not in df.columns:
        return {}
    vals = df["step5_selection_mode"].fillna("").astype(str)
    return dict(vals.value_counts().to_dict())


# =========================================================
# Main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 Step 7 v2: MMseqs2-family capping & group-safe monitor split")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Step 6 records JSONL")
    parser.add_argument("--out_main_jsonl", type=str, required=True, help="Main train JSONL")
    parser.add_argument("--out_monitor_jsonl", type=str, required=True, help="Monitor JSONL")
    parser.add_argument("--out_dropped_jsonl", type=str, required=True, help="Dropped by family capping JSONL")
    parser.add_argument("--out_summary_json", type=str, required=True, help="Summary JSON")
    parser.add_argument("--max_per_group", type=int, default=200, help="Max samples per family/target proxy group")
    parser.add_argument("--monitor_ratio", type=float, default=0.01, help="Target monitor ratio at group level")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--drop_out_of_bins", action="store_true", help="Drop rows that fall outside the manual bins before splitting")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    out_main = Path(args.out_main_jsonl)
    out_monitor = Path(args.out_monitor_jsonl)
    out_dropped = Path(args.out_dropped_jsonl)
    out_summary = Path(args.out_summary_json)

    print("=" * 80, flush=True)
    print("[START] Phase-1 Step 7 v2: MMseqs2-family capping & monitor split", flush=True)
    print(f"[INPUT]  input_jsonl       = {input_path}", flush=True)
    print(f"[OUTPUT] out_main_jsonl   = {out_main}", flush=True)
    print(f"[OUTPUT] out_monitor_jsonl= {out_monitor}", flush=True)
    print(f"[OUTPUT] out_dropped_jsonl= {out_dropped}", flush=True)
    print(f"[OUTPUT] out_summary_json = {out_summary}", flush=True)
    print(f"[PARAM]  max_per_group    = {args.max_per_group}", flush=True)
    print(f"[PARAM]  monitor_ratio    = {args.monitor_ratio}", flush=True)
    print(f"[PARAM]  random_state     = {args.random_state}", flush=True)
    print(f"[PARAM]  drop_out_of_bins = {args.drop_out_of_bins}", flush=True)
    print("=" * 80, flush=True)

    df = load_jsonl(input_path)
    df = ensure_required_columns(df)

    bins_dict = make_bins_dict()
    df = add_bins(df, bins_dict)
    df["group_key"] = df.apply(choose_group_key, axis=1)

    input_count = len(df)
    out_of_bins_count = int(df["is_out_of_bins"].sum())

    if args.drop_out_of_bins:
        df = df.loc[~df["is_out_of_bins"]].copy()

    # Step 7A: family/target proxy capping
    df_capped, df_dropped = apply_family_capping(
        df=df,
        max_per_group=args.max_per_group,
        random_state=args.random_state,
    )

    # Step 7B: group-safe monitor split
    df_main, df_monitor = stratified_monitor_split_by_group(
        df=df_capped,
        monitor_ratio=args.monitor_ratio,
        random_state=args.random_state,
        min_per_nonempty_bin=1,
    )

    # save
    save_jsonl(df_main, out_main)
    save_jsonl(df_monitor, out_monitor)
    save_jsonl(df_dropped, out_dropped)

    group_counts_before = df["group_key"].value_counts().to_dict()
    group_counts_after = df_capped["group_key"].value_counts().to_dict()

    summary = {
        "step7_version": "family_capping_monitor_split_v2",
        "group_key_semantics": "prefer receptor family/cluster identifiers (especially Step6X MMseqs2 receptor_seq_cluster); fallback to pdb_id, then parent_task_id",
        "sort_semantics": "family capping priority = step5_selected_by_sampling -> keep_prob -> score_final -> random_tiebreak",
        "monitor_split_semantics": "group-safe monitor split by bin_key on one representative row per group_key; all rows of a chosen MMseqs2 family/target group go to monitor",
        "input_count": int(input_count),
        "out_of_bins_count_before_optional_drop": out_of_bins_count,
        "after_optional_out_of_bins_drop_count": int(len(df)),
        "after_family_capping_count": int(len(df_capped)),
        "family_capping_dropped_count": int(len(df_dropped)),
        "monitor_count": int(len(df_monitor)),
        "main_train_count": int(len(df_main)),
        "monitor_ratio_target": float(args.monitor_ratio),
        "monitor_ratio_actual_rows": float(len(df_monitor) / len(df_capped)) if len(df_capped) > 0 else 0.0,
        "max_per_group": int(args.max_per_group),
        "random_state": int(args.random_state),
        "num_groups_before": int(len(group_counts_before)),
        "num_groups_after": int(len(group_counts_after)),
        "largest_groups_before_top20": top_n_counts(group_counts_before, 20),
        "largest_groups_after_top20": top_n_counts(group_counts_after, 20),
        "main_patch_summary": summarize_numeric(df_main["pocket_num_residues"]),
        "main_peptide_summary": summarize_numeric(df_main["peptide_length"]),
        "main_rbsa_summary": summarize_numeric(df_main["rBSA_proxy"]),
        "monitor_patch_summary": summarize_numeric(df_monitor["pocket_num_residues"]),
        "monitor_peptide_summary": summarize_numeric(df_monitor["peptide_length"]),
        "monitor_rbsa_summary": summarize_numeric(df_monitor["rBSA_proxy"]),
        "main_bin_counts": df_main["bin_key"].value_counts().to_dict(),
        "monitor_bin_counts": df_monitor["bin_key"].value_counts().to_dict(),
        "main_selection_mode_counts": summarize_selection_modes(df_main),
        "monitor_selection_mode_counts": summarize_selection_modes(df_monitor),
        "main_sampling_selected_ratio": float(df_main["step5_selected_by_sampling"].mean()) if "step5_selected_by_sampling" in df_main.columns and len(df_main) > 0 else 0.0,
        "monitor_sampling_selected_ratio": float(df_monitor["step5_selected_by_sampling"].mean()) if "step5_selected_by_sampling" in df_monitor.columns and len(df_monitor) > 0 else 0.0,
    }

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] input_count                 = {input_count}", flush=True)
    print(f"[DONE] after_family_capping_count = {len(df_capped)}", flush=True)
    print(f"[DONE] family_capping_dropped     = {len(df_dropped)}", flush=True)
    print(f"[DONE] monitor_count              = {len(df_monitor)}", flush=True)
    print(f"[DONE] main_train_count           = {len(df_main)}", flush=True)
    print(f"[DONE] saved main      -> {out_main}", flush=True)
    print(f"[DONE] saved monitor   -> {out_monitor}", flush=True)
    print(f"[DONE] saved dropped   -> {out_dropped}", flush=True)
    print(f"[DONE] saved summary   -> {out_summary}", flush=True)


if __name__ == "__main__":
    main()

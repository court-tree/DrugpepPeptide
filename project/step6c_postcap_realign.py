from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from step6_stratified_sampling import (
    apply_bins,
    build_lookup,
    ensure_required_columns,
    largest_remainder_quota,
    load_table,
    make_bins_dict,
    save_jsonl,
    summarize_numeric,
)
from step7_family_capping_monitor_split_v2 import choose_group_key


def add_group_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "group_key" not in df.columns:
        df["group_key"] = df.apply(choose_group_key, axis=1)
    return df


def add_row_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df))
    return df


def add_sort_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "step5_selected_by_sampling" not in df.columns:
        df["step5_selected_by_sampling"] = False
    if "score_final" not in df.columns:
        df["score_final"] = 0.0
    if "sample_weight" not in df.columns:
        df["sample_weight"] = 0.0

    df["sort_is_sampling"] = df["step5_selected_by_sampling"].astype(int)
    df["sort_sample_weight"] = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(0.0)
    df["sort_score_final"] = pd.to_numeric(df["score_final"], errors="coerce").fillna(0.0)
    return df


def drop_existing_lookup_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    overlap_cols = [
        "count_ref",
        "count_gen",
        "P_ref",
        "P_gen",
        "density_ratio",
        "sample_weight",
        "sample_weight_raw",
    ]
    present = [col for col in overlap_cols if col in df.columns]
    if present:
        df = df.drop(columns=present)
    return df


def target_bin_quota(lookup: pd.DataFrame, total_count: int) -> dict[tuple[str, str, str], int]:
    cols = ["patch_bin", "peptide_bin", "rbsa_bin"]
    bin_keys = lookup[cols].apply(lambda r: tuple(r), axis=1)
    p_ref = pd.Series(lookup["P_ref"].values, index=bin_keys)
    quota = largest_remainder_quota(p_ref, total_count)
    return {k: int(v) for k, v in quota.to_dict().items()}


def select_task_floor(df_main: pd.DataFrame) -> set[int]:
    if "parent_task_id" not in df_main.columns:
        return set()
    task_best = (
        df_main.sort_values(
            ["sort_sample_weight", "sort_score_final", "row_id"],
            ascending=[False, False, True],
        )
        .groupby("parent_task_id", as_index=False)
        .head(1)
    )
    return set(task_best["row_id"].tolist())


def thin_main_to_quota(
    df_main: pd.DataFrame,
    quota_by_bin: dict[tuple[str, str, str], int],
    protected_row_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["patch_bin", "peptide_bin", "rbsa_bin"]
    df = df_main.copy()
    df["bin_key"] = df[cols].apply(lambda r: tuple(r), axis=1)

    kept_parts: list[pd.DataFrame] = []
    dropped_parts: list[pd.DataFrame] = []

    for bin_key, sub in df.groupby("bin_key", sort=False):
        sub = sub.copy()
        quota = quota_by_bin.get(bin_key, len(sub))
        protected = sub.loc[sub["row_id"].isin(protected_row_ids)].copy()
        flexible = sub.loc[~sub["row_id"].isin(protected_row_ids)].copy()

        keep_n = max(quota, len(protected))
        keep_n = min(keep_n, len(sub))
        need_flexible = max(0, keep_n - len(protected))

        flexible = flexible.sort_values(
            ["sort_sample_weight", "sort_score_final", "sort_is_sampling", "row_id"],
            ascending=[False, False, False, True],
        )
        kept = pd.concat([protected, flexible.head(need_flexible)], ignore_index=True)
        dropped = flexible.iloc[need_flexible:].copy()
        if not dropped.empty:
            dropped["drop_reason"] = "step6c_overrepresented_bin"

        kept_parts.append(kept)
        if not dropped.empty:
            dropped_parts.append(dropped)

    kept_df = pd.concat(kept_parts, ignore_index=True) if kept_parts else df.iloc[:0].copy()
    dropped_df = pd.concat(dropped_parts, ignore_index=True) if dropped_parts else df.iloc[:0].copy()
    return kept_df, dropped_df


def refill_from_donor_pool(
    df_main_kept: pd.DataFrame,
    df_donor: pd.DataFrame,
    quota_by_bin: dict[tuple[str, str, str], int],
    max_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    cols = ["patch_bin", "peptide_bin", "rbsa_bin"]
    main = df_main_kept.copy()
    donor = df_donor.copy()
    main["bin_key"] = main[cols].apply(lambda r: tuple(r), axis=1)
    donor["bin_key"] = donor[cols].apply(lambda r: tuple(r), axis=1)

    current_bin_counts = main.groupby("bin_key", sort=False).size().to_dict()
    current_group_counts = main["group_key"].value_counts().to_dict()

    selected_parts: list[pd.DataFrame] = []
    donor_unused_parts: list[pd.DataFrame] = []
    refill_counts: dict[str, int] = {}

    for bin_key, sub in donor.groupby("bin_key", sort=False):
        sub = sub.copy()
        target = quota_by_bin.get(bin_key, 0)
        current = int(current_bin_counts.get(bin_key, 0))
        deficit = max(0, target - current)
        refill_counts[str(bin_key)] = deficit

        if deficit <= 0:
            donor_unused_parts.append(sub)
            continue

        sub = sub.sort_values(
            ["sort_sample_weight", "sort_score_final", "sort_is_sampling", "row_id"],
            ascending=[False, False, False, True],
        )

        chosen_ids: list[int] = []
        for _, row in sub.iterrows():
            if len(chosen_ids) >= deficit:
                break
            group_key = str(row["group_key"])
            if int(current_group_counts.get(group_key, 0)) >= max_per_group:
                continue
            chosen_ids.append(int(row["row_id"]))
            current_group_counts[group_key] = int(current_group_counts.get(group_key, 0)) + 1

        chosen = sub.loc[sub["row_id"].isin(chosen_ids)].copy()
        unused = sub.loc[~sub["row_id"].isin(chosen_ids)].copy()
        if not chosen.empty:
            chosen["step6c_added_from_donor"] = True
            chosen["split"] = "main_train"
            selected_parts.append(chosen)
        if not unused.empty:
            donor_unused_parts.append(unused)

    selected_df = pd.concat(selected_parts, ignore_index=True) if selected_parts else donor.iloc[:0].copy()
    unused_df = pd.concat(donor_unused_parts, ignore_index=True) if donor_unused_parts else donor.iloc[:0].copy()
    return selected_df, unused_df, refill_counts


def normalize_optional_donor_table(
    df_donor_raw: pd.DataFrame,
    df_main: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str, bool]:
    """
    Step7 may legitimately produce an empty dropped file when family capping drops nothing.
    In that case, treat donor pool as empty instead of requiring full Step6-compatible columns.
    """
    if df_donor_raw is None or len(df_donor_raw) == 0:
        empty = df_main.iloc[:0].copy()
        return empty, "empty_donor_pool", "empty_donor_pool", True

    try:
        df_donor, donor_patch_size_source, donor_rbsa_source = ensure_required_columns(df_donor_raw, "df_donor")
        return df_donor, donor_patch_size_source, donor_rbsa_source, False
    except ValueError:
        empty = df_main.iloc[:0].copy()
        return empty, "empty_donor_pool", "empty_donor_pool", True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-1 Step 6C: family-aware post-Step7 realignment"
    )
    parser.add_argument("--gen_file", type=str, required=True, help="Step7 main JSONL/CSV")
    parser.add_argument("--donor_file", type=str, required=True, help="Step7 dropped JSONL/CSV")
    parser.add_argument("--ref_file", type=str, required=True, help="Reference JSONL/CSV")
    parser.add_argument("--out_survived", type=str, required=True, help="Main JSONL after family-aware repair")
    parser.add_argument("--out_lookup", type=str, required=True, help="Lookup CSV")
    parser.add_argument("--out_scored", type=str, required=True, help="Scored full JSONL")
    parser.add_argument("--out_summary", type=str, required=True, help="Summary JSON")
    parser.add_argument("--alpha", type=float, default=1.0, help="Compatibility field")
    parser.add_argument("--epsilon_gen", type=float, default=1e-5)
    parser.add_argument("--epsilon_ref", type=float, default=1e-6)
    parser.add_argument("--random_state", type=int, default=42, help="Compatibility field")
    parser.add_argument("--bins_mode", type=str, default="manual")
    parser.add_argument("--ensure_task_floor", action="store_true")
    parser.add_argument("--target_keep_ratio", type=float, default=1.0, help="Compatibility field")
    parser.add_argument("--max_per_group", type=int, required=True, help="Must match Step7 family cap")
    args = parser.parse_args()

    gen_path = Path(args.gen_file)
    donor_path = Path(args.donor_file)
    ref_path = Path(args.ref_file)
    out_survived = Path(args.out_survived)
    out_lookup = Path(args.out_lookup)
    out_scored = Path(args.out_scored)
    out_summary = Path(args.out_summary)

    print("=" * 80, flush=True)
    print("[START] Phase-1 Step 6C: Family-aware post-Step7 realignment", flush=True)
    print(f"[INPUT] gen_file     = {gen_path}", flush=True)
    print(f"[INPUT] donor_file   = {donor_path}", flush=True)
    print(f"[INPUT] ref_file     = {ref_path}", flush=True)
    print(f"[OUTPUT] survived    = {out_survived}", flush=True)
    print(f"[OUTPUT] lookup      = {out_lookup}", flush=True)
    print(f"[OUTPUT] scored      = {out_scored}", flush=True)
    print(f"[OUTPUT] summary     = {out_summary}", flush=True)
    print(f"[PARAM]  max_per_group = {args.max_per_group}", flush=True)
    print("=" * 80, flush=True)

    df_main_raw = load_table(gen_path)
    df_donor_raw = load_table(donor_path)
    df_ref_raw = load_table(ref_path)

    df_main, main_patch_size_source, main_rbsa_source = ensure_required_columns(df_main_raw, "df_main")
    df_donor, donor_patch_size_source, donor_rbsa_source, donor_was_empty = normalize_optional_donor_table(
        df_donor_raw, df_main
    )
    df_ref, ref_patch_size_source, ref_rbsa_source = ensure_required_columns(df_ref_raw, "df_ref")

    df_main = add_group_key(add_row_id(df_main))
    df_donor = add_group_key(add_row_id(df_donor))
    df_main = drop_existing_lookup_columns(df_main)
    df_donor = drop_existing_lookup_columns(df_donor)

    bins_dict = make_bins_dict(mode=args.bins_mode)
    df_main = apply_bins(df_main, bins_dict)
    df_donor = apply_bins(df_donor, bins_dict)
    df_ref = apply_bins(df_ref, bins_dict)

    lookup = build_lookup(
        df_gen=df_main,
        df_ref=df_ref,
        bins_dict=bins_dict,
        epsilon_gen=args.epsilon_gen,
        epsilon_ref=args.epsilon_ref,
    )

    cols = ["patch_bin", "peptide_bin", "rbsa_bin"]
    lookup_slice = lookup[
        cols + ["count_ref", "count_gen", "P_ref", "P_gen", "density_ratio", "sample_weight"]
    ]
    df_main_scored = pd.merge(df_main, lookup_slice, on=cols, how="left")
    df_donor_scored = pd.merge(df_donor, lookup_slice, on=cols, how="left")

    df_main_scored["sample_weight"] = df_main_scored["sample_weight"].fillna(0.0)
    df_donor_scored["sample_weight"] = df_donor_scored["sample_weight"].fillna(0.0)
    df_main_scored["is_kept"] = False
    df_donor_scored["is_kept"] = False

    df_main_scored = add_sort_keys(df_main_scored)
    df_donor_scored = add_sort_keys(df_donor_scored)

    protected_row_ids = select_task_floor(df_main_scored) if args.ensure_task_floor else set()
    quota_by_bin = target_bin_quota(lookup, len(df_main_scored))

    df_main_thinned, df_main_dropped_step6c = thin_main_to_quota(
        df_main_scored,
        quota_by_bin=quota_by_bin,
        protected_row_ids=protected_row_ids,
    )

    df_refill, _df_donor_unused, refill_counts = refill_from_donor_pool(
        df_main_kept=df_main_thinned,
        df_donor=df_donor_scored,
        quota_by_bin=quota_by_bin,
        max_per_group=args.max_per_group,
    )

    df_survived = pd.concat([df_main_thinned, df_refill], ignore_index=True)
    if "split" not in df_survived.columns:
        df_survived["split"] = "main_train"
    else:
        df_survived["split"] = df_survived["split"].where(df_survived["split"].notna(), "main_train")
        df_survived.loc[df_survived["split"].astype(str).str.strip().eq(""), "split"] = "main_train"
    df_survived["is_kept"] = True

    kept_ids = set(df_survived["row_id"].tolist())
    donor_kept_ids = set(df_refill["row_id"].tolist())
    df_main_scored["is_kept"] = df_main_scored["row_id"].isin(kept_ids)
    df_donor_scored["is_kept"] = df_donor_scored["row_id"].isin(donor_kept_ids)
    df_scored = pd.concat([df_main_scored, df_donor_scored], ignore_index=True)

    out_survived.parent.mkdir(parents=True, exist_ok=True)
    out_lookup.parent.mkdir(parents=True, exist_ok=True)
    out_scored.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    save_jsonl(df_survived, out_survived)
    lookup.to_csv(out_lookup, index=False)
    save_jsonl(df_scored, out_scored)

    summary = {
        "step6c_version": "family_aware_postcap_realign_v2",
        "bins_mode": args.bins_mode,
        "alpha_compat": args.alpha,
        "target_keep_ratio_compat": args.target_keep_ratio,
        "ensure_task_floor": args.ensure_task_floor,
        "max_per_group": args.max_per_group,
        "main_input_count": int(len(df_main)),
        "donor_input_count": int(len(df_donor)),
        "donor_pool_empty": bool(donor_was_empty),
        "ref_input_count": int(len(df_ref)),
        "survived_count": int(len(df_survived)),
        "survived_ratio_vs_main": float(len(df_survived) / len(df_main)) if len(df_main) > 0 else 0.0,
        "task_floor_kept_count": int(len(protected_row_ids)),
        "step6c_thinned_from_main_count": int(len(df_main_dropped_step6c)),
        "step6c_added_from_donor_count": int(len(df_refill)),
        "refill_counts_by_bin": refill_counts,
        "main_patch_size_source": main_patch_size_source,
        "donor_patch_size_source": donor_patch_size_source,
        "ref_patch_size_source": ref_patch_size_source,
        "main_rbsa_source": main_rbsa_source,
        "donor_rbsa_source": donor_rbsa_source,
        "ref_rbsa_source": ref_rbsa_source,
        "survived_patch_summary": summarize_numeric(df_survived["pocket_num_residues"]),
        "survived_peptide_summary": summarize_numeric(df_survived["peptide_length"]),
        "survived_rbsa_summary": summarize_numeric(df_survived["rBSA_proxy"]),
        "group_key_semantics": "reuse Step7 group_key semantics; prefer Step6X MMseqs2 receptor_seq_cluster, fallback to pdb_id, then parent_task_id",
        "note": (
            "Step 6C first thins over-represented bins within step7_main, then refills deficit bins only from "
            "step7_dropped donors whose group_key remains below max_per_group. It never exceeds the Step7 MMseqs2-family cap."
        ),
    }

    if "parent_task_id" in df_survived.columns:
        summary["survived_task_count"] = int(df_survived["parent_task_id"].nunique())

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] main_input_count          = {len(df_main)}", flush=True)
    print(f"[DONE] donor_input_count         = {len(df_donor)}", flush=True)
    print(f"[DONE] survived_count           = {len(df_survived)}", flush=True)
    print(f"[DONE] step6c_thinned_from_main = {len(df_main_dropped_step6c)}", flush=True)
    print(f"[DONE] step6c_added_from_donor  = {len(df_refill)}", flush=True)
    print(f"[DONE] saved survived  -> {out_survived}", flush=True)
    print(f"[DONE] saved lookup    -> {out_lookup}", flush=True)
    print(f"[DONE] saved scored    -> {out_scored}", flush=True)
    print(f"[DONE] saved summary   -> {out_summary}", flush=True)


if __name__ == "__main__":
    main()

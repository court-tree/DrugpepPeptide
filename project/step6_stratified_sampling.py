from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# IO helpers
# =========================================================
def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def summarize_numeric(series: pd.Series) -> dict[str, float]:
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


# =========================================================
# Column normalization
# =========================================================
def ensure_required_columns(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()

    # rBSA
    if "rBSA_proxy" in df.columns:
        rbsa_source = "rBSA_proxy"
    elif "rBSA" in df.columns:
        df["rBSA_proxy"] = df["rBSA"]
        rbsa_source = "rBSA"
    elif "rBSA_raw" in df.columns:
        df["rBSA_proxy"] = df["rBSA_raw"]
        rbsa_source = "rBSA_raw"
    else:
        raise ValueError(f"{name} 缺少 rBSA_proxy / rBSA / rBSA_raw 列")

    # patch size
    if "pocket_num_residues" in df.columns:
        patch_size_source = "pocket_num_residues"
    elif "pocket_size_6A" in df.columns:
        df["pocket_num_residues"] = df["pocket_size_6A"]
        patch_size_source = "pocket_size_6A_proxy"
    elif "n_contact_residues_step4" in df.columns:
        df["pocket_num_residues"] = df["n_contact_residues_step4"]
        patch_size_source = "n_contact_residues_step4_proxy"
    else:
        raise ValueError(f"{name} 缺少 pocket_num_residues / pocket_size_6A / n_contact_residues_step4 列")

    required = ["pocket_num_residues", "peptide_length", "rBSA_proxy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要列: {missing}")

    # quality score
    if "score_final" not in df.columns:
        rbsa = pd.to_numeric(df["rBSA_proxy"], errors="coerce").fillna(0.0)

        if "contact_coverage_6A" in df.columns:
            cov6 = pd.to_numeric(df["contact_coverage_6A"], errors="coerce").fillna(0.0)
        else:
            cov6 = pd.Series(0.0, index=df.index)

        if "covalent_bias_risk" in df.columns:
            bias = pd.to_numeric(df["covalent_bias_risk"], errors="coerce").fillna(0.0)
        else:
            bias = pd.Series(0.0, index=df.index)

        df["score_final"] = 0.60 * rbsa + 0.30 * cov6 - 0.15 * bias

    return df, patch_size_source, rbsa_source


# =========================================================
# Binning
# =========================================================
def make_bins_dict(mode: str = "manual") -> dict[str, list]:
    if mode != "manual":
        raise ValueError("当前 Step 6 修正版仍建议使用 mode='manual'")

    return {
        "patch": [-0.1, 25, 50, 100, np.inf],
        "peptide": [-0.1, 8, 14, np.inf],
        "rbsa": [-0.01, 0.35, 0.70, np.inf],
    }


def apply_bins(df: pd.DataFrame, bins_dict: dict) -> pd.DataFrame:
    df = df.copy()

    patch_cat = pd.cut(df["pocket_num_residues"], bins=bins_dict["patch"], include_lowest=True, right=False)
    peptide_cat = pd.cut(df["peptide_length"], bins=bins_dict["peptide"], include_lowest=True, right=False)
    rbsa_cat = pd.cut(df["rBSA_proxy"], bins=bins_dict["rbsa"], include_lowest=True, right=False)

    df["is_out_of_bins"] = patch_cat.isna() | peptide_cat.isna() | rbsa_cat.isna()

    # 使用字符串 bin，避免 merge/categorical 兼容问题
    df["patch_bin"] = patch_cat.astype(str)
    df["peptide_bin"] = peptide_cat.astype(str)
    df["rbsa_bin"] = rbsa_cat.astype(str)

    return df


def _interval_labels_from_breaks(breaks: list) -> list[str]:
    labels = []
    for i in range(len(breaks) - 1):
        left = breaks[i]
        right = breaks[i + 1]
        labels.append(str(pd.Interval(left, right, closed="left")))
    return labels


def build_lookup(
    df_gen: pd.DataFrame,
    df_ref: pd.DataFrame,
    bins_dict: dict,
    epsilon_gen: float,
    epsilon_ref: float,
) -> pd.DataFrame:
    cols = ["patch_bin", "peptide_bin", "rbsa_bin"]

    df_gen_clean = df_gen.loc[~df_gen["is_out_of_bins"]].copy()
    df_ref_clean = df_ref.loc[~df_ref["is_out_of_bins"]].copy()

    if df_gen_clean.empty:
        raise ValueError("df_gen_clean 在分箱后为空，请检查 bins_dict 与 df_gen 数值范围。")
    if df_ref_clean.empty:
        raise ValueError("df_ref_clean 在分箱后为空，请检查 bins_dict 与 df_ref 数值范围。")

    # 用 pd.cut(...).astype(str) 直接生成 full grid 标签，保证和 apply_bins 完全一致
    patch_labels = pd.cut(
        pd.Series([(bins_dict["patch"][i] + (0 if np.isinf(bins_dict["patch"][i+1]) else bins_dict["patch"][i+1])) / 2
                   if not np.isinf(bins_dict["patch"][i+1]) else bins_dict["patch"][i] + 1
                   for i in range(len(bins_dict["patch"]) - 1)]),
        bins=bins_dict["patch"],
        include_lowest=True,
        right=False,
    ).astype(str).tolist()

    peptide_labels = pd.cut(
        pd.Series([(bins_dict["peptide"][i] + (0 if np.isinf(bins_dict["peptide"][i+1]) else bins_dict["peptide"][i+1])) / 2
                   if not np.isinf(bins_dict["peptide"][i+1]) else bins_dict["peptide"][i] + 1
                   for i in range(len(bins_dict["peptide"]) - 1)]),
        bins=bins_dict["peptide"],
        include_lowest=True,
        right=False,
    ).astype(str).tolist()

    rbsa_labels = pd.cut(
        pd.Series([(bins_dict["rbsa"][i] + (0 if np.isinf(bins_dict["rbsa"][i+1]) else bins_dict["rbsa"][i+1])) / 2
                   if not np.isinf(bins_dict["rbsa"][i+1]) else bins_dict["rbsa"][i] + 0.1
                   for i in range(len(bins_dict["rbsa"]) - 1)]),
        bins=bins_dict["rbsa"],
        include_lowest=True,
        right=False,
    ).astype(str).tolist()

    full_index = pd.MultiIndex.from_product(
        [patch_labels, peptide_labels, rbsa_labels],
        names=cols,
    )

    p_ref = (
        df_ref_clean.groupby(cols, observed=False)
        .size()
        .reindex(full_index, fill_value=0)
        .reset_index(name="count_ref")
    )
    p_gen = (
        df_gen_clean.groupby(cols, observed=False)
        .size()
        .reindex(full_index, fill_value=0)
        .reset_index(name="count_gen")
    )

    total_ref = p_ref["count_ref"].sum()
    total_gen = p_gen["count_gen"].sum()
    if total_ref <= 0:
        raise ValueError("参考集有效计数为 0。")
    if total_gen <= 0:
        raise ValueError("生成集有效计数为 0。")

    p_ref["P_ref"] = p_ref["count_ref"] / total_ref
    p_gen["P_gen"] = p_gen["count_gen"] / total_gen

    lookup = pd.merge(
        p_ref[cols + ["count_ref", "P_ref"]],
        p_gen[cols + ["count_gen", "P_gen"]],
        on=cols,
        how="outer",
    )

    lookup["count_ref"] = lookup["count_ref"].fillna(0).astype(int)
    lookup["count_gen"] = lookup["count_gen"].fillna(0).astype(int)
    lookup["P_ref"] = lookup["P_ref"].fillna(0.0)
    lookup["P_gen"] = lookup["P_gen"].fillna(0.0)

    lookup.loc[lookup["count_ref"] == 0, "P_ref"] = epsilon_ref
    lookup["P_ref"] = lookup["P_ref"] / lookup["P_ref"].sum()

    lookup["density_ratio"] = lookup["P_ref"] / (lookup["P_gen"] + epsilon_gen)

    lookup["sample_weight_raw"] = np.log1p(lookup["density_ratio"])
    max_val = float(lookup["sample_weight_raw"].max())
    if max_val <= 0:
        lookup["sample_weight"] = 0.0
    else:
        lookup["sample_weight"] = lookup["sample_weight_raw"] / max_val

    return lookup
# =========================================================
# Deterministic quota sampling
# =========================================================
def largest_remainder_quota(weights: pd.Series, total_quota: int) -> pd.Series:
    if total_quota <= 0:
        return pd.Series(0, index=weights.index, dtype=int)

    weights = weights.fillna(0.0).astype(float)
    if weights.sum() <= 0:
        return pd.Series(0, index=weights.index, dtype=int)

    raw = weights / weights.sum() * total_quota
    floor = np.floor(raw).astype(int)
    remainder = raw - floor
    need = int(total_quota - floor.sum())

    quota = pd.Series(floor, index=weights.index, dtype=int)

    if need > 0:
        order = remainder.sort_values(ascending=False).index[:need]
        quota.loc[order] += 1

    return quota


def select_with_task_floor_and_bin_quota(
    df_scored: pd.DataFrame,
    lookup: pd.DataFrame,
    target_keep_count: int,
    ensure_task_floor: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    cols = ["patch_bin", "peptide_bin", "rbsa_bin"]
    df = df_scored.copy()

    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df))

    df["sort_key_1"] = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(0.0)
    df["sort_key_2"] = pd.to_numeric(df["score_final"], errors="coerce").fillna(0.0)

    kept_row_ids = set()
    task_floor_kept_count = 0

    # -------------------------------------------------
    # Stage A: 每个 task 先保 1 条
    # -------------------------------------------------
    if ensure_task_floor and "parent_task_id" in df.columns:
        task_best = (
            df.sort_values(["sort_key_1", "sort_key_2", "row_id"], ascending=[False, False, True])
            .groupby("parent_task_id", as_index=False)
            .head(1)
        )
        kept_row_ids.update(task_best["row_id"].tolist())
        task_floor_kept_count = len(task_best)

    # -------------------------------------------------
    # Stage B: 按 bin 配额补齐
    # -------------------------------------------------
    adjusted_target_keep_count = max(target_keep_count, len(kept_row_ids))

    clean_mask = ~df["is_out_of_bins"]
    clean_df = df.loc[clean_mask].copy()

    bin_keys = lookup[cols].apply(lambda r: tuple(r), axis=1)
    p_ref = pd.Series(lookup["P_ref"].values, index=bin_keys)
    quota = largest_remainder_quota(p_ref, adjusted_target_keep_count)

    kept_df = df.loc[df["row_id"].isin(kept_row_ids)].copy()
    if not kept_df.empty:
        kept_bin_counts = kept_df.groupby(cols, observed=False).size()
    else:
        kept_bin_counts = pd.Series(dtype=int)

    gen_bin_counts = clean_df.groupby(cols, observed=False).size()

    # 关键修复：不要再用 Series.loc[tuple]，改成 dict 访问
    remaining_quota_dict = quota.to_dict()

    for key, val in kept_bin_counts.items():
        if key in remaining_quota_dict:
            remaining_quota_dict[key] = max(0, int(remaining_quota_dict[key]) - int(val))

    clipped_quota_dict = remaining_quota_dict.copy()
    for key, current_quota in list(clipped_quota_dict.items()):
        already_kept_in_bin = int(kept_bin_counts.get(key, 0))
        available_in_bin = int(gen_bin_counts.get(key, 0))
        remaining_available = max(0, available_in_bin - already_kept_in_bin)
        clipped_quota_dict[key] = min(int(current_quota), remaining_available)

    clean_df["bin_key"] = clean_df[cols].apply(lambda r: tuple(r), axis=1)
    clean_df = clean_df.loc[~clean_df["row_id"].isin(kept_row_ids)].copy()
    clean_df = clean_df.sort_values(
        ["bin_key", "sort_key_1", "sort_key_2", "row_id"],
        ascending=[True, False, False, True]
    )

    selected_extra_ids = []
    for key, sub in clean_df.groupby("bin_key", sort=False):
        q = int(clipped_quota_dict.get(key, 0))
        if q <= 0:
            continue
        selected_extra_ids.extend(sub.head(q)["row_id"].tolist())

    kept_row_ids.update(selected_extra_ids)

    # -------------------------------------------------
    # Stage C: 若还没达到目标总量，全局按权重补齐
    # -------------------------------------------------
    if len(kept_row_ids) < adjusted_target_keep_count:
        need = adjusted_target_keep_count - len(kept_row_ids)
        extra_pool = (
            df.loc[~df["row_id"].isin(kept_row_ids)]
            .sort_values(["sort_key_1", "sort_key_2", "row_id"], ascending=[False, False, True])
            .head(need)
        )
        kept_row_ids.update(extra_pool["row_id"].tolist())

    out = df.loc[df["row_id"].isin(kept_row_ids)].copy()
    out["is_kept"] = True

    stats = {
        "task_floor_kept_count": int(task_floor_kept_count),
        "adjusted_target_keep_count": int(adjusted_target_keep_count),
        "final_kept_count": int(len(out)),
    }
    return out, stats


# =========================================================
# Main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-1 Step 6 (fixed): deterministic stratified quota sampling"
    )
    parser.add_argument("--gen_file", type=str, required=True, help="候选池文件: jsonl/csv")
    parser.add_argument("--ref_file", type=str, required=True, help="真实参考集文件: jsonl/csv")
    parser.add_argument("--out_survived", type=str, required=True, help="采样后保留样本 JSONL")
    parser.add_argument("--out_lookup", type=str, required=True, help="三维联合分布特征表 CSV")
    parser.add_argument("--out_scored", type=str, required=True, help="全量打分后 JSONL")
    parser.add_argument("--out_summary", type=str, required=True, help="summary JSON")
    parser.add_argument("--alpha", type=float, default=1.0, help="目标保留比例缩放")
    parser.add_argument("--epsilon_gen", type=float, default=1e-5, help="生成分布防除零项")
    parser.add_argument("--epsilon_ref", type=float, default=1e-6, help="参考分布未覆盖格子的最小探索概率")
    parser.add_argument("--random_state", type=int, default=42, help="兼容字段；本版不依赖随机抽样")
    parser.add_argument("--bins_mode", type=str, default="manual", help="当前建议 manual")
    parser.add_argument("--ensure_task_floor", action="store_true", help="每个 parent_task_id 至少保 1 条")
    parser.add_argument("--target_keep_ratio", type=float, default=0.40, help="最终保留比例（相对 gen_input_count）")
    args = parser.parse_args()

    gen_path = Path(args.gen_file)
    ref_path = Path(args.ref_file)

    out_survived = Path(args.out_survived)
    out_lookup = Path(args.out_lookup)
    out_scored = Path(args.out_scored)
    out_summary = Path(args.out_summary)

    print("=" * 80, flush=True)
    print("[START] Phase-1 Step 6 fixed: Deterministic Stratified Sampling", flush=True)
    print(f"[INPUT] gen_file   = {gen_path}", flush=True)
    print(f"[INPUT] ref_file   = {ref_path}", flush=True)
    print(f"[OUTPUT] survived  = {out_survived}", flush=True)
    print(f"[OUTPUT] lookup    = {out_lookup}", flush=True)
    print(f"[OUTPUT] scored    = {out_scored}", flush=True)
    print(f"[OUTPUT] summary   = {out_summary}", flush=True)
    print("=" * 80, flush=True)

    df_gen_raw = load_table(gen_path)
    df_ref_raw = load_table(ref_path)

    df_gen, gen_patch_size_source, gen_rbsa_source = ensure_required_columns(df_gen_raw, "df_gen")
    df_ref, ref_patch_size_source, ref_rbsa_source = ensure_required_columns(df_ref_raw, "df_ref")

    bins_dict = make_bins_dict(mode=args.bins_mode)

    df_gen = apply_bins(df_gen, bins_dict)
    df_ref = apply_bins(df_ref, bins_dict)

    lookup = build_lookup(
        df_gen=df_gen,
        df_ref=df_ref,
        bins_dict=bins_dict,
        epsilon_gen=args.epsilon_gen,
        epsilon_ref=args.epsilon_ref,
    )

    cols = ["patch_bin", "peptide_bin", "rbsa_bin"]
    df_gen_scored = pd.merge(
        df_gen,
        lookup[
            cols + [
                "count_ref",
                "count_gen",
                "P_ref",
                "P_gen",
                "density_ratio",
                "sample_weight",
            ]
        ],
        on=cols,
        how="left",
    )

    if "row_id" not in df_gen_scored.columns:
        df_gen_scored["row_id"] = np.arange(len(df_gen_scored))

    df_gen_scored["sample_weight"] = df_gen_scored["sample_weight"].fillna(0.0)
    df_gen_scored["is_kept"] = False

    target_keep_count = int(round(len(df_gen_scored) * args.target_keep_ratio * args.alpha))
    target_keep_count = max(1, target_keep_count)

    df_survived, keep_stats = select_with_task_floor_and_bin_quota(
        df_scored=df_gen_scored,
        lookup=lookup,
        target_keep_count=target_keep_count,
        ensure_task_floor=args.ensure_task_floor,
    )

    kept_row_ids = set(df_survived["row_id"].tolist())
    df_gen_scored["is_kept"] = df_gen_scored["row_id"].isin(kept_row_ids)

    out_survived.parent.mkdir(parents=True, exist_ok=True)
    out_lookup.parent.mkdir(parents=True, exist_ok=True)
    out_scored.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    save_jsonl(df_survived, out_survived)
    lookup.to_csv(out_lookup, index=False)
    save_jsonl(df_gen_scored, out_scored)

    ref_out_of_bins_count = int(df_ref["is_out_of_bins"].sum())

    def serialize_bins(bins: list) -> list[str]:
        out = []
        for x in bins:
            if x == np.inf:
                out.append("inf")
            else:
                out.append(str(x))
        return out

    summary = {
        "step6_version": "global_stratified_sampling_v5_quota_fullgrid",
        "bins_mode": args.bins_mode,
        "bins_dict": {
            "patch": serialize_bins(bins_dict["patch"]),
            "peptide": serialize_bins(bins_dict["peptide"]),
            "rbsa": serialize_bins(bins_dict["rbsa"]),
        },
        "alpha": args.alpha,
        "epsilon_gen": args.epsilon_gen,
        "epsilon_ref": args.epsilon_ref,
        "random_state": args.random_state,
        "ensure_task_floor": args.ensure_task_floor,
        "target_keep_ratio": args.target_keep_ratio,
        "target_keep_count": int(target_keep_count),
        "adjusted_target_keep_count": int(keep_stats["adjusted_target_keep_count"]),
        "task_floor_kept_count": int(keep_stats["task_floor_kept_count"]),
        "gen_input_count": int(len(df_gen)),
        "ref_input_count": int(len(df_ref)),
        "gen_out_of_bins_count": int(df_gen_scored["is_out_of_bins"].sum()),
        "ref_out_of_bins_count": ref_out_of_bins_count,
        "survived_count": int(len(df_survived)),
        "survived_ratio": float(len(df_survived) / len(df_gen)) if len(df_gen) > 0 else 0.0,
        "gen_patch_size_source": gen_patch_size_source,
        "ref_patch_size_source": ref_patch_size_source,
        "gen_rbsa_source": gen_rbsa_source,
        "ref_rbsa_source": ref_rbsa_source,
        "patch_size_semantics": "proxy_from_step4_if_missing",
        "lookup_nonzero_ref_bins": int((lookup["count_ref"] > 0).sum()),
        "lookup_nonzero_gen_bins": int((lookup["count_gen"] > 0).sum()),
        "sample_weight_summary": summarize_numeric(df_gen_scored["sample_weight"]),
        "gen_patch_summary": summarize_numeric(df_gen["pocket_num_residues"]),
        "gen_peptide_summary": summarize_numeric(df_gen["peptide_length"]),
        "gen_rbsa_summary": summarize_numeric(df_gen["rBSA_proxy"]),
        "ref_patch_summary": summarize_numeric(df_ref["pocket_num_residues"]),
        "ref_peptide_summary": summarize_numeric(df_ref["peptide_length"]),
        "ref_rbsa_summary": summarize_numeric(df_ref["rBSA_proxy"]),
    }

    if "parent_task_id" in df_survived.columns:
        summary["survived_task_count"] = int(df_survived["parent_task_id"].nunique())

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] gen_input_count = {len(df_gen)}", flush=True)
    print(f"[DONE] survived_count  = {len(df_survived)}", flush=True)
    print(
        f"[DONE] survived_ratio  = {len(df_survived) / len(df_gen):.4f}"
        if len(df_gen) > 0 else "[DONE] survived_ratio = 0.0",
        flush=True
    )
    print(f"[DONE] saved survived  -> {out_survived}", flush=True)
    print(f"[DONE] saved lookup    -> {out_lookup}", flush=True)
    print(f"[DONE] saved scored    -> {out_scored}", flush=True)
    print(f"[DONE] saved summary   -> {out_summary}", flush=True)


if __name__ == "__main__":
    main()
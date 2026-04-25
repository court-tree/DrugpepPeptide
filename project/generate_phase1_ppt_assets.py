from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(r"e:\pep")
PROJECT_ROOT = ROOT / "project"
RUN_ROOT = PROJECT_ROOT / "runs" / "full_run"
SMOKE_ROOT = PROJECT_ROOT / "runs" / "smoke_test"
OUT_DIR = PROJECT_ROOT / "ppt_phase1"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_first_jsonl(path: Path, n: int = 3) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for _, line in zip(range(n), fh):
            rows.append(json.loads(line))
    return rows


def setup_output() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_flow_overview(step1: dict, step5: dict, step6: dict, step7: dict) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f6f1")

    title = "PeptideCLIP Phase-1 Full Pipeline Overview"
    ax.text(0.05, 0.93, title, fontsize=24, fontweight="bold", color="#16213e")
    ax.text(
        0.05,
        0.885,
        "From raw complex structures to a balanced peptide-patch training set",
        fontsize=13,
        color="#4f5d75",
    )

    steps = [
        ("Step 1\nX-ray QC", f"62,180 input\n40,457 passed", "#d8e2dc"),
        ("Step 2\nBidirectional tasks", "153,294 tasks\nA→B / B→A", "#cde3f8"),
        ("Step 3\nAnchor-driven windows", "4,096,888 raw candidates", "#fde2b8"),
        ("Step 4\nPhysical post-scoring", "1,083,526 survivors", "#ffd6d6"),
        ("Step 5\nDedup + probabilistic sampling", "203,086 kept\nTop-k=5", "#d9c2f0"),
        ("Step 6\nReference-aware stratified keep", f"{step6['survived_count']:,} kept", "#bee1e6"),
        (
            "Step 7\nFamily cap + monitor split",
            f"{step7['main_train_count']:,} main\n{step7['monitor_count']:,} monitor",
            "#cdeac0",
        ),
    ]

    x_positions = [0.05, 0.18, 0.31, 0.44, 0.57, 0.70, 0.83]
    box_w = 0.11
    box_h = 0.27
    y = 0.48

    for idx, ((label, detail, color), x) in enumerate(zip(steps, x_positions), start=1):
        patch = FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            linewidth=1.5,
            edgecolor="#30475e",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + 0.01, y + box_h - 0.05, label, fontsize=14, fontweight="bold", color="#1b263b")
        ax.text(x + 0.01, y + 0.06, detail, fontsize=12, color="#243b53")
        if idx < len(steps):
            ax.annotate(
                "",
                xy=(x + box_w + 0.01, y + box_h / 2),
                xytext=(x + box_w + 0.035, y + box_h / 2),
                arrowprops=dict(arrowstyle="->", lw=2, color="#1d3557"),
            )

    notes = [
        "Phase-1 scientific goal: cut native peptide-like fragments from real interfaces, not random short-chain neighbors.",
        "Key innovation sits in Step 5: absolute dedup + eligible pool + weighted sampling keeps interface quality while preserving 8-20 aa diversity.",
        "The current full run is based on the upgraded project/ pipeline, not the older prototype scripts.",
    ]
    yy = 0.28
    for note in notes:
        ax.text(0.07, yy, f"- {note}", fontsize=13, color="#283618")
        yy -= 0.07

    ax.text(
        0.05,
        0.06,
        f"QC pass rate: {step1['pass_rate'] * 100:.1f}%   |   Step 5 retention vs Step 4: {step5['total_candidates_after_step5'] / step5['total_candidates_before_step5'] * 100:.1f}%   |   Main train set: {step7['main_train_count']:,}",
        fontsize=12,
        color="#495057",
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "flow_overview.png", bbox_inches="tight")
    plt.close(fig)


def draw_step1_failures(step1: dict) -> None:
    reasons = step1["reason_counts"].copy()
    reasons.pop("pass", None)
    labels = [
        "insufficient\nchain instances",
        "delta SASA\nfailed",
        "low interface\narea",
        "empty\nstructure",
        "no contacting\npairs",
    ]
    values = [
        reasons["insufficient_chain_instances"],
        reasons["delta_sasa_failed"],
        reasons["low_interface_buried_area"],
        reasons["empty_structure"],
        reasons["no_contacting_chain_pairs"],
    ]

    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=160)
    fig.patch.set_facecolor("#faf7f2")
    bars = ax.bar(labels, values, color=["#d62828", "#f77f00", "#fcbf49", "#6c757d", "#457b9d"])
    ax.set_title("Step 1 QC Rejection Reasons", fontsize=20, fontweight="bold")
    ax.set_ylabel("Structure count")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 150, f"{val:,}", ha="center", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "step1_failures.png", bbox_inches="tight")
    plt.close(fig)


def draw_stage_reduction(step1: dict, step5: dict, step6: dict, step7: dict) -> None:
    labels = [
        "Step1\npassed",
        "Step2\ntasks",
        "Step3\ncandidates",
        "Step4\nscored",
        "Step5\nsampled",
        "Step6\nbalanced",
        "Step7\nmain+monitor",
    ]
    values = [
        step1["n_passed"],
        step5["num_parent_tasks"],
        step5["step3_input_count"],
        step5["step4_input_count"],
        step5["total_candidates_after_step5"],
        step6["survived_count"],
        step7["input_count"],
    ]
    colors = ["#588157", "#4ea8de", "#f4a261", "#e76f51", "#8d5fd3", "#2a9d8f", "#6a994e"]

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    fig.patch.set_facecolor("#f8f9fa")
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Full-Run Data Funnel", fontsize=20, fontweight="bold")
    ax.set_ylabel("Record count")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.ticklabel_format(style="plain", axis="y")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.01, f"{val:,}", ha="center", fontsize=10, rotation=0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "stage_reduction.png", bbox_inches="tight")
    plt.close(fig)


def draw_step5_story(step5: dict) -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=160)
    fig.patch.set_facecolor("#fbf8ff")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(4, 93, "Step 5: Dedup + Probabilistic Sampling", fontsize=22, fontweight="bold", color="#2b2d42")
    ax.text(4, 87, "Purpose: avoid near-duplicate windows and keep peptide length diversity within 8-20 aa", fontsize=12.5, color="#4a4e69")

    panels = [
        (5, 52, 25, 24, "#ffd6e0", "1. Absolute dedup", "Drop near-clones with high IoU and similar anchors."),
        (37, 52, 25, 24, "#dfe7fd", "2. Eligible pool", "Require contact coverage >= 0.5, enough contacts, and acceptable rBSA."),
        (69, 52, 25, 24, "#d8f3dc", "3. Weighted sampling", "Coverage becomes sampling weight instead of hard greedy top-k."),
    ]
    for x, y, w, h, color, title, text in panels:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor=color, edgecolor="#495057", linewidth=1.5))
        ax.text(x + 1.5, y + h - 5, title, fontsize=15, fontweight="bold", color="#22223b")
        ax.text(x + 1.5, y + 6, text, fontsize=11.5, color="#343a40", wrap=True)

    ax.annotate("", xy=(31, 64), xytext=(36, 64), arrowprops=dict(arrowstyle="->", lw=2, color="#495057"))
    ax.annotate("", xy=(63, 64), xytext=(68, 64), arrowprops=dict(arrowstyle="->", lw=2, color="#495057"))

    before = step5["total_candidates_before_step5"]
    dedup_drop = step5["total_abs_redundant_dropped"]
    topk_drop = step5["total_topk_dropped"]
    after = step5["total_candidates_after_step5"]
    length_buckets = step5["final_length_bucket"]

    ax.text(6, 38, f"Before Step 5: {before:,}", fontsize=16, fontweight="bold", color="#1d3557")
    ax.text(6, 31, f"Absolute redundancy dropped: {dedup_drop:,}", fontsize=14, color="#c1121f")
    ax.text(6, 24, f"Sampling / fallback dropped: {topk_drop:,}", fontsize=14, color="#6a4c93")
    ax.text(6, 17, f"After Step 5: {after:,}", fontsize=16, fontweight="bold", color="#2a9d8f")

    bucket_items = [
        ("8-10 aa", length_buckets["8-10"], "#457b9d"),
        ("11-15 aa", length_buckets["11-15"], "#2a9d8f"),
        ("16-20 aa", length_buckets["16-20"], "#f4a261"),
    ]
    base_x = 58
    for idx, (label, value, color) in enumerate(bucket_items):
        x = base_x + idx * 12
        height = 22 * (value / max(v for _, v, _ in bucket_items))
        ax.add_patch(Rectangle((x, 14), 8, height, facecolor=color, edgecolor="none"))
        ax.text(x + 4, 12, label, ha="center", va="top", fontsize=10.5)
        ax.text(x + 4, 15 + height, f"{value:,}", ha="center", fontsize=10.5)
    ax.text(57, 41, "Final peptide-length distribution", fontsize=14, fontweight="bold", color="#22223b")
    ax.text(57, 35, "Sampling keeps short, medium, and longer fragments alive.", fontsize=11.5, color="#495057")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "step5_story.png", bbox_inches="tight")
    plt.close(fig)


def draw_case_fragment(sample_rows: list[dict]) -> None:
    row_short = sample_rows[0]
    row_long = sample_rows[2]

    fig, ax = plt.subplots(figsize=(14, 7.6), dpi=160)
    fig.patch.set_facecolor("#f8f5f2")
    ax.set_xlim(22, 40)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(22, 9.2, "Example: one parent task can keep multiple valid fragments", fontsize=22, fontweight="bold", color="#1d3557")
    ax.text(22, 8.5, "PDB 1a15 | receptor chain A | peptide-source chain B | sampled from the same task", fontsize=12.5, color="#495057")

    for pos in range(24, 39):
        ax.add_patch(Rectangle((pos, 5.9), 0.9, 1.0, facecolor="#dee2e6", edgecolor="white"))
        ax.text(pos + 0.45, 6.4, str(pos), ha="center", va="center", fontsize=9)

    ax.text(22.1, 6.45, "source chain", fontsize=12, fontweight="bold")

    ax.add_patch(Rectangle((24, 3.9), 9, 1.0, facecolor="#f4a261", edgecolor="#6d6875"))
    ax.text(22.1, 4.4, "kept window A", fontsize=12, fontweight="bold", color="#bc6c25")
    ax.text(33.3, 4.4, "9 aa | 24-32 | coverage 1.00", fontsize=12, color="#7f5539")

    ax.add_patch(Rectangle((24, 1.9), 15, 1.0, facecolor="#2a9d8f", edgecolor="#6d6875"))
    ax.text(22.1, 2.4, "kept window B", fontsize=12, fontweight="bold", color="#1b4332")
    ax.text(39.3, 2.4, "15 aa | 24-38 | coverage 0.93", fontsize=12, color="#2d6a4f", ha="right")

    ax.annotate("anchor", xy=(28.5, 5.8), xytext=(28.5, 7.8), ha="center", arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#e63946"), color="#e63946", fontsize=11)

    ax.text(
        22,
        0.55,
        "Interpretation: the pipeline does not collapse to only one short top-scoring clone; it preserves distinct, physically valid fragment lengths from the same interface.",
        fontsize=12.5,
        color="#343a40",
    )

    plt.tight_layout()
    fig.savefig(OUT_DIR / "case_fragment.png", bbox_inches="tight")
    plt.close(fig)


def build_markdown(
    step1: dict,
    step5: dict,
    step6: dict,
    step6c: dict,
    step7: dict,
    step8: dict,
    sample_rows: list[dict],
) -> None:
    md = f"""---
title: "PeptideCLIP 阶段一可视化汇报"
author: "Codex for session1"
date: "2026-04-12"
lang: "zh-CN"
---

# PeptideCLIP 阶段一

高保真 Peptide-Patch 数据底座构建

- 目标：从真实蛋白复合物界面中切出可训练的 peptide-patch 片段
- 核心问题：切到的是“受体主导的界面片段”，而不是原链邻域的随机短截段
- 当前汇报基于 `project/` 下升级版全流程与 `project/runs/full_run`

# 研究目标

要解决的不是“怎么把链剪短”，而是三件事：

- 先筛掉结构质量差、界面不可靠的复合物
- 再围绕真实接触锚点生成候选片段
- 最后在保证物理合理性的同时，保留长度多样性和家族分布平衡

这套流程最终服务于后续模型训练所需的主训练集与 monitor 集。

# 全流程总览

![](flow_overview.png){{ width=95% }}

# Step 1 结构质控

![](step1_failures.png){{ width=78% }}

- 输入复合物：{step1['n_total']:,}
- 通过质控：{step1['n_passed']:,}，通过率 {step1['pass_rate'] * 100:.1f}%
- 最大拦截原因：`insufficient_chain_instances`，说明很多结构天然不满足后续双向切片需求

# 数据漏斗

![](stage_reduction.png){{ width=88% }}

- Step 2 把通过质控的复合物展开为 {step5['num_parent_tasks']:,} 个双向任务
- Step 3 由真实接触锚点扩展出 {step5['step3_input_count']:,} 个原始候选窗口
- Step 4 用 rBSA、contact coverage、patch size 等物理特征把候选收缩到 {step5['step4_input_count']:,}

# Step 5 核心创新

![](step5_story.png){{ width=93% }}

- Step 5 前输入：{step5['total_candidates_before_step5']:,}
- 绝对冗余去除：{step5['total_abs_redundant_dropped']:,}
- 最终保留：{step5['total_candidates_after_step5']:,}
- 长度分布：8-10 aa {step5['final_length_bucket']['8-10']:,}；11-15 aa {step5['final_length_bucket']['11-15']:,}；16-20 aa {step5['final_length_bucket']['16-20']:,}

# 真实切片案例

![](case_fragment.png){{ width=90% }}

- 同一个 parent task 可以保留多个“不是同一个克隆”的有效片段
- 示例任务来自 `1a15 / B_as_peptide__A_as_receptor`
- 样例短片段：{sample_rows[0]['peptide_start_resseq']}-{sample_rows[0]['peptide_end_resseq']}，长度 {sample_rows[0]['peptide_length']} aa，coverage={sample_rows[0]['contact_coverage_6A']:.2f}
- 样例长片段：{sample_rows[2]['peptide_start_resseq']}-{sample_rows[2]['peptide_end_resseq']}，长度 {sample_rows[2]['peptide_length']} aa，coverage={sample_rows[2]['contact_coverage_6A']:.2f}

# 阶段一后半程的平衡化处理

- Step 6：参考真实分布做分层保留，从 {step5['total_candidates_after_step5']:,} 压到 {step6['survived_count']:,}
- Step 6C：在不打破 family cap 的前提下再次重对齐，主集合保留 {step6c['survived_count']:,}
- Step 7：家族上限 + monitor split，得到 main {step7['main_train_count']:,} / monitor {step7['monitor_count']:,}

当前 monitor 占比约 {step7['monitor_ratio_actual_rows'] * 100:.2f}%。

# 最终产出 Step 8

- Step 8 成功写出样本：{step8['success_count']:,}，错误数 {step8['error_count']:,}
- split 结果：main_train {step8['split_counts']['main_train']:,}；monitor {step8['split_counts']['monitor']:,}
- 采样来源：sampling {step8['selection_mode_counts']['sampling']:,}；fallback {step8['selection_mode_counts']['fallback_score']:,}
- 端点情况：天然末端 {step8['terminal_native_count']:,}，占比 {step8['terminal_native_ratio'] * 100:.1f}%
- 封端覆盖：N cap {step8['n_cap_count']:,} ({step8['n_cap_ratio'] * 100:.1f}%)；C cap {step8['c_cap_count']:,} ({step8['c_cap_ratio'] * 100:.1f}%)

# 目前结果

- 全流程已经从“切片”走到了“可训练数据集组织”
- 当前最终可用样本数：{step8['success_count']:,}
- 其中主训练集 {step8['split_counts']['main_train']:,}，monitor 集 {step8['split_counts']['monitor']:,}
- 主集合 peptide 长度中位数：{step7['main_peptide_summary']['median']:.0f} aa
- 主集合 patch 大小中位数：{step7['main_patch_summary']['median']:.0f}
- 主集合 rBSA 中位数：{step7['main_rbsa_summary']['median']:.3f}

# 可以向老师强调的点

- 这不是简单切肽，而是“以真实界面锚点为中心”的片段生成
- Step 5 没有采用死板 greedy Top-k，而是用了“先去冗余，再按接触密度带权抽样”
- 后续又做了参考分布对齐、family cap 和 monitor split，数据集更适合训练与评估
- 整体上，阶段一已经形成了从结构质控到训练样本组织的完整闭环
"""
    (OUT_DIR / "phase1_presentation.md").write_text(md, encoding="utf-8")


def main() -> None:
    setup_output()
    step1 = load_json(RUN_ROOT / "step1" / "step1_summary.json")
    step5 = load_json(RUN_ROOT / "step5" / "step5_summary.json")
    step6 = load_json(RUN_ROOT / "step6_align" / "step6_summary.json")
    step6c = load_json(RUN_ROOT / "step6c_align" / "step6c_main_summary.json")
    step7 = load_json(RUN_ROOT / "step7" / "step7_summary.json")
    step8 = load_json(RUN_ROOT / "step8" / "lmdb" / "step8_summary.json")
    sample_rows = read_first_jsonl(SMOKE_ROOT / "step5" / "step5_final.jsonl", n=3)

    draw_flow_overview(step1, step5, step6, step7)
    draw_step1_failures(step1)
    draw_stage_reduction(step1, step5, step6, step7)
    draw_step5_story(step5)
    draw_case_fragment(sample_rows)
    build_markdown(step1, step5, step6, step6c, step7, step8, sample_rows)


if __name__ == "__main__":
    main()

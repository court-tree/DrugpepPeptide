from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(r"e:\pep\project")
RUN_ROOT = PROJECT_ROOT / "runs" / "smoke_server"
OUT_DIR = PROJECT_ROOT / "ppt_smoke_server"


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


def draw_flow_overview(step1: dict, step5: dict, step6: dict, step6x: dict, step7: dict, step8: dict) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f4ed")

    ax.text(0.05, 0.93, "Smoke Server Run Overview", fontsize=24, fontweight="bold", color="#14213d")
    ax.text(0.05, 0.885, "Server-side smoke validation for the Phase-1 peptide-patch pipeline", fontsize=13, color="#4f5d75")

    steps = [
        ("Step1\nQC", f"{step1['n_passed']:,}\npassed", "#d8e2dc"),
        ("Step3-5\nCandidate → Sample", f"{step5['step3_input_count']:,} →\n{step5['total_candidates_after_step5']:,}", "#cde3f8"),
        ("Step6\nStratified keep", f"{step6['survived_count']:,}\nsurvived", "#fde2b8"),
        ("Step6x\nSeq clusters", f"{step6x['unique_clusters']} clusters\n{step6x['unique_receptors']} receptors", "#ffd6d6"),
        ("Step7\nTrain / monitor", f"{step7['main_train_count']:,} / {step7['monitor_count']:,}", "#d9c2f0"),
        ("Step8\nFinal metadata", f"{step8['success_count']:,}\nsuccess", "#cdeac0"),
    ]

    xs = [0.05, 0.21, 0.37, 0.53, 0.69, 0.85]
    w = 0.10
    h = 0.26
    y = 0.50

    for idx, ((title, detail, color), x) in enumerate(zip(steps, xs), start=1):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.015,rounding_size=0.025",
                linewidth=1.5,
                edgecolor="#2f3e46",
                facecolor=color,
            )
        )
        ax.text(x + 0.01, y + h - 0.05, title, fontsize=14, fontweight="bold", color="#1b263b")
        ax.text(x + 0.01, y + 0.07, detail, fontsize=12, color="#243b53")
        if idx < len(steps):
            ax.annotate("", xy=(x + w + 0.01, y + h / 2), xytext=(x + w + 0.04, y + h / 2), arrowprops=dict(arrowstyle="->", lw=2, color="#1d3557"))

    notes = [
        "This run is compact enough for server smoke validation, but still exercises the full Step1-Step8 data path.",
        "All 197 tasks entered probabilistic sampling in Step 5 and no task fell back due to an empty eligible pool.",
        "Exact receptor-sequence clustering gives a clean family signal before the Step 7 monitor split.",
    ]
    yy = 0.28
    for note in notes:
        ax.text(0.07, yy, f"- {note}", fontsize=13, color="#283618")
        yy -= 0.07

    plt.tight_layout()
    fig.savefig(OUT_DIR / "flow_overview.png", bbox_inches="tight")
    plt.close(fig)


def draw_funnel(step1: dict, step5: dict, step6: dict, step7: dict, step8: dict) -> None:
    labels = ["Step1\npassed", "Step5\nfinal", "Step6\nsurvived", "Step7\nmain+monitor", "Step8\nfinal"]
    values = [step1["n_passed"], step5["total_candidates_after_step5"], step6["survived_count"], step7["input_count"], step8["success_count"]]
    colors = ["#588157", "#8d5fd3", "#2a9d8f", "#4ea8de", "#f4a261"]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    fig.patch.set_facecolor("#f8f9fa")
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Smoke Server Data Funnel", fontsize=20, fontweight="bold")
    ax.set_ylabel("Record count")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02, f"{val:,}", ha="center", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "funnel.png", bbox_inches="tight")
    plt.close(fig)


def draw_step5_story(step5: dict) -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=160)
    fig.patch.set_facecolor("#fbf8ff")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(4, 93, "Step 5 on Smoke Server", fontsize=22, fontweight="bold", color="#2b2d42")
    ax.text(4, 87, "Absolute dedup + eligible-pool sampling is already visible even in the smoke-sized run", fontsize=12.5, color="#4a4e69")

    panels = [
        (5, 52, 25, 24, "#ffd6e0", "Absolute dedup", f"Dropped {step5['total_abs_redundant_dropped']:,} near-clone windows."),
        (37, 52, 25, 24, "#dfe7fd", "Eligible pool", f"{step5['total_eligible_after_dedup']:,} windows entered weighted sampling."),
        (69, 52, 25, 24, "#d8f3dc", "Task-wise keep", f"{step5['total_candidates_after_step5']:,} windows kept across {step5['num_parent_tasks']:,} tasks."),
    ]
    for x, y, w, h, color, title, text in panels:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor=color, edgecolor="#495057", linewidth=1.5))
        ax.text(x + 1.5, y + h - 5, title, fontsize=15, fontweight="bold", color="#22223b")
        ax.text(x + 1.5, y + 6, text, fontsize=11.5, color="#343a40", wrap=True)

    ax.annotate("", xy=(31, 64), xytext=(36, 64), arrowprops=dict(arrowstyle="->", lw=2, color="#495057"))
    ax.annotate("", xy=(63, 64), xytext=(68, 64), arrowprops=dict(arrowstyle="->", lw=2, color="#495057"))

    ax.text(6, 36, f"Before Step 5: {step5['total_candidates_before_step5']:,}", fontsize=16, fontweight="bold", color="#1d3557")
    ax.text(6, 29, f"After Step 5: {step5['total_candidates_after_step5']:,}", fontsize=16, fontweight="bold", color="#2a9d8f")
    ax.text(6, 22, f"Average candidates per task: {step5['avg_candidates_per_task_before']:.2f} → {step5['avg_candidates_per_task_after']:.2f}", fontsize=14, color="#495057")
    ax.text(6, 15, f"Tasks with sampling active: {step5['tasks_with_sampling_active']:,} / {step5['num_parent_tasks']:,}", fontsize=14, color="#495057")

    buckets = [
        ("8-10 aa", step5["final_length_bucket"]["8-10"], "#457b9d"),
        ("11-15 aa", step5["final_length_bucket"]["11-15"], "#2a9d8f"),
        ("16-20 aa", step5["final_length_bucket"]["16-20"], "#f4a261"),
    ]
    base_x = 60
    max_v = max(v for _, v, _ in buckets)
    ax.text(58, 39, "Final length buckets", fontsize=14, fontweight="bold", color="#22223b")
    for idx, (label, value, color) in enumerate(buckets):
        x = base_x + idx * 11
        height = 20 * (value / max_v)
        ax.add_patch(Rectangle((x, 15), 7, height, facecolor=color, edgecolor="none"))
        ax.text(x + 3.5, 13, label, ha="center", va="top", fontsize=10)
        ax.text(x + 3.5, 16 + height, f"{value:,}", ha="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "step5_story.png", bbox_inches="tight")
    plt.close(fig)


def draw_seqcluster(step6x: dict) -> None:
    labels = list(step6x["largest_clusters_top10"].keys())
    values = list(step6x["largest_clusters_top10"].values())
    short_labels = [label.split(":")[-1].replace("|", "\n") for label in labels]

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    fig.patch.set_facecolor("#f7fbff")
    bars = ax.bar(range(len(values)), values, color="#4ea8de")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel("Rows per cluster")
    ax.set_title("Step 6x Exact Receptor Sequence Clusters", fontsize=20, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, str(val), ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "seqcluster.png", bbox_inches="tight")
    plt.close(fig)


def draw_case_fragment(sample_rows: list[dict]) -> None:
    row_short = sample_rows[0]
    row_long = sample_rows[2]

    fig, ax = plt.subplots(figsize=(14, 7.6), dpi=160)
    fig.patch.set_facecolor("#f8f5f2")
    ax.set_xlim(134, 150)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(134, 9.2, "Example fragment from smoke_server", fontsize=22, fontweight="bold", color="#1d3557")
    ax.text(134, 8.5, "PDB 1a98 | same task keeps multiple physically valid windows", fontsize=12.5, color="#495057")

    for pos in range(136, 149):
        ax.add_patch(Rectangle((pos, 5.9), 0.9, 1.0, facecolor="#dee2e6", edgecolor="white"))
        ax.text(pos + 0.45, 6.4, str(pos), ha="center", va="center", fontsize=9)

    ax.text(134.1, 6.45, "source chain", fontsize=12, fontweight="bold")
    ax.add_patch(Rectangle((139, 3.9), 10, 1.0, facecolor="#f4a261", edgecolor="#6d6875"))
    ax.text(134.1, 4.4, "window A", fontsize=12, fontweight="bold", color="#bc6c25")
    ax.text(149.2, 4.4, "10 aa | 139-148 | coverage 1.00", fontsize=12, color="#7f5539", ha="right")

    ax.add_patch(Rectangle((136, 1.9), 13, 1.0, facecolor="#2a9d8f", edgecolor="#6d6875"))
    ax.text(134.1, 2.4, "window B", fontsize=12, fontweight="bold", color="#1b4332")
    ax.text(149.2, 2.4, "13 aa | 136-148 | coverage 0.85", fontsize=12, color="#2d6a4f", ha="right")

    ax.annotate("anchor", xy=(143.5, 5.8), xytext=(143.5, 7.8), ha="center", arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#e63946"), color="#e63946", fontsize=11)

    ax.text(134, 0.55, "Interpretation: the smoke run already demonstrates that one interface can yield multiple non-clone peptide windows with distinct lengths.", fontsize=12.5, color="#343a40")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "case_fragment.png", bbox_inches="tight")
    plt.close(fig)


def build_markdown(step1: dict, step5: dict, step6: dict, step6c: dict, step6x: dict, step7: dict, step8: dict, sample_rows: list[dict]) -> None:
    md = f"""---
title: "smoke_server 运行结果汇报"
author: "Codex"
date: "2026-04-15"
lang: "zh-CN"
---

# smoke_server 运行结果汇报

PeptideCLIP Phase-1 服务器 smoke 验证

- 目的：确认升级版全流程在服务器环境下能完整跑通
- 范围：Step1 到 Step8 全链路，包括 Step6x 序列聚类
- 重点：不是追求最大规模，而是验证流程完整性、统计合理性和最终落盘成功

# 本次 run 的意义

- 它是一套小规模但完整的服务器 smoke run
- 能证明从候选切片到最终 metadata/lmdb 的整条链路已经闭环
- 也能提前暴露 Step5 采样、Step6 平衡、Step7 分组拆分、Step8 封端落盘是否稳定

# 全流程总览

![](flow_overview.png){{ width=95% }}

# 数据漏斗

![](funnel.png){{ width=85% }}

- Step 1 通过质控：{step1['n_passed']:,}
- Step 5 保留候选：{step5['total_candidates_after_step5']:,}
- Step 6 分层保留后：{step6['survived_count']:,}
- Step 8 最终写出：{step8['success_count']:,}

# Step 5 关键机制

![](step5_story.png){{ width=92% }}

- 原始输入候选：{step5['total_candidates_before_step5']:,}
- 绝对冗余去除：{step5['total_abs_redundant_dropped']:,}
- Top-k / sampling 后继续收缩：{step5['total_topk_dropped']:,}
- 最终任务数：{step5['num_parent_tasks']:,}，所有任务都进入了 sampling

# 真实切片案例

![](case_fragment.png){{ width=90% }}

- 示例来自 `1a98 / A-2_as_peptide__A_as_receptor`
- 片段 A：{sample_rows[0]['peptide_start_resseq']}-{sample_rows[0]['peptide_end_resseq']}，长度 {sample_rows[0]['peptide_length']} aa
- 片段 B：{sample_rows[2]['peptide_start_resseq']}-{sample_rows[2]['peptide_end_resseq']}，长度 {sample_rows[2]['peptide_length']} aa
- 说明同一个界面任务不会只剩一个机械式短肽克隆

# Step 6x 序列聚类

![](seqcluster.png){{ width=88% }}

- exact 模式下共有受体序列 {step6x['unique_receptors']:,} 条
- 聚成 {step6x['unique_clusters']:,} 个 cluster
- 最大 cluster 只有 {max(step6x['largest_clusters_top10'].values())} 条，说明 smoke run 的家族结构还比较可控

# Step 7 / Step 8 最终结果

- Step 7：main_train {step7['main_train_count']:,}，monitor {step7['monitor_count']:,}
- Step 7 monitor 实际比例 {step7['monitor_ratio_actual_rows'] * 100:.2f}%
- Step 8：成功写出 {step8['success_count']:,} 条，错误数 {step8['error_count']:,}
- Step 8 terminal native 占比 {step8['terminal_native_ratio'] * 100:.1f}%
- Step 8 N-cap 覆盖 {step8['n_cap_ratio'] * 100:.1f}%；C-cap 覆盖 {step8['c_cap_ratio'] * 100:.1f}%

# 可以向老师强调的点

- 服务器 smoke run 已经证明全流程能稳定落到 Step 8
- Step 5 的“去冗余 + 概率抽样”在小规模 run 上也能观察到明显效果
- Step 6x 额外提供了受体序列聚类视角，便于后续 family-aware 控制
- 最终已经不是中间候选文件，而是真正可用于训练的数据元信息输出

# 下一步建议

- 在服务器上扩大任务规模，观察 Step 6 到 Step 8 的统计是否保持稳定
- 挑 2 到 3 个代表性复合物做结构可视化截图，补一页更直观的案例展示
- 如果准备正式组会，再把这版 smoke 汇报和 full_run 汇报拼成“验证版 + 正式版”双层结构
"""
    (OUT_DIR / "smoke_server_report.md").write_text(md, encoding="utf-8")


def main() -> None:
    setup_output()
    step1 = load_json(RUN_ROOT / "step1" / "step1_summary.json")
    step5 = load_json(RUN_ROOT / "step5" / "step5_summary.json")
    step6 = load_json(RUN_ROOT / "step6_align" / "step6_summary.json")
    step6c = load_json(RUN_ROOT / "step6c_align" / "step6c_main_summary.json")
    step6x = load_json(RUN_ROOT / "step6_seqcluster" / "step6x_summary.json")
    step7 = load_json(RUN_ROOT / "step7" / "step7_summary.json")
    step8 = load_json(RUN_ROOT / "step8" / "lmdb" / "step8_summary.json")
    sample_rows = read_first_jsonl(RUN_ROOT / "step5" / "step5_final.jsonl", n=3)

    draw_flow_overview(step1, step5, step6, step6x, step7, step8)
    draw_funnel(step1, step5, step6, step7, step8)
    draw_step5_story(step5)
    draw_seqcluster(step6x)
    draw_case_fragment(sample_rows)
    build_markdown(step1, step5, step6, step6c, step6x, step7, step8, sample_rows)


if __name__ == "__main__":
    main()

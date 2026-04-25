from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(r"e:\pep\project")
RUN_ROOT = PROJECT_ROOT / "runs" / "smoke_server"
PDB_DIR = Path(r"e:\pep\download")
OUT_DIR = PROJECT_ROOT / "ppt_complex_trace_1adu"
WSL_PROJECT_ROOT = "/mnt/e/pep/project"
WSL_PDB_DIR = "/mnt/e/pep/download"
WSL_PYTHON = "/mnt/e/pep/.venv/bin/python"

TARGET = {
    "pdb_id": "1adu",
    "receptor_chain_id": "A",
    "peptide_source_chain_id": "B",
}


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("pdb_id") == TARGET["pdb_id"]
        and row.get("receptor_chain_id") == TARGET["receptor_chain_id"]
        and row.get("peptide_source_chain_id") == TARGET["peptide_source_chain_id"]
    ]


def load_step_data() -> dict[str, list[dict]]:
    mapping = {
        "step2": RUN_ROOT / "step2" / "step2_tasks.jsonl",
        "step3": RUN_ROOT / "step3" / "step3_candidates.jsonl",
        "step4": RUN_ROOT / "step4" / "step4_features.jsonl",
        "step5": RUN_ROOT / "step5" / "step5_final.jsonl",
        "step6": RUN_ROOT / "step6_align" / "step6_survived.jsonl",
        "step7_main": RUN_ROOT / "step7" / "step7_main.jsonl",
        "step7_monitor": RUN_ROOT / "step7" / "step7_monitor.jsonl",
        "step8": RUN_ROOT / "step8" / "lmdb" / "final_metadata.jsonl",
    }
    out: dict[str, list[dict]] = {}
    for step, path in mapping.items():
        rows = load_rows(path)
        out[step] = filter_rows(rows)
    return out


def sort_candidate_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("peptide_start_resseq", -1)),
            int(row.get("peptide_end_resseq", -1)),
            str(row.get("candidate_id", "")),
        ),
    )


def draw_step_counts(data: dict[str, list[dict]]) -> None:
    labels = ["Step2\ntask", "Step3\ncandidates", "Step4\nscored", "Step5\nkept", "Step6\nsurvived", "Step7\nmain", "Step8\nfinal"]
    values = [
        len(data["step2"]),
        len(data["step3"]),
        len(data["step4"]),
        len(data["step5"]),
        len(data["step6"]),
        len(data["step7_main"]),
        len(data["step8"]),
    ]
    colors = ["#adb5bd", "#4ea8de", "#f4a261", "#8d5fd3", "#2a9d8f", "#6a994e", "#d62828"]

    fig, ax = plt.subplots(figsize=(12.5, 7), dpi=160)
    fig.patch.set_facecolor("#f8f9fa")
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Tracking One Complex Through the Pipeline", fontsize=20, fontweight="bold")
    ax.set_ylabel("Rows for 1adu / receptor A / peptide B")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(values) * 0.02, str(val), ha="center", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "step_counts.png", bbox_inches="tight")
    plt.close(fig)


def draw_window_evolution(data: dict[str, list[dict]]) -> None:
    step_rows = {
        "Step3 all candidates": sort_candidate_rows(data["step3"]),
        "Step5 kept windows": sort_candidate_rows(data["step5"]),
        "Step8 final sample": sort_candidate_rows(data["step8"]),
    }
    all_rows = step_rows["Step3 all candidates"]
    xmin = min(int(r["peptide_start_resseq"]) for r in all_rows) - 5
    xmax = max(int(r["peptide_end_resseq"]) for r in all_rows) + 5

    fig, ax = plt.subplots(figsize=(14, 8), dpi=160)
    fig.patch.set_facecolor("#fffdf7")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 42)
    ax.set_xlabel("Peptide residue index on source chain B")
    ax.set_yticks([])
    ax.set_title("Window Evolution on One Fixed Complex", fontsize=20, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.2)

    y_start = 34
    colors = {
        "Step3 all candidates": "#4ea8de",
        "Step5 kept windows": "#8d5fd3",
        "Step8 final sample": "#d62828",
    }

    for label, rows in step_rows.items():
        ax.text(xmin + 1, y_start + 1.2, label, fontsize=13, fontweight="bold", color="#1d3557")
        y = y_start
        for row in rows:
            start = int(row["peptide_start_resseq"])
            end = int(row["peptide_end_resseq"])
            width = end - start + 1
            ax.add_patch(Rectangle((start, y), width, 0.8, facecolor=colors[label], edgecolor="white", alpha=0.9))
            y -= 1
        y_start = y - 3

    ax.text(
        xmin + 1,
        2,
        "Interpretation: Step3 proposes many overlapping windows from several local hotspots; Step5 keeps 5 representative windows; Step8 retains 1 final training sample.",
        fontsize=11.5,
        color="#495057",
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "window_evolution.png", bbox_inches="tight")
    plt.close(fig)


def draw_step4_scatter(data: dict[str, list[dict]]) -> None:
    step4 = data["step4"]
    step5_ids = {row["candidate_id"] for row in data["step5"]}
    step8_ids = {row["candidate_id"] for row in data["step8"]}

    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    fig.patch.set_facecolor("#f8fbff")

    x_all = [float(r.get("contact_coverage_6A", 0.0)) for r in step4]
    y_all = [float(r.get("rBSA_raw", r.get("rBSA_proxy", 0.0))) for r in step4]
    ax.scatter(x_all, y_all, s=45, color="#a8dadc", edgecolors="none", label="Step4 candidates")

    step5_rows = [r for r in step4 if r["candidate_id"] in step5_ids]
    ax.scatter(
        [float(r.get("contact_coverage_6A", 0.0)) for r in step5_rows],
        [float(r.get("rBSA_raw", r.get("rBSA_proxy", 0.0))) for r in step5_rows],
        s=80,
        color="#6a4c93",
        label="Selected at Step5",
    )

    step8_rows = [r for r in step4 if r["candidate_id"] in step8_ids]
    ax.scatter(
        [float(r.get("contact_coverage_6A", 0.0)) for r in step8_rows],
        [float(r.get("rBSA_raw", r.get("rBSA_proxy", 0.0))) for r in step8_rows],
        s=110,
        color="#d62828",
        label="Final Step8 sample",
    )

    ax.set_xlabel("contact_coverage_6A")
    ax.set_ylabel("rBSA_raw")
    ax.set_title("Step4 Physical Features and Later Selection", fontsize=20, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "step4_scatter.png", bbox_inches="tight")
    plt.close(fig)


def draw_final_card(data: dict[str, list[dict]]) -> None:
    row = data["step8"][0]

    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=160)
    fig.patch.set_facecolor("#fbf7f0")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(5, 92, "Final Surviving Sample for This Complex", fontsize=22, fontweight="bold", color="#1d3557")
    ax.add_patch(FancyBboxPatch((5, 18), 90, 62, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#fffaf0", edgecolor="#adb5bd", linewidth=1.5))

    lines = [
        f"PDB: {row['pdb_id']}   receptor: {row['receptor_chain_id']}   peptide source: {row['peptide_source_chain_id']}",
        f"Candidate ID: {row['candidate_id']}",
        f"Residue window: {row['peptide_start_resseq']}-{row['peptide_end_resseq']}   length: {row['peptide_length']} aa",
        f"contact_coverage_6A: {float(row['contact_coverage_6A']):.3f}",
        f"rBSA_raw: {float(row['rBSA_raw']):.3f}",
        f"score_final: {float(row['score_final']):.3f}",
        f"split: {row.get('split', 'n/a')}",
        f"N-cap: {bool(row.get('has_n_cap'))}   C-cap: {bool(row.get('has_c_cap'))}",
        f"terminal_native: {bool(row.get('is_terminal_native', False))}",
        f"peptide_sequence: {row.get('peptide_sequence', '')}",
    ]
    y = 72
    for line in lines:
        ax.text(9, y, line, fontsize=13, color="#343a40")
        y -= 6

    ax.text(
        5,
        8,
        "This is the unique window from this complex that remains after Step6 stratified keep, Step7 split, and Step8 capping/finalization.",
        fontsize=11.5,
        color="#495057",
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "final_card.png", bbox_inches="tight")
    plt.close(fig)


def export_trace_pdbs(parent_task_id: str, final_candidate_id: str) -> tuple[bool, str]:
    viz_dir = OUT_DIR / "pdb_exports"
    viz_dir.mkdir(parents=True, exist_ok=True)

    win_to_wsl = {
        str(RUN_ROOT / "step3" / "step3_candidates.jsonl"): f"{WSL_PROJECT_ROOT}/runs/smoke_server/step3/step3_candidates.jsonl",
        str(RUN_ROOT / "step5" / "step5_final.jsonl"): f"{WSL_PROJECT_ROOT}/runs/smoke_server/step5/step5_final.jsonl",
        str(RUN_ROOT / "step8" / "lmdb" / "final_metadata.jsonl"): f"{WSL_PROJECT_ROOT}/runs/smoke_server/step8/lmdb/final_metadata.jsonl",
        str(viz_dir / "step3_task_overlay.pdb"): f"{WSL_PROJECT_ROOT}/ppt_complex_trace_1adu/pdb_exports/step3_task_overlay.pdb",
        str(viz_dir / "step5_task_overlay.pdb"): f"{WSL_PROJECT_ROOT}/ppt_complex_trace_1adu/pdb_exports/step5_task_overlay.pdb",
        str(viz_dir / "step8_final_candidate_overlay.pdb"): f"{WSL_PROJECT_ROOT}/ppt_complex_trace_1adu/pdb_exports/step8_final_candidate_overlay.pdb",
        str(viz_dir / "step8_interfaces"): f"{WSL_PROJECT_ROOT}/ppt_complex_trace_1adu/pdb_exports/step8_interfaces",
    }

    commands = [
        f'"{WSL_PYTHON}" "{WSL_PROJECT_ROOT}/visualize_export_pdb.py" --input_jsonl "{win_to_wsl[str(RUN_ROOT / "step3" / "step3_candidates.jsonl")]}" --pdb_dir "{WSL_PDB_DIR}" --output_pdb "{win_to_wsl[str(viz_dir / "step3_task_overlay.pdb")]}" --parent_task_id "{parent_task_id}" --receptor_scope full --max_task_candidates 12',
        f'"{WSL_PYTHON}" "{WSL_PROJECT_ROOT}/visualize_export_pdb.py" --input_jsonl "{win_to_wsl[str(RUN_ROOT / "step5" / "step5_final.jsonl")]}" --pdb_dir "{WSL_PDB_DIR}" --output_pdb "{win_to_wsl[str(viz_dir / "step5_task_overlay.pdb")]}" --parent_task_id "{parent_task_id}" --receptor_scope full --max_task_candidates 12',
        f'"{WSL_PYTHON}" "{WSL_PROJECT_ROOT}/visualize_export_pdb.py" --input_jsonl "{win_to_wsl[str(RUN_ROOT / "step8" / "lmdb" / "final_metadata.jsonl")]}" --pdb_dir "{WSL_PDB_DIR}" --output_pdb "{win_to_wsl[str(viz_dir / "step8_final_candidate_overlay.pdb")]}" --candidate_id "{final_candidate_id}" --receptor_scope patch',
        f'"{WSL_PYTHON}" "{WSL_PROJECT_ROOT}/export_protein_interface_pdbs.py" --input_jsonl "{win_to_wsl[str(RUN_ROOT / "step8" / "lmdb" / "final_metadata.jsonl")]}" --pdb_dir "{WSL_PDB_DIR}" --pdb_id "{TARGET["pdb_id"]}" --receptor_chain_id "{TARGET["receptor_chain_id"]}" --peptide_source_chain_id "{TARGET["peptide_source_chain_id"]}" --output_dir "{win_to_wsl[str(viz_dir / "step8_interfaces")]}" --max_candidates 1 --sort_by rbsa --include_full_receptor --include_source_full_chain',
    ]

    env = os.environ.copy()
    try:
        for cmd in commands:
            subprocess.run(
                ["powershell", "-Command", f'wsl bash -lc \'{cmd}\''],
                check=True,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
    except subprocess.CalledProcessError as exc:
        return False, f"PDB export skipped in this session: {exc}"
    return True, "PDB exports generated successfully."


def write_markdown(data: dict[str, list[dict]], pdb_export_ok: bool, pdb_export_note: str) -> None:
    step2 = data["step2"][0]
    step8 = data["step8"][0]
    task_id = data["step3"][0]["parent_task_id"]

    # Count Step3 hotspot starts for a compact narrative.
    hotspot_counts = Counter((int(r["peptide_start_resseq"]), int(r["peptide_end_resseq"])) for r in data["step3"])
    hotspot_text = ", ".join(f"{a}-{b} ({n})" for (a, b), n in hotspot_counts.most_common(5))

    md = f"""---
title: "单一复合物全流程追踪：1adu"
author: "Codex"
date: "2026-04-15"
lang: "zh-CN"
---

# 单一复合物全流程追踪

固定案例：`1adu / receptor A / peptide source B`

- 这页汇报不再讲整体数据集，而是只追踪一个复合物
- 目的是让老师直观看到：一个复合物如何从原始任务逐步变成最终训练样本
- 当前基于 `project/runs/smoke_server`

# 案例背景

- Step2 中该复合物对应 1 个有向任务
- `parent_task_id = {task_id}`
- 受体链 A，来源肽链 B
- source chain 上存在多个可切片热点，因此很适合做流程追踪

# 单个复合物的行数变化

![](step_counts.png){{ width=88% }}

- Step3 生成 30 个候选窗口
- Step5 保留 5 个代表窗口
- Step6 以后只剩 1 个最终样本

# 窗口是如何收缩的

![](window_evolution.png){{ width=95% }}

- Step3 在同一条 source chain 上提出大量重叠窗口
- Step5 把这些窗口压缩成 5 个代表候选
- Step8 最终只保留 1 个样本进入训练元数据

Step3 高频窗口示例：{hotspot_text}

# Step4 打分如何影响后续选择

![](step4_scatter.png){{ width=82% }}

- 横轴是 `contact_coverage_6A`
- 纵轴是 `rBSA_raw`
- 紫色点是 Step5 选中的窗口，红点是最后进入 Step8 的样本

# 最终保留下来的样本

![](final_card.png){{ width=92% }}

- 最终窗口：{step8['peptide_start_resseq']}-{step8['peptide_end_resseq']}
- 长度：{step8['peptide_length']} aa
- split：{step8.get('split', 'n/a')}
- `has_n_cap = {bool(step8.get('has_n_cap'))}`，`has_c_cap = {bool(step8.get('has_c_cap'))}`

# 可视化结构文件

- 结构导出状态：{"已生成" if pdb_export_ok else "本次会话未成功导出"}
- 说明：{pdb_export_note}
- 如果导出成功，可在 `pdb_exports/` 下继续用 PyMOL/Chimera 截图补充到 PPT

# 这个案例能说明什么

- 同一个复合物在 Step3 会出现大量重叠候选
- Step5 先做去冗余和带权采样，留下少量代表窗口
- Step6 到 Step8 再结合分层保留、split 和封端，最终变成真正可训练的单条样本
- 用一个固定复合物跟踪，比只看总统计更能解释流程的行为逻辑
"""
    (OUT_DIR / "complex_trace_report.md").write_text(md, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_step_data()
    if not data["step3"] or not data["step8"]:
        raise SystemExit("Target complex was not found across the required steps.")

    draw_step_counts(data)
    draw_window_evolution(data)
    draw_step4_scatter(data)
    draw_final_card(data)
    pdb_export_ok, pdb_export_note = export_trace_pdbs(
        parent_task_id=str(data["step3"][0]["parent_task_id"]),
        final_candidate_id=str(data["step8"][0]["candidate_id"]),
    )
    write_markdown(data, pdb_export_ok, pdb_export_note)


if __name__ == "__main__":
    main()

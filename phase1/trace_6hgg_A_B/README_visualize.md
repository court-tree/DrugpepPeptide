# 6hgg A->B 全流程可视化说明

这个目录追踪的是：

- `PDB`: `6hgg`
- `direction`: `B_as_peptide__A_as_receptor`
- `task_id`: `fd0f122a-dcab-4074-bd30-a8df33efe10f`
- `final candidate`: `11451315-1bfd-4492-adad-9b06886b1121`

## 每一步文件

- Step 1 结构质控记录：[step1_6hgg.json](E:\pep\phase1\trace_6hgg_A_B\step1_6hgg.json)
- Step 2 任务定义：[step2_task.jsonl](E:\pep\phase1\trace_6hgg_A_B\step2_task.jsonl)
- Step 3 候选列表：[step3_candidates.jsonl](E:\pep\phase1\trace_6hgg_A_B\step3_candidates.jsonl)
- Step 4 通过物理过滤：[step4_features.jsonl](E:\pep\phase1\trace_6hgg_A_B\step4_features.jsonl)
- Step 5 task 内抽样保留：[step5_final.jsonl](E:\pep\phase1\trace_6hgg_A_B\step5_final.jsonl)
- Step 6 主集保留：[step6_main.jsonl](E:\pep\phase1\trace_6hgg_A_B\step6_main.jsonl)
- Step 6 本任务被丢弃：[step6_dropped.jsonl](E:\pep\phase1\trace_6hgg_A_B\step6_dropped.jsonl)
- Step 7 最终 metadata：[step7_final_metadata.jsonl](E:\pep\phase1\trace_6hgg_A_B\step7_final_metadata.jsonl)
- 汇总：[trace_summary.json](E:\pep\phase1\trace_6hgg_A_B\trace_summary.json)

## 这条链路的数量变化

- Step 2: `1` 个 task
- Step 3: `16` 个候选
- Step 4: `16` 个通过
- Step 5: `4` 个保留
- Step 6 main: `3` 个保留
- Step 6 dropped: `0`
- Step 7 final metadata: `4` 条

说明：

- 这个 task 在 Step 4 没有被刷掉，因为 16 条候选都非常强。
- Step 5 开始做“代表性保留”，从 16 条收缩到 4 条。
- Step 6 在主训练集里保留了 3 条；第 4 条没有进 `main`，但仍在 Step 7 里，因为 Step 7 是 `main + monitor` 一起汇总。

## 推荐看的可视化文件

- Step 1 / Step 2：直接看原始复合物  
  `E:\pep\download\6hgg.cif`
- Step 3 候选叠加  
  [step3_candidates_overlay.pdb](E:\pep\phase1\trace_6hgg_A_B\step3_candidates_overlay.pdb)
- Step 4 通过候选叠加  
  [step4_survived_overlay.pdb](E:\pep\phase1\trace_6hgg_A_B\step4_survived_overlay.pdb)
- Step 5 保留候选叠加  
  [step5_selected_overlay.pdb](E:\pep\phase1\trace_6hgg_A_B\step5_selected_overlay.pdb)
- Step 6 主集保留叠加  
  [step6_kept_overlay.pdb](E:\pep\phase1\trace_6hgg_A_B\step6_kept_overlay.pdb)
- Step 7 最终 20 aa 单候选叠加  
  [step7_final_overlay.pdb](E:\pep\phase1\trace_6hgg_A_B\step7_final_overlay.pdb)
- Step 7 patch + peptide 单独导出  
  [6hgg_01_A_B_11451315-1bf.pdb](E:\pep\phase1\trace_6hgg_A_B\step7_patch_exports\6hgg_01_A_B_11451315-1bf.pdb)

## PyMOL 命令

先看 Step 1 / Step 2 的原始复合物：

```pml
load E:/pep/download/6hgg.cif, complex
hide everything, all
bg_color white
show cartoon, chain A
color gray80, chain A
show cartoon, chain B
color wheat, chain B
orient
zoom visible, 8
```

看 Step 3 候选池怎么铺开：

```pml
load E:/pep/phase1/trace_6hgg_A_B/step3_candidates_overlay.pdb, step3
hide everything, all
show cartoon, chain R
color gray80, chain R
show cartoon, chain S
color wheat, chain S
show sticks, chain A or chain B or chain C or chain D or chain E or chain F or chain G or chain H or chain I or chain J or chain K or chain L or chain M or chain N or chain O or chain P
spectrum count, rainbow, chain A or chain B or chain C or chain D or chain E or chain F or chain G or chain H or chain I or chain J or chain K or chain L or chain M or chain N or chain O or chain P
orient chain R
zoom visible, 8
```

看 Step 5 代表性保留后的 4 条：

```pml
load E:/pep/phase1/trace_6hgg_A_B/step5_selected_overlay.pdb, step5
hide everything, all
show cartoon, chain R
color gray80, chain R
show cartoon, chain S
color wheat, chain S
show sticks, not chain R+S
spectrum count, rainbow, not chain R+S
orient
zoom visible, 8
```

看 Step 6 主训练集里真正留下的 3 条：

```pml
load E:/pep/phase1/trace_6hgg_A_B/step6_kept_overlay.pdb, step6
hide everything, all
show cartoon, chain R
color gray80, chain R
show cartoon, chain S
color wheat, chain S
show sticks, not chain R+S
spectrum count, rainbow, not chain R+S
orient
zoom visible, 8
```

只看最终那条 20 aa：

```pml
load E:/pep/phase1/trace_6hgg_A_B/step7_final_overlay.pdb, final20
hide everything, all
show cartoon, chain R
color gray80, chain R
show cartoon, chain S
color wheat, chain S
show sticks, chain P
color orange, chain P
orient chain P
zoom visible, 7
```

想看“受体 patch + 最终 peptide”的结合形状，用这个：

```pml
load E:/pep/phase1/trace_6hgg_A_B/step7_patch_exports/6hgg_01_A_B_11451315-1bf.pdb, patch_final
hide everything, all
bg_color white
show cartoon, chain F
color gray85, chain F
show cartoon, chain L
color wheat, chain L
show surface, chain X
set transparency, 0.35, chain X
color cyan, chain X
show sticks, chain P
color orange, chain P
orient chain P
zoom chain X or chain P, 6
```

## 每一步你该看什么

- Step 1：这个复合物是不是一个像样的蛋白-蛋白复合物，A/B 两条链有没有真实界面。
- Step 2：A 当 receptor、B 当 peptide source 这个方向是否合理。
- Step 3：16 个候选是不是沿着同一片真实界面展开，而不是到处乱跳。
- Step 4：这一例里 Step 3 和 Step 4 基本一样，说明这些候选本身已经很强。
- Step 5：从密集重叠的候选里，保留了 8/9/11/20 aa 四条代表窗口。
- Step 6：主训练集最终保留 3 条，说明没有把这个 task 压缩到只剩 1 条。
- Step 7：最终 20 aa 片段 `RFNRPFLMIIVDHFTWSIFF` 与 patch 接触完整，`coverage=1.0`，`longest_contact_run=20`，很适合放 PPT。

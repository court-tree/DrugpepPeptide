# Step 5 单独导出的 4 条代表片段

这 4 个文件都来自：

- [step5_final.jsonl](E:\pep\phase1\trace_6hgg_A_B\step5_final.jsonl)
- 复合物：`6hgg`
- 方向：`A <- B`

## 文件对应关系

- [step5_8aa_387_394.pdb](E:\pep\phase1\trace_6hgg_A_B\step5_single_candidates\step5_8aa_387_394.pdb)
  - candidate_id: `7e50cfcd-fc2d-48c0-9e8d-967dc9a4142d`
  - peptide range: `B 387–394`
  - length: `8`

- [step5_9aa_389_397.pdb](E:\pep\phase1\trace_6hgg_A_B\step5_single_candidates\step5_9aa_389_397.pdb)
  - candidate_id: `d556050c-bad4-45bc-bd14-75ca8ce8fc33`
  - peptide range: `B 389–397`
  - length: `9`

- [step5_11aa_387_397.pdb](E:\pep\phase1\trace_6hgg_A_B\step5_single_candidates\step5_11aa_387_397.pdb)
  - candidate_id: `d4cf72ad-d9dc-4194-807e-b9e6cdbe440e`
  - peptide range: `B 387–397`
  - length: `11`

- [step5_20aa_371_390.pdb](E:\pep\phase1\trace_6hgg_A_B\step5_single_candidates\step5_20aa_371_390.pdb)
  - candidate_id: `11451315-1bfd-4492-adad-9b06886b1121`
  - peptide range: `B 371–390`
  - length: `20`

## PyMOL 通用显示命令

每个文件都可以直接用同一套命令看：

```pml
hide everything, all
bg_color white
show cartoon, chain R
color slate, chain R
show sticks, chain P
color tv_orange, chain P
orient chain P
zoom chain P, 7
```

## 说明

这些文件都是“单候选导出”，因此：

- `chain R` = receptor
- `chain S` = source peptide full chain
- `chain P` = 当前这一个候选片段

你现在不需要再关心 overlay 里 `A/B/C/D` 链和长度的映射问题了。

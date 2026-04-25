# Step4 长度自适应 Coverage 阈值实验报告

## 实验目的

本次实验想验证一个问题：

> Step4 中固定使用 `contact_coverage >= 0.5`，是否是导致长肽样本偏少的主要原因？

实验假设是：

- 固定的 coverage 阈值可能对长肽过于严格。
- 长肽可能由一个紧密结合核心区加上两端非接触延伸段组成。
- 因此，如果对长肽适当放宽 `contact_coverage` 阈值，理论上可能增加 15-20 aa 长肽候选的保留数量。

## 对比运行

| 运行版本 | Step4 coverage 规则 | 其他关键过滤条件 |
|---|---|---|
| `full_run_v2` | 统一使用 `contact_coverage >= 0.5` | `avg_contact_count >= 3.5` |
| `full_run_v3` | 长度自适应 coverage：8-10 aa `>= 0.50`，11-14 aa `>= 0.40`，15-20 aa `>= 0.30` | `avg_contact_count >= 3.5` |

两个版本使用相同的 Step1-Step3 输入。`full_run_v3` 复用了 `full_run_v2` 的 Step3 候选，因此主要变量就是 Step4 的 coverage 阈值。

## 总体结果

| 指标 | `full_run_v2` | `full_run_v3` | 变化 |
|---|---:|---:|---:|
| 最终样本数 | 171,123 | 171,042 | -81 |
| 平均 peptide 长度 | 9.492 | 9.490 | 几乎不变 |
| 平均 `avg_contact_count` | 4.928 | 4.926 | 几乎不变 |
| 最小 `contact_coverage` | 0.5 | 0.4 | 阈值变化已生效 |
| 平均 `contact_coverage` | 0.9181 | 0.9179 | 几乎不变 |
| 15-20 aa 总数 | 2,213 | 2,200 | -13 |

## 长度分布

| Peptide 长度 | `full_run_v2` | `full_run_v3` | 变化 |
|---:|---:|---:|---:|
| 8 | 56,329 | 56,494 | +165 |
| 9 | 41,045 | 40,885 | -160 |
| 10 | 39,746 | 39,655 | -91 |
| 11 | 18,059 | 18,072 | +13 |
| 12 | 7,983 | 7,976 | -7 |
| 13 | 3,865 | 3,860 | -5 |
| 14 | 1,883 | 1,900 | +17 |
| 15 | 945 | 915 | -30 |
| 16 | 581 | 604 | +23 |
| 17 | 383 | 369 | -14 |
| 18 | 140 | 153 | +13 |
| 19 | 84 | 86 | +2 |
| 20 | 80 | 73 | -7 |

## 结果解释

长度自适应 coverage 阈值确实生效了，因为最终数据集中 `contact_coverage` 的最小值从 `0.5` 降到了 `0.4`。

但是，这个改动没有明显增加长肽数量。15-20 aa 样本数不仅没有增加，反而从 `2,213` 轻微下降到 `2,200`。

这说明：

> 固定的 `contact_coverage >= 0.5` 很可能不是长肽不足的主要瓶颈。

换句话说，长肽并不是主要死在 Step4 的 coverage 阈值上。

## 结论

本次实验是一个有价值的反证实验：

> 按长度放宽 Step4 coverage 阈值，并不能显著改善长肽样本比例。

因此，长肽不足的问题更可能出现在 Step4 之前或之后：

- Step3 的 task-level top candidates 仍然容易被高平均接触数的短窗口占据，导致长肽进入后续流程的机会不足。
- Step5 虽然已经限制每个 task 最多保留 2 条 8-mer，但 9-10 aa 窗口仍然会大量占据剩余名额。
- Step6 的长肽优先去重只能保留已经进入 Step6 的长肽；如果长肽在 Step3-Step5 阶段数量已经很少，Step6 无法凭空补回来。

## 下一步建议

不建议继续把主要精力放在调 Step4 coverage 阈值上。

更干净的下一轮实验是调整 Step3 的候选保留策略，让每个 task 在进入 Step4 之前就保留不同长度段的候选。当前已实现的新策略为：

| 长度段 | 每个 task 最多保留数 |
|---|---:|
| 8-10 aa | 6 |
| 11-14 aa | 6 |
| 15-20 aa | 4 |

这样总候选上限仍然控制在 `16`，但可以保证长肽有机会进入 Step4 和 Step5，而不是在 Step3 的全局 top 排序中被短窗口提前挤掉。如果某个长度段候选不足，剩余名额会按 `avg_contact_count` 从其他未选候选中回填。

---

# 当前 Phase-1 算法逻辑总结

这版算法的主线是：

> 从真实 PPI 复合物中寻找少数界面热点锚点，围绕热点生成连续 8-20 aa 多肽窗口；随后对单条候选做质量过滤，再在合格候选中用带权抽样和轻量长度约束保留代表片段；最后做 receptor + peptide 联合同源去重并生成最终 metadata。

## Step 1：结构级质控

**目标：**  
只保留结构质量和界面质量基本可靠的复合物。

**逻辑：**

- 读取 mmCIF 结构
- 保留有效蛋白链数量足够的结构
- 检查链间是否存在有效接触
- 要求最小界面面积
- 过滤蛋白链不足、无有效链间接触、界面面积过小的结构

**作用：**  
保证后续候选生成不是建立在低质量或无界面的结构上。

## Step 2：双向任务生成

**目标：**  
把结构级复合物拆成 receptor-peptide source 任务。

**逻辑：**

- 对每对有接触的链生成两个方向：
  - A 作为 receptor，B 作为 peptide source
  - B 作为 receptor，A 作为 peptide source
- 每个任务记录 `source_file`、`pdb_id`、`receptor_chain_id`、`peptide_source_chain_id` 和链对信息

**作用：**  
把复合物转成后续可枚举窗口的任务单位。

## Step 3：热点锚点驱动的候选生成

**目标：**  
不是全链盲目滑窗，而是在真实界面中选择少数热点锚点，并围绕这些热点生成候选多肽窗口。

**逻辑：**

- 先找到 peptide source chain 上与 receptor 有接触的 seed residues
- 对每个 seed residue 计算 anchor 自身直接接触到的 receptor residue 数，以及 anchor 周围局部窗口的平均接触数
- anchor 必须满足 `min_anchor_contact_count = 2`
- anchor 排序优先级为 `anchor_direct_contact_count`，然后是局部窗口 `avg_contact_count`
- 对 anchor 做 NMS，避免相邻热点重复
- 每个 task 最多保留 `max_anchors_per_task = 3`
- 围绕热点 anchor 枚举 `8-20 aa` 连续窗口
- 同一窗口边界只保留更优候选
- 每个 task 最多输出 `max_candidates_per_task = 16`
- 输出前按长度段保留候选：
  - 8-10 aa 最多 6 条
  - 11-14 aa 最多 6 条
  - 15-20 aa 最多 4 条
- 如果某个长度段不足，则用剩余高 `avg_contact_count` 候选回填

**作用：**  
控制锚点数量和候选数量，让候选来自少数真实热点区域，而不是密集重复接触点；同时避免 Step3 的纯 top16 被短窗口完全占据。

## Step 4：单条候选质量过滤

**目标：**  
逐一检查 Step 3 生成的候选窗口，剔除断裂、弱接触或质量不佳的单条序列。

**逻辑：**

- 回原始结构重新切出候选窗口
- 检查窗口边界是否一致、主链是否连续
- 重新计算 `avg_contact_count` 和 `contact_coverage`
- 当前保守基线过滤条件为 `avg_contact_count >= 3.5` 和 `contact_coverage >= 0.5`
- 不在 Step 4 做近重复去冗余

**作用：**  
保证进入抽样阶段的候选，至少是结构连续、接触充分、单条质量说得过去的片段。

## Step 5：带权抽样与轻量长度多样性控制

**目标：**  
在 Step 4 通过质量过滤的候选中，随机保留代表片段，同时避免最短 8-mer 过度占据结果。

**逻辑：**

- 每个 task 最多保留 `max_keep_per_task = 4`
- 抽样权重为 `sampling_weight = avg_contact_count`
- 平均接触数越高，被抽中的概率越大
- 加入 task 内 8-mer 上限 `max_len8_per_task = 2`
- 如果一个 task 中还有非 8-mer 候选，则最多保留 2 条 8-mer
- 如果该 task 只有 8-mer 候选，则允许 8-mer 兜底补满

**作用：**  
在保持接触质量优先的同时，防止同一个 task 全部被 8-mer 占满，提高 8-20 aa 长度多样性。

## Step 6：联合 receptor + peptide 同源去重

**目标：**  
去掉真正高度重复的样本，但不因为 receptor 相似就误删不同 peptide。

**逻辑：**

- 给每条候选补 receptor sequence 和 peptide sequence
- 代表优先级为 peptide length 更长优先，然后是 `avg_contact_count` 更高优先，最后是 `contact_coverage` 更高优先
- 使用 peptide k-mer 索引加速候选重复对查找
- 两条样本只有同时满足 `receptor_identity >= 0.85`、`peptide_identity >= 0.85`、`shorter_peptide_length / longer_peptide_length >= 0.70` 才视为重复
- 如果短肽被长肽高度覆盖，则优先保留长肽
- 如果长度差异大、覆盖不足 0.70，则两者都保留
- 去重后随机划分 `main_train` 和 `monitor`，monitor 比例为 `0.10`

**作用：**  
保留 receptor 和 peptide 的联合多样性，只删除真正近重复样本。

## Step 7：最终 metadata 生成

**目标：**  
把最终样本整理成后续训练可使用的数据格式。

**逻辑：**

- 读取 Step 6 的 main 和 monitor
- 回原始结构定位最终 peptide window
- 生成 peptide sequence、receptor local patch residue ids、peptide residue ids、split 标记和 proxy cap 信息
- patch 提取半径为 `patch_cutoff = 6.0`

**输出：**

- `final_metadata.jsonl`

## 当前核心默认参数

| Step | 参数 | 当前值 |
|---|---|---:|
| Step 3 | `min_anchor_contact_count` | 2 |
| Step 3 | `max_anchors_per_task` | 3 |
| Step 3 | `max_candidates_per_task` | 16 |
| Step 3 | `max_candidates_len_8_10` | 6 |
| Step 3 | `max_candidates_len_11_14` | 6 |
| Step 3 | `max_candidates_len_15_20` | 4 |
| Step 4 | `min_avg_contact_count` | 3.5 |
| Step 4 | `min_contact_coverage` | 0.5 |
| Step 5 | `max_keep_per_task` | 4 |
| Step 5 | `max_len8_per_task` | 2 |
| Step 6 | `receptor_identity_threshold` | 0.85 |
| Step 6 | `peptide_identity_threshold` | 0.85 |
| Step 6 | `peptide_min_coverage` | 0.70 |
| Step 7 | `patch_cutoff` | 6.0 |

## 一句话总结

当前算法逻辑是：

> 先从真实界面中挑少数热点锚点并生成 8-20 aa 连续候选；再过滤掉单条质量不合格的片段；随后按平均接触数带权抽样，并限制每个 task 中 8-mer 的数量；最后只有 receptor 和 peptide 同时高度同源且长度覆盖充分时才去重，生成最终训练 metadata。

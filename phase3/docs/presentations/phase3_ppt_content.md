# PepCLIP Phase-3 数据集构建算法 PPT 内容稿

用途：向老师讲清楚 Phase-3 数据集到底打算怎么实现，而不是只列版本规定。  
核心叙事：先把当前要落地的 V1 数据集构建算法讲清楚，再说明 V1.5/V2/V3 是沿着同一条流水线逐步增强，而不是另起炉灶。

建议页数：18 页  
建议风格：科研汇报，重流程、重输入输出、重为什么这样做  
一句话主线：

```text
Phase-3 先用真实 receptor-peptide complex 构建可信强监督数据集；
再用 exact same-sequence 构象证据扩展为构象增强训练；
最后把 similar / motif conformer 作为先验，而不是强监督。
```

---

## Slide 1｜标题页

**标题**

PepCLIP Phase-3 数据集构建算法

**副标题**

从真实 receptor-peptide complex 到构象增强微调数据集

**页脚信息**

Phase-3 Dataset Construction Plan  
V1 baseline + V1.5/V2/V3 extension roadmap

**讲述重点**

这份 PPT 要回答三个问题：

```text
1. Phase-3 数据集第一版怎么构建？
2. 每一步输入、处理、输出是什么？
3. 后续构象增强版本如何自然接到这条流水线上？
```

---

## Slide 2｜Phase-3 要解决什么问题

**标题**

Phase-3 的目标：真实结构监督微调

**Phase-2 已经完成**

```text
teacher data
-> dual-tower pretraining
-> paired retrieval validation
-> raw Top-K retrieval
```

**Phase-3 要补上的监督**

Phase-2 更偏向学习：

```text
receptor interface <-> peptide sequence / peptide representation
```

Phase-3 要加入真实结构监督：

```text
receptor interface
<->
真实 peptide bound conformer
```

**核心目标**

让模型不仅知道：

```text
哪条 peptide 可能结合这个 receptor interface
```

还要进一步知道：

```text
这条 peptide 的哪种 bound conformer 更适合这个 receptor interface
```

---

## Slide 3｜总实现路线

**标题**

总体路线：先构建可信 V1，再逐步增强构象监督

**路线图**

```text
V1:
    真实 receptor-peptide bound pair 数据集
    positive_strong_bound only

V1.5:
    exact same-sequence conformer evidence
    默认只做证据统计和审计

V2:
    same-cluster weak positive
    conformer_hard_negative ranking

V3:
    similar sequence / motif conformer prior
    prior only，不作为强监督边
```

**为什么要分版本**

```text
V1 先回答：真实 bound conformer 微调有没有用？
V1.5 回答：同序列构象证据够不够支撑增强？
V2 回答：构象增强训练是否提升 receptor-conformer 匹配？
V3 回答：相似序列 / motif prior 是否能提供额外信息？
```

**讲述重点**

版本路线不是替代实现算法，而是把同一条数据集构建流水线拆成可验证、可归因的阶段。

---

## Slide 4｜V1 数据集的定义

**标题**

V1 是 Phase-3 的最小可用数据集

**V1 核心样本**

```text
anchor_i =
receptor_interface_i
+
true_bound_conformer_i
```

**V1 只生成一种训练监督边**

```text
positive_strong_bound:
    receptor_interface_i <-> true_bound_conformer_i
```

**V1 不做**

```text
positive_aug_same_cluster
conformer_hard_negative
similar sequence prior
motif prior
multi-positive contrastive loss
ranking loss
```

**V1 的目的**

先建立一个干净问题：

```text
真实复合物里的 receptor interface 和 peptide bound conformer
能否作为 Phase-3 微调的有效强监督？
```

---

## Slide 5｜V1 输入数据从哪里来

**标题**

第一步：收集真实 receptor-peptide complex 候选

**优先数据源**

```text
Q-BioLiP
BioLiP peptide entries
PepBDB
Propedia
PDB protein-peptide complex records
```

**每条候选记录需要包含**

```text
source_database
source_entry_id
pdb_id
biological_assembly_id
complex_structure_file
receptor_chain_id
peptide_chain_id
peptide residue range
source confidence / annotation
```

**不优先使用**

```text
只有 peptide sequence
只有活性信息
没有明确 receptor-bound structure
没有 receptor-peptide chain 对应关系
```

**讲述重点**

Phase-3 的强监督必须来自真实复合物，不从未配对 peptide 库中直接生成正样本。

---

## Slide 6｜V1 算法总流程

**标题**

V1 数据集构建流水线

**流程**

```text
1. 收集真实复合物候选
2. 多数据库候选去重
3. 重建 biological assembly
4. 定位 receptor chain 和 peptide chain
5. 过滤 peptide chain
6. 验证 receptor-peptide 接触
7. 定义 receptor interface
8. 提取 true_bound_conformer
9. 质量检查与 anchor 生成
10. 防泄露 split
11. 导出 Track A / Track B 微调数据
12. 生成 dataset_audit
```

**输出结果**

```text
receptor_peptide_pair.jsonl
peptide_conformer_evidence.jsonl
conformer_cluster.jsonl
track_a_<split>.jsonl
track_b_<split>.jsonl
phase3_summary.json
dataset_audit.json
```

**讲述重点**

V1 不是规则清单，而是一条完整的数据处理流水线：原始结构记录进来，训练可用的 receptor/interface 和 peptide conformer 样本出去。

---

## Slide 7｜V1 Step 1-2：候选收集与去重

**标题**

先保证每个真实复合物只进入一次

**Step 1：候选收集**

从多个数据库收集 receptor-peptide complex 记录，每条记录必须能定位到：

```text
pdb_id
assembly_id
receptor_chain
peptide_chain
peptide sequence / residue range
structure file
```

**Step 2：多数据库去重**

最低去重键：

```text
pdb_id
assembly_id
receptor_chain
peptide_chain
```

如果同一个复合物出现在多个数据库：

```text
保留一条主记录
保存 source_dbs = [Q-BioLiP, BioLiP, PepBDB, ...]
```

**目的**

```text
避免同一个 anchor 重复进入训练；
避免同一个真实复合物同时泄露到 train / val / test。
```

---

## Slide 8｜V1 Step 3-5：结构重建与 peptide 过滤

**标题**

只保留能形成标准 Phase-3 样本的 peptide chain

**Step 3：重建 biological assembly**

```text
读取 PDB/mmCIF
-> 使用 biological assembly
-> 定位 receptor chain
-> 定位 peptide chain
```

不只使用 asymmetric unit，因为可能包含晶体堆积接触或缺少真实 biological interface。

**Step 4：peptide chain 过滤**

只保留：

```text
8 <= peptide_length <= 20
```

过滤：

```text
length < 8
length > 20
残基不连续
backbone 坐标缺失
非标准残基过多
缺失残基过多
不是独立 peptide chain
和 receptor 共价连接的特殊情况
```

**Step 5：length_group**

```text
8-10 aa   -> short_8_10
11-15 aa  -> medium_short_11_15
16-20 aa  -> medium_long_16_20
```

---

## Slide 9｜V1 Step 6-7：接触验证与 interface 定义

**标题**

确认 receptor 和 peptide 真的形成结构界面

**Step 6：heavy atom 接触验证**

基础规则：

```text
min_heavy_atom_distance <= 5.0 A
```

并按 peptide 长度检查：

```text
contact_count
interface_residue_count
```

**长度分档阈值的原因**

```text
短肽接触数自然较少，但每个残基更关键；
中长肽接触范围更大，需要更高接触数量保证真实界面。
```

**Step 7：定义 receptor interface**

保存两个区域：

```text
interface_5A:
    receptor 中距离 peptide heavy atom <= 5 A 的残基
    用于 QC 和核心接触统计

context_10A:
    receptor 中距离 peptide heavy atom <= 10 A 的残基
    用于模型输入，提供周围结构环境
```

---

## Slide 10｜V1 Step 8-9：true_bound_conformer 与 anchor

**标题**

true_bound_conformer 是 V1 强监督的核心

**Step 8：提取 true_bound_conformer**

从真实复合物 peptide chain 中提取：

```text
peptide_sequence
backbone_coords: N, CA, C, O
heavy_atom_coords: all non-H atoms
```

生成：

```text
true_bound_conformer_id
```

**Step 9：质量检查**

如果出现：

```text
backbone 原子缺失
残基不连续
sequence 对不上
missing_ratio 超标
坐标质量太差
```

则：

```text
删除整个 anchor
```

**硬规则**

不能删除 true_bound_conformer 后继续保留 receptor interface。因为 V1 的强正样本标签就来自这个真实 bound conformer。

---

## Slide 11｜V1 Step 10：split 与防泄露

**标题**

split 是数据集可信度的关键步骤

**为什么不能随机切 pair**

随机 pair split 可能导致：

```text
同一 peptide sequence 同时出现在 train 和 test；
同一 receptor family 同时出现在 train 和 test；
同一 PDB 或高度相似结构跨 split 泄露。
```

**推荐 split 层级**

```text
1. pair-level split
2. peptide exact sequence split
3. receptor family split
4. strict split
```

**每个样本保存**

```text
split_group
split
peptide_sequence_key
receptor_family_key
pdb_key
```

**V1 目标**

至少保证主评估不被明显重复样本污染；后续 V1.5/V2 再继续处理 conformer leakage。

---

## Slide 12｜V1 输出表与训练入口

**标题**

V1 输出要直接服务 Phase-3 微调

**核心输出**

```text
receptor_peptide_pair.jsonl:
    anchor / receptor / peptide pair metadata

peptide_conformer_evidence.jsonl:
    true_bound_conformer evidence

conformer_cluster.jsonl:
    V1 中主要记录 bound conformer 的初始 cluster / identity 信息

track_a_<split>.jsonl:
    sequence-side / receptor-side training input

track_b_<split>.jsonl:
    conformer-side / 3D training input

dataset_audit.json:
    counts, filters, split statistics, leakage checks
```

**V1 训练**

```text
train:
    positive_strong_bound only

val/test:
    positive_strong_bound only
```

**讲述重点**

V1 的成果不是“最终构象增强”，而是一套可以训练、可以评估、可以审计的真实结构强监督数据集。

---

## Slide 13｜V1.5：exact same-sequence 构象证据层

**标题**

V1.5 不改变主训练目标，只补齐构象证据

**V1.5 接在 V1 后面**

输入：

```text
V1 anchors
V1 true_bound_conformers
PDB SEQRES / mmCIF exact-match evidence
```

处理：

```text
对每个 anchor peptide sequence 搜索 exact same-sequence conformers
要求 identity = 100%, coverage = 100%, same length, no gap
```

硬规则：

```text
exact_pool = [true_bound_conformer] + searched_exact_conformers
```

输出：

```text
external_exact_match_conformers
peptide_conformer_evidence
conformer_cluster
bound_conformer_cluster_mapping
conformer_mining_summary
dataset_audit
```

**用途**

默认只做覆盖率、聚类质量、bound conformer mapping 和 leakage 审计，为 V2 做准备。

---

## Slide 14｜V2：构象增强训练怎么接入

**标题**

V2 在 V1/V1.5 基础上生成 supervised_edges

**V2 输入**

```text
V1 anchors
V1.5 exact_pool
exact_same_sequence conformer clusters
bound_cluster_id
```

**V2 生成三类监督边**

```text
positive_strong_bound:
    true_bound_conformer
    edge_weight = 1.0

positive_aug_same_cluster:
    same sequence + same bound_cluster
    edge_weight = 0.3-0.5
    train only

conformer_hard_negative:
    same sequence + different cluster
    edge_weight = 0.2-0.5
    train only
    rank_loss_only
```

**关键边界**

V2 不重新定义正样本来源。真实强正样本仍来自 V1 的 true_bound_conformer；V2 只是在其周围增加同序列构象增强。

---

## Slide 15｜V2 训练损失与语义边界

**标题**

V2 的训练改变在 loss，不在 V1 强监督定义

**总损失**

```text
L_total =
L_clip_1D
+ alpha * L_multipos_3D
+ beta * L_multipos_fusion
+ lambda * L_conformer_rank
```

**分支使用规则**

| 分支 | 使用样本 | 损失 |
|---|---|---|
| 1D sequence branch | `positive_strong_bound` only | ordinary CLIP |
| 3D conformer branch | strong positive + same-cluster weak positive | multi-positive CLIP |
| fusion branch | strong positive + same-cluster weak positive | multi-positive CLIP |
| ranking branch | true bound vs same-sequence different-cluster | margin ranking |

**最重要语义**

```text
conformer_hard_negative 不是 peptide-level negative。
它只表示同一 peptide 的另一个构象状态
在当前 receptor interface 下不如 true_bound_conformer 匹配。
```

**禁止**

```text
不能进入 1D sequence branch
不能进入普通 CLIP negative pool
不能进入 multi-positive denominator
```

---

## Slide 16｜V3：similar / motif prior 怎么接入

**标题**

V3 引入相似构象先验，但不进入强监督边

**V3 动机**

当 exact same-sequence conformer 较少时，可以搜索：

```text
similar sequence conformer
motif family conformer
```

这些信息代表：

```text
相似 peptide / motif family 的构象先验
```

不是：

```text
当前 peptide sequence 的真实构象监督
```

**V3 输出**

```text
conformer_prior.parquet
```

字段：

```text
anchor_id
conformer_id
prior_type
identity_to_query
similarity_to_query
coverage_to_query
contact_core_identity
contact_core_similarity
reason
```

**硬边界**

V3 prior 不参与：

```text
exact_same_sequence cluster
supervised_edges
positive_aug_same_cluster
conformer_hard_negative
```

---

## Slide 17｜主评估与诊断评估

**标题**

无论哪个版本，主评估都必须保持清洁

**主评估只使用**

```text
positive_strong_bound
```

回答：

```text
模型能否识别真实 receptor-peptide bound pair？
```

**不进入主评估**

```text
positive_aug_same_cluster
conformer_hard_negative
similar_sequence_conformer_prior
motif_conformer_prior
```

**V2/V3 可选诊断**

```text
conformer ranking diagnostic:
    给定 receptor interface
    比较 true_bound_conformer
    和 same-sequence different-cluster conformer 的分数

prior diagnostic:
    分析 similar / motif prior 是否提供额外排序信息
```

**讲述重点**

主指标衡量真实配对检索；诊断指标衡量构象增强策略。两者必须分开汇报。

---

## Slide 18｜最终实施计划

**标题**

阶段三数据集构建的实际实施顺序

**实施顺序**

```text
Step 1:
    完成 V1 数据集构建
    输出 positive_strong_bound only 的训练/验证/测试集

Step 2:
    用 V1 跑 Phase-3 baseline fine-tuning
    得到是否有效的第一组结果

Step 3:
    构建 V1.5 exact same-sequence conformer evidence
    完成覆盖率、聚类、泄露审计

Step 4:
    若 V1.5 证据足够，再实现 V2 supervised_edges
    加入 same-cluster weak positive 和 conformer_hard_negative

Step 5:
    V2 稳定后，再实现 V3 conformer_prior
```

**交付物**

```text
V1:
    Phase-3 baseline dataset + audit + baseline training result

V1.5:
    conformer evidence report + cluster mapping + audit

V2:
    supervised_edges + multi-positive/ranking training ablation

V3:
    conformer_prior + prior ablation / reranking analysis
```

**收束句**

这条路线的核心是：先用真实复合物建立可信强监督基线，再用同序列构象证据做可审计增强，最后才引入相似序列和 motif 先验。


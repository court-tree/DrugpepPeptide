# PepCLIP Phase-3 构象增强微调数据集构建算法逻辑｜完整修正版

## 0. 总目标

构建用于 PepCLIP Phase-3 微调的真实结构监督数据集：

```text
真实 receptor interface
↔
真实 peptide sequence
↔
真实 peptide bound conformer
```

Phase-3 的目标不是单纯学习：

```text
receptor ↔ peptide sequence
```

而是进一步学习：

```text
receptor interface ↔ peptide conformer
```

也就是让模型知道：

```text
哪条 peptide 可能结合当前 receptor interface
+
这条 peptide 的哪种构象状态更适合当前 receptor interface
```

核心监督必须来自真实 receptor-peptide complex。构象库只能围绕真实 bound conformer 做增强，不能替代真实复合物中的配对监督。

---

# 1. 核心硬规则

## 1.1 长度范围固定为 8–20 aa

为保持与 Phase-2 预训练数据分布一致，Phase-3 只保留长度为 8–20 aa 的 peptide。

```text
length < 8:
    discard

8 <= length <= 20:
    enter Phase-3 dataset

length > 20:
    exclude from current Phase-3
    save to future_long_peptide_set
```

在 8–20 aa 内部分三档：

```text
8–10 aa:
    short peptide group

11–15 aa:
    medium-short peptide group

16–20 aa:
    medium-long peptide group
```

不同长度组使用不同的：

```text
接触验证阈值
相似构象搜索阈值
RMSD 聚类阈值
```

---

## 1.2 true_bound_conformer 是 anchor 核心

每个 anchor 的定义是：

```text
anchor_i =
receptor_interface_i
+
true_bound_conformer_i
```

如果 true_bound_conformer 质量不合格，则删除整个 anchor。

不能做：

```text
删除 true_bound_conformer
但保留 anchor
```

因为 anchor 的强正样本标签就来自 true_bound_conformer。

---

## 1.3 true_bound_conformer 必须加入 exact_pool

构建同序列构象库时，必须执行：

```text
exact_pool = [true_bound_conformer] + searched_exact_conformers
```

不能只用：

```text
exact_pool = searched_exact_conformers
```

否则可能出现 true_bound_conformer 无法映射到构象簇的问题。

---

## 1.4 构象增强边默认只在 train split 启用

构象增强边包括：

```text
positive_aug_same_cluster
conformer_hard_negative
```

它们默认只在 train split 中启用。

val/test 主评估只使用：

```text
positive_strong_bound
```

val/test 可以额外做 conformer ranking diagnostic，但必须单独报告，不能和主评估混在一起。

---

## 1.5 conformer_hard_negative 不是 peptide-level negative

`conformer_hard_negative` 的含义是：

```text
同一 peptide sequence
但属于不同构象簇
```

它不是说：

```text
这个 peptide sequence 不结合 receptor
```

而是说：

```text
在当前 receptor interface 下，
这个构象状态比 true_bound_conformer 更不匹配
```

所以它只能进入：

```text
conformer branch ranking loss
fusion branch ranking loss
```

不能进入：

```text
1D sequence branch
普通 CLIP loss
普通 batch negative pool
multi-positive contrastive loss 的 denominator
```

---

## 1.6 similar sequence / motif 只作为 prior

相似序列构象和 motif 构象只进入：

```text
conformer_prior.parquet
```

不参与：

```text
exact_same_sequence cluster
supervised_edges 生成
positive_aug_same_cluster
conformer_hard_negative
```

原因是：

```text
降低 sequence identity 后，
搜到的构象不再是同一条 peptide 的构象，
而是相似 peptide / motif family 的构象先验。
```

---

# 2. 数据对象定义

## 2.1 Anchor

每个 anchor 来自真实 receptor-peptide complex。

```text
anchor_i =
receptor_interface_i
+
true_bound_conformer_i
```

它表示一个真实结构级正配对：

```text
receptor interface ↔ peptide bound conformer
```

---

## 2.2 true_bound_conformer

从真实复合物中的 peptide chain 提取。

保存：

```text
peptide_sequence
backbone_coords: N, CA, C, O
heavy_atom_coords: 所有非氢原子
```

这是最可靠的强正样本构象。

---

## 2.3 exact same-sequence conformer

对 anchor peptide sequence，在 PDB 中搜索完全同序列构象。

要求：

```text
identity = 100%
coverage = 100%
same length
no gap
```

这类构象可以参与：

```text
exact_same_sequence cluster
positive_aug_same_cluster
conformer_hard_negative
```

---

## 2.4 similar sequence / motif conformer prior

当 exact_pool 太少时，可以搜索相似序列或局部 motif 构象。

但它们只能作为：

```text
conformer prior
```

不能作为 supervised edge。

---

# 3. Part A：构建真实 receptor-peptide anchor

## Step 1：收集真实复合物候选

优先使用真实 receptor-peptide complex 数据库：

```text
Q-BioLiP
BioLiP2 peptide entries
PepBDB
Propedia
PepX
PepBind
PDB protein-peptide complex records
```

每条候选记录：

```text
candidate_id
source_db
pdb_id
assembly_id
receptor_chain
peptide_chain
peptide_sequence
peptide_length
source_annotation
```

不优先使用只有 peptide sequence 或活性信息、但没有明确 receptor-bound structure 的数据库。

---

## Step 2：多数据库候选去重

同一个复合物可能被多个数据库收录，所以必须先去重。

最低去重键：

```text
pdb_id
assembly_id
receptor_chain
peptide_chain
```

如果多个数据库提供同一条记录，只保留一条主记录，同时保存来源列表：

```text
source_dbs = [Q-BioLiP, BioLiP2, PepBDB, ...]
```

目的：

```text
避免同一个 anchor 重复进入训练
避免同一个真实复合物同时出现在 train / val / test
```

---

## Step 3：重建 biological assembly

对每条候选：

```text
下载 PDB/mmCIF
→ 重建 biological assembly
→ 定位 receptor chain
→ 定位 peptide chain
```

不能只用 asymmetric unit，因为 asymmetric unit 可能包含晶体堆积接触，也可能缺少真实 biological interface。

---

## Step 4：过滤 peptide chain

只保留：

```text
8 <= peptide_length <= 20
```

过滤掉：

```text
length < 8
length > 20
残基不连续
backbone 坐标缺失
非标准残基太多
缺失残基太多
不是独立 peptide chain
和 receptor 共价连接的特殊情况
```

---

## Step 5：分配 length_group

根据 peptide_length 分配长度组：

```text
8–10 aa:
    length_group = short_8_10

11–15 aa:
    length_group = medium_short_11_15

16–20 aa:
    length_group = medium_long_16_20
```

---

## Step 6：验证 receptor-peptide 接触

使用 heavy atom 距离判断 receptor 和 peptide 是否真实形成界面。

基础规则：

```text
min_heavy_atom_distance ≤ 5.0 Å
```

接触数量按长度分档。

```text
8–10 aa:
    contact_count ≥ 4–5
    interface_residue_count ≥ 3

11–15 aa:
    contact_count ≥ 5–8
    interface_residue_count ≥ 4

16–20 aa:
    contact_count ≥ 8–10
    interface_residue_count ≥ 5
```

目的：

```text
确认这个 receptor interface 和 peptide bound conformer
确实可以作为结构级正配对。
```

没有通过接触验证的候选不能作为强正样本。

---

## Step 7：定义 receptor interface

保存两个区域。

### interface_5A

```text
receptor 中任意 heavy atom
距离 peptide 任意 heavy atom ≤ 5 Å 的残基
```

用途：

```text
QC
接触统计
核心界面定义
```

### context_10A

```text
receptor 中任意 heavy atom
距离 peptide 任意 heavy atom ≤ 10 Å 的残基
```

用途：

```text
模型输入
提供 receptor interface 周围结构环境
```

---

## Step 8：提取 true_bound_conformer

从同一个真实复合物中的 peptide chain 提取：

```text
peptide_sequence
backbone_coords: N, CA, C, O
heavy_atom_coords: 所有非氢原子
```

生成：

```text
true_bound_conformer_id
```

---

## Step 9：检查 true_bound_conformer 质量

如果 true_bound_conformer 出现以下问题，删除整个 anchor：

```text
backbone 原子缺失
残基不连续
peptide sequence 对不上
missing_ratio 超标
坐标质量太差
```

不能只从 exact_pool 中删除 true_bound_conformer 后继续保留 anchor。

通过质量检查后，生成 anchor：

```text
anchor_i =
receptor_interface_i
+
true_bound_conformer_i
```

---

# 4. Part B：split 与防泄露

## Step 10：先做 split，再做构象增强

构象增强前必须先完成 split。

推荐 split 类型：

```text
1. pair-level split
   只做 sanity check

2. peptide exact sequence split
   test peptide sequence 不出现在 train

3. receptor family split
   test receptor family 不出现在 train

4. strict split
   test peptide 和 train peptide 不高度相似
   test receptor 和 train receptor 不高度相似
```

每个 anchor 保存：

```text
split_group
split
```

---

## Step 11：防止 conformer leakage

构象库搜索可能把 val/test 的 true_bound_conformer 搜进 train 的 conformer pool。

所以每个 conformer instance 必须保存：

```text
source_anchor_id
source_split
is_true_bound
```

训练集增强时必须过滤：

```text
train augmentation 不能使用 val/test 的 true_bound_conformer
```

否则会出现构象泄露，导致评估虚高。

---

# 5. Part C：构建构象库

## Step 12：完全同序列构象搜索

对每个 anchor 的 peptide sequence，在 PDB 中搜索完全同序列片段：

```text
identity = 100%
coverage = 100%
same length
no gap
```

得到：

```text
searched_exact_conformers
```

---

## Step 13：强制加入 true_bound_conformer

必须执行：

```text
exact_pool = [true_bound_conformer] + searched_exact_conformers
```

每个 anchor 的 exact_pool 至少包含自己的 true_bound_conformer。

---

## Step 14：exact_pool 防泄露过滤

对 train split 的 anchor：

```text
不能使用 val/test true_bound_conformer 作为构象增强来源
```

所以要过滤：

```text
source_split in [val, test] and is_true_bound == True
```

但注意：

```text
当前 anchor 自己的 true_bound_conformer 不受这个过滤影响
```

也就是说：

```text
anchor 自己的 true_bound_conformer 必须保留
其他 split 的 true_bound_conformer 不能作为 train augmentation
```

---

## Step 15：相似序列构象搜索

如果 exact_pool 数量太少，可以启动相似序列构象搜索。

但相似序列构象只能进入：

```text
conformer_prior
```

不能进入：

```text
supervised_edges
exact_same_sequence cluster
```

动态阈值按长度分档。

### 8–10 aa

```text
exact search:
    identity = 100%
    coverage = 100%
    same length
    no gap

similar search:
    默认不放宽 identity
    最多作为 motif / family prior
```

短肽每个残基都很关键，不建议放宽 identity。

---

### 11–15 aa

```text
exact search:
    identity = 100%
    coverage = 100%
    same length
    no gap

similar search:
    identity ≥ 90–95%
    similarity ≥ 95%
    coverage ≥ 95–100%
    contact-core identity / similarity 高保守
```

允许少量保守突变，但只进入 conformer_prior。

---

### 16–20 aa

```text
exact search:
    identity = 100%
    coverage = 100%
    same length
    no gap

similar search:
    identity ≥ 85–90%
    similarity ≥ 90–95%
    coverage ≥ 90–100%
    contact-core identity / similarity 高保守
```

中长肽完全同序列构象更难搜到，可以适度扩大相似序列构象库，但仍然不能作为正样本或 conformer_hard_negative。

---

## Step 16：接触核心位点约束

对每个 anchor，根据真实复合物确定 peptide 上真正接触 receptor 的位置：

```text
peptide_contact_positions
```

相似序列搜索时必须检查：

```text
contact-core identity
contact-core similarity
```

原则：

```text
全局 identity 可以适度放宽
但 contact-core positions 必须高度保守
```

因为接触核心位点变化可能直接改变 receptor-peptide 结合模式。

---

## Step 17：构象清洗

对所有 conformer instance 进行清洗。

过滤：

```text
backbone 原子缺失
残基编号不连续
序列对不上
坐标质量差
B-factor 异常高
分辨率太低
局部断裂
```

注意：

```text
如果被清洗掉的是 anchor 自己的 true_bound_conformer，
则删除整个 anchor。
```

---

# 6. Part D：同序列构象聚类

## Step 18：只对 exact same-sequence conformers 聚类

参与聚类的只能是：

```text
identity = 100%
coverage = 100%
same length
no gap
```

不允许把以下内容混入 exact cluster：

```text
similar_sequence_conformer_prior
motif_conformer_prior
```

---

## Step 19：按 backbone RMSD 聚类

流程：

```text
同一个 peptide sequence 的 exact_pool
→ backbone atoms 对齐
→ 计算 pairwise backbone RMSD
→ 根据 length_group 设置 RMSD 阈值
→ 聚类
```

RMSD 阈值按长度分档：

```text
8–10 aa:
    RMSD threshold = 1.0–1.2 Å

11–15 aa:
    RMSD threshold = 1.2–1.6 Å

16–20 aa:
    RMSD threshold = 1.6–2.0 Å
```

这些是 pilot 初始值，后续可以根据真实 RMSD 分布调整。

---

## Step 20：映射 true_bound_conformer 到 bound_cluster

对每个 anchor：

```text
true_bound_conformer_i
→ 所属 exact_same_sequence conformer cluster
```

得到：

```text
bound_cluster_id
```

后续所有同簇增强和构象困难负样本，都围绕 `bound_cluster_id` 判断。

---

# 7. Part E：生成监督边和先验表

## Step 21：生成 supervised_edges

`supervised_edges` 只保存真正参与训练监督的边。

允许的 edge_type：

```text
positive_strong_bound
positive_aug_same_cluster
conformer_hard_negative
```

字段：

```text
anchor_id
interface_id
conformer_id
edge_type
edge_weight
loss_scope
split
reason
```

---

## Step 22：positive_strong_bound

真实复合物中的 bound conformer：

```text
receptor_i ↔ true_bound_conformer_i
```

标记：

```text
edge_type = positive_strong_bound
edge_weight = 1.0
loss_scope = sequence_branch + conformer_branch + fusion_branch
```

split 行为：

```text
train / val / test 都保留
val/test 主评估只使用它
```

含义：

```text
这是最可靠的 receptor-interface–peptide-conformer 强正样本。
```

---

## Step 23：positive_aug_same_cluster

如果 train split 中，exact_pool 有其他 conformer 和 true_bound_conformer 属于同一 cluster：

```text
receptor_i ↔ same_sequence_same_cluster_conformer
```

标记：

```text
edge_type = positive_aug_same_cluster
edge_weight = 0.3–0.5
loss_scope = conformer_branch + fusion_branch
split = train only
```

含义：

```text
完全同序列
且构象与 true_bound_conformer 接近
可以作为弱正增强
```

硬规则：

```text
positive_aug_same_cluster 只在 train split 启用
如果启用，3D / fusion 分支必须使用 multi-positive contrastive loss
不能使用普通单正样本 CLIP loss
```

---

## Step 24：conformer_hard_negative

如果 train split 中，同一 peptide sequence 还有其他构象簇：

```text
receptor_i ↔ same_sequence_different_cluster_conformer
```

标记：

```text
edge_type = conformer_hard_negative
edge_weight = 0.2–0.5
loss_scope = rank_loss_only + conformer_branch + fusion_branch
split = train only
```

含义：

```text
不是说这个 peptide sequence 不结合 receptor
而是说在当前 receptor interface 下
这个构象状态比 true_bound_conformer 更不匹配
```

硬规则：

```text
conformer_hard_negative 只在 train split 启用
conformer_hard_negative 不进入 1D sequence 分支
conformer_hard_negative 不进入任何普通 CLIP loss
conformer_hard_negative 不进入普通负样本池
conformer_hard_negative 只进入 ranking loss
```

---

## Step 25：ordinary in-batch negatives 不写入 supervised_edges

普通 batch 内负样本由训练时动态产生。

不要提前写成：

```text
R1 ↔ P2
R1 ↔ P3
R1 ↔ P4
...
```

原因：

```text
数据量会爆炸
容易把潜在真阳性写死成负样本
不符合 CLIP 动态 batch negative 逻辑
```

正确做法：

```text
batch 内非对角线默认作为动态负样本
但需要 conflict mask
```

如果非对角线中出现：

```text
same peptide sequence
same anchor
same receptor family
potential same motif family
conformer_hard_negative
```

应根据规则 mask 掉，避免错误负样本。

---

## Step 26：similar_sequence_conformer_prior 单独成表

相似序列构象和 motif 构象不放入 supervised_edges。

单独保存为：

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

prior_type：

```text
similar_sequence_conformer_prior
motif_conformer_prior
```

用途：

```text
构象统计
family-level 构象先验
可选的辅助分析
可选的后续预训练/弱约束
```

默认不参与 Phase-3 supervised fine-tuning。

---

# 8. Part F：训练损失函数

## Step 27：总损失

最终训练损失：

```text
L_total =
L_clip_1D
+ α L_multipos_3D
+ β L_multipos_fusion
+ λ L_conformer_rank
```

---

## Step 28：1D sequence 分支

1D 分支只使用：

```text
positive_strong_bound
```

训练目标：

```text
receptor interface ↔ peptide sequence
```

不使用：

```text
positive_aug_same_cluster
conformer_hard_negative
similar_sequence_conformer_prior
motif_conformer_prior
```

原因：

```text
同序列不同构象在 1D 上完全一样，
不能一会儿当正样本，一会儿当负样本。
```

---

## Step 29：3D conformer 分支使用 multi-positive contrastive loss

3D 分支的正样本集合：

```text
positive_set(R_i) =
{
    true_bound_conformer_i,
    same_cluster_aug_conformers
}
```

权重：

```text
true_bound_conformer_i:
    1.0

same_cluster_aug_conformers:
    0.3–0.5
```

因此必须使用：

```text
multi-positive contrastive loss
```

不能使用普通单正样本 CLIP loss。

普通动态负样本来自 batch 内其他 anchor 的 peptide conformers，但要使用 conflict mask。

---

## Step 30：fusion 分支也使用 multi-positive contrastive loss

fusion 分支输入：

```text
peptide sequence embedding
+
peptide conformer embedding
```

它同样有多个正样本：

```text
true_bound_conformer
same_cluster_aug_conformer
```

所以也必须使用 multi-positive contrastive loss。

---

## Step 31：conformer_hard_negative 使用 ranking loss

对于：

```text
C+ = true_bound_conformer
C- = same_sequence_different_cluster_conformer
```

计算：

```text
s_pos = score(R, C+)
s_alt = score(R, C-)
```

希望：

```text
s_pos > s_alt + margin
```

损失：

```text
L_conformer_rank =
max(0, margin - s_pos + s_alt)
```

这个损失只作用于：

```text
conformer_branch
fusion_branch
```

不作用于：

```text
sequence_branch
```

语义：

```text
当前 receptor interface 更匹配真实 bound conformer，
而不是同一 peptide 的其他构象状态。
```

不是：

```text
这个 peptide sequence 是负样本。
```

---

# 9. Part G：val/test 主评估规则

## Step 32：val/test 主评估只使用 positive_strong_bound

val/test 的主评估样本必须来自真实 receptor-peptide complex：

```text
receptor interface
↔
true_bound_conformer
```

也就是：

```text
edge_type = positive_strong_bound
```

不使用：

```text
positive_aug_same_cluster
conformer_hard_negative
similar_sequence_conformer_prior
motif_conformer_prior
```

原因：

```text
val/test 要回答的是：
模型能否识别真实 receptor-peptide bound pair？

不是：
构象增强策略本身是否合理。
```

---

## Step 33：可选 conformer ranking diagnostic

可以额外做诊断评估：

```text
给定 R_i
比较 true_bound_conformer_i
和 same_sequence_different_cluster_conformer
的分数
```

但这只能作为：

```text
diagnostic metric
```

不能作为主指标。

---

# 10. 输出表设计

## 10.1 receptor_peptide_anchor.csv

```text
anchor_id
source_dbs
pdb_id
assembly_id
receptor_chain
peptide_chain
peptide_sequence
peptide_length
length_group
interface_id
true_bound_conformer_id
bound_cluster_id
split_group
split
contact_count
min_heavy_atom_distance
interface_residue_count
qc_status
qc_flags
```

---

## 10.2 receptor_interface.parquet

```text
interface_id
anchor_id
pdb_id
assembly_id
receptor_chain
interface_residues_5A
context_residues_10A
coords_path
```

---

## 10.3 peptide_conformer_instance.parquet

```text
conformer_id
peptide_sequence
peptide_length
length_group
pdb_id
chain_id
start_residue
end_residue
evidence_tier
source_context
source_anchor_id
source_split
is_true_bound
identity_to_query
similarity_to_query
coverage_to_query
contact_core_identity
contact_core_similarity
backbone_coords_path
heavy_atom_coords_path
missing_ratio
qc_status
```

---

## 10.4 peptide_conformer_cluster.parquet

```text
cluster_id
peptide_sequence
peptide_length
length_group
cluster_scope
num_instances
representative_conformer_id
mean_backbone_rmsd
max_backbone_rmsd
rmsd_threshold
```

cluster_scope：

```text
exact_same_sequence
similar_sequence_family
local_motif
```

注意：

```text
Phase-3 supervised_edges 只使用 exact_same_sequence cluster。
similar_sequence_family 和 local_motif 只作为 prior / analysis。
```

---

## 10.5 supervised_edges.parquet

只保存训练监督边：

```text
anchor_id
interface_id
conformer_id
edge_type
edge_weight
loss_scope
split
reason
```

允许的 edge_type：

```text
positive_strong_bound
positive_aug_same_cluster
conformer_hard_negative
```

split 行为：

```text
positive_strong_bound:
    train / val / test 都保留

positive_aug_same_cluster:
    train only

conformer_hard_negative:
    train only
    rank_loss_only
```

---

## 10.6 conformer_prior.parquet

只保存非监督先验：

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

prior_type：

```text
similar_sequence_conformer_prior
motif_conformer_prior
```

---

# 11. 总体伪代码

```python
# =====================================================
# Step 1: Collect and deduplicate candidates
# =====================================================

candidates = collect_candidates_from_complex_databases()

candidates = deduplicate_candidates(
    candidates,
    keys=["pdb_id", "assembly_id", "receptor_chain", "peptide_chain"]
)


# =====================================================
# Step 2: Build true receptor-peptide anchors
# =====================================================

anchors = []

for candidate in candidates:

    structure = load_biological_assembly(
        pdb_id=candidate.pdb_id,
        assembly_id=candidate.assembly_id
    )

    receptor = get_chain(structure, candidate.receptor_chain)
    peptide = get_chain(structure, candidate.peptide_chain)

    if not peptide_length_ok(peptide, min_len=8, max_len=20):
        continue

    length_group = assign_length_group(
        peptide_length=len(peptide.sequence)
    )

    if peptide_chain_quality_bad(peptide):
        continue

    contact_thresholds = get_contact_thresholds(length_group)

    contact = compute_heavy_atom_contact(
        receptor=receptor,
        peptide=peptide,
        cutoff=5.0
    )

    if contact.min_distance > 5.0:
        continue

    if contact.contact_count < contact_thresholds.min_contact_count:
        continue

    if contact.interface_residue_count < contact_thresholds.min_interface_residue_count:
        continue

    interface_5A = extract_receptor_interface(
        receptor=receptor,
        peptide=peptide,
        cutoff=5.0
    )

    context_10A = extract_receptor_context(
        receptor=receptor,
        peptide=peptide,
        cutoff=10.0
    )

    true_bound_conformer = extract_peptide_conformer(
        peptide=peptide,
        atoms=["backbone", "heavy"]
    )

    if true_bound_conformer_quality_bad(true_bound_conformer):
        # Hard rule:
        # delete the entire anchor
        continue

    peptide_contact_positions = get_peptide_contact_positions(
        receptor=receptor,
        peptide=peptide,
        cutoff=5.0
    )

    anchor = create_anchor(
        receptor_interface=interface_5A,
        receptor_context=context_10A,
        true_bound_conformer=true_bound_conformer,
        contact_positions=peptide_contact_positions,
        length_group=length_group,
        metadata=contact
    )

    anchors.append(anchor)


# =====================================================
# Step 3: Assign split before augmentation
# =====================================================

anchors = assign_split(
    anchors,
    mode="peptide_sequence_split / receptor_family_split / strict_split"
)


# =====================================================
# Step 4: Build conformer pools
# =====================================================

for anchor in anchors:

    true_c = anchor.true_bound_conformer

    searched_exact = search_exact_same_sequence_conformers(
        sequence=anchor.peptide_sequence,
        identity=1.0,
        coverage=1.0,
        no_gap=True
    )

    # Hard rule:
    # true_bound_conformer must be included in exact_pool
    exact_pool = [true_c] + searched_exact

    exact_pool = clean_conformers(exact_pool)

    # If true_c failed during cleaning, delete entire anchor
    if true_c not in exact_pool:
        delete_anchor(anchor)
        continue

    # Leakage control:
    # train augmentation cannot use val/test true_bound_conformer
    exact_pool = remove_leakage_conformers(
        exact_pool=exact_pool,
        current_anchor=anchor,
        rule="train cannot use val/test true_bound_conformer except its own true_c"
    )

    save_conformer_instances(exact_pool)

    # Similar sequence / motif conformers only go to prior table
    if exact_pool_too_small(exact_pool):

        prior_thresholds = get_similar_search_thresholds(
            length_group=anchor.length_group
        )

        prior_pool = search_similar_sequence_conformers(
            sequence=anchor.peptide_sequence,
            thresholds=prior_thresholds,
            contact_core_positions=anchor.contact_positions
        )

        save_to_conformer_prior(prior_pool)


# =====================================================
# Step 5: Cluster exact same-sequence conformers
# =====================================================

for sequence in unique_anchor_sequences:

    exact_conformers = get_exact_same_sequence_conformers(
        sequence=sequence
    )

    # Do not include similar sequence / motif priors
    exact_conformers = filter_exact_only(exact_conformers)

    rmsd_threshold = get_rmsd_threshold_by_length_group(
        length_group=assign_length_group(len(sequence))
    )

    clusters = cluster_by_backbone_rmsd(
        conformers=exact_conformers,
        threshold=rmsd_threshold
    )

    save_clusters(clusters)


# =====================================================
# Step 6: Map true_bound_conformer to bound_cluster
# =====================================================

for anchor in anchors:

    bound_cluster = find_cluster(
        conformer=anchor.true_bound_conformer,
        clusters=get_exact_clusters(anchor.peptide_sequence)
    )

    if bound_cluster is None:
        delete_anchor(anchor)
        continue

    anchor.bound_cluster_id = bound_cluster.cluster_id


# =====================================================
# Step 7: Generate supervised edges
# =====================================================

for anchor in anchors:

    # Strong positive: train / val / test
    add_supervised_edge(
        anchor_id=anchor.id,
        interface_id=anchor.interface_id,
        conformer_id=anchor.true_bound_conformer_id,
        edge_type="positive_strong_bound",
        edge_weight=1.0,
        loss_scope=["sequence_branch", "conformer_branch", "fusion_branch"],
        split=anchor.split,
        reason="true receptor-peptide bound conformer"
    )

    # Augmentation edges are train-only
    if anchor.split != "train":
        continue

    # Weak positives: same sequence, same cluster
    for c in same_cluster_conformers(anchor):

        if c.id == anchor.true_bound_conformer_id:
            continue

        add_supervised_edge(
            anchor_id=anchor.id,
            interface_id=anchor.interface_id,
            conformer_id=c.id,
            edge_type="positive_aug_same_cluster",
            edge_weight=0.4,
            loss_scope=["conformer_branch", "fusion_branch"],
            split="train",
            reason="exact same sequence and same conformer cluster"
        )

    # Conformer hard negatives: same sequence, different cluster
    for c in different_cluster_conformers(anchor):

        add_supervised_edge(
            anchor_id=anchor.id,
            interface_id=anchor.interface_id,
            conformer_id=c.id,
            edge_type="conformer_hard_negative",
            edge_weight=0.3,
            loss_scope=["rank_loss_only", "conformer_branch", "fusion_branch"],
            split="train",
            reason="same sequence but different conformer cluster"
        )
```

---

# 12. 训练伪代码

```python
# =====================================================
# Batch construction
# =====================================================

batch = sample_train_anchors()

# 1D branch:
# only strong positives
strong_sequences = batch.true_bound_sequences


# 3D / fusion CLIP candidates:
# include strong positives and same-cluster weak positives
# exclude conformer_hard_negative from ordinary CLIP candidates
clip_conformer_candidates = build_clip_candidates(
    batch=batch,
    include_strong_positive=True,
    include_same_cluster_aug=True,
    exclude_conformer_hard_negative=True,
    exclude_prior=True
)

positive_mask_3d = build_multi_positive_mask(
    batch=batch,
    candidates=clip_conformer_candidates,
    positive_types=[
        "positive_strong_bound",
        "positive_aug_same_cluster"
    ]
)

positive_weights_3d = build_positive_weights(
    strong_weight=1.0,
    weak_same_cluster_weight=0.4
)

valid_negative_mask_3d = build_valid_negative_mask(
    batch=batch,
    candidates=clip_conformer_candidates,
    mask_same_sequence_conflicts=True,
    mask_same_anchor_conflicts=True,
    mask_same_receptor_family_conflicts=True,
    mask_conformer_hard_negative=True
)


# =====================================================
# Loss
# =====================================================

loss_1d = clip_loss_single_positive(
    receptor=batch.receptors,
    peptide_sequence=strong_sequences,
    conflict_mask=batch.sequence_conflict_mask
)

loss_3d = multi_positive_clip_loss(
    receptor=batch.receptors,
    peptide_conformers=clip_conformer_candidates,
    positive_mask=positive_mask_3d,
    positive_weights=positive_weights_3d,
    negative_mask=valid_negative_mask_3d
)

loss_fusion = multi_positive_clip_loss(
    receptor=batch.receptors,
    peptide_fusion=clip_conformer_candidates,
    positive_mask=positive_mask_3d,
    positive_weights=positive_weights_3d,
    negative_mask=valid_negative_mask_3d
)


# conformer_hard_negative only enters ranking loss
loss_rank = 0

rank_pairs = sample_conformer_hard_negative_pairs(batch)

for pair in rank_pairs:

    R = pair.receptor
    C_pos = pair.true_bound_conformer
    C_alt = pair.alternative_conformer

    s_pos_3d = score_3d(R, C_pos)
    s_alt_3d = score_3d(R, C_alt)

    s_pos_fusion = score_fusion(R, C_pos)
    s_alt_fusion = score_fusion(R, C_alt)

    loss_rank_3d = max(
        0,
        margin - s_pos_3d + s_alt_3d
    )

    loss_rank_fusion = max(
        0,
        margin - s_pos_fusion + s_alt_fusion
    )

    loss_rank += pair.weight * (
        loss_rank_3d + loss_rank_fusion
    )


loss_total = (
    loss_1d
    + alpha * loss_3d
    + beta * loss_fusion
    + lambda_rank * loss_rank
)
```

---

# 13. 版本推进与实现门槛

## 13.1 总体版本判断

Phase-3 采用分版本实现是当前最稳妥的路线。核心原因是不同版本回答的问题不同：

```text
V1:
    真实 receptor-peptide complex 能否提供有效强监督。

V1.5:
    同序列构象证据是否足够、是否干净、是否存在泄露。

V2:
    在 V1/V1.5 审计通过后，构象增强监督是否提升 3D / fusion 学习。

V3:
    similar sequence / motif conformer prior 是否能提供辅助排序、解释或弱约束信息。
```

因此后续实现不得把 V1、V1.5、V2、V3 的逻辑混在同一个默认 builder 中。每一层都必须有独立输出、独立审计和显式开关。

---

## 13.2 V1 必须先闭环

V1 是 Phase-3 的基准版本，只允许使用：

```text
positive_strong_bound
```

V1 不依赖外部构象库，不启用同序列构象增强，不启用 ranking loss，不使用 prior。它的任务是验证：

```text
真实 receptor interface
↔
真实 true_bound_conformer
```

能否作为可训练、可评估、可解释的 Phase-3 微调监督。

V1 完成标准：

```text
anchor 构建稳定
true_bound_conformer QC 通过
split / leakage audit 通过
Track A / Track B 可被训练代码读取
val/test 主评估只使用 positive_strong_bound
baseline fine-tuning 有可解释结果
```

只有 V1 闭环后，才进入 V1.5 / V2。

---

## 13.3 V1.5 是证据层，不是训练层

V1.5 的默认定位是构象证据审计层，而不是训练增强层。

V1.5 可以生成：

```text
exact_pool
peptide_conformer_instance
peptide_conformer_cluster
bound_cluster_id
conformer leakage audit
```

但默认不改变训练目标，不把 `positive_aug_same_cluster` 或 `conformer_hard_negative` 写入正式训练输入。

V1.5 必须报告以下进入 V2 的门槛指标：

```text
exact_pool_coverage
anchors_with_multiple_exact_conformers
anchors_with_same_cluster_aug
anchors_with_different_cluster_hard_negative
true_bound_cluster_mapping_success_rate
leakage_violation_count
cluster_rmsd_distribution
source_split_distribution
```

如果：

```text
true_bound_cluster_mapping_success_rate 不足；
leakage_violation_count 非零且无法解释；
anchors_with_same_cluster_aug 或 anchors_with_different_cluster_hard_negative 太少；
```

则 V2 不应作为正式训练版本，只能作为小规模 ablation 或继续修正 V1.5。

---

## 13.4 V2 必须作为独立增强层

V2 才允许生成正式的：

```text
supervised_edges
```

允许的监督边仍然只有：

```text
positive_strong_bound
positive_aug_same_cluster
conformer_hard_negative
```

其中：

```text
positive_strong_bound:
    train / val / test 都保留；
    是主评估唯一使用的 edge_type。

positive_aug_same_cluster:
    train only；
    只进入 conformer_branch / fusion_branch；
    需要 multi-positive contrastive loss。

conformer_hard_negative:
    train only；
    只进入 conformer_branch / fusion_branch ranking loss；
    不进入 1D sequence branch；
    不进入普通 CLIP loss；
    不进入普通 batch negative pool。
```

V2 的成功标准不是替代 V1 主评估，而是在同样的 `positive_strong_bound` val/test 主评估下，验证构象增强是否改善 3D / fusion 表征。

---

## 13.5 V3 只能作为 prior 层

V3 引入：

```text
similar_sequence_conformer_prior
motif_conformer_prior
```

这些内容只能进入：

```text
conformer_prior
```

不能进入：

```text
exact_same_sequence cluster
supervised_edges
positive_aug_same_cluster
conformer_hard_negative
val/test 主评估
```

V3 的默认用途是：

```text
family-level 构象统计
prior ablation
retrieval reranking
生物学解释
可选弱约束实验
```

V3 不应被描述为强监督数据集的一部分。

---

## 13.6 strict split 需要工程化定义

后续实现 strict split 时，必须显式写入参数和审计结果。至少包括：

```text
peptide exact sequence split rule
peptide similarity threshold
peptide coverage threshold
receptor family definition
receptor similarity threshold
PDB/source-family leakage rule
conformer source_split leakage rule
```

`dataset_audit` 中必须保存这些参数，避免不同实现者得到不可比较的 train / val / test。

---

# 14. 最终定版原则

```text
1. Phase-3 peptide length 固定为 8–20 aa，
   与 Phase-2 预训练长度范围保持一致。

2. 8–20 aa 内部分三档：
   8–10 aa
   11–15 aa
   16–20 aa

3. 不同长度组使用不同的：
   接触验证阈值
   相似构象搜索阈值
   RMSD 聚类阈值

4. true_bound_conformer 必须通过质量检查；
   如果不通过，删除整个 anchor。

5. true_bound_conformer 必须强制加入 exact_pool。

6. split / 去重 / 防泄露必须在构象增强前完成。

7. train augmentation 不能使用 val/test 的 true_bound_conformer。

8. 构象增强边默认只在 train split 启用。

9. val/test 主评估只使用 positive_strong_bound。

10. positive_aug_same_cluster 只作为 train split 的弱正增强；
    若启用，3D/fusion 分支必须使用 multi-positive contrastive loss。

11. conformer_hard_negative 只进入 conformer/fusion ranking loss；
    不进入 1D sequence branch；
    不进入任何普通 CLIP loss；
    不进入普通负样本池。

12. ordinary in-batch negatives 不提前写入 supervised_edges；
    由 dataloader / loss 动态生成，并使用 conflict mask。

13. similar sequence / motif 相关内容只作为 prior；
    不参与 exact_same_sequence cluster；
    不参与 supervised_edges 生成。

14. supervised_edges 只允许：
    positive_strong_bound
    positive_aug_same_cluster
    conformer_hard_negative

15. conformer_prior 单独保存：
    similar_sequence_conformer_prior
    motif_conformer_prior
```

一句话总结：

```text
先用真实 receptor-peptide complex 确定强正样本 anchor；
长度固定为 8–20 aa，并分为 8–10、11–15、16–20 三档；
true_bound_conformer 必须通过 QC 并强制加入 exact_pool；
只对 exact same-sequence conformers 聚类；
train split 中同簇构象作为 multi-positive 弱增强；
同序列不同簇构象作为 conformer_hard_negative，只进入 ranking loss；
val/test 主评估只使用 positive_strong_bound；
similar sequence / motif 只作为 prior；
普通负样本由 batch 动态产生并进行 conflict mask。
```

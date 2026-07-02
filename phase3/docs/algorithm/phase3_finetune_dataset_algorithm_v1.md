# PepCLIP Phase-3 Fine-tune Dataset Algorithm V1

本文定义 Phase-3 第一版微调数据集的简化构建算法。它是后续重新实现代码前的工程规格，不是旧实现说明。

V1 只做一件事：

```text
从真实 receptor-peptide complex 中构建强监督 positive_strong_bound 数据集。
```

V1 不做构象增强，不做同序列构象挖掘训练，不做相似序列或 motif prior，不生成构象困难负样本。

---

## 1. V1 目标

Phase-3 V1 的目标是构建可用于 fine-tuning 的真实结构监督数据集：

```text
receptor interface
↔
peptide sequence
↔
true bound peptide conformer
```

每个训练样本必须来自一个真实 receptor-peptide complex。核心监督边为：

```text
edge_type = positive_strong_bound
interface_id ↔ true_bound_conformer_id
```

V1 回答的问题是：

```text
真实复合物中的 receptor interface 和 true_bound_conformer
能否作为 Phase-3 微调的有效强监督？
```

---

## 2. V1 版本边界

V1 允许：

```text
真实 receptor-peptide complex 候选收集
多来源候选去重
biological assembly 结构解析
peptide chain 质量过滤
receptor-peptide 接触验证
receptor interface 提取
true_bound_conformer 提取和 QC
防泄露 split
Track A / Track B fine-tuning 数据导出
dataset audit
```

V1 禁止：

```text
positive_aug_same_cluster
conformer_hard_negative
similar_sequence_conformer_prior
motif_conformer_prior
multi-positive contrastive loss 输入
conformer ranking loss 输入
把外部构象替代 true_bound_conformer
把未配对 peptide sequence 当作正样本
```

如果某个步骤需要外部构象库才能成立，则该逻辑不属于 V1。

---

## 3. 核心数据对象

### 3.1 Candidate

候选记录来自真实 receptor-peptide complex 数据源。每条候选至少需要能定位：

```text
candidate_id
source_db
source_entry_id
pdb_id
assembly_id
complex_structure_path
receptor_chain_id
peptide_chain_id
peptide_residue_start
peptide_residue_end
source_annotation
```

如果一个来源只有 peptide sequence、活性信息或未配对 peptide library，则不能进入 V1 候选。

### 3.2 Anchor

V1 的基本样本是 anchor：

```text
anchor_i =
receptor_interface_i
+
true_bound_conformer_i
```

anchor 表示一个真实结构级正配对。只要 `true_bound_conformer_i` 不合格，整个 anchor 必须删除。

### 3.3 Receptor Interface

从真实复合物 receptor chain 中提取两个区域：

```text
interface_5A:
    receptor 中任意 heavy atom 距离 peptide 任意 heavy atom <= 5 A 的残基

context_10A:
    receptor 中任意 heavy atom 距离 peptide 任意 heavy atom <= 10 A 的残基
```

`interface_5A` 用于 QC 和核心界面定义，`context_10A` 用于模型输入。

### 3.4 True Bound Conformer

从同一个真实复合物的 peptide chain 中提取：

```text
peptide_sequence
backbone_coords: N, CA, C, O
heavy_atom_coords: all non-H atoms
```

这是 V1 唯一允许作为强正样本的 peptide conformer。

---

## 4. 输入数据源

V1 只使用 curated receptor-peptide annotation 作为正样本标签来源。

Tier 1 主来源：

```text
Q-BioLiP
BioLiP / BioLiP2 peptide entries
```

Tier 2 扩展来源：

```text
PepBDB
Propedia
```

暂不作为 V1 强监督来源：

```text
只有 peptide sequence 的数据库
只有活性或亲和力信息但没有复合物结构的数据
没有明确 receptor chain / peptide chain 对应关系的数据
外部孤立 peptide conformer 数据
相似序列或 motif 构象库
raw PDB protein-peptide mining
```

PDB/mmCIF 文件在 V1 中只能作为 curated source 引用的坐标仓库，不能作为独立正样本标签数据库。

---

## 5. V1 主流程

```text
1. 收集真实复合物候选
2. 多来源候选去重
3. 加载或重建 biological assembly
4. 定位 receptor chain 和 peptide chain
5. 过滤 peptide chain
6. 验证 receptor-peptide 接触
7. 提取 receptor interface
8. 提取 true_bound_conformer
9. 检查 true_bound_conformer 质量
10. 生成 anchor
11. 做防泄露 split
12. 导出 V1 fine-tuning 数据
13. 生成 summary 和 dataset_audit
```

---

## 6. Step 1：收集候选

把不同数据源统一成一张候选表：

```text
candidate_id
source_db
source_entry_id
pdb_id
assembly_id
complex_structure_path
receptor_chain_id
peptide_chain_id
peptide_residue_start
peptide_residue_end
source_annotation
source_confidence
```

要求：

```text
必须能定位结构文件；
必须能定位 receptor chain；
必须能定位 peptide chain；
必须能说明该 peptide 来自真实 complex。
```

---

## 7. Step 2：候选去重

同一复合物可能被多个数据库收录。进入结构解析前必须去重。

优先去重键：

```text
pdb_id
assembly_id
receptor_chain_id
peptide_chain_id
peptide_residue_start
peptide_residue_end
```

如果残基范围缺失，降级为：

```text
pdb_id
assembly_id
receptor_chain_id
peptide_chain_id
```

多来源命中时：

```text
保留一条主记录；
合并 source_db 列表；
保留全部 source_entry_id；
记录最高 source_confidence 或来源优先级。
```

---

## 8. Step 3：重建 biological assembly

对每条去重候选：

```text
load mmCIF/PDB
select biological assembly
locate receptor chain
locate peptide chain
```

V1 不应只依赖 asymmetric unit。原因：

```text
asymmetric unit 可能包含晶体堆积接触；
asymmetric unit 可能缺少真实 biological interface；
Phase-3 需要真实 biological complex 中的 receptor-peptide interface。
```

如果无法确定 assembly：

```text
记录 qc_flag；
必要时丢弃候选；
不得静默退化为不可信结构。
```

实际实现必须区分两类情况：

```text
source_provided_complex:
    curated source 已经提供可用 receptor-peptide complex 文件。
    例如 BioLiP source PDB、PepBDB curated complex、Propedia recovered complex。
    这类记录需要在 audit 中标记 assembly_policy = source_provided_complex。

source_provided_separate_receptor_peptide:
    curated source 分别提供 receptor 和 peptide 结构文件，但来源已声明它们属于同一个 biological interaction。
    例如 Q-BioLiP receptor/ligand 文件。
    这类记录需要在 audit 中标记 assembly_policy = source_provided_separate_receptor_peptide。

gemmi_transform_to_assembly:
    输入是普通 PDB/mmCIF 且记录提供明确 biological_assembly_id。
    builder 必须使用结构文件中的 assembly metadata 重建该 assembly。
```

如果普通结构文件没有 biological assembly metadata，默认必须拒绝，不允许静默使用 asymmetric unit。只有在显式调试参数允许时，才可以使用 asymmetric-unit fallback，并且必须写入 `dataset_audit`。

---

## 9. Step 4：过滤 peptide chain

长度范围固定为：

```text
8 <= peptide_length <= 20
```

长度处理：

```text
length < 8:
    discard

8 <= length <= 20:
    keep

length > 20:
    exclude from V1
    optionally save to future_long_peptide_set
```

长度分组：

```text
8-10:
    length_group = short_8_10

11-15:
    length_group = medium_short_11_15

16-20:
    length_group = medium_long_16_20
```

过滤条件：

```text
残基不连续
backbone 原子缺失
坐标无法重建 peptide sequence
非标准残基过多
missing residue ratio 超标
不是独立 peptide chain
与 receptor 共价连接且无法作为普通复合物处理
```

---

## 10. Step 5：验证 receptor-peptide 接触

使用 heavy atom 距离确认 receptor 和 peptide 真实接触。

基础规则：

```text
min_heavy_atom_distance <= 5.0 A
```

同时计算：

```text
contact_count
interface_residue_count
peptide_contact_positions
min_heavy_atom_distance
```

初始接触阈值按长度分组：

```text
short_8_10:
    contact_count >= 4
    interface_residue_count >= 3

medium_short_11_15:
    contact_count >= 5
    interface_residue_count >= 4

medium_long_16_20:
    contact_count >= 8
    interface_residue_count >= 5
```

这些阈值是 V1 初始工程值，后续可以根据真实分布在配置中调整，但每次运行必须写入 `dataset_audit`。

没有通过接触验证的候选不能生成 anchor。

---

## 11. Step 6：提取 receptor interface

对通过接触验证的候选，提取并保存：

```text
interface_id
anchor_id
pdb_id
assembly_id
receptor_chain_id
interface_residues_5A
context_residues_10A
coords_path
```

其中：

```text
interface_residues_5A:
    receptor heavy atom 到 peptide heavy atom <= 5 A 的 receptor residues

context_residues_10A:
    receptor heavy atom 到 peptide heavy atom <= 10 A 的 receptor residues
```

如果 `interface_residues_5A` 为空或小于长度分组要求，删除候选。

---

## 12. Step 7：提取 true_bound_conformer

从同一 complex 的 peptide chain 提取 true bound conformer：

```text
true_bound_conformer_id
peptide_sequence
peptide_length
length_group
backbone_coords_path
heavy_atom_coords_path
source_anchor_id
source_split
is_true_bound = true
```

硬规则：

```text
true_bound_conformer 是 anchor 的核心；
如果 true_bound_conformer 不合格，删除整个 anchor；
不能删除 conformer 后保留 receptor interface。
```

---

## 13. Step 8：true_bound_conformer QC

以下情况删除整个 anchor：

```text
N / CA / C / O backbone 原子缺失
关键接触残基坐标缺失
残基编号不连续
peptide sequence 与坐标重建结果不一致
missing_ratio 超标
altloc / occupancy 无法稳定解析
坐标质量过差
```

QC 结果必须记录：

```text
qc_status
qc_flags
missing_ratio
failed_reason
```

---

## 14. Step 9：生成 anchor 表

通过全部 QC 后生成 anchor：

```text
anchor_id
source_dbs
pdb_id
assembly_id
receptor_chain_id
peptide_chain_id
peptide_sequence
peptide_length
length_group
interface_id
true_bound_conformer_id
contact_count
interface_residue_count
min_heavy_atom_distance
peptide_contact_positions
qc_status
qc_flags
```

V1 中每个 anchor 只有一个强正监督：

```text
anchor_id
interface_id
true_bound_conformer_id
edge_type = positive_strong_bound
edge_weight = 1.0
```

---

## 15. Step 10：split 与防泄露

V1 不允许只依赖随机 pair-level split 作为主评估。

至少支持以下 split 方案：

```text
pair_level:
    sanity check only

peptide_exact_sequence:
    val/test peptide sequence 不出现在 train

receptor_family:
    val/test receptor family 不出现在 train

strict:
    同时控制 peptide similarity 和 receptor similarity
```

V1 默认主评估建议使用：

```text
peptide_exact_sequence split
```

如果 receptor family annotation 可用，则同时报告 receptor_family 或 strict split。

每个 anchor 必须保存：

```text
split_group
split
peptide_sequence_key
receptor_family_key
pdb_key
```

`receptor_family_key` 不能只是 exact receptor sequence hash。实际实现应支持：

```text
1. external receptor_family_map:
   例如 MMseqs cluster、UniProt family、或人工审计后的 receptor family。

2. built-in sequence-similarity cluster:
   当没有外部 map 时，使用 receptor sequence similarity + coverage 阈值生成 family。
   阈值必须写入 dataset_audit。
```

当前内置 family split 是确定性的 greedy sequence-similarity clustering。它不是 MMseqs 的替代品，但不再等同于 exact sequence split。后续如果有 MMseqs/UniProt family map，应通过外部 map 覆盖内置结果。

`strict` split 不能使用简单组合键。必须使用并查集/连通分量逻辑：

```text
如果两个 anchors 共享 peptide_sequence_key，
或共享 receptor_family_key，
或共享 pdb_key，
则必须属于同一个 split_group。
```

`dataset_audit` 必须报告：

```text
same_peptide_sequence_cross_split_count
same_pdb_cross_split_count
same_receptor_family_cross_split_count
receptor_family_method
receptor_family_identity_threshold / receptor_family_map
```

---

## 16. Step 11：导出数据

V1 推荐输出：

```text
receptor_peptide_anchor.csv
receptor_interface.parquet
peptide_true_bound_conformer.parquet
positive_strong_bound_edges.parquet
track_a_train.jsonl
track_a_val.jsonl
track_a_test.jsonl
track_b_train.jsonl
track_b_val.jsonl
track_b_test.jsonl
phase3_v1_summary.json
dataset_audit.json
```

### 16.1 Track A

Track A 面向 sequence-side / receptor-side 输入：

```text
sample_id
anchor_id
split
receptor_sequence
receptor_patch_residue_ids
receptor_patch_seq_indices
receptor_patch_sequence
receptor_sequence_or_patch
receptor_residue_ids
peptide_sequence
peptide_length
length_group
avg_contact_count
contact_coverage
receptor_key
peptide_key
edge_type = positive_strong_bound
```

最终训练格式必须能被 `phase2.pepclip.data.PepCLIPDataset` 直接读取。也就是说，Track A 不能只输出 `receptor_sequence_or_patch` 这种说明性字段，必须显式输出默认训练字段：

```text
receptor_patch_sequence
peptide_sequence
```

### 16.2 Track B

Track B 面向 conformer-side / 3D 输入：

```text
sample_id
anchor_id
split
interface_id
true_bound_conformer_id
receptor_coords_path
receptor_patch_coords_path
interface_residues_5A
context_residues_10A
receptor_context_residue_count
receptor_context_atom_count
receptor_interface_atom_count
receptor_peptide_contact_pair_count_5A
patch_residue_ids
peptide_residue_ids
patch_atoms
receptor_atoms
peptide_atoms
patch_cutoff
receptor_key
peptide_key
peptide_sequence_id
receptor_family_30_id
receptor_interface_key
peptide_sequence
backbone_coords_path
heavy_atom_coords_path
edge_type = positive_strong_bound
```

最终训练格式必须能被 `phase2.pepclip.data.PepCLIP3DDataset` 直接读取。因此 `patch_atoms` 或 `receptor_atoms` 与 `peptide_atoms` 是训练必需字段；`receptor_coords_path` / `receptor_patch_coords_path` 只是审计和人工检查 sidecar，不能作为唯一 3D 输入。

`receptor_coords_path` / `receptor_patch_coords_path` 指向的 receptor 3D patch 文件至少包含：

```text
receptor_context_residues
receptor_context_atoms
receptor_context_backbone_atoms
receptor_interface_residues
receptor_interface_atoms
receptor_peptide_contact_pairs_5A
```

其中 `receptor_context_atoms` 来自 10A context receptor patch，`receptor_interface_atoms` 来自 5A interface receptor subset，`receptor_peptide_contact_pairs_5A` 记录 receptor context atoms 与 peptide heavy atoms 之间的真实接触边。若 receptor context backbone 不完整，V1 必须拒绝该 anchor，并在 audit 中记为 `receptor_patch_quality`。

---

## 17. dataset_audit 必须记录

```text
input_candidate_count
deduplicated_candidate_count
filtered_by_missing_structure
filtered_by_missing_chain
filtered_by_length
filtered_by_peptide_chain_quality
filtered_by_contact
filtered_by_interface_quality
filtered_by_true_bound_conformer_quality
final_anchor_count
split_counts
length_group_counts
unique_peptide_count
unique_receptor_count
unique_pdb_count
qc_flag_counts
leakage_checks
parameter_snapshot
input_file_hashes
output_file_hashes
```

`parameter_snapshot` 至少包括：

```text
min_peptide_length
max_peptide_length
contact_cutoff
context_cutoff
length_group_contact_thresholds
split_mode
random_seed
source_priority
```

---

## 18. V1 伪代码

```python
candidates = collect_candidates()

candidates = deduplicate_candidates(
    candidates,
    keys=[
        "pdb_id",
        "assembly_id",
        "receptor_chain_id",
        "peptide_chain_id",
        "peptide_residue_start",
        "peptide_residue_end",
    ],
)

anchors = []

for candidate in candidates:
    structure = load_biological_assembly(
        candidate.complex_structure_path,
        assembly_id=candidate.assembly_id,
    )

    receptor = select_chain(structure, candidate.receptor_chain_id)
    peptide = select_chain(structure, candidate.peptide_chain_id)

    if receptor is None or peptide is None:
        audit.reject(candidate, "missing_chain")
        continue

    peptide_sequence = reconstruct_sequence(peptide)
    peptide_length = len(peptide_sequence)

    if not 8 <= peptide_length <= 20:
        audit.reject(candidate, "length_out_of_range")
        continue

    length_group = assign_length_group(peptide_length)

    if peptide_chain_quality_bad(peptide):
        audit.reject(candidate, "peptide_chain_quality")
        continue

    contact = compute_receptor_peptide_contacts(
        receptor=receptor,
        peptide=peptide,
        cutoff=5.0,
    )

    thresholds = get_contact_thresholds(length_group)

    if contact.min_heavy_atom_distance > 5.0:
        audit.reject(candidate, "no_heavy_atom_contact")
        continue

    if contact.contact_count < thresholds.min_contact_count:
        audit.reject(candidate, "low_contact_count")
        continue

    if contact.interface_residue_count < thresholds.min_interface_residue_count:
        audit.reject(candidate, "low_interface_residue_count")
        continue

    interface = extract_receptor_interface(
        receptor=receptor,
        peptide=peptide,
        interface_cutoff=5.0,
        context_cutoff=10.0,
    )

    true_bound_conformer = extract_true_bound_conformer(
        peptide=peptide,
        peptide_sequence=peptide_sequence,
    )

    if true_bound_conformer_quality_bad(true_bound_conformer):
        audit.reject(candidate, "true_bound_conformer_quality")
        continue

    anchor = make_anchor(
        candidate=candidate,
        interface=interface,
        true_bound_conformer=true_bound_conformer,
        contact=contact,
        length_group=length_group,
    )

    anchors.append(anchor)

anchors = assign_splits(
    anchors,
    mode="peptide_exact_sequence",
    seed=20260630,
)

edges = [
    make_positive_strong_bound_edge(anchor)
    for anchor in anchors
]

export_anchor_table(anchors)
export_interface_table(anchors)
export_true_bound_conformer_table(anchors)
export_positive_strong_bound_edges(edges)
export_track_a(anchors, edges)
export_track_b(anchors, edges)
export_dataset_audit(anchors, audit)
```

---

## 19. V1 完成标准

V1 完成后必须满足：

```text
所有样本来自真实 receptor-peptide complex
所有 peptide 长度在 8-20 aa
所有 anchor 都有合格 true_bound_conformer
所有训练监督边都是 positive_strong_bound
val/test 主评估只使用 positive_strong_bound
dataset_audit 中无无法解释的泄露
Track A / Track B 可被 Phase-3 baseline fine-tuning 代码读取
```

---

## 20. 后续版本接口

V1 输出是后续版本的输入，但 V1 本身不实现增强逻辑。

V1.5 可以读取：

```text
receptor_peptide_anchor.csv
peptide_true_bound_conformer.parquet
```

用于构建：

```text
exact same-sequence conformer evidence
conformer clusters
bound_cluster_id mapping
conformer leakage audit
```

V2 可以在 V1.5 审计通过后额外生成：

```text
supervised_edges.parquet
```

V3 可以在 V2 稳定后额外生成：

```text
conformer_prior.parquet
```

这些后续产物不得反向改变 V1 anchor 定义和 V1 主评估规则。

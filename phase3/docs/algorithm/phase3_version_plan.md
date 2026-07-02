# PepCLIP Phase-3 版本路线

本文用于区分 Phase-3 当前可落地的数据集版本与后续构象增强微调方案。核心原则是：

```text
V1 先证明真实 bound conformer 微调是否有效；
V1.5 只补齐 exact same-sequence 构象证据与审计；
V2 再启用同序列构象增强训练；
V3 最后引入 similar sequence / motif prior。
```

不要把 V2/V3 的构象增强逻辑直接并入最小可用 V1。否则数据构建、训练损失和评估语义会同时变化，实验结果很难归因。

## 总体版本划分

| Version | 定位 | 训练监督 | 默认训练使用 | 主要目的 |
|---|---|---|---|---|
| Phase-3 V1 | 真实 bound pair 微调集 | `positive_strong_bound` | 是 | 验证真实复合物强监督能否提升 receptor-peptide 检索 |
| Phase-3 V1.5 | exact same-sequence 构象证据层 | `positive_strong_bound`，构象证据仅审计 | 否 | 统计同序列构象覆盖率、聚类质量和 bound conformer 映射 |
| Phase-3 V2 | 构象增强训练 | `positive_strong_bound`、`positive_aug_same_cluster`、`conformer_hard_negative` | 仅 train split | 训练模型区分同一 peptide 的不同构象适配性 |
| Phase-3 V3 | similar / motif prior 扩展 | V2 监督 + 独立 `conformer_prior` | 可选 prior，不作为强监督 | 引入相似序列或 motif family 的构象先验 |

## Phase-3 V1：真实 Bound Pair 微调集

V1 是当前最小可用版本。只使用真实 receptor-peptide complex 中的强监督：

```text
receptor_interface_i <-> true_bound_conformer_i
```

V1 做：

```text
真实 receptor-peptide complex
-> anchor
-> true_bound_conformer
-> QC
-> split
-> positive_strong_bound
-> ordinary CLIP fine-tuning
```

V1 不做：

```text
positive_aug_same_cluster
conformer_hard_negative
similar sequence prior
motif prior
multi-positive contrastive loss
ranking loss
```

V1 的实验问题是：

```text
真实 bound conformer 微调能不能提升 receptor-peptide 检索？
```

## Phase-3 V1.5：Exact Same-Sequence 构象证据

V1.5 在 V1 的 anchor 基础上增加完整的同序列构象证据，但默认不改变训练目标。

V1.5 做：

```text
exact_pool = [true_bound_conformer] + searched_exact_conformers
exact same-sequence conformer QC
exact same-sequence RMSD clustering
bound_cluster_id mapping
conformer evidence statistics
coverage / leakage / cluster audit
```

V1.5 默认不做：

```text
positive_aug_same_cluster training
conformer_hard_negative training
multi-positive loss
ranking loss
```

V1.5 的实验问题是：

```text
真实 bound conformer 是否能稳定映射到同序列构象簇？
同序列外部构象证据覆盖率是否足够支持 V2？
```

当前 `phase3/conformer_v1` 更接近这一层：它补齐 full-PDB exact-match conformer evidence，并且不把外部构象变成新的 receptor-positive pair。

## Phase-3 V2：构象增强训练

V2 是构象增强微调主体。它开始把同一 peptide sequence 的构象簇结构用于训练。

V2 启用：

```text
positive_strong_bound
positive_aug_same_cluster
conformer_hard_negative
multi-positive contrastive loss
conformer/fusion ranking loss
train-only augmentation
```

V2 的关键语义：

```text
positive_aug_same_cluster:
    同一 peptide sequence
    与 true_bound_conformer 属于同一构象簇
    只作为 train split 的弱正增强

conformer_hard_negative:
    同一 peptide sequence
    但属于不同构象簇
    表示该构象状态对当前 receptor interface 不如 true_bound_conformer 匹配
```

`conformer_hard_negative` 不是 peptide-level negative，不能进入：

```text
1D sequence branch
普通 CLIP loss
普通 batch negative pool
multi-positive contrastive loss 的 denominator
```

它只能进入：

```text
conformer branch ranking loss
fusion branch ranking loss
```

V2 的实验问题是：

```text
在真实 bound pair 微调之外，
同序列同簇弱正样本和同序列异簇 hard negative
是否能提升构象级 receptor-conformer 匹配能力？
```

## Phase-3 V3：Similar / Motif Prior 扩展

V3 才引入相似序列和 motif family 的构象先验。它们不是同一条 peptide 的真实构象，不能作为 supervised positive edge。

V3 只允许写入：

```text
conformer_prior.parquet
```

prior 类型包括：

```text
similar_sequence_conformer_prior
motif_conformer_prior
```

V3 不允许 similar / motif conformer 参与：

```text
exact_same_sequence cluster
supervised_edges 生成
positive_aug_same_cluster
conformer_hard_negative
```

V3 的实验问题是：

```text
相似 peptide / motif family 的构象先验
是否能作为辅助信息提高检索或排序表现？
```

## 跨版本硬规则

以下规则所有版本都必须遵守：

```text
1. Phase-3 peptide length 固定为 8-20 aa。
2. length < 8 直接丢弃。
3. length > 20 不进入当前 Phase-3，保存到 future_long_peptide_set。
4. true_bound_conformer 是 anchor 核心。
5. 如果 true_bound_conformer 质量不合格，删除整个 anchor。
6. true_bound_conformer 必须强制加入 exact_pool。
7. split / 去重 / 防泄露必须在构象增强前完成。
8. train augmentation 不能使用 val/test 的 true_bound_conformer。
9. val/test 主评估只使用 positive_strong_bound。
10. ordinary in-batch negatives 不提前写入 supervised_edges，由 dataloader / loss 动态生成，并使用 conflict mask。
```

## 允许的监督边

`supervised_edges` 只允许以下 edge type：

```text
positive_strong_bound
positive_aug_same_cluster
conformer_hard_negative
```

其中：

```text
positive_strong_bound:
    V1 起启用
    train / val / test 主评估均可使用

positive_aug_same_cluster:
    V2 起启用
    默认只用于 train split

conformer_hard_negative:
    V2 起启用
    默认只用于 train split
    只进入 conformer/fusion ranking loss
```

`similar_sequence_conformer_prior` 和 `motif_conformer_prior` 不属于 supervised edge。

## 推荐落地顺序

```text
1. 固化 V1 / V1-beta 数据集与 ordinary CLIP fine-tuning baseline。
2. 跑通 V1.5 exact same-sequence conformer evidence，并完成覆盖率与泄露审计。
3. 在 V1.5 审计通过后，实现 V2 的 supervised_edges 和训练损失。
4. 仅当 V2 有稳定收益后，再实现 V3 的 conformer_prior。
```

一句话总结：

```text
V1 解决真实强监督；
V1.5 解决同序列构象证据；
V2 解决构象增强训练；
V3 解决相似序列 / motif 先验。
```

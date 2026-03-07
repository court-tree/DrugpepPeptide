import torch
import torch.nn as nn
import torch.distributed as dist

class ClipLoss(nn.Module):
    def __init__(self, cache_labels=True):
        super().__init__()
        self.cache_labels = cache_labels
        
        # 缓存 labels 以减少重复计算
        self.prev_num_logits = 0
        self.labels = {}
        
        # ✨【改进】自动检测分布式环境，防止手动传参出错
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def forward(self, pocket_features, peptide_features, logit_scale):
        device = pocket_features.device
        
        # 1. 分布式训练处理：收集所有 GPU 上的特征
        # 即使 world_size=1，这段逻辑也能安全跳过
        if self.world_size > 1:
            # GatherLayer.apply 返回的是 tuple，需要 cat 成一个大 Tensor
            all_pocket_features = torch.cat(GatherLayer.apply(pocket_features), dim=0)
            all_peptide_features = torch.cat(GatherLayer.apply(peptide_features), dim=0)
        else:
            all_pocket_features = pocket_features
            all_peptide_features = peptide_features

        # 2. 计算相似度矩阵 (Cosine Similarity * Temperature)
        # 结果 shape: [Global_Batch, Global_Batch]
        # 注意：这里计算了全量矩阵，对于 4090 显存来说 (Batch 128*2) 完全没压力
        logits_per_pocket = logit_scale * all_pocket_features @ all_peptide_features.T
        logits_per_peptide = logits_per_pocket.T

        # 3. 生成标签 (Labels)
        # 对比学习的标准假设：第 k 个口袋 匹配 第 k 个多肽
        num_logits = logits_per_pocket.shape[0]
        
        # 动态生成并缓存 Labels
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        # 4. 计算双向交叉熵损失
        loss_p2m = nn.functional.cross_entropy(logits_per_pocket, labels)
        loss_m2p = nn.functional.cross_entropy(logits_per_peptide, labels)
        
        # 总损失
        total_loss = (loss_p2m + loss_m2p) / 2
        
        return total_loss

# --- 辅助类：用于 DDP 环境下的梯度回传 ---
# 这是一个标准的 DDP Hack，确保梯度能流过 all_gather
class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 1. 准备容器
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        # 2. 收集数据 (无梯度)
        dist.all_gather(output, x)
        # 3. 返回 Tuple
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        # 1. 堆叠梯度
        all_gradients = torch.stack(grads)
        # 2. 累加所有 GPU 的梯度 (All-Reduce)
        dist.all_reduce(all_gradients)
        # 3. 只取回属于当前 GPU 的那部分梯度
        return all_gradients[dist.get_rank()]
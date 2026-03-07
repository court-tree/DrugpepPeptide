import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import sys
import argparse
import random

# --- 1. 路径设置 (将 src 加入路径以便导入模块) ---
sys.path.append("./src")
from model import DrugPeptideCLIP
from loss import ClipLoss
from dataset_lmdb import PeptidePocketDataset, collate_fn

# 尝试导入 unicore 的 Dictionary
try:
    from unicore.data import Dictionary
except ImportError:
    print("❌ Critical: Uni-Core not installed properly.")
    exit(1)

# --- 2. 模拟 Uni-Mol 的配置类 ---
# 这些参数必须与下载的 pocket_pretrain.pt 权重匹配
class PocketConfig:
    def __init__(self):
        self.encoder_layers = 15
        self.encoder_embed_dim = 512
        self.encoder_ffn_embed_dim = 2048
        self.encoder_attention_heads = 64
        self.max_seq_len = 512
        self.dropout = 0.1
        self.emb_dropout = 0.1
        self.attention_dropout = 0.1
        self.activation_dropout = 0.0
        self.pooler_dropout = 0.0
        self.delta_pair_repr_dim = 1
        self.activation_fn = "gelu"
        self.post_ln = False
        self.mode = "train"


# ==========================================
# 📊 免费批改引擎：计算 In-Batch Top-1 / Top-3 准确率
# ==========================================
def calc_in_batch_acc(p_embed, m_embed):
    """
    不需要额外显存，直接利用当前 batch 算准确率
    """
    # 1. 算相似度矩阵: [Batch, Dim] x [Dim, Batch] -> [Batch, Batch]
    logits = p_embed @ m_embed.T
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device)
    
    # 2. 拿到前 3 名的索引
    # 假设 Batch=8，由于选项只有 8 个，Top-3 相当于闭眼猜中率 37.5%
    k_max = min(3, batch_size) 
    _, top_indices = logits.topk(k_max, dim=1)
    
    # 3. 算正确率
    top1 = top_indices[:, :1].eq(labels.view(-1, 1)).sum().item() / batch_size * 100
    top3 = top_indices[:, :3].eq(labels.view(-1, 1).expand_as(top_indices[:, :3])).sum().item() / batch_size * 100
    
    return top1, top3

def main():
    # =====================================================
    # A. 配置区域 (根据您的 3060 笔记本调整)
    # =====================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8          # 3060 显存较小，建议从 4 或 8 开始
    LR = 1e-4               # 学习率
    EPOCHS = 5
    VAL_INTERVAL = 500
    VAL_TRIALS = 20
    VAL_CANDIDATES = 100
    
    # 路径配置 (请根据实际情况确认)
    LMDB_PATH = "E:/DrugPeptide/ppi_pretrain.lmdb"
    DICT_PATH = "E:/DrugPeptide/src/UniMol_Repo/unimol/data/dict.txt" # Uni-Mol 字典路径
    ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
    POCKET_WEIGHTS = "E:/DrugPeptide/checkpts/pocket_pre_220816.pt" # 预训练权重路径

    print(f"🚀 Training Device: {DEVICE}")
    print(f"📂 Loading Data from: {LMDB_PATH}")

    # =====================================================
    # B. 准备数据与字典
    # =====================================================
    # 1. 加载字典
    if not os.path.exists(DICT_PATH):
        print(f"⚠️ Dictionary not found at {DICT_PATH}. Please check path.")
        # 如果实在找不到，这里可以插入一段代码生成临时字典，但推荐用官方的
        return
    dictionary = Dictionary.load(DICT_PATH)
    dictionary.add_symbol("[MASK]", is_special=True)
    print(f"✅ Dictionary loaded. Size: {len(dictionary)}")

    # 2. 加载 ESM Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)

    # 3. 初始化 Dataset
    dataset = PeptidePocketDataset(
        lmdb_path=LMDB_PATH,
        tokenizer=tokenizer,
        dictionary=dictionary,
        max_atoms=256,    # 显存优化：限制原子数
        max_seq_len=64    # 显存优化：限制多肽长度
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,    # Windows 下必须设为 0
        pin_memory=True
    )

    # =====================================================
    # C. 初始化模型
    # =====================================================
    print("🛠️ Initializing Model...")
    pocket_config = PocketConfig()
    
    model = DrugPeptideCLIP(
        pocket_model_config=pocket_config,
        pocket_dict=dictionary,
        esm_model_name=ESM_MODEL,
        projection_dim=512,
        freeze_esm_layers=6,  # 冻结前 6 层，节省显存
        pocket_pooling='mean' # 使用平均池化 (更稳定)
    ).to(DEVICE)
    # ==========================================
    # 🛡️ 智能冻结策略：保护核心大脑，只训练“外围接口”
    # ==========================================
    print("🛡️ Freezing pre-trained encoders to stabilize training...")

    for name, param in model.named_parameters():
        # 只要参数名字里包含以下关键词，就放开训练：
        # 1. embed_tokens: 必须重训的 Uni-Mol 106种原子词典
        # 2. proj / projection: CLIP 必须训练的两个线性投影层
        # 3. logit_scale: CLIP 的温度超参数
        # 严格限定只放开我们自己写的最后两层投影，以及底层的字典和温度参数
        if "embed_tokens" in name or "pocket_proj" in name or "peptide_proj" in name or "logit_scale" in name:
            param.requires_grad = True
        else:
            # 其他所有层（无论你把 ESM 叫什么名字，也无论是哪层 Transformer）全部无情冻结！
            param.requires_grad = False

    # 打印出真正参与训练的层，让我们做到心里有数
    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"✅ Active Trainable Layers ({len(trainable_params)} total):")
    for name in trainable_params:
        print(f"   - {name}")
    # ==========================================
    # 加载 Uni-Mol 预训练权重 (重要！)
    if os.path.exists(POCKET_WEIGHTS):
        model.load_pocket_weights(POCKET_WEIGHTS)
    else:
        print(f"⚠️ Pretrained weights not found at {POCKET_WEIGHTS}. Training from scratch (NOT RECOMMENDED).")

    # =====================================================
    # D. 优化器与 Loss
    # =====================================================
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = ClipLoss(cache_labels=True).to(DEVICE)

    # =====================================================
    # E. 训练循环
    # =====================================================
    print("🔥 Start Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        step_count = 0
        
        for step, batch in enumerate(loader):
            if batch is None: continue # 跳过空 Batch

            # 1. Move to GPU
            pocket_tokens = batch["pocket_src_tokens"].to(DEVICE)
            pocket_coord = batch["pocket_src_coord"].to(DEVICE)
            pocket_dist = batch["pocket_src_distance"].to(DEVICE)
            pocket_edge = batch["pocket_src_edge_type"].to(DEVICE)
            
            pep_ids = batch["peptide_input_ids"].to(DEVICE)
            pep_mask = batch["peptide_attention_mask"].to(DEVICE)

            # 2. Forward
            # 注意：forward 返回 (p_embed, m_embed, logit_scale)
            p_embed, m_embed, logit_scale = model(
                pocket_src_tokens=pocket_tokens,
                pocket_src_distance=pocket_dist,
                pocket_src_coord=pocket_coord,
                pocket_src_edge_type=pocket_edge,
                peptide_input_ids=pep_ids,
                peptide_attention_mask=pep_mask
            )

            # 3. Compute Loss
            loss = loss_fn(p_embed, m_embed, logit_scale)

            # 4. Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪防止爆炸
            optimizer.step()

            # Logging
            total_loss += loss.item()
            step_count += 1
            
            if step % 10 == 0:
                with torch.no_grad():
                    # 调用我们刚写的免费引擎
                    top1, top3 = calc_in_batch_acc(p_embed, m_embed)
                print(f"  Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f} | Top-1: {top1:.1f}% | Top-3: {top3:.1f}%")

        # Epoch 结束
        avg_loss = total_loss / step_count if step_count > 0 else 0
        print(f"✅ Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")
        
        # 保存 Checkpoint
        torch.save(model.state_dict(), f"E:/DrugPeptide/checkpts/epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()

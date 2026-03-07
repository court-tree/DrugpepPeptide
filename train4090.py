import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
import os
import sys
import argparse
import random

# ==========================================
# 🚀 新增：分布式训练库
# ==========================================
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

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
    logits = p_embed @ m_embed.T
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device)
    
    k_max = min(3, batch_size) 
    _, top_indices = logits.topk(k_max, dim=1)
    
    top1 = top_indices[:, :1].eq(labels.view(-1, 1)).sum().item() / batch_size * 100
    top3 = top_indices[:, :3].eq(labels.view(-1, 1).expand_as(top_indices[:, :3])).sum().item() / batch_size * 100
    
    return top1, top3

def main():
    # =====================================================
    # 🌟 新增：初始化分布式环境 (DDP)
    # =====================================================
    dist.init_process_group(backend="nccl") # 英伟达显卡专用通信后端
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device("cuda", local_rank)
    
    # 仅在主进程（GPU 0）上打印信息，避免终端信息刷屏
    is_master = (local_rank == 0)

    # =====================================================
    # A. 配置区域 (已针对双卡 4090 优化)
    # =====================================================
    BATCH_SIZE = 128        # 🚀 4090 显存极大，从 8 提升至 128 (双卡总共256)
    LR = 3e-4               # Batch Size 变大，学习率也可适当调高
    EPOCHS = 10
    
    # ⚠️ 请将以下路径修改为您 CentOS 服务器上的实际路径
    LMDB_PATH = "./data/ppi_pretrain.lmdb"
    DICT_PATH = "./data/dict.txt" 
    POCKET_WEIGHTS = "./checkpts/pocket_pre_220816.pt" 
    ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
    SAVE_DIR = "./checkpts"

    if is_master:
        print(f"🚀 Training Device: Dual RTX 4090 (DDP Mode)")
        print(f"📂 Loading Data from: {LMDB_PATH}")
        os.makedirs(SAVE_DIR, exist_ok=True)

    # =====================================================
    # B. 准备数据与字典
    # =====================================================
    if not os.path.exists(DICT_PATH):
        if is_master: print(f"⚠️ Dictionary not found at {DICT_PATH}.")
        return
        
    dictionary = Dictionary.load(DICT_PATH)
    dictionary.add_symbol("[MASK]", is_special=True)

    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)

    dataset = PeptidePocketDataset(
        lmdb_path=LMDB_PATH,
        tokenizer=tokenizer,
        dictionary=dictionary,
        max_atoms=256,    
        max_seq_len=64    
    )
    
    # 🚀 新增：分布式数据采样器（确保两张卡分到不同的数据）
    sampler = DistributedSampler(dataset)

    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,  # 使用 sampler 时不能设置 shuffle=True
        collate_fn=collate_fn,
        num_workers=16,   # 🚀 利用您的 72 核 CPU 进行多线程数据加载
        pin_memory=True   # 加速内存到显存的转移
    )

    # =====================================================
    # C. 初始化模型
    # =====================================================
    if is_master: print("🛠️ Initializing Model...")
    pocket_config = PocketConfig()
    
    model = DrugPeptideCLIP(
        pocket_model_config=pocket_config,
        pocket_dict=dictionary,
        esm_model_name=ESM_MODEL,
        projection_dim=512,
        freeze_esm_layers=6,  
        pocket_pooling='mean' 
    ).to(DEVICE)

    # 🛡️ 智能冻结策略
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "pocket_proj" in name or "peptide_proj" in name or "logit_scale" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    if is_master:
        trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
        print(f"✅ Active Trainable Layers ({len(trainable_params)} total)")

    # 加载预训练权重
    if os.path.exists(POCKET_WEIGHTS):
        model.load_pocket_weights(POCKET_WEIGHTS)

    # 🚀 新增：将模型包装进 DDP
    # find_unused_parameters=True 是因为冻结了部分层，防止 DDP 报错
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # =====================================================
    # D. 优化器与 Loss
    # =====================================================
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = ClipLoss(cache_labels=True).to(DEVICE)
    
    # 🚀 新增：AMP 混合精度缩放器
    scaler = GradScaler()

    # =====================================================
    # E. 训练循环
    # =====================================================
    if is_master: print("🔥 Start Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        # ⚠️ 关键：每个 epoch 必须调用 set_epoch，否则 DDP 的打乱顺序会失效
        sampler.set_epoch(epoch)
        
        total_loss = 0
        step_count = 0
        
        for step, batch in enumerate(loader):
            if batch is None: continue 

            # 1. Move to GPU
            pocket_tokens = batch["pocket_src_tokens"].to(DEVICE)
            pocket_coord = batch["pocket_src_coord"].to(DEVICE)
            pocket_dist = batch["pocket_src_distance"].to(DEVICE)
            pocket_edge = batch["pocket_src_edge_type"].to(DEVICE)
            pep_ids = batch["peptide_input_ids"].to(DEVICE)
            pep_mask = batch["peptide_attention_mask"].to(DEVICE)

            # 🚀 2. Forward (使用 AMP 半精度前向传播)
            with autocast():
                p_embed, m_embed, logit_scale = model(
                    pocket_src_tokens=pocket_tokens,
                    pocket_src_distance=pocket_dist,
                    pocket_src_coord=pocket_coord,
                    pocket_src_edge_type=pocket_edge,
                    peptide_input_ids=pep_ids,
                    peptide_attention_mask=pep_mask
                )
                loss = loss_fn(p_embed, m_embed, logit_scale)

            # 🚀 3. Backward (使用 Scaler 反向传播)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # 缩放回来以便进行梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            
            scaler.step(optimizer)
            scaler.update()

            # Logging
            total_loss += loss.item()
            step_count += 1
            
            if step % 20 == 0 and is_master:
                with torch.no_grad():
                    top1, top3 = calc_in_batch_acc(p_embed, m_embed)
                print(f"  Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f} | Top-1: {top1:.1f}% | Top-3: {top3:.1f}%")

        # Epoch 结束 (仅主进程执行打印和保存)
        if is_master:
            avg_loss = total_loss / step_count if step_count > 0 else 0
            print(f"✅ Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")
            
            # ⚠️ 注意：DDP 模式下，真实的 model 藏在 model.module 里
            torch.save(model.module.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pt"))

    if is_master:
        print("🎉 Training Completed!")
        # 销毁进程组
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
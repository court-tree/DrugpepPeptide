import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel
import sys
import os

# --- 🛠️ 终极路径修复逻辑 ---
# 1. 获取当前脚本所在目录 (E:\DrugPeptide\src)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 暴力搜索：在 src 下寻找包含 'models' 子文件夹的 'unimol' 文件夹
found = False
# 遍历 src 下的一级子文件夹 (例如 UniMol_Repo)
for folder_name in os.listdir(current_dir):
    folder_path = os.path.join(current_dir, folder_name)
    if os.path.isdir(folder_path):
        # 检查是否存在 unimol/models (标准结构)
        candidate_1 = os.path.join(folder_path, "unimol", "models")
        # 检查是否存在 models (扁平结构)
        candidate_2 = os.path.join(folder_path, "models")
        
        if os.path.exists(candidate_1):
            # 找到了！真正的包在 folder_path 下
            if folder_path not in sys.path:
                sys.path.insert(0, folder_path)
            print(f"✅ Found Uni-Mol package in: {folder_path}")
            found = True
            break
        elif os.path.exists(candidate_2) and folder_name == "unimol":
             # 找到了！真正的包就是 folder_path 本身
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            print(f"✅ Found Uni-Mol package in: {current_dir}")
            found = True
            break

if not found:
    print("❌ Error: Could not verify Uni-Mol folder structure automatically.")

try:
    # 尝试导入
    from unimol.models.unimol import UniMolModel
    print("✅ Successfully imported UniMolModel.")
except ImportError as e:
    print(f"⚠️ Import Failed: {e}")
    # 打印一下当前的 unimol 是从哪里加载的（如果有的话）
    try:
        import unimol
        print(f"   Configured unimol path: {unimol.__file__}")
    except:
        pass
    
    class UniMolModel(nn.Module): 
        def __init__(self, config, dictionary): 
            super().__init__()
            print("💀 DUMMY MODEL INIT")
            
class DrugPeptideCLIP(nn.Module):
    def __init__(self, 
                 pocket_model_config, 
                 pocket_dict,
                 esm_model_name="facebook/esm2_t30_150M_UR50D",
                 projection_dim=512,
                 freeze_esm_layers=15,   # ✨ 新增：冻结层数 (建议冻结一半)
                 pocket_pooling="cls"):  # ✨ 新增：可选 'cls' 或 'mean'
        """
        Args:
            pocket_pooling: 'cls' (需要 dataset 有 <bos> token) 或 'mean' (对原子取平均)
        """
        super().__init__()
        self.pocket_pooling = pocket_pooling
        
        # -----------------------------------------------------------
        # 1. Pocket Encoder (Uni-Mol)
        # -----------------------------------------------------------
        self.pocket_encoder = UniMolModel(pocket_model_config, pocket_dict)
        self.pocket_hidden_dim = pocket_model_config.encoder_embed_dim
        
        # -----------------------------------------------------------
        # 2. Peptide Encoder (ESM-2)
        # -----------------------------------------------------------
        print(f"Loading ESM-2: {esm_model_name}...")
        self.peptide_encoder = EsmModel.from_pretrained(esm_model_name)
        self.peptide_hidden_dim = self.peptide_encoder.config.hidden_size
        
        # ✨ 显存优化：冻结 ESM-2 部分层
        if freeze_esm_layers > 0:
            print(f"❄️ Freezing first {freeze_esm_layers} layers of ESM-2")
            # 冻结 Embedding
            self.peptide_encoder.embeddings.requires_grad_(False)
            # 冻结 Encoder Layers
            for layer in self.peptide_encoder.encoder.layer[:freeze_esm_layers]:
                layer.requires_grad_(False)
        
        # -----------------------------------------------------------
        # 3. Projection Heads (升级为 GELU)
        # -----------------------------------------------------------
        self.pocket_proj = nn.Sequential(
            nn.Linear(self.pocket_hidden_dim, self.pocket_hidden_dim),
            nn.GELU(),  # ✨ 优化：ReLU -> GELU
            nn.Linear(self.pocket_hidden_dim, projection_dim)
        )
        
        self.peptide_proj = nn.Sequential(
            nn.Linear(self.peptide_hidden_dim, self.peptide_hidden_dim),
            nn.GELU(),  # ✨ 优化：ReLU -> GELU
            nn.Linear(self.peptide_hidden_dim, projection_dim)
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, 
                pocket_src_tokens, pocket_src_distance, pocket_src_coord, pocket_src_edge_type,
                peptide_input_ids, peptide_attention_mask):
        
        # === A. Pocket Encoding ===
        unimol_out = self.pocket_encoder(
            src_tokens=pocket_src_tokens,
            src_distance=pocket_src_distance,
            src_coord=pocket_src_coord,
            src_edge_type=pocket_src_edge_type,
            encoder_masked_tokens=None
        )
        # encoder_out: [Batch, Seq_Len, Hidden_Dim]
        pocket_feats = unimol_out[0] 

        # ✨ 修复逻辑：智能池化 ✨
        if self.pocket_pooling == 'cls':
            # 警告：请确保 dataset 里 src_tokens 第一个位置是 <s> (id=0)
            pocket_vec = pocket_feats[:, 0, :]
        
        elif self.pocket_pooling == 'mean':
            # 修复：只对非 Padding 的部分求平均
            # Uni-Mol 的 padding index 通常是 1 (dictionary.pad())
            # 创建 mask: [Batch, Seq_Len, 1] (1 为有效，0 为 padding)
            pad_idx = 1 
            mask = (pocket_src_tokens.ne(pad_idx)).unsqueeze(-1).type_as(pocket_feats)
            
            # Sum(有效特征) / Count(有效原子)
            sum_feats = (pocket_feats * mask).sum(dim=1)
            mean_feats = sum_feats / (mask.sum(dim=1) + 1e-9) # 防止除零
            pocket_vec = mean_feats
            
        else:
            raise ValueError(f"Unknown pooling type: {self.pocket_pooling}")

        pocket_embed = self.pocket_proj(pocket_vec)

        # === B. Peptide Encoding ===
        esm_out = self.peptide_encoder(
            input_ids=peptide_input_ids,
            attention_mask=peptide_attention_mask
        )
        # ESM-2 的 [CLS] 始终在 0 位，且 attention_mask 已经处理了 padding
        # 但为了保险，也可以用 masked mean，不过通常取 CLS 效果最好
        peptide_vec = esm_out.last_hidden_state[:, 0, :]
        peptide_embed = self.peptide_proj(peptide_vec)

        # === C. Normalization ===
        pocket_embed = pocket_embed / (pocket_embed.norm(dim=1, keepdim=True) + 1e-6)
        peptide_embed = peptide_embed / (peptide_embed.norm(dim=1, keepdim=True) + 1e-6)

        return pocket_embed, peptide_embed, self.logit_scale.exp()
    
    def load_pocket_weights(self, path):
        """加载权重，智能跳过形状不匹配的层"""
        if not os.path.exists(path):
            print(f"⚠️ Weight file not found: {path}")
            return
            
        print(f"Loading pocket weights from {path}...")
        state_dict = torch.load(path, map_location="cpu")
        
        if "model" in state_dict: state_dict = state_dict["model"]
            
        new_state_dict = {}
        model_state_dict = self.pocket_encoder.state_dict()
        
        skipped_keys = []
        loaded_keys = []

        for k, v in state_dict.items():
            # 1. 清洗前缀
            clean_k = k.replace("pocket_encoder.", "").replace("module.", "")
            
            # 2. 尝试匹配模型中的键
            # 有些版本的 checkpoint 前缀可能不同，这里做一点模糊匹配
            if clean_k in model_state_dict:
                target_key = clean_k
            elif f"encoder.{clean_k}" in model_state_dict:
                target_key = f"encoder.{clean_k}"
            elif f"gbf.{clean_k}" in model_state_dict:
                target_key = f"gbf.{clean_k}"
            elif f"emb.{clean_k}" in model_state_dict: # 处理 embedding 的不同命名
                 target_key = f"emb.{clean_k}"
            else:
                # 如果完全找不到对应的层，就跳过
                continue

            # 3. 关键步骤：检查形状是否匹配
            if target_key in model_state_dict:
                model_shape = model_state_dict[target_key].shape
                checkpoint_shape = v.shape
                
                if model_shape == checkpoint_shape:
                    new_state_dict[target_key] = v
                    loaded_keys.append(target_key)
                else:
                    # 形状不匹配（比如 embed_tokens），跳过加载
                    skipped_keys.append(f"{target_key} (ckpt: {checkpoint_shape} vs model: {model_shape})")
        
        # 4. 加载匹配的权重
        missing, unexpected = self.pocket_encoder.load_state_dict(new_state_dict, strict=False)
        
        print(f"✅ Weights loaded successfully!")
        print(f"   - Loaded layers: {len(loaded_keys)}")
        print(f"   - Skipped layers (Size Mismatch): {len(skipped_keys)}")
        if len(skipped_keys) > 0:
            print(f"     Example skipped: {skipped_keys[0]}")
        print(f"   - Missing layers (Initialized from scratch): {len(missing)}")
import lmdb
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

# 尝试导入 unicore，如果失败则使用 Dummy 字典
try:
    from unicore.data import Dictionary
except ImportError:
    print("⚠️ Unicore not found. Using dummy dictionary logic.")
    class Dictionary:
        def bos(self): return 0
        def pad(self): return 1
        def eos(self): return 2
        def unk(self): return 3
        def index(self, sym): return 3

# Uni-Mol 支持的重原子集合
VALID_ATOMS = {'C', 'N', 'O', 'S', 'P', 'F', 'CL', 'BR', 'I'}

class PeptidePocketDataset(Dataset):
    def __init__(self, lmdb_path, tokenizer, dictionary, max_atoms=256, max_seq_len=64):
        self.lmdb_path = lmdb_path
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        self.max_atoms = max_atoms
        self.max_seq_len = max_seq_len
        
        # 1. 获取特殊 Token 的 ID
        self.bos_idx = dictionary.bos()
        self.pad_idx = dictionary.pad()
        self.eos_idx = dictionary.eos()

        # ==========================================
        # 🛡️ 多进程 Bug 修复核心点 1：只拿 keys，不要保持 env 打开
        # ==========================================
        # 临时打开 LMDB 获取数据集的所有 key，然后立即关闭
        tmp_env = lmdb.open(
            self.lmdb_path, 
            subdir=True, 
            readonly=True, 
            lock=False, 
            readahead=False, 
            meminit=False
        )
        with tmp_env.begin() as txn:
            self.keys = [key for key, _ in txn.cursor()]
        tmp_env.close() # 必须关闭，防止主进程带着打开的句柄去 fork 子进程
        
        # 真正的 env 初始化为 None，留给 worker 进程自己去创建
        self.env = None

        print(f"Dataset loaded: {len(self.keys)} samples.")

    def _init_env(self):
        # ==========================================
        # 🛡️ 多进程 Bug 修复核心点 2：懒加载机制
        # ==========================================
        # 当 DataLoader 的子进程第一次尝试拿数据时，才真正打开它专属的 LMDB 环境
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path, 
                subdir=True, 
                readonly=True, 
                lock=False, 
                readahead=False, 
                meminit=False
            )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # 每次读取前确保当前进程的 LMDB 环境已经打开
        self._init_env()

        with self.env.begin() as txn:
            byte_data = txn.get(self.keys[index])
        
        # 容错：如果读取失败，回退到第一条
        if byte_data is None:
            with self.env.begin() as txn:
                byte_data = txn.get(self.keys[0])
                
        data = pickle.loads(byte_data)

        # =======================================================
        # 1. 处理口袋 (Pocket)
        # =======================================================
        raw_atoms = data['pocket_atoms']
        coords = data['pocket_coords']

        # A. 动态过滤原子
        clean_atoms = []
        clean_coords = []
        for i, a in enumerate(raw_atoms):
            a_upper = a.upper()
            if a_upper in VALID_ATOMS:
                clean_atoms.append(a_upper)
                clean_coords.append(coords[i])
            else:
                clean_atoms.append('C') # 未知原子映射为 C
                clean_coords.append(coords[i])
        
        clean_coords = np.array(clean_coords, dtype=np.float32)

        # B. 截断 (预留 2 个位置给 BOS 和 EOS)
        if len(clean_atoms) > self.max_atoms - 2:
            clean_atoms = clean_atoms[:self.max_atoms - 2]
            clean_coords = clean_coords[:self.max_atoms - 2]

        # C. 构建 Token 序列: [BOS] + Atoms + [EOS]
        atom_ids = [self.dictionary.index(a) for a in clean_atoms]
        src_tokens = torch.tensor([self.bos_idx] + atom_ids + [self.eos_idx], dtype=torch.long)
        
        # D. 构建坐标序列: [0,0,0] + Coords + [0,0,0]
        bos_coord = np.zeros((1, 3), dtype=np.float32)
        eos_coord = np.zeros((1, 3), dtype=np.float32)
        src_coord = torch.tensor(
            np.concatenate([bos_coord, clean_coords, eos_coord], axis=0), 
            dtype=torch.float32
        )

        # E. 计算距离矩阵 [N+2, N+2]
        src_distance = torch.norm(src_coord[:, None, :] - src_coord[None, :, :], dim=-1)
        
        # F. 边类型 [N+2, N+2] (全 0)
        src_edge_type = torch.zeros((len(src_tokens), len(src_tokens)), dtype=torch.long)

        # =======================================================
        # 2. 处理多肽 (Peptide)
        # =======================================================
        seq_str = data['pep_seq']
        tokenized = self.tokenizer(
            seq_str, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_len
        )
        
        return {
            "pocket_src_tokens": src_tokens,
            "pocket_src_coord": src_coord,
            "pocket_src_distance": src_distance,
            "pocket_src_edge_type": src_edge_type,
            "peptide_input_ids": tokenized['input_ids'].squeeze(0),
            "peptide_attention_mask": tokenized['attention_mask'].squeeze(0),
            "pad_idx": self.pad_idx
        }

# ===================================================================
# Collate Function
# ===================================================================
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None

    pad_idx = batch[0]['pad_idx']

    # 1. 对齐口袋数据
    max_atom_len = max([item['pocket_src_tokens'].size(0) for item in batch])
    
    batch_tokens = []
    batch_coords = []
    batch_dists = []
    batch_edges = []

    for item in batch:
        num_atoms = item['pocket_src_tokens'].size(0)
        pad_len = max_atom_len - num_atoms
        
        # Pad Tokens
        tokens = torch.cat([item['pocket_src_tokens'], torch.full((pad_len,), pad_idx, dtype=torch.long)])
        
        # Pad Coords (填 0)
        coords = torch.cat([item['pocket_src_coord'], torch.zeros((pad_len, 3))])
        
        # Pad Distance (填 0)
        d = item['pocket_src_distance']
        d = torch.nn.functional.pad(d, (0, pad_len, 0, pad_len), value=0)
        
        # Pad Edge Types (填 0)
        e = item['pocket_src_edge_type']
        e = torch.nn.functional.pad(e, (0, pad_len, 0, pad_len), value=0)
        
        batch_tokens.append(tokens)
        batch_coords.append(coords)
        batch_dists.append(d)
        batch_edges.append(e)

    # 2. 对齐多肽数据
    peptide_ids = torch.stack([item['peptide_input_ids'] for item in batch])
    peptide_masks = torch.stack([item['peptide_attention_mask'] for item in batch])

    return {
        "pocket_src_tokens": torch.stack(batch_tokens),
        "pocket_src_coord": torch.stack(batch_coords),
        "pocket_src_distance": torch.stack(batch_dists),
        "pocket_src_edge_type": torch.stack(batch_edges),
        "peptide_input_ids": peptide_ids,
        "peptide_attention_mask": peptide_masks
    }
import os
import torch
import numpy as np
from torch_scatter import scatter_max
from torch.utils.data import DataLoader,Dataset
import pickle

class VSmodel(torch.nn.Module):
    def __init__(self, embedding_tensor_path, random_samples_path):
        super(VSmodel, self).__init__()
        embedding1 = np.load(embedding_tensor_path)
        self.register_buffer('mol_embs',  torch.from_numpy(embedding1.T).float())
        embedding2 = np.load(random_samples_path)
        self.register_buffer('random_samples',  torch.from_numpy(embedding2.T).float())
        source = embedding_tensor_path.split("/")[2]
        index = embedding_tensor_path.split("/")[-1].split(".")[0]
        self.mol_path_name = f"{source}_{index}"
        print('model loaded!')
    def forZscore(self,input_array):
        sampled_score = input_array.mm(self.random_samples)
        medians = torch.median(sampled_score, dim=1, keepdim=True)[0] #torch.median returns a tuple (values,indicecs)
        mads = torch.median(torch.abs(sampled_score - medians), dim=1, keepdim=True)[0]
        return medians, mads
    
    def forward(self, input_array,pocket_index):
        scores = input_array.mm(self.mol_embs)
        medians, mads = self.forZscore(input_array)
        scores = 0.6745 * (scores - medians) / (mads + 1e-6)
        max_scores,_ = scatter_max(scores, pocket_index,dim=0)
        topk_result = torch.topk(max_scores.contiguous(), 100000, largest=True,dim=1)
        return topk_result
    
class PocketDataset(Dataset):
    def __init__(self, pocket_reps_path, pocket_name_path=None, lambda_name=lambda x: os.path.splitext(os.path.basename(x))[0]):
        if pocket_reps_path.endswith(".pkl"):
            with open(pocket_reps_path, "rb") as f:
                pocket_reps = pickle.load(f)
        else:
            pocket_reps = np.load(pocket_name_path), np.load(pocket_reps_path)

        pocket_dict = {}
        for name,emb in zip(*pocket_reps):
            name = lambda_name(name)
            try:
                pocket_dict[name].append(emb)
            except: 
                pocket_dict[name] = [emb]
        self.pocket_name = list(pocket_dict.keys())
        self.pocket_emb = [np.array(x) for x in pocket_dict.values()]
            
        
    def __len__(self):
        return len(self.pocket_name)
    
    def __getitem__(self, idx):
        return self.pocket_name[idx], torch.tensor(self.pocket_emb[idx]).float()


def pocket_collate_fn(batch):
    names = []
    embs = []
    pocket_index = []
    for i,(n,e) in enumerate(batch):
        names.append(n)
        embs.append(e)
        pocket_index += [i for _ in range(e.size(0))]
    return names, torch.cat(embs),torch.tensor(pocket_index,dtype=torch.long)

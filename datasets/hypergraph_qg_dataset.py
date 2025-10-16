import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class HypergraphQGDataset(Dataset):
    def __init__(self, json_path, input_dim=768, target_dim=64):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.input_dim = input_dim
        self.target_dim = target_dim

        # Projection layer to match model's expected input size
        self.project = nn.Identity()
        if input_dim != target_dim:
            self.project = nn.Linear(input_dim, target_dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        q_nodes = torch.tensor(item["q_nodes"], dtype=torch.float)  # (Nq, 768)
        k_nodes = torch.tensor(item["k_nodes"], dtype=torch.float)  # (Nk, 768)

        # Apply projection if needed
        q_nodes = self.project(q_nodes)
        k_nodes = self.project(k_nodes)

        return {
            "q_nodes": q_nodes,                          # (Nq, 256)
            "k_nodes": k_nodes,                          # (Nk, 256)
            "hyperedges": item["hyperedges"],            # list of [q_idx, k1, k2, ...]
            "labels": torch.tensor(item["labels"], dtype=torch.long)  # (T,)
        }
import torch
import torch.nn as nn
from modules.hypergraph_transformer_layer import HypergraphTransformerLayer

class HypergraphEncoder(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4, num_layers=3, dropout=0.3):
        super().__init__()  # Modern super() call
        self.layers = nn.ModuleList([  # ✅ Fixed typo
            HypergraphTransformerLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, q_nodes, k_nodes, hyperedges):
        # Process full batch at once (no manual batching needed)
        for layer in self.layers:
            q_nodes = layer(q_nodes, k_nodes, hyperedges)  # ✅ Uses batch support
        return q_nodes
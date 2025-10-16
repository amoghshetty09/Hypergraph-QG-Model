import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HypergraphTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, q_nodes, k_nodes, hyperedges):
        """
        q_nodes: (B, Q, H)
        k_nodes: (B, K, H)
        hyperedges: List of B lists, where each list is [ [q_idx, k1, k2, ...], ... ]
        """
        B, Q, H = q_nodes.shape
        _, K, _ = k_nodes.shape
        device = q_nodes.device

        Q_proj = self.query(q_nodes).view(B, Q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, heads, Q, d)
        K_proj = self.key(k_nodes).view(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)    # (B, heads, K, d)
        V_proj = self.value(k_nodes).view(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, heads, K, d)

        updated_q_nodes = torch.zeros_like(q_nodes)  # (B, Q, H)

        for b in range(B):
            Q_len = q_nodes.size(1)
            K_len = k_nodes.size(1)

            for edge in hyperedges[b]:
                if len(edge) < 2:
                    continue

                # Clamp q_idx and k_idxs to within batch-padded bounds
                q_idx = min(edge[0], Q_len - 1)
                k_idxs = [min(k, K_len - 1) for k in edge[1:]]

                k_idxs_tensor = torch.tensor(k_idxs, dtype=torch.long, device=device)
                q_vec = Q_proj[b, :, q_idx, :].unsqueeze(1)  # (heads, 1, d)
                k_vecs = K_proj[b, :, k_idxs_tensor, :]      # (heads, k, d)
                v_vecs = V_proj[b, :, k_idxs_tensor, :]      # (heads, k, d)

                scores = torch.matmul(q_vec, k_vecs.transpose(-1, -2)) / math.sqrt(self.head_dim)
                attn_weights = F.softmax(scores, dim=-1)
                attn_output = torch.matmul(attn_weights, v_vecs)  # (heads, 1, d)

                attn_output = attn_output.squeeze(1).permute(1, 0).contiguous().view(-1)  # (H,)
                updated_q_nodes[b, q_idx, :] += attn_output

            updated_q_nodes[b] = updated_q_nodes[b] / (len(hyperedges[b]) + 1e-6)

        out = self.norm1(q_nodes + self.dropout(updated_q_nodes))
        out = self.norm2(out + self.dropout(self.ffn(out)))

        return out
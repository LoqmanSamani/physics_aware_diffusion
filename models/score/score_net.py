import torch
import torch.nn as nn


class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        pass




class TimeEmbedding(nn.Module):
    """time embedding used to embedd diffusion
    time in both teacher and student models"""
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, time_):
        return self.mlp(time_.unsqueeze(-1))


class GraphTransformer(nn.Module):
    """attention-based message passing"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.edge_project = nn.Linear(3, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, nodes, edges, edge_index):

        i, j = edge_index
        query_i = self.query(nodes[i])
        key_j = self.key(nodes[j])
        value_j = self.value(nodes[j])
        edge_ij = self.edge_project(edges)

        attn_logits = (query_i * (key_j + edge_ij)).sum(dim=-1)
        attn = torch.Softmax(attn_logits, dim=0)

        message = attn.unsqueeze(-1) * (value_j + edge_ij)
        agg = torch.zero_like(nodes)
        agg.index_add(0, i, message)

        return self.norm(nodes + self.out(agg))


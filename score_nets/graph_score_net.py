import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_add
import math
from typing import Optional


class GraphScoreNet(nn.Module):
    """graph score transformer"""
    def __init__(self, atom_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1, *args) -> None:
        super().__init__()
        self.position_encoder = PositionalEncoding(hidden_dim)
        self.initializer = NodeInitializer(atom_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.out_head = OutputHead(hidden_dim)

    def forward(self, data: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor,
                time_: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_nodes = data.size(0)
        pos_features = self.position_encoder(data)
        nodes = self.initializer(atom_features, time_, num_nodes, batch)
        nodes = nodes + pos_features
        if edge_index.numel() > 0:
            edges = compute_edge_features(data, edge_index)
            for layer in self.layers:
                nodes = layer(nodes, edges, edge_index)
        score = self.out_head(nodes, data)
        return score


class OutputHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.coeff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        self.bias_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, node_features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        coeff = self.coeff_mlp(node_features)
        bias = self.bias_mlp(node_features)
        score = coeff * positions + bias
        return score


class PositionalEncoding(nn.Module):
    """encode 3d positions into high-dimensional features"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.mlp(data)


class SinusoidalTimeEmbedding(nn.Module):
    """sinusoidal time embedding"""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    """time embedding"""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, time_: torch.Tensor) -> torch.Tensor:
        """
        time_: (B,) or (B, 1) or scalar
        returns: (B, embed_dim)
        """
        if time_.dim() == 0:
            time_ = time_.unsqueeze(0)
        if time_.dim() == 2:
            time_ = time_.squeeze(-1)
        return self.mlp(time_.unsqueeze(-1))


class GraphTransformer(nn.Module):
    """attention-based message passing layer."""
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_project = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        i, j = edge_index
        query_i = self.query(nodes[i])
        key_j = self.key(nodes[j])
        value_j = self.value(nodes[j])
        edge_ij = self.edge_project(edges)
        attn_logits = (query_i * (key_j + edge_ij)).sum(dim=-1) / (self.hidden_dim ** 0.5)
        attn = scatter_softmax(attn_logits, i, dim=0)
        attn = self.dropout(attn)
        message = attn.unsqueeze(-1) * (value_j + edge_ij)
        agg = torch.zeros_like(nodes)
        agg = scatter_add(message, i, dim=0, out=agg)
        nodes = self.norm1(nodes + self.output(agg))
        nodes = self.norm2(nodes + self.feedforward(nodes))
        return nodes


class NodeInitializer(nn.Module):
    """initialize node embeddings with atom features and time"""
    def __init__(self, atom_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.atom_project = nn.Linear(atom_dim, hidden_dim)
        #self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.time_embed = TimeEmbedding(hidden_dim)

    def forward(self, atom_features: torch.Tensor, time_: torch.Tensor,
                num_nodes: int, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        arguments:
            atom_features: (N, atom_dim)
            time_: (batch_size,) or scalar diffusion timestep
            num_nodes: int, number of nodes
            batch: (N,) batch assignment (optional)
        returns:
            (N, hidden_dim) initialized node features
        """
        h_atom = self.atom_project(atom_features)
        time_emb = self.time_embed(time_)
        if batch is not None:
            h_time = time_emb[batch]
        else:
            if time_emb.shape[0] == num_nodes:
                h_time = time_emb
            else:
                h_time = time_emb.expand(num_nodes, -1)
        return h_atom + h_time


def compute_edge_features(data: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    compute translation-invariant edge features: e_ij = x_i - x_j
    arguments:
        data: (N, 3) atomic coordinates
        edge_index: (2, E) edge connectivity
    returns:
        (E, 4) [relative_x, relative_y, relative_z, distance]
    """
    i, j = edge_index
    relative_pos = data[i] - data[j]
    distance = torch.norm(relative_pos, dim=-1, keepdim=True)
    edge_features = torch.cat([relative_pos, distance], dim=-1)
    return edge_features
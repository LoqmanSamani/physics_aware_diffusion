import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_add
import math
from typing import Optional



class EnergyNet(nn.Module):
    """
    energy graph transformer used as energy net of the model
    """
    def __init__(self, atom_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1, * args) -> None:
        super().__init__()
        self.initializer = NodeInitializer(atom_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        #self.energy_head = EnergyHead(hidden_dim)
        self.out_head = OutputHead(hidden_dim)

    def forward(self, data: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor,
                time_: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        arguments:
            data: (N, 3) molecular coordinates
            atom_features: (N, atom_dim) atom type features
            edge_index: (2, E) edge connectivity
            time_: (batch_size,) or scalar diffusion timestep
            batch: (N,) batch assignment for each node (optional)
        returns:
            scalar log p_theta(x,t)
        """
        num_nodes = data.size(0)
        nodes = self.initializer(atom_features, time_, num_nodes, batch)
        edges = compute_edge_features(data, edge_index)
        for layer in self.layers:
            nodes = layer(nodes, edges, edge_index)
        #logp = self.energy_head(nodes, batch)
        logp = self.out_head(nodes, batch)
        return logp


class EnergyHead(nn.Module):
    """energy head (node → scalar)"""
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_features: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        node_energy = self.mlp(node_features).squeeze(-1)
        if batch is None:
            return node_energy.sum()
        else:
            return scatter_add(node_energy, batch, dim=0)


class OutputHead(nn.Module):
    """used in models with no energy computation"""
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, node_features: torch.Tensor, *args) -> torch.Tensor:
            return self.mlp(node_features)


class TimeEmbedding(nn.Module):
    """time embedding for diffusion timestep"""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, time_: torch.Tensor) -> torch.Tensor:
        """
        arguments:
            time_: (batch_size,) or scalar
        returns:
            (batch_size, embed_dim) or (1, embed_dim)
        """
        if time_.dim() == 0:
            time_ = time_.unsqueeze(0)
        return self.mlp(time_.unsqueeze(-1))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb



class GraphTransformer(nn.Module):
    """attention-based message passing layer"""
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

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        arguments:
            nodes: (N, hidden_dim)
            edges: (E, 4) [dx, dy, dz, distance]
            edge_index: (2, E) [source, target]
        returns:
            (N, hidden_dim) updated node features
        """
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
    """
    creates the initial node embeddings for a graph by projecting
    atom features into a hidden space and adding a diffusion timestep
    embedding so that every node is aware of the current noise level.
    n_i^(0) = project(a_i) + project(t)
    """
    def __init__(self, atom_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.atom_project = nn.Linear(atom_dim, hidden_dim)
        self.time_embed = TimeEmbedding(hidden_dim)

    def forward(self, atom_features: torch.Tensor, time_: torch.Tensor, num_nodes: int,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            h_time = time_emb.expand(num_nodes, -1)
        return h_atom + h_time


def compute_edge_features(data: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    converts 3d coordinates (data) and graph connectivity (edge_index)
    into translation-invariant geometric edge features consisting of relative
    displacement and distance, making graph neural network to reason
    about spatial relationships between nodes.
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
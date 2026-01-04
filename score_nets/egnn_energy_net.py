import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_add
import math
from typing import Optional


class EGNNEnergyNet(nn.Module):
    """energy-based EGNN"""
    def __init__(self, atom_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.initializer = NodeInitializer(atom_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])
        self.energy_head = EnergyHead(hidden_dim)
        #self.out_head = OutputHead(hidden_dim)

    def forward(self, x: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor, time_: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.initializer(atom_features, time_, x.size(0), batch)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        energy = self.energy_head(h, batch)
        #score = self.out_head(h)
        return energy


class OutputHead(nn.Module):
    """simple mlp head that outputs 3d noise prediction"""
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        #nn.init.zeros_(self.mlp[-1].weight)
        #nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(node_features)



class EnergyHead(nn.Module):
    """energy head (node → scalar)"""
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
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


class EGNNLayer(nn.Module):
    """
    equivariant graph neural network (EGNN) layer
    updates both node features and coordinates
    """
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor):
        i, j = edge_index
        rel = x[i] - x[j]
        dist2 = (rel ** 2).sum(dim=-1, keepdim=True)
        m_ij = self.edge_mlp(torch.cat([h[i], h[j], dist2], dim=-1))
        coord_weights = self.coord_mlp(m_ij)
        delta_x = rel * coord_weights
        x = x + scatter_add(delta_x, i, dim=0)
        m_i = scatter_add(m_ij, i, dim=0)
        h = self.norm(self.node_mlp(torch.cat([h, m_i], dim=-1)))
        return h, x

class NodeInitializer(nn.Module):
    """
    creates the initial node embeddings by projecting
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


def compute_score(model: nn.Module, x: torch.Tensor, *args, **kwargs):
    x.requires_grad_(True)
    energy = model(x, *args, **kwargs)
    score = -torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
    return score
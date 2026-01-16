import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_add
import math
from typing import Optional


class GraphEnergyNet(nn.Module):
    """
    energy-based graph transformer for conservative score parameterization.
    the score is computed as: s_θ(x,t) = ∇_x log p_θ(x,t) = ∇_x E_θ(x,t)
    where E_θ is the energy function (log probability).
    the computed logp (output of this network)
    is then used to compute score through torch.autograd.grad()
    """
    def __init__(self, atom_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1, *args) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.initializer = NodeInitializer(atom_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.energy_head = EnergyHead(hidden_dim)

    def forward(self, coords: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor,
                t: torch.Tensor, batch_idx: Optional[torch.Tensor] = None, reduce: bool = True) -> torch.Tensor:
        """
        arguments:
            coords: (number of atoms, 3) atom coordinates
            atom_features: (number of atoms, atom_dim) atom type features
            edge_index: (2, E) edge connectivity
            t: (number of atoms, ) continuous diffusion time
            batch_idx: (number of atoms, ) specifies each atom's origin
                   (atoms belong to the same molecule has the same index)
                   it is created automatically if PyG dataloader is used.
            reduce: specifies whether computed energy per atom is
                    summed per molecule (by default this is done!)
        returns:
            scalar log p_theta(x,t) - outputs a scalar per molecule in batch
                                      (if reduce is True in energy head)
        """
        num_nodes = coords.size(0)
        coords_features = self.pos_encoder(coords)
        nodes = self.initializer(atom_features, t, num_nodes)
        nodes = nodes + coords_features
        if edge_index.numel() > 0:
            edges = compute_edge_features(coords, edge_index)
            for layer in self.layers:
                nodes = layer(nodes, edges, edge_index)
        logp = self.energy_head(nodes, batch_idx=None, reduce = False)
        return logp

class EnergyHead(nn.Module):
    """
    maps node embeddings to scalar energies and sums them
    ψ : R^K → R, then score = ∇_x Σ_i ψ(n^(L)_i)
    """
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_features: torch.Tensor, batch_idx: Optional[torch.Tensor] = None, reduce: bool = False) -> torch.Tensor:
        """maps each node to a scalar energy and optionally sums over the molecule(s)."""
        node_energy = self.mlp(node_features).squeeze(-1)  # (num_atoms,)

        if not reduce or batch_idx is None:
            return node_energy
        return scatter_add(node_energy, batch_idx, dim=0)


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

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(coords)


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

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (number of atoms, ) or scalar
        returns: (number of atoms, embed_dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 2:
            t = t.squeeze(-1)
        return self.mlp(t.unsqueeze(-1))


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


class GraphTransformer(nn.Module):
    """
    attention-based message passing layer
    n^(l+1) = φ^(l)(n^(l), e) where e_ij = x_i - x_j
    """
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
            nodes: (number of atoms, hidden_dim)
            edges: (E, 4) [dx, dy, dz, distance]
            edge_index: (2, E) [source, target]
        returns:
            (number of atoms, hidden_dim) updated node features
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
        agg = torch.zeros_like(nodes, dtype=message.dtype)
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

    def forward(self, atom_features: torch.Tensor, t: torch.Tensor,
                num_nodes: int, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        arguments:
            atom_features: (number of atoms, atom_dim)
            t: (number fo atoms, ) continuous time in range (0, 1)
            num_nodes: int, number of nodes
            batch: (number of atoms, ) batch assignment (optional)
        returns:
            (number of atoms, hidden_dim) initialized node features
        """
        h_atom = self.atom_project(atom_features)
        time_emb = self.time_embed(t)
        if batch is not None:
            h_time = time_emb[batch]
        else:
            if time_emb.shape[0] == num_nodes:
                h_time = time_emb
            else:
                h_time = time_emb.expand(num_nodes, -1)
        return h_atom + h_time


def compute_edge_features(coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    compute translation-invariant edge features: e_ij = x_i - x_j
    arguments:
        coords: (number of atoms, 3) atomic coordinates
        edge_index: (2, E) edge connectivity
    returns:
        (E, 4) [relative_x, relative_y, relative_z, distance]
    """
    i, j = edge_index
    relative_pos = coords[i] - coords[j]
    distance = torch.norm(relative_pos, dim=-1, keepdim=True)
    edge_features = torch.cat([relative_pos, distance], dim=-1)
    return edge_features
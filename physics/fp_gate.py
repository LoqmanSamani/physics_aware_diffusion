import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_std

class FPGate(nn.Module):
    """gating network that decides whether to compute FP residual"""
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def fp_gate_features(self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor,
                         batch_idx: torch.Tensor, num_molecules: int) -> torch.Tensor:
        """
        aggregate atomic features to molecular level for gate decision
        arguments:
            x: (num_atoms, 3) - atomic positions
            t: (num_molecules,) - time per molecule
            score: (num_atoms, 3) - score per atom
            batch_idx: (num_atoms,) - which molecule each atom belongs to
            num_molecules: int - total number of molecules
        returns:
            features: (num_molecules, 6) - one feature vector per molecule
        """
        score_norm_per_atom = score.norm(dim=-1, keepdim=True)
        score_norm_per_mol = scatter_mean(score_norm_per_atom, batch_idx, dim=0, dim_size=num_molecules)
        score_std_per_mol = scatter_std(score_norm_per_atom, batch_idx, dim=0, dim_size=num_molecules)
        x_norm_per_atom = x.norm(dim=-1, keepdim=True)
        x_norm_per_mol = scatter_mean(x_norm_per_atom, batch_idx, dim=0, dim_size=num_molecules)
        t = t.reshape(num_molecules, 1)
        t_features = torch.cat([t, t ** 2, torch.exp(-t)], dim=-1)
        features = torch.cat([score_norm_per_mol, t_features, score_std_per_mol, x_norm_per_mol], dim=-1)
        return features



import torch
import torch.nn as nn

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

    def fp_gate_features(self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """
        concat heuristic features that correlate with FP violation regions
        arguments:
            x: shape (batch_size, *data_dims) - input data
            t: shape (batch_size,) - diffusion times
            score: shape (batch_size, *data_dims) - predicted scores
        returns:
            features: shape (batch_size, feature_dim)
        """
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        score_flat = score.reshape(batch_size, -1)
        score_norm = score_flat.norm(dim=-1, keepdim=True)
        t = t.reshape(batch_size, 1)
        t_features = torch.cat([t, t ** 2, torch.exp(-t)], dim=-1)
        score_std = score_flat.std(dim=-1, keepdim=True)
        x_norm = x_flat.norm(dim=-1, keepdim=True)
        return torch.cat([score_norm, t_features, score_std, x_norm], dim=-1) # shape(batch_size, 6)



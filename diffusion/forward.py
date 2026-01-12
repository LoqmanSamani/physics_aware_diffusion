import torch
import torch.nn as nn
from diffusion.schedules import LinearVS
from typing import Tuple



class ForwardVP(nn.Module):
    """continuous-time vp-sde forward process"""
    def __init__(self, variance_scheduler: LinearVS, eps: float = 1e-5):
        super().__init__()
        self.vs = variance_scheduler
        self.eps = eps

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """forward diffusion: x_t = √(1 - σ²(t)) · x₀ + σ(t) · ε
        arguments:
            x0: (number of atoms, 3) clean coordinates
            noise: (number of atoms, 3) standard Gaussian noise
            t: (number of atoms, ) continuous time in range (0, 1)
        returns:
            xt: (number of atoms, 3) noised coordinates
            true_score: (number of atoms, 3) true score ∇_x log p(x_t | x_0)
        """
        assert torch.all(t > 0.0) and torch.all(t < 1.0), "Time must be in (0, 1)"
        variance = self.vs.get_variance(t)  # σ²(t)
        std = torch.sqrt(variance)  # σ(t)
        signal_coeff = torch.sqrt(1.0 - variance)  # √(1 - σ²(t))
        while std.dim() < x0.dim():
            std = std.unsqueeze(-1)
            signal_coeff = signal_coeff.unsqueeze(-1)
            variance = variance.unsqueeze(-1)
        xt = signal_coeff * x0 + std * noise
        true_score = -noise / (std + self.eps)
        return xt, true_score

import torch
import torch.nn as nn
from typing import Tuple


class ForwardVP(nn.Module):
    """forward diffusion process of variance preserving sde"""
    def __init__(self, variance_scheduler: nn.Module, *args) -> None:
        super().__init__()
        self.vs = variance_scheduler

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, t_index: torch.Tensor) -> Tuple:
        assert t_index.min() >= 0 and t_index.max() < self.vs.num_steps
        variance = self.vs.get_variance(t_index)
        signal_coeff = torch.sqrt(1.0 - variance)
        noise_coeff = torch.sqrt(variance)
        while signal_coeff.dim() < x0.dim():
            signal_coeff = signal_coeff.unsqueeze(-1)
            noise_coeff = noise_coeff.unsqueeze(-1)
        xt = signal_coeff * x0 + noise_coeff * noise
        true_score = -noise / (noise_coeff + 1e-8)
        return xt, true_score

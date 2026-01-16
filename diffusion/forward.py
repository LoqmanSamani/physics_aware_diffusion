import torch
import torch.nn as nn
from diffusion.schedules import LinearVS
from typing import Tuple


class ForwardVP(nn.Module):
    """forward diffusion process for vp-sde
    transition kernel: p(x_t | x_0) = N(x_t; α(t)x_0, σ²(t)I)
    """
    def __init__(self, variance_scheduler: LinearVS, eps: float = 1e-8):
        super().__init__()
        self.vs = variance_scheduler
        self.eps = eps

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """sample from transition kernel and compute true score
        arguments:
            x0: (batch, ..., dims) clean data
            noise: (batch, ..., dims) standard Gaussian noise
            t: (batch,) continuous time in [0, 1]
        returns:
            xt: (batch, ..., dims) noised data
            score: (batch, ..., dims) true score ∇_x log p(x_t | x_0)
        """
        alpha = self.vs.alpha(t)  # α(t)
        std = self.vs.std(t)  # σ(t)
        while alpha.dim() < x0.dim():
            alpha = alpha.unsqueeze(-1)
            std = std.unsqueeze(-1)
        # x_t = α(t)x_0 + σ(t)ε
        xt = alpha * x0 + std * noise
        # ∇_x log p(x_t | x_0) = -(x_t - α(t)x_0) / σ²(t)
        #                      = -ε / σ(t)
        score = -noise / (std + self.eps)
        return xt, score

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """return parameters of p(x_t | x_0)
        arguments:
            x0: (batch, ..., dims) clean data
            t: (batch, ) continuous time in [0, 1]
        returns:
            mean: (batch, ..., dims) α(t)x_0
            std: (batch, ..., dims) σ(t)
        """
        alpha = self.vs.alpha(t)
        std = self.vs.std(t)
        while alpha.dim() < x0.dim():
            alpha = alpha.unsqueeze(-1)
            std = std.unsqueeze(-1)
        mean = alpha * x0
        std = std.expand_as(x0)
        return mean, std
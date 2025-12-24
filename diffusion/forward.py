import torch
import torch.nn as nn


class ForwardVP(nn.Module):
    """forward diffusion process of variance preserving sde"""
    def __init__(self, variance_scheduler: nn.Module) -> None:
        super().__init__()
        self.vs = variance_scheduler

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, t_index: torch.Tensor) -> torch.Tensor:

        variance = self.vs.get_variance(t_index)
        signal_coeff = torch.sqrt(1.0 - variance)
        noise_coeff = torch.sqrt(variance)

        while signal_coeff.dim() < x0.dim():
            signal_coeff = signal_coeff.unsqueeze(-1)
            noise_coeff = noise_coeff.unsqueeze(-1)

        xt = signal_coeff * x0 + noise_coeff * noise

        return xt


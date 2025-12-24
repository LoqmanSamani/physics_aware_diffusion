import torch
import torch.nn as nn


class ReverseVP(nn.Module):
    """reverse diffusion process of variance preserving sde"""
    def __init__(self, variance_scheduler: nn.Module) -> None:
        super().__init__()
        self.vs = variance_scheduler

    def forward(self, xt: torch.Tensor, noise_pred: torch.Tensor, t_index: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        dt = self.vs.dt
        beta_t = self.vs.betas[t_index]
        variance_t = self.vs.get_variance(t_index)
        sigma_t = torch.sqrt(variance_t)

        while beta_t.dim() < xt.dim():
            beta_t = beta_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)

        score = -noise_pred / (sigma_t + 1e-8)
        drift = -0.5 * beta_t * xt - beta_t * score
        drift = drift * dt
        diffusion = torch.sqrt(beta_t * dt) * noise
        xt_prev = xt + drift + diffusion

        return xt_prev
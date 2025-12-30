import torch
import torch.nn as nn



class ReverseVP(nn.Module):
    """reverse diffusion process using reverse-time sde"""
    def __init__(self, variance_scheduler: nn.Module, *args) -> None:
        super().__init__()
        self.vs = variance_scheduler

    def forward(self, xt: torch.Tensor, score_pred: torch.Tensor, t_index: torch.Tensor,
                noise: torch.Tensor) -> torch.Tensor:
        """
        The reverse-time sde is:
        dx = [f(x,t) - g²(t)∇_x log p_t(x)] dt + g(t) d̄w
        for discrete time stepping with euler-maruyama:
        x_{t-1} = x_t + [f(x_t,t) - g²(t) * score] * dt + g(t) * √dt * noise
        """
        assert t_index.min() >= 0 and t_index.max() < self.vs.num_steps
        dt = self.vs.dt
        drift_coeff = self.vs.get_drift_coeff(t_index)
        diffusion_coeff = self.vs.get_diffusion_coeff(t_index)
        while drift_coeff.dim() < xt.dim():
            drift_coeff = drift_coeff.unsqueeze(-1)
            diffusion_coeff = diffusion_coeff.unsqueeze(-1)
        drift_term = drift_coeff * xt - (diffusion_coeff ** 2) * score_pred
        diffusion_term = diffusion_coeff * noise
        xt_prev = xt + drift_term * dt + diffusion_term * torch.sqrt(torch.tensor(dt))
        return xt_prev
import torch
import torch.nn as nn

class ReverseVP(nn.Module):
    """reverse diffusion ddpm-style"""
    def __init__(self, variance_scheduler: nn.Module, *args) -> None:
        super().__init__()
        self.vs = variance_scheduler

    def forward(self, xt: torch.Tensor, noise_pred: torch.Tensor, t_index: torch.Tensor,
                noise: torch.Tensor) -> torch.Tensor:

        assert t_index.min() >= 0 and t_index.max() < self.vs.num_steps
        variance_t = self.vs.get_variance(t_index)
        alpha_t = 1.0 - variance_t
        t_prev = torch.maximum(t_index - 1, torch.zeros_like(t_index))
        variance_t_prev = self.vs.get_variance(t_prev)
        while variance_t.dim() < xt.dim():
            variance_t = variance_t.unsqueeze(-1)
            alpha_t = alpha_t.unsqueeze(-1)
            variance_t_prev = variance_t_prev.unsqueeze(-1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_variance_t = torch.sqrt(variance_t)
        x0_pred = (xt - sqrt_variance_t * noise_pred) / (sqrt_alpha_t + 1e-8)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        sqrt_alpha_t_prev = torch.sqrt(1.0 - variance_t_prev)
        sqrt_variance_t_prev = torch.sqrt(variance_t_prev)
        if t_index[0].item() == 0:
            xt_prev = sqrt_alpha_t_prev * x0_pred
        else:
            xt_prev = sqrt_alpha_t_prev * x0_pred + sqrt_variance_t_prev * noise
        return xt_prev


"""
class ReverseVP(nn.Module):
    def __init__(self, variance_scheduler: nn.Module, *args) -> None:
        super().__init__()
        self.vs = variance_scheduler

    def forward(self, xt: torch.Tensor, noise_pred: torch.Tensor, t_index: torch.Tensor,
                noise: torch.Tensor) -> torch.Tensor:
        assert t_index.min() >= 0 and t_index.max() < self.vs.num_steps
        sigma_t = self.vs.get_std(t_index)
        beta_t = self.vs.betas[t_index]
        while beta_t.dim() < xt.dim():
            beta_t = beta_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)
        dt = 1.0
        score = -noise_pred / (sigma_t + 1e-8)
        drift = (-0.5 * beta_t * xt - beta_t * score) * dt
        diffusion = torch.sqrt(beta_t * dt) * noise
        xt_prev = xt + drift + diffusion
        return xt_prev

class ReverseVP(nn.Module):
    #reverse diffusion process of variance preserving sde
    def __init__(self, variance_scheduler: nn.Module, *args) -> None:
        super().__init__()
        self.vs = variance_scheduler

    def forward(self, xt: torch.Tensor, noise_pred: torch.Tensor, t_index: torch.Tensor,
                noise: torch.Tensor) -> torch.Tensor:
        assert t_index.min() >= 0 and t_index.max() < self.vs.num_steps
        dt = self.vs.dt
        beta_t = self.vs.betas[t_index]
        sigma_t = self.vs.get_std(t_index)
        while beta_t.dim() < xt.dim():
            beta_t = beta_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)
        # drift = -0.5 * beta_t * xt - beta_t * noise_pred / (sigma_t + 1e-8)
        score = -noise_pred / (sigma_t + 1e-8)
        #drift = -0.5 * beta_t * xt - beta_t * noise_pred / (sigma_t + 1e-8)
        drift = (-0.5 * beta_t * xt + beta_t * score) * dt
        diffusion = torch.sqrt(beta_t * dt) * noise
        xt_prev = (xt + drift + diffusion)
        return xt_prev
"""
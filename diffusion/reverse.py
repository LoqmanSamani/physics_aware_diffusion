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
    #reverse diffusion process using reverse-time sde

    def __init__(self, variance_scheduler: nn.Module, *args) -> None:
        super().__init__()
        self.vs = variance_scheduler

    def forward(self, xt: torch.Tensor, score_pred: torch.Tensor, t_index: torch.Tensor,
                noise: torch.Tensor) -> torch.Tensor:
        
        #The reverse-time sde is:
        #dx = [f(x,t) - g²(t)∇_x log p_t(x)] dt + g(t) d̄w
        #for discrete time stepping with euler-maruyama:
        #x_{t-1} = x_t + [f(x_t,t) - g²(t) * score] * dt + g(t) * √dt * noise
        
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
"""





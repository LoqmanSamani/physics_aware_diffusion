import torch
import torch.nn as nn
from diffusion.schedules import LinearVS


class ReverseVP(nn.Module):
    """
    reverse-time dynamics for vp-sde with explicit mode switching
    modes:
        - 'sde' : stochastic reverse sde (teacher / sampling)
        - 'ode' : deterministic probability flow ode (distillation / student)
    """
    def __init__(self, variance_scheduler: LinearVS, eps: float = 1e-5, default_mode: str = "ode"):
        super().__init__()
        assert default_mode in ("sde", "ode")
        self.vs = variance_scheduler
        self.eps = eps
        self.default_mode = default_mode

    def _broadcast_coeff(self, coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while coeff.dim() < x.dim():
            coeff = coeff.unsqueeze(-1)
        return coeff

    def _reverse_sde_drift(self, x, score, t):
        """
        drift of reverse-time sde:
        f(x,t) - g(t)^2 * score
        """
        beta_half = self.vs.get_drift_coeff(t)          # -½β(t)
        g = self.vs.get_diffusion_coeff(t)              # √β(t)
        beta_half = self._broadcast_coeff(beta_half, x)
        g = self._broadcast_coeff(g, x)
        return beta_half * x - (g ** 2) * score

    def _reverse_ode_drift(self, x, score, t):
        """
        drift of probability flow ode:
        f(x,t) - ½ g(t)^2 * score
        """
        beta_half = self.vs.get_drift_coeff(t)          # -½β(t)
        g = self.vs.get_diffusion_coeff(t)
        beta_half = self._broadcast_coeff(beta_half, x)
        g = self._broadcast_coeff(g, x)
        return beta_half * x - 0.5 * (g ** 2) * score

    def forward(self, xt: torch.Tensor, score: torch.Tensor, t: torch.Tensor, dt,
                *, noise: torch.Tensor = None, mode: str = None) -> torch.Tensor:
        """
        perform one reverse-time step.
        arguments:
            xt: current state at time t
            score: ∇_x log p_t(x)
            t: current time (in (0, 1))
            dt: positive time step size
            noise: Gaussian noise (required for sde mode)
            mode: 'sde' or 'ode' (overrides default_mode)
        """
        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, device=xt.device, dtype=xt.dtype)
        while dt.dim() < xt.dim():
            dt = dt.unsqueeze(-1)
        assert torch.all(dt > 0), "dt must be positive"
        mode = mode or self.default_mode
        assert mode in ("sde", "ode")
        if mode == "sde":
            if noise is None:
                noise = torch.randn_like(xt)
            drift = self._reverse_sde_drift(xt, score, t)
            g = self.vs.get_diffusion_coeff(t)
            g = self._broadcast_coeff(g, xt)
            xt_prev = xt + drift * dt + g * noise * torch.sqrt(dt)
        else:  # 'ode'
            drift = self._reverse_ode_drift(xt, score, t)
            xt_prev = xt + drift * dt
        return xt_prev




class ReverseDDPM(nn.Module):
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
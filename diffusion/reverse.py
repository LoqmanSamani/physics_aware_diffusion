import torch
import torch.nn as nn
from diffusion.schedules import LinearVS
from typing import Optional


class ReverseVP(nn.Module):
    """reverse diffusion process for vp-sde
    reverse sde: dx = [f(x,t) - g²(t)∇_x log p_t(x)]dt + g(t)dw̄
    """
    def __init__(self, variance_scheduler: LinearVS):
        super().__init__()
        self.vs = variance_scheduler

    def forward(self, xt: torch.Tensor, score: torch.Tensor, t: torch.Tensor, dt,* ,last_step: bool = False) -> torch.Tensor:
        """single reverse euler-maruyama step
        arguments:
            xt: (batch, ..., dims) current state
            score: (batch, ..., dims) score estimate ∇_x log p_t(x)
            t: (batch,) current time
            dt: scalar time step (negative for reverse)
        returns:
            x_prev: (batch, ..., dims) previous state
        """
        if not torch.is_tensor(dt):
            assert (dt < 0.0), "dt must be negative!"
            dt = torch.tensor(dt, device=xt.device, dtype=xt.dtype)

        f = self.vs.drift_coeff(t)  # -½β(t)
        g = self.vs.diffusion_coeff(t)  # √β(t)
        g_squared = self.vs.beta(t)  # β(t)
        while f.dim() < xt.dim():
            f = f.unsqueeze(-1)
            g = g.unsqueeze(-1)
            g_squared = g_squared.unsqueeze(-1)
        # reverse drift: f(x,t) - g²(t)·score
        drift = f * xt - g_squared * score
        # diffusion term
        if last_step:
            noise = torch.zeros_like(xt)
        else:
            noise = torch.randn_like(xt)
        diffusion = g * noise
        # euler-maruyama step
        x_prev = xt + drift * dt + diffusion * torch.sqrt(torch.abs(dt))
        return x_prev

    def probability_flow_ode(self, xt: torch.Tensor, score: torch.Tensor, t: torch.Tensor, dt) -> torch.Tensor:
        """single ode step (deterministic sampling)
           dx = [f(x,t) - ½g²(t)∇_x log p_t(x)]dt
        arguments:
            xt: (batch, ..., dims) current state
            score: (batch, ..., dims) score estimate
            t: (batch, ) current time
            dt: scalar time step
        returns:
            x_prev: (batch, ..., dims) previous state
        """
        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, device=xt.device, dtype=xt.dtype)
        f = self.vs.drift_coeff(t)
        g_squared = self.vs.beta(t)
        while f.dim() < xt.dim():
            f = f.unsqueeze(-1)
            g_squared = g_squared.unsqueeze(-1)
        # ode drift
        drift = f * xt - 0.5 * g_squared * score
        # euler step
        x_prev = xt + drift * dt
        return x_prev




class ReverseDDPM(nn.Module):
    """
    ddpm-style reverse process compatible with vp-sde forward process used in this study
    Uses predicted noise (ε) instead of score
    ddpm reverse step:
        x_{t-1} = (1/√α_t) * [x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t)] + σ_t * z
    where:
        - α_t = 1 - β_t (for small timestep)
        - ᾱ_t = exp(-∫₀ᵗ β(s)ds) = α²(t)
        - β_t = β(t) * Δt (discretized)
        - σ_t = √β_t for ancestral sampling, or 0 for ddpm
    """
    def __init__(self, variance_scheduler, eps: float = 1e-5, eta: float = 1.0):
        """
        arguments:
            variance_scheduler: LinearVS
            eps: small constant for numerical stability
            eta: controls stochasticity (1.0 = ddpm, 0.0 = ddim)
        """
        super().__init__()
        self.vs = variance_scheduler
        self.eps = eps
        self.eta = eta

    def _broadcast_coeff(self, coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """broadcast scalar/1d tensor to match x dimensions"""
        while coeff.dim() < x.dim():
            coeff = coeff.unsqueeze(-1)
        return coeff

    def _to_4d(self, x, ref):
        return x.view(-1, *([1] * (ref.dim() - 1)))

    def forward(self, xt: torch.Tensor, pred_noise: torch.Tensor,
                t: torch.Tensor, t_prev: torch.Tensor,
                *, noise: Optional[torch.Tensor] = None, deterministic: bool = False) -> torch.Tensor:
        """
        perform one ddpm reverse step from t to t_prev
        """
        alpha_t_sq = self.vs.alpha_squared(t)  # ᾱ_t = α²(t)
        alpha_t = torch.sqrt(alpha_t_sq)  # √ᾱ_t
        alpha_prev_sq = self.vs.alpha_squared(t_prev)  # ᾱ_{t-1}
        alpha_prev = torch.sqrt(alpha_prev_sq)  # √ᾱ_{t-1}
        sigma_t_sq = self.vs.variance(t)  # σ²_t = 1 - ᾱ_t
        sigma_t = torch.sqrt(sigma_t_sq + self.eps)  # σ_t
        sigma_prev_sq = self.vs.variance(t_prev)  # σ²_{t-1} = 1 - ᾱ_{t-1}
        alpha_t = self._broadcast_coeff(alpha_t, xt)
        alpha_prev = self._broadcast_coeff(alpha_prev, xt)
        sigma_t = self._broadcast_coeff(sigma_t, xt)
        sigma_prev_sq = self._broadcast_coeff(sigma_prev_sq, xt)
        # x0 = (x_t - σ_t * ε_θ) / √ᾱ_t
        pred_x0 = (xt - sigma_t * pred_noise) / (alpha_t + self.eps)
        direction = (xt - alpha_t * pred_x0) / (sigma_t + self.eps)
        sigma_t_sq_4d = self._to_4d(sigma_t_sq, xt)
        sigma_prev_sq_4d = self._to_4d(sigma_prev_sq, xt)
        variance = sigma_t_sq_4d - sigma_prev_sq_4d
        variance = torch.clamp(variance, min=0.0)
        if self.eta == 1.0:
            sigma_noise = torch.sqrt(variance)
        else:
            sigma_prev_sq_broadcast = self._broadcast_coeff(sigma_prev_sq, xt)
            sigma_t_sq_broadcast = self._broadcast_coeff(sigma_t_sq, xt)
            sigma_noise = self.eta * torch.sqrt(
                (variance * sigma_prev_sq_broadcast) / (sigma_t_sq_broadcast + self.eps)
            )
        sigma_noise = self._broadcast_coeff(sigma_noise, xt)
        # μ = √ᾱ_{t-1} * x0 + √(σ²_{t-1} - σ²_noise) * direction
        coeff_x0 = alpha_prev
        coeff_direction = torch.sqrt(torch.clamp(sigma_prev_sq_4d - sigma_noise ** 2, min=0.0))
        #print(coeff_x0.shape)
        #print(pred_x0.shape)
        #print(coeff_direction.shape)
        #print(direction.shape)
        mean = coeff_x0 * pred_x0 + coeff_direction * direction
        if t_prev.max() > self.eps and self.eta > 0 and not deterministic:
            if noise is None:
                noise = torch.randn_like(xt)
            x_prev = mean + sigma_noise * noise
        else:
            x_prev = mean
        return x_prev

class ReverseDDPMSimplified(nn.Module):
    """simplified ddpm reverse process"""
    def __init__(self, variance_scheduler, eps: float = 1e-5, clip_x0: bool = False):
        super().__init__()
        self.vs = variance_scheduler
        self.eps = eps
        self.clip_x0 = clip_x0

    def _broadcast_coeff(self, coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while coeff.dim() < x.dim():
            coeff = coeff.unsqueeze(-1)
        return coeff

    # xt, pred_noise, t_batch, t_prev_batch, deterministic=deterministic
    def forward(self, xt: torch.Tensor, pred_noise: torch.Tensor,
                t: torch.Tensor, t_prev: torch.Tensor,
                *, noise: Optional[torch.Tensor] = None,
                deterministic: bool = False) -> torch.Tensor:
        """
        simplified ddpm step.
        arguments:
            xt: current state
            pred_noise: Network's predicted noise
            t: current time
            t_prev: previous time (t_prev < t)
            noise: optional noise for sampling
            deterministic: if True, use ddim (no noise)
        """
        alpha_t = torch.sqrt(self.vs.alpha_squared(t))
        alpha_prev = torch.sqrt(self.vs.alpha_squared(t_prev))
        sigma_t = self.vs.std(t)
        sigma_prev = self.vs.std(t_prev)
        alpha_t = self._broadcast_coeff(alpha_t, xt)
        alpha_prev = self._broadcast_coeff(alpha_prev, xt)
        sigma_t = self._broadcast_coeff(sigma_t, xt)
        sigma_prev = self._broadcast_coeff(sigma_prev, xt)
        # x0 = (xt - σ_t * ε) / α_t
        pred_x0 = (xt - sigma_t * pred_noise) / (alpha_t + self.eps)
        if self.clip_x0:
            pred_x0 = torch.clamp(pred_x0, min=-1.0, max=1.0)
        # μ_{t→t_prev} = α_{t_prev}/α_t * (xt - σ_t²/(σ_t) * ε) + σ_{t_prev}²/(σ_t) * α_t * ε
        # simplified: μ = α_{t_prev} * pred_x0 + σ_{t_prev} * ε * (√(1 - σ²_{t_prev}/σ²_t) - 1)
        direction_to_xt = (xt - alpha_t * pred_x0) / (sigma_t + self.eps)
        mean = alpha_prev * pred_x0 + sigma_prev * direction_to_xt
        if not deterministic and t_prev.max() > self.eps:
            variance = (sigma_prev ** 2 / (sigma_t ** 2 + self.eps)) * (sigma_t ** 2 - sigma_prev ** 2)
            variance = torch.clamp(variance, min=0.0)
            sigma = torch.sqrt(variance)
            if noise is None:
                noise = torch.randn_like(xt)
            x_prev = mean + sigma * noise
        else:
            x_prev = mean
        return x_prev
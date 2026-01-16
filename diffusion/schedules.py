import torch
import torch.nn as nn
import numpy as np


class LinearVS(nn.Module):
    """Linear variance schedule for VP-SDE

    β(t) = β_min + t(β_max - β_min)
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        super().__init__()
        if not (0.0 < beta_min < beta_max):
            raise ValueError("Require 0 < beta_min < beta_max")
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """β(t) = β_min + t(β_max - β_min)"""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """∫₀ᵗ β(s) ds = β_min·t + ½(β_max - β_min)·t²"""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Mean coefficient: α(t) = exp(-½∫₀ᵗ β(s) ds)"""
        return torch.exp(-0.5 * self.integral_beta(t))

    def alpha_squared(self, t: torch.Tensor) -> torch.Tensor:
        """α²(t) = exp(-∫₀ᵗ β(s) ds)"""
        return torch.exp(-self.integral_beta(t))

    def variance(self, t: torch.Tensor) -> torch.Tensor:
        """Variance: σ²(t) = 1 - α²(t)"""
        return 1.0 - self.alpha_squared(t)

    def std(self, t: torch.Tensor) -> torch.Tensor:
        """Standard deviation: σ(t) = √(1 - α²(t))"""
        return torch.sqrt(self.variance(t))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio: SNR(t) = α²(t) / σ²(t)"""
        alpha_sq = self.alpha_squared(t)
        var = self.variance(t)
        return alpha_sq / (var + 1e-8)

    def drift_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient: f(x,t) = -½β(t)x
        Returns: -½β(t)"""
        return -0.5 * self.beta(t)

    def diffusion_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient: g(t) = √β(t)"""
        return torch.sqrt(self.beta(t))



class CosineVS(nn.Module):
    """cosine variance schedule"""
    def __init__(self, num_steps: int = 1000, s: float = 0.008):
        super().__init__()
        self.num_steps = num_steps
        self.s = s
        t = torch.linspace(0, 1, num_steps)
        alphas_cumprod = torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2

        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        variance = 1 - alphas_cumprod
        betas = torch.cat([
            torch.tensor([0.0]),
            1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        ])
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("variance", variance)
        self.register_buffer("betas", betas)

    def get_variance(self, t_index: torch.Tensor) -> torch.Tensor:
        return self.variance[t_index]

    def get_std(self, t_index: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.variance[t_index])

    def get_drift_coeff(self, t_index: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.betas[t_index]

    def get_diffusion_coeff(self, t_index: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.betas[t_index])
import torch
import torch.nn as nn
import numpy as np



class LinearVS(nn.Module):
    """variance schedule with minimum variance clipping to prevent explosion"""
    def __init__(self, beta_start=0.1, beta_end=20.0, min_variance=1e-5):
        super().__init__()
        if not (0.0 < beta_start < beta_end):
            raise ValueError("Require 0 < beta_start < beta_end")
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.min_variance = min_variance # critical for training stability (it prevents true score exploding)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """β(t)"""
        return self.beta_start + (self.beta_end - self.beta_start) * t

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """∫₀ᵗ β(s) ds"""
        return self.beta_start * t + 0.5 * (self.beta_end - self.beta_start) * t**2

    def get_variance(self, t: torch.Tensor) -> torch.Tensor:
        """σ²(t) = 1 - exp(-∫β), clipped to minimum value"""
        variance = 1.0 - torch.exp(-self.integral_beta(t))
        #return torch.clamp(variance, min=self.min_variance)
        return variance

    def get_std(self, t: torch.Tensor) -> torch.Tensor:
        """σ(t) = √(σ²(t))"""
        return torch.sqrt(self.get_variance(t))

    def get_drift_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """-½ β(t)"""
        return -0.5 * self.beta(t)

    def get_diffusion_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """√β(t)"""
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
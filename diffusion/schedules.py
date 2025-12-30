import torch
import torch.nn as nn



class LinearVS(nn.Module):
    def __init__(self, num_steps: int = 1000, beta_start: float = 0.1, beta_end: float = 20.0,
                 start: float = 0.0, end: float = 1.0, *args) -> None:
        """
        linear variance schedule for vp-sde
        β(t) = β_min + t * (β_max - β_min)
        """
        super().__init__()
        self.num_steps = num_steps
        self.start = start
        self.end = end
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if not (0.0 < beta_start < beta_end):
            raise ValueError(f"Must satisfy 0 < beta_start < beta_end")
        self.dt = (end - start) / num_steps
        t = torch.linspace(start, end, num_steps)
        betas = beta_start + (beta_end - beta_start) * t
        integral_beta = beta_start * t + 0.5 * (beta_end - beta_start) * t ** 2

        self.register_buffer("t", t)
        self.register_buffer("betas", betas)
        self.register_buffer("integral_beta", integral_beta)

    def get_variance(self, t_index: torch.Tensor) -> torch.Tensor:
        """get variance for vp sde: σ²(t) = 1 - exp(-∫₀ᵗ β(s) ds)"""
        return 1.0 - torch.exp(-self.integral_beta[t_index])

    def get_std(self, t_index: torch.Tensor) -> torch.Tensor:
        """get standard deviation: σ(t) = √(1 - exp(-∫₀ᵗ β(s) ds))"""
        return torch.sqrt(self.get_variance(t_index))

    def get_drift_coeff(self, t_index: torch.Tensor) -> torch.Tensor:
        """get drift coefficient: -0.5 * β(t)"""
        return -0.5 * self.betas[t_index]

    def get_diffusion_coeff(self, t_index: torch.Tensor) -> torch.Tensor:
        """get diffusion coefficient: √β(t)"""
        return torch.sqrt(self.betas[t_index])
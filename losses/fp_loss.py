import torch

def energy_fokker_planck_loss(residual: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return alpha * (residual ** 2).mean()

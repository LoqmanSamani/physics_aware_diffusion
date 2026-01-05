import torch

def energy_fokker_planck_loss(residual1: torch.Tensor, residual2: torch.Tensor, dim: float, alpha: float = 5e-4) -> torch.Tensor:
    return (alpha * ((residual1 * residual2) / (dim**2))).mean()

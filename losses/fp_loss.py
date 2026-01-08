import torch

def fokker_planck_loss(fp_residual: torch.Tensor, alpha: float = 5e-4) -> torch.Tensor:
    return alpha * (fp_residual ** 2).mean()


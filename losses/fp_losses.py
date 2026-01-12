import torch

def snr_fokker_planck_loss(r1, r2, variance, gamma=5.0, alpha=5e-4, *args):
    """compute snr weighted fokker-planck loss with independent fokker-planck residuals"""
    snr = (1.0 - variance) / variance.clamp(min=1e-8)
    gamma_t = torch.full_like(snr, gamma)
    weight = torch.minimum(snr, gamma_t)
    return alpha * 0.5 * (weight * (r1**2 + r2**2)).mean()


def fokker_planck_loss(r1: torch.Tensor, r2, alpha: float = 5e-4, *args) -> torch.Tensor:
    """compute fokker-planck loss with independent fokker-planck residuals"""
    return alpha * 0.5 * (r1**2 + r2**2).mean()


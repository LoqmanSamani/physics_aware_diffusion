import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor, *args) -> torch.Tensor:
    return ((pred - target) ** 2).mean()

def dsm_loss(pred_score: torch.Tensor, true_score: torch.Tensor, *args) -> torch.Tensor:
    """standard dsm loss for vp-sde diffusion models"""
    se = ((pred_score - true_score)**2).mean(dim=-1)
    return se.mean()

def snr_weighted_loss(pred_noise: torch.Tensor, target_noise: torch.Tensor,
                      variance: torch.Tensor, batch_idx: torch.Tensor, *args) -> torch.Tensor:
    """noise prediction loss weighted by the signal-to-noise ratio (snr)"""
    snr = (1 - variance) / variance.clamp(min=1e-8)
    snr_per_atom = snr[batch_idx]
    while snr_per_atom.dim() < target_noise.dim():
        snr_per_atom = snr_per_atom.unsqueeze(-1)
    return ((pred_noise - target_noise) ** 2 * snr_per_atom).mean()

def min_snr_weighted_loss(pred_noise: torch.Tensor, target_noise: torch.Tensor, variance: torch.Tensor,
                          batch_idx: torch.Tensor, gamma: float = 5.0, *args) -> torch.Tensor:
    """snr-weighted noise prediction loss with the snr capped at a maximum value"""
    snr = (1 - variance) / variance.clamp(min=1e-8)
    weight = torch.minimum(snr, torch.tensor(gamma, device=snr.device))
    weight_per_atom = weight[batch_idx]
    while weight_per_atom.dim() < target_noise.dim():
        weight_per_atom = weight_per_atom.unsqueeze(-1)
    return ((pred_noise - target_noise) ** 2 * weight_per_atom).mean()
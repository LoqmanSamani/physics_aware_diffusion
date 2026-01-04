import torch
import torch.nn.functional as F


def snr_weighted_loss(pred_noise, target_noise, variance, batch_idx, *args):
    snr = (1 - variance) / variance.clamp(min=1e-8)
    snr_per_atom = snr[batch_idx]
    while snr_per_atom.dim() < target_noise.dim():
        snr_per_atom = snr_per_atom.unsqueeze(-1)
    loss = ((pred_noise - target_noise) ** 2 * snr_per_atom).mean()
    return loss


def min_snr_weighted_loss(pred_noise, target_noise, variance, batch_idx, gamma=5.0, *args):
    snr = (1 - variance) / variance.clamp(min=1e-8)
    weight = torch.minimum(snr, torch.tensor(gamma, device=snr.device))
    weight_per_atom = weight[batch_idx]
    while weight_per_atom.dim() < target_noise.dim():
        weight_per_atom = weight_per_atom.unsqueeze(-1)
    loss = ((pred_noise - target_noise) ** 2 * weight_per_atom).mean()
    return loss


def loss_fn(pred_noise, target_noise, *args):
    return F.mse_loss(pred_noise, target_noise)
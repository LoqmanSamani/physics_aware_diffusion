import torch
import torch.nn.functional as F


def snr_weighted_loss(pred_noise, target_noise, variance, batch_idx, *args):
    snr = (1 - variance) / variance.clamp(min=1e-8)
    snr_per_atom = snr[batch_idx]
    while snr_per_atom.dim() < target_noise.dim():
        snr_per_atom = snr_per_atom.unsqueeze(-1)
    loss = ((pred_noise - target_noise) ** 2 * snr_per_atom).mean()
    return loss

"""
def min_snr_weighted_loss(pred_noise: torch.Tensor, target_noise: torch.Tensor, time_per_atom: torch.Tensor, vs, gamma: float = 5.0):
    variance = vs.get_variance(time_per_atom)
    snr = (1.0 - variance) / variance.clamp(min=1e-8)
    weight = torch.minimum(snr, torch.tensor(gamma, device=snr.device))
    while weight.dim() < pred_noise.dim():
        weight = weight.unsqueeze(-1)
    loss = ((pred_noise - target_noise) ** 2 * weight).mean()
    return loss
"""

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


def score_matching_loss(
    pred_score: torch.Tensor,
    true_score: torch.Tensor,
    variance: torch.Tensor,
    batch_idx: torch.Tensor,
):
    """
    Correct VP-SDE score matching loss:
        E[ σ_t^2 || s_theta(x_t,t) - s_true ||^2 ]
    """

    # variance = σ_t^2, shape: [num_steps] or [num_atoms]
    var_per_atom = variance[batch_idx]

    while var_per_atom.dim() < pred_score.dim():
        var_per_atom = var_per_atom.unsqueeze(-1)

    loss = ((pred_score - true_score) ** 2 * var_per_atom).mean()
    return loss

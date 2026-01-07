import torch
import torch.nn as nn


def weak_fp_residual(energy_net: nn.Module, x: torch.Tensor, atom_features: torch.Tensor,
                     edge_index: torch.Tensor, t_mol: torch.Tensor, batch: torch.Tensor, scheduler: nn.Module,
                     v_sigma: float = 1e-4, h_s: float = 1e-3, h_d: float = 5e-4) -> torch.Tensor:
    """
    per-atom weak fokker–planck residual estimator.
    returns:
        fp_residual: (num_atoms,)
    """
    num_atoms, dim = x.shape
    t_atom = t_mol[batch]
    t_norm = t_atom.float() / (scheduler.num_steps - 1)
    beta_t = scheduler.betas[t_atom]
    beta_t_exp = beta_t.unsqueeze(-1)
    v = v_sigma * torch.randn_like(x)
    x_plus = (x + v).requires_grad_(True)
    x_minus = (x - v).requires_grad_(True)
    logp_plus = energy_net(x_plus, atom_features, edge_index, t_norm, batch, reduce=False)
    logp_minus = energy_net(x_minus, atom_features, edge_index, t_norm, batch, reduce=False)
    s_plus = score_from_energy(logp_plus, x_plus)
    s_minus = score_from_energy(logp_minus, x_minus)
    div_score = ((v / v_sigma) * (s_plus - s_minus) / (2.0 * v_sigma)).sum(dim=-1)
    score_sq = (s_plus ** 2).sum(dim=-1)
    drift = -0.5 * beta_t_exp * x_plus
    drift_term = (drift * s_plus).sum(dim=-1)
    div_drift = -0.5 * beta_t * dim
    x_detached = x_plus.detach().requires_grad_(True)
    t_plus = torch.clamp(t_norm + h_s, 0.0, 1.0)
    t_minus = torch.clamp(t_norm - h_d, 0.0, 1.0)
    logp_t_plus = energy_net(x_detached, atom_features, edge_index, t_plus, batch, reduce=False)
    logp_t = energy_net(x_detached, atom_features, edge_index, t_norm, batch, reduce=False)
    logp_t_minus = energy_net(x_detached, atom_features, edge_index, t_minus, batch, reduce=False)
    numerator = h_d**2 * logp_t_plus + (h_d**2 - h_s**2) * logp_t - h_s**2 * logp_t_minus
    denominator = h_s * h_d * (h_s + h_d)
    dlogp_dt = numerator / denominator
    fp_residual = (0.5 * beta_t * (div_score + score_sq) - drift_term - div_drift - dlogp_dt)
    return fp_residual

def score_from_energy(logp: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """
    score function: computes score from energy: s_θ(x,t) = ∇_x log p_θ(x,t)
    arguments:
        logp: scalar or (batch_size,) tensor of log probabilities (output of energy net)
        data: (N, 3) coordinates that require gradients
    returns:
        (N, 3) score vectors ∇_x log p_θ(x,t)
    """
    if logp.dim() == 0:
        logp_sum = logp
    else:
        logp_sum = logp.sum()
    score = torch.autograd.grad(outputs=logp_sum, inputs=data, create_graph=True, retain_graph=True)[0]
    return score
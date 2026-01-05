import torch
import torch.nn as nn


def weak_fp_residual(energy_net: nn.Module, data: torch.Tensor, atom_features: torch.Tensor,
                     edge_index: torch.Tensor, time_: torch.Tensor, batch: torch.Tensor,
                     scheduler: nn.Module, v_sigma: float = 1e-4, h_s: float = 1e-3, h_d: float = 5e-4) -> torch.Tensor:
    """
    weak Fokker-Planck residual estimator
    R̃(x,t;v) = 0.5*g²(t)*[(v/σ)ᵀ(s_θ(x+v,t) - s_θ(x-v,t))/(2σ) + ||s_θ(x+v,t)||²]
                - ⟨f(x+v,t), s_θ(x+v,t)⟩ - div_x(f(x+v,t)) - ∂_t log p_θ(x+v,t)
    """
    beta_t = scheduler.betas[time_]
    g_sq = beta_t
    v = v_sigma * torch.randn_like(data)
    data_plus = data + v
    data_minus = data - v
    data_plus.requires_grad_(True)
    data_minus.requires_grad_(True)
    logp_plus = energy_net(data_plus, atom_features, edge_index, time_, batch)
    logp_minus = energy_net(data_minus, atom_features, edge_index, time_, batch)
    s_plus = score_from_energy(logp_plus, data_plus)
    s_minus = score_from_energy(logp_minus, data_minus)
    div_term = ((v / v_sigma) * (s_plus - s_minus) / (2.0 * v_sigma)).sum()
    score_sq = (s_plus ** 2).sum()

    if beta_t.dim() == 0:
        beta_t_expanded = beta_t
    else:
        beta_t_expanded = beta_t[batch]
        while beta_t_expanded.dim() < data_plus.dim():
            beta_t_expanded = beta_t_expanded.unsqueeze(-1)

    drift = -0.5 * beta_t_expanded * data_plus
    drift_term = (drift * s_plus).sum()
    total_dim = data_plus.numel()
    div_drift = -0.5 * beta_t * total_dim

    if beta_t.dim() > 0:
        div_drift = div_drift.sum()

    data_plus_detached = data_plus.detach().requires_grad_(True)
    time_plus = torch.clamp(time_ + h_s, 0.0, 1.0)
    time_minus = torch.clamp(time_ - h_d, 0.0, 1.0)
    logp_t_plus = energy_net(data_plus_detached, atom_features, edge_index, time_plus, batch)
    logp_t = energy_net(data_plus_detached, atom_features, edge_index, time_, batch)
    logp_t_minus = energy_net(data_plus_detached, atom_features, edge_index, time_minus, batch)
    numerator = h_d ** 2 * logp_t_plus + (h_d ** 2 - h_s ** 2) * logp_t - h_s ** 2 * logp_t_minus
    denominator = h_s * h_d * (h_s + h_d)
    dlogp_dt = numerator / denominator
    fp_residual = (0.5 * g_sq * (div_term + score_sq) - drift_term - div_drift - dlogp_dt)
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
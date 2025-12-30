import torch
import torch.nn as nn


def weak_fp_residual(energy_net: nn.Module, data: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor, time_: torch.Tensor,
                           batch: torch.Tensor, scheduler: nn.Module, v_sigma: float = 1e-2, h_time: float = 1e-3) -> torch.Tensor:
    """
    weak Fokker-Planck residual estimator
    """
    beta_t = scheduler.betas[time_]
    v = v_sigma * torch.randn_like(data)
    data_plus  = data + v
    data_minus = data - v
    logp_plus = energy_net(data_plus, atom_features, edge_index, time_, batch)
    logp_minus = energy_net(data_minus, atom_features, edge_index, time_, batch)
    s_plus = score_from_energy(logp_plus, data_plus)
    s_minus = score_from_energy(logp_minus, data_minus)
    div_term = (v * (s_plus - s_minus)).sum(dim=-1) / (2.0 * v_sigma**2)
    score_sq = (s_plus ** 2).sum(dim=-1)
    drift = -0.5 * beta_t * data_plus
    drift_term = (drift * s_plus).sum(dim=-1)
    logp_t_plus = energy_net(data_plus, atom_features, edge_index, time_ + h_time, batch)
    logp_t_minus = energy_net(data_plus, atom_features, edge_index, time_ - h_time, batch)
    dlogp_dt = (logp_t_plus - logp_t_minus) / (2.0 * h_time)
    fp_residual = (0.5 * beta_t * (div_term + score_sq) - drift_term - dlogp_dt)
    return fp_residual


def score_from_energy(logp: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """
    s_theta(x,t) = ∇_x log p_theta(x,t)
    """
    return torch.autograd.grad(logp.sum(), data, create_graph=True)[0]
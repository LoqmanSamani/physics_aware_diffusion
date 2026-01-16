import torch
import torch.nn as nn
from physics.derive_score import score_from_energy


def heavy_fp_residual(energy_net: nn.Module, x: torch.Tensor, atom_features: torch.Tensor,
                      edge_index: torch.Tensor, t: torch.Tensor, batch_idx: torch.Tensor,
                      scheduler: nn.Module, seed: int | None = None, sigma: float = 1e-4,
                      h_s: float = 1e-3, h_d: float = 5e-4) -> torch.Tensor:
    """per-atom weak Fokker–Planck residual estimator using finite-difference
    approximations for spatial and temporal derivatives"""
    if seed is not None:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
    else:
        gen = None
    _, dim = x.shape
    # β_t = g^2(t)
    beta_t = scheduler.variance(t)
    beta_t_exp = beta_t.unsqueeze(-1)
    # sample v ~ N(0, σ^2 I)
    v = sigma * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
    # x ± v
    x_plus = (x + v).requires_grad_(True)
    x_minus = (x - v).requires_grad_(True)

    # log p_t^θ(x ± v)
    logp_p = energy_net(x_plus, atom_features, edge_index, t, batch_idx, reduce = False)
    logp_m = energy_net(x_minus, atom_features, edge_index, t, batch_idx, reduce = False)
    # s^θ(x ± v) = ∇_x log p_t^θ(x ± v)
    score_plus = score_from_energy(logp_p, x_plus)
    score_minus = score_from_energy(logp_m, x_minus)
    # weak estimator of div_x s^θ(x,t)
    # (v/σ)^T [s(x+v) - s(x−v)] / (2σ)
    div_s = ((v / sigma) * (score_plus - score_minus) / (2.0 * sigma)).sum(dim=-1)
    # || s^θ(x+v,t) ||^2
    s_sq = (score_plus ** 2).sum(dim=-1)
    # drift f(x+v,t) = -½ β_t x
    drift_vec = -0.5 * beta_t_exp * x_plus
    # ⟨ f(x+v,t), s^θ(x+v,t) ⟩
    drift = (drift_vec * score_plus).sum(dim=-1)
    # div f(x+v,t) = -½ β_t d
    div_drift = -0.5 * beta_t * dim
    # finite-difference estimator of ∂_t log p_t^θ(x)
    x_ = x.detach().requires_grad_(True)
    t_p = torch.clamp(t + h_d, 0.0, 1.0)
    t_m = torch.clamp(t - h_s, 0.0, 1.0)

    logp_t_p = energy_net(x_, atom_features, edge_index, t_p, batch_idx, reduce = True)
    logp_t = energy_net(x_, atom_features, edge_index, t, batch_idx, reduce = True)
    logp_t_m = energy_net(x_, atom_features, edge_index, t_m, batch_idx, reduce = True)

    # numerator
    num = (h_s ** 2 * logp_t_p + (h_d ** 2 - h_s ** 2) * logp_t - h_d ** 2 * logp_t_m)
    # denominator
    den = h_s * h_d * (h_s + h_d)
    # ∂_t log p_t^θ(x)
    dlogp_dt_mol = num / den # per molecule
    dlogp_dt = dlogp_dt_mol[batch_idx] # per atom
    # final weak FP residual
    r = (0.5 * beta_t * (div_s + s_sq) - drift - div_drift - dlogp_dt)
    return r



def light_fp_residual(energy_net: nn.Module, x: torch.Tensor, atom_features: torch.Tensor,
                      edge_index: torch.Tensor, t: torch.Tensor, batch_idx: torch.Tensor,
                      scheduler: nn.Module, seed: int | None = None, h_s: float = 1e-3,
                      h_d: float = 5e-4, ) -> torch.Tensor:
    """lightweight weak fokker–planck residual estimator using Hutchinson’s
    trace estimator for the score divergence"""
    if seed is not None:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
    else:
        gen = None
    _, dim = x.shape
    beta_t = scheduler.variance(t)
    beta_t_exp = beta_t.unsqueeze(-1)
    x = x.requires_grad_(True)
    logp = energy_net(x, atom_features, edge_index, t, batch_idx, reduce = False)
    score = score_from_energy(logp, x)
    # hutchinson divergence estimator: div s(x)
    eps = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
    #eps = torch.randn_like(x)

    # g(x) = s(x) · eps
    #score_eps = (score * eps).sum()
    score_eps = (score * eps).sum(dim=-1).sum()
    # ∇_x (s(x) · eps)
    grad_score_eps = torch.autograd.grad(
        score_eps, x, create_graph=True, retain_graph=True
    )[0]
    # εᵀ ∇_x (s · ε)
    div_s = (grad_score_eps * eps).sum(dim=-1)
    # ||s(x)||^2
    s_sq = (score ** 2).sum(dim=-1)
    # drift f(x,t) = -½ β_t x
    drift_vec = -0.5 * beta_t_exp * x
    drift = (drift_vec * score).sum(dim=-1)
    # div f(x,t) = -½ β_t d
    div_drift = -0.5 * beta_t * dim
    # ∂_t log p_t(x) via finite differences
    x_ = x.detach().requires_grad_(True)
    t_p = torch.clamp(t + h_d, 0.0, 1.0)
    t_m = torch.clamp(t - h_s, 0.0, 1.0)

    logp_t_p = energy_net(x_, atom_features, edge_index, t_p, batch_idx, reduce = True)
    logp_t = energy_net(x_, atom_features, edge_index, t, batch_idx, reduce = True)
    logp_t_m = energy_net(x_, atom_features, edge_index, t_m, batch_idx, reduce = True)

    num = (h_s ** 2 * logp_t_p + (h_d ** 2 - h_s ** 2) * logp_t - h_d ** 2 * logp_t_m)
    den = h_s * h_d * (h_s + h_d)

    dlogp_dt_mol = num / den  # per molecule
    dlogp_dt = dlogp_dt_mol[batch_idx]  # per atom
    # final fp residual
    r = (0.5 * beta_t * (div_s + s_sq) - drift - div_drift - dlogp_dt)
    return r
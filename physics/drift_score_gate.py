import torch

def drift_score_gating(self, x0: torch.Tensor, score: torch.Tensor, noise: torch.Tensor,
                       t_atom: torch.Tensor, batch_idx: torch.Tensor,
                       k: float = 1.0, c: float = 1.0, t_max: float = 1.0) -> torch.Tensor:
    """
    compute binary mask for which atoms/molecules to apply fp regularization.
    arguments:
        x0: [n_atoms, 3] atom positions
        score: [n_atoms, 3] model score
        noise: [n_atoms, 3] sampled noise
        t_atom: [n_atoms] time for each atom
        batch_idx: [n_atoms] batch index / molecule id
        k: ddr threshold
        c: score norm factor threshold
        t_max: maximum time to apply fp
    returns:
        mask: [n_atoms] boolean mask, True = apply fp
    """
    variance = self.forward_vp.vs.get_variance(t_atom)
    beta_t = variance
    drift_vec = 0.5 * beta_t.unsqueeze(-1) * x0
    diff_vec = torch.sqrt(variance).unsqueeze(-1) * noise
    ddr = drift_vec.norm(dim=-1) / diff_vec.norm(dim=-1)
    score_norm = score.norm(dim=-1) ** 2
    score_mean = score_norm.mean()
    mask = (ddr > k) & (score_norm > c * score_mean) & (t_atom < t_max)
    #score_mean_per_mol = scatter_mean(score_norm, batch_idx, dim=0)
    #score_mean_atom = score_mean_per_mol[batch_idx]
    #mask = (ddr > k) & (score_norm > c * score_mean_atom) & (t_atom < t_max)
    #mask = torch.sigmoid(beta1 * (ddr - k)) * torch.sigmoid(beta2 * (score_norm - c * score_mean))
    return mask

import torch

def fp_gate(self, x0: torch.Tensor, t_atom: torch.Tensor, k: float, t_max: float, *args):
    """
    drift–diffusion dominance gate for fp regularization.
    returns per-atom boolean mask.
    """
    beta_t = self.forward_vp.vs.beta(t_atom)          # β(t)
    sqrt_beta = torch.sqrt(beta_t + 1e-8)             # √β(t)
    ddr = 0.5 * sqrt_beta * x0.norm(dim=-1)           # (n_atoms,)
    mask = (ddr > k) & (t_atom < t_max)
    return mask


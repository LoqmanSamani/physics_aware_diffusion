import torch


def fp_gate(x0: torch.Tensor, t_atom: torch.Tensor, variance_scheduler,
            k: float = 1.2, t_max: float = 0.2,
            distance_threshold: float = 1.3,
            snr_min: float = 0.5):
    """
    multi-criteria drift–diffusion dominance gate for FP regularization.
    arguments:
        x0: atom positions
        t_atom: diffusion time per atom
        variance_scheduler: provides beta(t) and alpha(t)
        k: ddr threshold
        t_max: maximum time for fp
        distance_threshold: minimum distance from origin
        snr_min: minimum signal-to-noise ratio
    """
    # core ddr criterion
    beta_t = variance_scheduler.beta(t_atom)
    sqrt_beta = torch.sqrt(beta_t + 1e-8)
    ddr = 0.5 * sqrt_beta * x0.norm(dim=-1)
    # distance criterion (focus on far-from-equilibrium atoms)
    distance = x0.norm(dim=-1)
    # snr criterion (avoid very noisy regime)
    alpha_t = variance_scheduler.alpha(t_atom)
    snr = alpha_t / (1 - alpha_t + 1e-8)
    gate = torch.sigmoid(5.0 * (ddr - k))
    mask = ((gate > 0.5) & (t_atom < t_max) & (distance > distance_threshold) & (snr > snr_min))
    #print("--------------------------------------------------------")
    #print("t: ", t_atom)
    #print("snr: ", snr)
    #print("distance: ", distance)
    #print("ddr: ", ddr)
    #print("--------------------------------------------------------")
    return mask

def fp_gate_(x0: torch.Tensor, t_atom: torch.Tensor, variance_scheduler, k: float, t_max: float, *args):
    """
    drift–diffusion dominance gate for fp regularization.
    returns per-atom boolean mask.
    """
    beta_t = variance_scheduler.beta(t_atom)          # β(t)
    sqrt_beta = torch.sqrt(beta_t + 1e-8)             # √β(t)
    ddr = 0.5 * sqrt_beta * x0.norm(dim=-1)           # (n_atoms,)
    mask = (ddr > k) & (t_atom < t_max)
    return mask
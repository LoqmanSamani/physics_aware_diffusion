import torch
from torch_scatter import scatter_mean

def fokker_planck_loss(r1: torch.Tensor, r2: torch.Tensor, batch_idx: torch.Tensor,* , alpha: float = 5e-4) -> torch.Tensor:
    """
    r1, r2: (num_atoms,)
    batch_idx: (num_atoms,)
    """
    # per-atom squared residuals
    atom_loss = 0.5 * (r1**2 + r2**2)  # (num_atoms,)
    # average per molecule
    mol_loss = scatter_mean(atom_loss, batch_idx, dim=0)  # (num_molecules,)
    return alpha * mol_loss.mean()


def snr_fokker_planck_loss(r1: torch.Tensor, r2: torch.Tensor, batch_idx: torch.Tensor, *,
                           variance: torch.Tensor, gamma: float = 5.0, alpha: float = 5e-4) -> torch.Tensor:
    """
    r1, r2, variance: (num_atoms,)
    batch_idx: (num_atoms,)
    """
    snr = (1.0 - variance) / variance.clamp(min=1e-8)
    gamma_t = torch.full_like(snr, gamma)
    weight = torch.minimum(snr, gamma_t)
    atom_loss = 0.5 * weight * (r1**2 + r2**2)  # (num_atoms,)
    mol_loss = scatter_mean(atom_loss, batch_idx, dim=0)  # (num_molecules,)

    return alpha * mol_loss.mean()


def mb_fokker_planck_loss(r1: torch.Tensor, r2: torch.Tensor, alpha: float = 5e-4) -> torch.Tensor:
    """specific loss used for mueller-brown experiment"""
    loss = 0.5 * (r1**2 + r2**2)  # (num_atoms,)
    return alpha * loss.mean()


def mb_snr_fokker_planck_loss(r1: torch.Tensor, r2: torch.Tensor, variance: torch.Tensor, gamma: float = 5.0, alpha: float = 5e-4) -> torch.Tensor:
    """specific snr loss used for mueller-brown experiment"""
    snr = (1.0 - variance) / variance.clamp(min=1e-8)
    gamma_t = torch.full_like(snr, gamma)
    weight = torch.minimum(snr, gamma_t)
    loss = 0.5 * weight * (r1**2 + r2**2)
    return alpha * loss.mean()
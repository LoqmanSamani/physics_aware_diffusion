import torch
from torch_scatter import scatter_mean


def distillation_loss(pred: torch.Tensor, target: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    atom_loss = ((pred - target) ** 2).sum(dim=-1)
    mol_loss = scatter_mean(atom_loss, batch_idx, dim=0)
    return mol_loss.mean()
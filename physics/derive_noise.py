import torch
import torch.nn as nn


def noise_from_energy(logp: torch.Tensor, xt: torch.Tensor, t: torch.Tensor, variance_scheduler: nn.Module, *args) -> torch.Tensor:
    """derive noise prediction from energy"""
    logp_sum = logp.sum()
    score = torch.autograd.grad(
        outputs=logp_sum,
        inputs=xt,
        create_graph=True,
        retain_graph=True
    )[0]
    std = variance_scheduler.get_std(t)
    while std.dim() < score.dim():
        std = std.unsqueeze(-1)
    return -std * score
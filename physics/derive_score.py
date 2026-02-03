import torch


def score_from_energy(logp: torch.Tensor, data: torch.Tensor, use_soft_clipping: bool = False) -> torch.Tensor:
    """derive score from energy, optionally apply soft clipping for numerical stability"""
    logp_sum = logp if logp.dim() == 0 else logp.sum()
    score = torch.autograd.grad(
        outputs=logp_sum,
        inputs=data,
        create_graph=True, # i thin this should be False when sampling for efficiency reasons
        retain_graph=True # i thin this should be False when sampling for efficiency reasons
    )[0]
    if use_soft_clipping:
        norm = torch.norm(score, dim=-1, keepdim=True)
        c = norm.mean().item() * 1.5
        score = score / (1 + norm / c)
    return score

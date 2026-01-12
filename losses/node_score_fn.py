import torch

def gaussian_score_fn(x_t, sigma_t, **kwargs):
    """∇ log N(0, σ²I) = -x / σ²"""
    if sigma_t.dim() == 1:
        sigma_t = sigma_t.unsqueeze(-1)
    return -x_t / (sigma_t.unsqueeze(-1) ** 2)
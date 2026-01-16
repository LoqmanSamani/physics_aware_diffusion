import torch

class Gaussian1NodeDataset(torch.utils.data.Dataset):
    """single-node isotropic Gaussian with analytic score"""
    def __init__(self, n_samples: int, sigma_min=0.1, sigma_max=2.0):
        self.n = n_samples
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        t = torch.rand(1)
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        x_t = (sigma_t * torch.randn(1, 3))
        return {
            "x_t": x_t,
            "atom_features": torch.ones(1, 1),
            "edge_index": torch.empty(2, 0, dtype=torch.long),
            "t": t,
            "sigma_t": sigma_t,
        }
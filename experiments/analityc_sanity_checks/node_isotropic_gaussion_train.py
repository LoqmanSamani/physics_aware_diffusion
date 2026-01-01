import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from score_nets.score_net import ScoreNet
from trainers.analytic_score_trainer import AnalyticScoreTrainer


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


def gaussian_score_fn(x_t, sigma_t, **kwargs):
    """∇ log N(0, σ²I) = -x / σ²"""
    if sigma_t.dim() == 1:
        sigma_t = sigma_t.unsqueeze(-1)
    return -x_t / (sigma_t.unsqueeze(-1) ** 2)



hidden_dim = 64
num_layers = 2
batch_size = 32
lr = 1e-3
epochs = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Gaussian1NodeDataset(n_samples = 1024)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

score_net = ScoreNet(
    atom_dim=1,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=0.0
)
print(sum(p.numel() for p in score_net.parameters()))

optimizer = torch.optim.Adam(score_net.parameters(), lr=lr)

trainer = AnalyticScoreTrainer(
    score_net=score_net,
    data_loader=data_loader,
    optimizer=optimizer,
    score_fn=gaussian_score_fn,
    epochs=epochs,
    device=device
)

trainer()








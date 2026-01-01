import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from score_nets.graph_score_net import GraphScoreNet
from trainers.node_score_trainer import AnalyticScoreTrainer
from pathlib import Path
from configs.load_config import load_config


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

project_root = Path(__file__).parent.parent.parent
config_path = project_root / "configs" / "node_isotropic_gaussian.yaml"
cfg = load_config(str(config_path))

device = cfg["experiment"]["device"]

dataset = Gaussian1NodeDataset(
    n_samples=cfg["dataset"]["n_samples"],
    sigma_min=cfg["dataset"]["sigma_min"],
    sigma_max=cfg["dataset"]["sigma_max"])
data_loader = DataLoader(
    dataset,
    batch_size=cfg["training"]["batch_size"],
    shuffle=True
)

score_net = GraphScoreNet(
    atom_dim=cfg["model"]["atom_dim"],
    hidden_dim=cfg["model"]["hidden_dim"],
    num_layers=cfg["model"]["num_layers"],
    dropout=cfg["model"]["dropout"]
)
#print(sum(p.numel() for p in score_net.parameters()))

optimizer = torch.optim.Adam(score_net.parameters(), lr=cfg["training"]["learning_rate"])

trainer = AnalyticScoreTrainer(
    score_net=score_net,
    data_loader=data_loader,
    optimizer=optimizer,
    score_fn=gaussian_score_fn,
    epochs=cfg["training"]["epochs"],
    device=device
)

losses = trainer()








import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.synthetic.node_dataset import Gaussian1NodeDataset
from score_nets.graph_score_net import GraphScoreNet
from trainers.node_score_trainer import AnalyticScoreTrainer
from pathlib import Path
from configs.load_config import load_config
from losses.node_score_fn import gaussian_score_fn


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








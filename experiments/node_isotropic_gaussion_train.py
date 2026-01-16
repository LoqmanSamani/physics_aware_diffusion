import torch
from torch.utils.data import DataLoader
from data.loaders.node_dataset import Gaussian1NodeDataset
from score_nets.graph_energy_net import GraphEnergyNet
from trainers.node_analytic_energy_trainer import AnalyticEnergyTrainer
from physics.derive_score import score_from_energy
from data.loaders.load_config import load_config
from losses.node_score_fn import gaussian_score_fn


config_path = "../configs/node_isotropic_gaussian.yaml"
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


energy_net = GraphEnergyNet(
    atom_dim=cfg["model"]["atom_dim"],
    hidden_dim=cfg["model"]["hidden_dim"],
    num_layers=cfg["model"]["num_layers"],
    edge_dim =cfg["model"]["edge_dim"],
    num_heads =cfg["model"]["num_heads"],
    dropout=cfg["model"]["dropout"]
)


optimizer = torch.optim.Adam(energy_net.parameters(), lr=cfg["training"]["learning_rate"])

trainer = AnalyticEnergyTrainer(
    energy_net=energy_net,
    score_from_energy = score_from_energy,
    data_loader=data_loader,
    optimizer=optimizer,
    score_fn=gaussian_score_fn,
    epochs=cfg["training"]["epochs"],
    device=device
)

if __name__ == "__main__":
    print(sum(p.numel() for p in energy_net.parameters()))
    losses = trainer()
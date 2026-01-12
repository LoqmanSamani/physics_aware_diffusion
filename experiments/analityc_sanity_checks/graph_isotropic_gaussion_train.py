import torch
from data.synthetic.graph_3d import GaussianPyGDataset
from torch_geometric.loader import DataLoader
from score_nets.graph_score_net import GraphScoreNet
from trainers.graph_score_trainer import AnalyticScoreTrainer
from losses.graph_score_fn import gaussian_score_fn
from pathlib import Path
from configs.load_config import load_config


project_root = Path(__file__).parent.parent.parent
config_path = project_root / "configs" / "graph_isotropic_gaussian.yaml"
cfg = load_config(str(config_path))

device = cfg["experiment"]["device"]

dataset = GaussianPyGDataset(
    n_samples=cfg["dataset"]["n_samples"],
    min_nodes=cfg["dataset"]["min_nodes"],
    max_nodes=cfg["dataset"]["max_nodes"])

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




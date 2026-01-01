import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from score_nets.graph_score_net import ScoreGraphNet
from trainers.graph_score_trainer import AnalyticScoreTrainer
from pathlib import Path
from configs.load_config import load_config



class GaussianPyGDataset(torch.utils.data.Dataset):
    """
    multi-node isotropic Gaussian dataset
    each graph can have variable number of nodes and edges.
    """
    def __init__(self, n_samples: int, min_nodes=1, max_nodes=5, sigma_min=0.1, sigma_max=2.0):
        self.n_samples = n_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        num_nodes = torch.randint(self.min_nodes, self.max_nodes + 1, (1,)).item()
        t = torch.rand(1)
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        x_t = sigma_t * torch.randn(num_nodes, 3)
        atom_features = torch.ones(num_nodes, 1)
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and torch.rand(1).item() < 0.5:
                    edges.append([i, j])
        if len(edges) == 0 and num_nodes > 1:
            edges = [[0, 1]]
        edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.empty(2, 0, dtype=torch.long)
        return Data(
            x=atom_features,
            pos=x_t,
            edge_index=edge_index,
            t=t,
            sigma_t=sigma_t
        )

def gaussian_score_fn(x_t, sigma_t, **kwargs):
    """∇ log N(0, σ²I) = -x / σ²"""
    return -x_t / (sigma_t.unsqueeze(-1) ** 2)


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

score_net = ScoreGraphNet(
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




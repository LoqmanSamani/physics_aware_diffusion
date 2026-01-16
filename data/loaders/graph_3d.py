import torch
from torch_geometric.data import Data

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
        t_ = torch.rand(1)
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t_
        x_t = sigma_t * torch.randn(num_nodes, 3)
        atom_features = torch.ones(num_nodes, 1)
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and torch.rand(1).item() < 0.5:
                    edges.append([i, j])
        if len(edges) == 0 and num_nodes > 1:
            edges = [[0, 1]]
        edge_index = torch.tensor(edges, dtype=torch.long).t_() if edges else torch.empty(2, 0, dtype=torch.long)
        t = torch.full((x_t.shape[0], ), t_.item())
        return Data(
            x=atom_features,
            pos=x_t,
            edge_index=edge_index,
            t=t,
            sigma_t=sigma_t
        )
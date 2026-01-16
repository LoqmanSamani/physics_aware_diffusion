import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List




class SyntheticMolecularDataset(torch.utils.data.Dataset):
    """synthetic molecular dataset"""
    def __init__(self, n_molecules: int = 100, n_atom_types: int = 3, min_atoms: int = 5, max_atoms: int = 10,
                 seed: int = 42, use_atom_types: bool = False) -> None:
        self.n_molecules = n_molecules
        self.n_atom_types = n_atom_types
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.use_atom_types = use_atom_types
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.molecules = self._generate_molecules()

    def _generate_molecules(self) -> List:
        molecules = []
        geometry_types = ['gaussian_blob', 'ring', 'linear_chain']
        for i in range(self.n_molecules):
            num_atoms = np.random.randint(self.min_atoms, self.max_atoms + 1)
            geometry = np.random.choice(geometry_types)

            if geometry == 'gaussian_blob':
                pos = self._create_gaussian_blob(num_atoms)
            elif geometry == 'ring':
                pos = self._create_ring(num_atoms)
            else:
                pos = self._create_linear_chain(num_atoms)

            pos = pos - pos.mean(dim=0, keepdim=True)

            # Always create one-hot encoded features based on n_atom_types
            if self.use_atom_types and self.n_atom_types > 1:
                atom_type_indices = torch.randint(0, self.n_atom_types, (num_atoms,))
            else:
                # Use a single atom type (e.g., type 0) for all atoms
                atom_type_indices = torch.zeros(num_atoms, dtype=torch.long)

            atom_features = torch.nn.functional.one_hot(
                atom_type_indices, num_classes=self.n_atom_types
            ).float()

            edge_index = self._create_edges(num_atoms, geometry)
            molecules.append({
                'pos': pos,
                'atom_features': atom_features,
                'edge_index': edge_index,
                'num_atoms': num_atoms,
                'geometry': geometry
            })
        return molecules


    def _create_gaussian_blob(self, num_atoms: int) -> torch.Tensor:
        """random 3D blob with unit variance"""
        return torch.randn(num_atoms, 3) * 0.8

    def _create_ring(self, num_atoms: int) -> torch.Tensor:
        """ring in xy-plane"""
        angles = torch.linspace(0, 2 * np.pi, num_atoms + 1)[:-1]
        radius = 1.0
        x = radius * torch.cos(angles)
        y = radius * torch.sin(angles)
        z = torch.zeros(num_atoms)
        return torch.stack([x, y, z], dim=1)

    def _create_linear_chain(self, num_atoms: int) -> torch.Tensor:
        """linear chain along x-axis"""
        x = torch.linspace(-1.0, 1.0, num_atoms)
        y = torch.zeros(num_atoms)
        z = torch.zeros(num_atoms)
        return torch.stack([x, y, z], dim=1)

    def _create_edges(self, num_atoms: int, geometry: str) -> torch.Tensor:
        """create edges based on geometry type"""
        edges = []
        if geometry == 'ring':
            for i in range(num_atoms):
                j = (i + 1) % num_atoms
                edges.append([i, j])
                edges.append([j, i])
        elif geometry == 'linear_chain':
            for i in range(num_atoms - 1):
                edges.append([i, i + 1])
                edges.append([i + 1, i])

        elif geometry == 'gaussian_blob':
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    edges.append([i, j])
                    edges.append([j, i])

        if len(edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t()

    def __len__(self):
        return self.n_molecules

    def __getitem__(self, idx):
        mol = self.molecules[idx]
        return Data(
            atom_features=mol['atom_features'].clone(),
            coords=mol['pos'].clone(),
            edge_index=mol['edge_index'].clone(),
            num_nodes=mol['num_atoms']  # <-- fix the warning
        )

    def get_molecule_info(self, idx):
        """Get metadata about a molecule"""
        mol = self.molecules[idx]
        return {
            'num_atoms': mol['num_atoms'],
            'geometry': mol['geometry'],
            'pos': mol['pos'].clone()
        }


def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = True, num_workers: int = 0):
    """create PyG dataloader with proper batching"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        follow_batch=['x']
    )
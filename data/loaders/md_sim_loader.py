import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
from contextlib import contextmanager
import os


@dataclass
class TrajectoryParams:
    """container for Langevin dynamics parameters"""
    num_steps: int
    dt: float
    temp: float
    friction: float
    mass: float
    kb: float
    t_eval: float

    def to_dict(self):
        return {'num_steps': self.num_steps, 'dt': self.dt, 'temp': self.temp,
                'friction': self.friction, 'mass': self.mass, 'kb': self.kb, 't_eval': self.t_eval}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_tensor(self, device='cpu'):
        """convert params to tensor for batching"""
        return torch.tensor([
            self.num_steps, self.dt, self.temp,
            self.friction, self.mass, self.kb, self.t_eval
        ], device=device, dtype=torch.float32)


class SyntheticMolecularDataset(torch.utils.data.Dataset):
    """
    synthetic molecular dataset with precomputed teacher trajectories
    this dataset stores:
        - initial molecular geometries
        - atom features and connectivity
        - precomputed teacher trajectories
        - trajectory parameters (can vary per molecule)
    """
    def __init__(self, n_molecules: int = 100, n_atom_types: int = 3, min_atoms: int = 5, max_atoms: int = 10,
                 seed: int = 42, use_atom_types: bool = False, include_trajectories: bool = False,
                 traj_params: Optional[TrajectoryParams] = None, vary_traj_params: bool = False,
                 teacher_model: Optional[torch.nn.Module] = None) -> None:
        """
        arguments:
            n_molecules: number of molecules to generate
            n_atom_types: number of different atom types
            min_atoms: minimum atoms per molecule
            max_atoms: maximum atoms per molecule
            seed: random seed
            use_atom_types: whether to use multiple atom types
            include_trajectories: whether to precompute trajectories
            traj_params: base trajectory parameters (required if include_trajectories=True)
            vary_traj_params: whether to randomly vary trajectory params per molecule
            teacher_model: teacher energy model for trajectory generation
        """
        self.n_molecules = n_molecules
        self.n_atom_types = n_atom_types
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.use_atom_types = use_atom_types
        self.include_trajectories = include_trajectories
        self.vary_traj_params = vary_traj_params
        self.base_traj_params = traj_params
        torch.manual_seed(seed)
        np.random.seed(seed)
        # generate base molecular structures
        self.molecules = self._generate_molecules()

        # precompute trajectories if requested
        if self.include_trajectories:
            if teacher_model is None:
                print("Warning: no teacher model provided. generating synthetic trajectories")
                self._generate_synthetic_trajectories()
            else:
                print("precomputing teacher trajectories...")
                self._compute_teacher_trajectories(teacher_model)

    def _generate_molecules(self) -> List[Dict]:
        """generate synthetic molecular structures"""
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
            # center the molecule
            pos = pos - pos.mean(dim=0, keepdim=True)
            # create atom features
            if self.use_atom_types and self.n_atom_types > 1:
                atom_type_indices = torch.randint(0, self.n_atom_types, (num_atoms,))
            else:
                atom_type_indices = torch.zeros(num_atoms, dtype=torch.long)
            atom_features = torch.nn.functional.one_hot(
                atom_type_indices, num_classes=self.n_atom_types
            ).float()
            edge_index = self._create_edges(num_atoms, geometry)
            # generate trajectory parameters for this molecule
            if self.include_trajectories:
                traj_params = self._generate_traj_params()
            else:
                traj_params = None
            molecules.append({
                'pos': pos,
                'atom_features': atom_features,
                'edge_index': edge_index,
                'num_atoms': num_atoms,
                'geometry': geometry,
                'traj_params': traj_params,
                'trajectory': None,  # will be filled later
                'initial_velocities': None
            })
        return molecules

    def _generate_traj_params(self) -> TrajectoryParams:
        """
        generate trajectory parameters, potentially with variation
        this allows training with diverse simulation conditions
        """
        if self.base_traj_params is None:
            # default parameters
            base = TrajectoryParams(
                num_steps=10, dt=0.001, temp=300.0, friction=1.0, mass=1.0, kb=1.380649e-23, t_eval=1e-5
            )
        else:
            base = self.base_traj_params

        if not self.vary_traj_params:
            return base

        # add controlled variation to parameters
        # temperature: ±20%
        temp = base.temp * (0.8 + 0.4 * np.random.rand())
        # friction: ±50%
        friction = base.friction * (0.5 + 1.0 * np.random.rand())
        # timestep: ±30%
        dt = base.dt * (0.7 + 0.6 * np.random.rand())
        # number of steps: ±50%
        num_steps = int(base.num_steps * (0.5 + 1.0 * np.random.rand()))
        num_steps = max(5, num_steps)  # ensure minimum steps
        return TrajectoryParams(
            num_steps=num_steps,
            dt=dt,
            temp=temp,
            friction=friction,
            mass=base.mass,  # keep mass fixed
            kb=base.kb,  # physical constant
            t_eval=base.t_eval
        )

    def _generate_synthetic_trajectories(self):
        """
        generate synthetic trajectories using simple physics simulation
        used when teacher model is not available (for testing)
        """
        print("generating synthetic trajectories (no teacher model)...")
        for mol in tqdm(self.molecules, desc="Synthetic trajectories"):
            params = mol['traj_params']
            num_atoms = mol['num_atoms']
            # simple harmonic potential trajectory (placeholder)
            trajectory = []
            x = mol['pos'].clone()
            # initialize velocities from Maxwell-Boltzmann
            kb_t = params.kb * params.temp
            v = torch.randn_like(x) * np.sqrt(kb_t / params.mass)
            mol['initial_velocities'] = v.clone()
            for _ in range(params.num_steps):
                # simple spring-like force toward origin
                force = -x * 0.1
                # langevin update
                alpha = np.exp(-params.friction * params.dt)
                sigma = np.sqrt(kb_t * (1 - alpha ** 2) / params.mass)
                v = alpha * v + (1 - alpha) / (params.friction * params.mass) * force
                v += sigma * torch.randn_like(v)
                x = x + params.dt * v
                trajectory.append(x.clone())
            mol['trajectory'] = torch.stack(trajectory)  # (num_steps, num_atoms, 3)

    def _compute_teacher_trajectories(self, teacher_model: torch.nn.Module):
        """
        precompute trajectories using teacher model
        this is the expensive operation done once during dataset creation
        """
        device = next(teacher_model.parameters()).device
        teacher_model.eval()
        with self.freeze_params(teacher_model):
            for mol in tqdm(self.molecules, desc="Computing teacher trajectories"):
                params = mol['traj_params']
                x0 = mol['pos'].to(device)
                atom_features = mol['atom_features'].to(device)
                edge_index = mol['edge_index'].to(device)
                # create batch index for single molecule
                batch_idx = torch.zeros(mol['num_atoms'], dtype=torch.long, device=device)
                # initialize velocities
                kb_t = params.kb * params.temp
                v = torch.randn_like(x0) * np.sqrt(kb_t / params.mass)
                mol['initial_velocities'] = v.cpu().clone()
                # run langevin dynamics with teacher forces
                trajectory = []
                x = x0.clone()
                alpha = np.exp(-params.friction * params.dt)
                sigma = np.sqrt(kb_t * (1 - alpha ** 2) / params.mass)
                for _ in range(params.num_steps):
                    # get forces from teacher model
                    x_input = x.clone().requires_grad_(True)
                    t_atom = torch.full((x.shape[0],), params.t_eval, device=device)
                    logp = teacher_model(x_input, atom_features, edge_index, t_atom, batch_idx)
                    score = torch.autograd.grad(logp.sum(), x_input)[0]
                    forces = score * kb_t
                    # langevin update
                    v = (alpha * v + (1 - alpha) / (params.friction * params.mass) * forces + sigma * torch.randn_like(v))
                    x = x + params.dt * v
                    trajectory.append(x.detach().cpu().clone())
                mol['trajectory'] = torch.stack(trajectory)  # (num_steps, num_atoms, 3)

    @contextmanager
    def freeze_params(self, module):
        old_requires_grad = []
        for p in module.parameters():
            old_requires_grad.append(p.requires_grad)
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p, rg in zip(module.parameters(), old_requires_grad):
                p.requires_grad_(rg)

    def _create_gaussian_blob(self, num_atoms: int) -> torch.Tensor:
        """random 3d blob with unit variance"""
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
        """
        return molecular data with optional precomputed trajectory
        note: trajectories are stored as lists to handle variable atom counts
        they will be properly batched in the custom collate function.
        """
        mol = self.molecules[idx]
        data = Data(
            atom_features=mol['atom_features'].clone(),
            coords=mol['pos'].clone(),
            edge_index=mol['edge_index'].clone(),
            num_nodes=mol['num_atoms']
        )
        # add trajectory data if available
        if self.include_trajectories and mol['trajectory'] is not None:
            data.has_trajectory = True
            data.mol_idx = idx
            params = mol['traj_params']
            data.traj_params = params.to_tensor()
        else:
            data.has_trajectory = False
            data.mol_idx = idx
        return data

    def save_to_disk(self, path: str, filename: str):
        """save dataset to disk for reuse"""
        filepath = os.path.join(path, filename)
        os.makedirs(path, exist_ok=True)
        torch.save({
            'molecules': self.molecules,
            'config': {
                'n_molecules': self.n_molecules,
                'n_atom_types': self.n_atom_types,
                'min_atoms': self.min_atoms,
                'max_atoms': self.max_atoms,
                'use_atom_types': self.use_atom_types,
                'include_trajectories': self.include_trajectories,
                'vary_traj_params': self.vary_traj_params,
                'base_traj_params': self.base_traj_params.to_dict() if self.base_traj_params else None
            }
        }, filepath)
        print(f"dataset saved to {filepath}")

    @classmethod
    def load_from_disk(cls, path: str, filename: str):
        """load precomputed dataset from disk"""
        filepath = os.path.join(path, filename)
        data = torch.load(filepath,  weights_only=False)
        config = data['config']

        # create dataset without regenerating
        dataset = cls(
            n_molecules=config['n_molecules'],
            n_atom_types=config['n_atom_types'],
            min_atoms=config['min_atoms'],
            max_atoms=config['max_atoms'],
            use_atom_types=config['use_atom_types'],
            include_trajectories=False
        )
        # load precomputed data
        dataset.molecules = data['molecules']
        dataset.include_trajectories = config['include_trajectories']
        dataset.vary_traj_params = config['vary_traj_params']
        if config['base_traj_params']:
            dataset.base_traj_params = TrajectoryParams.from_dict(config['base_traj_params'])

        print(f"Dataset loaded from {path}")
        return dataset

    def get_trajectories_for_batch(self, mol_indices: List[int]) -> Dict:
        """
        retrieve trajectories for a batch of molecule indices.
        returns:
            dictionary containing:
            - trajectories: List of trajectory tensors (each: num_steps, num_atoms, 3)
            - params: List of TrajectoryParams objects
            - num_atoms: List of atom counts
        """
        trajectories = []
        params_list = []
        num_atoms_list = []
        for idx in mol_indices:
            mol = self.molecules[idx]
            if mol['trajectory'] is not None:
                trajectories.append(mol['trajectory'])
                params_list.append(mol['traj_params'])
                num_atoms_list.append(mol['num_atoms'])
            else:
                raise ValueError(f"Molecule {idx} does not have a trajectory")
        return {
            'trajectories': trajectories,
            'params': params_list,
            'num_atoms': num_atoms_list
        }

    def get_molecule_info(self, idx):
        """get metadata about a molecule"""
        mol = self.molecules[idx]
        info = {
            'num_atoms': mol['num_atoms'],
            'geometry': mol['geometry'],
            'pos': mol['pos'].clone()
        }

        if self.include_trajectories:
            info['traj_params'] = mol['traj_params']
            if mol['trajectory'] is not None:
                info['trajectory_shape'] = mol['trajectory'].shape

        return info


def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = True):
    """create PyG dataloader with custom collate for trajectory handling"""

    def custom_collate(data_list):
        from torch_geometric.data import Batch
        batch = Batch.from_data_list(data_list)
        if all(hasattr(d, "mol_idx") for d in data_list):
            batch.mol_indices = [d.mol_idx for d in data_list]
        #if hasattr(data_list[0], "has_trajectory") and data_list[0].has_trajectory:
        #    batch.mol_indices = [data.mol_idx for data in data_list]
        return batch
    # create loader with custom collate
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate
    )
    # store dataset reference in loader for trajectory access
    loader.dataset_ref = dataset
    return loader
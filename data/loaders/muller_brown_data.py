import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional


class MullerBrownPotential:
    """Müller-Brown potential for 2d toy system"""
    def __init__(self, kbt: float = 23.0):
        self.kbt = kbt

    def energy(self, x: torch.Tensor):
        """compute Müller-Brown potential energy"""
        x_pos = x[..., 0]
        y_pos = x[..., 1]

        term1 = -200 * torch.exp(-(x_pos - 1) ** 2 - 10 * y_pos ** 2)
        term2 = -100 * torch.exp(-x_pos ** 2 - 10 * (y_pos - 0.5) ** 2)
        term3 = -170 * torch.exp(
            -6.5 * (0.5 + x_pos) ** 2 +
            11 * (x_pos + 0.5) * (y_pos - 1.5) -
            6.5 * (y_pos - 1.5) ** 2
        )
        term4 = 15 * torch.exp(
            0.7 * (1 + x_pos) ** 2 +
            0.6 * (x_pos + 1) * (y_pos - 1) +
            0.7 * (y_pos - 1) ** 2
        )
        return term1 + term2 + term3 + term4

    def force(self, x: torch.Tensor):
        """
        compute forces as -∇U(x)
        returns gradient with same shape as x
        """
        x_tensor = x.detach().requires_grad_(True)
        energy = self.energy(x_tensor)
        grad = torch.autograd.grad(
            energy.sum(),
            x_tensor,
            create_graph=False
        )[0]
        return -grad


def langevin_sampling(
        potential: MullerBrownPotential,
        n_steps: int = 50000,
        dt: float = 5e-3,
        mass: float = 0.5,
        kbt: float = 23.0,
        save_every: int = 50,
        initial_pos: torch.Tensor = None,
        log_freq: int = 1000,
        device: str = 'cuda'
):
    """generate samples using Langevin dynamics"""
    if initial_pos is None:
        x = torch.randn(2, device=device) * 0.5
    else:
        x = torch.tensor(initial_pos, device=device, dtype=torch.float32)
    v = torch.randn(2, device=device) * np.sqrt(kbt / mass)
    # friction coefficient
    gamma = 1.0
    # pre-compute constants
    noise_scale = np.sqrt(2 * gamma * kbt * dt / mass)

    samples = []
    print(f"Running Langevin dynamics for {n_steps} steps...")
    for step in range(n_steps):
        force = potential.force(x.unsqueeze(0)).squeeze(0)
        # V=velocity update (Langevin integrator)
        noise = torch.randn_like(v) * noise_scale
        v = v - gamma * v * dt + (force / mass) * dt + noise
        # position update
        x = x + v * dt
        if step % save_every == 0:
            samples.append(x.cpu().clone())
        if (step + 1) % log_freq == 0:
            print(f"Step {step + 1}/{n_steps}")
    samples = torch.stack(samples)
    print(f"Generated {len(samples)} samples")
    return samples


class MolecularDataset(Dataset):
    """Dataset for molecular configurations"""
    def __init__(self, samples: torch.Tensor):
        """samples: tensor of shape (n_samples, 2) for 2d positions"""
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def generate_muller_brown_data(
        n_samples: int = 100000,
        init_pos: Optional[None] = None,
        dt: float = 0.005,
        mass: float = 0.5,
        save_every: int = 50,
        kbt: float = 23.0,
        log_freq: int = 1000,
        device: str = 'cuda'
):
    """
    generate training data from Müller-Brown potential
    returns:
        dataset: MolecularDataset object
        potential: MullerBrownPotential object (for evaluation)
    """
    n_steps = n_samples * save_every # specifies number of samples
    potential = MullerBrownPotential(kbt=kbt)
    # generate samples via Langevin dynamics
    samples = langevin_sampling(
        potential = potential,
        n_steps = n_steps,
        initial_pos = init_pos,
        dt = dt,
        mass = mass,
        kbt = kbt,
        save_every = save_every,
        log_freq = log_freq,
        device = device
    )
    # normalize samples
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
    samples_normalized = (samples - mean) / std

    dataset = MolecularDataset(samples_normalized)
    dataset.mean = mean
    dataset.std = std

    return dataset, potential
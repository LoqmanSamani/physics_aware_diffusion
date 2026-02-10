import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class MullerBrownPotential:
    """Müller-Brown potential for 2D toy system"""

    def __init__(self, kBT=23.0):
        self.kBT = kBT

    def energy(self, x):
        """
        Compute Müller-Brown potential energy
        x: tensor of shape (..., 2) where x[..., 0] is x-coord, x[..., 1] is y-coord
        """
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

    def force(self, x):
        """
        Compute forces as -∇U(x)
        Returns gradient with same shape as x
        """
        x_tensor = x.detach().requires_grad_(True)
        energy = self.energy(x_tensor)

        # Compute gradient
        grad = torch.autograd.grad(
            energy.sum(),
            x_tensor,
            create_graph=False
        )[0]

        return -grad  # Force is negative gradient


def langevin_sampling(
        potential,
        n_steps=5_000_000,
        dt=0.005,
        mass=0.5,
        kBT=23.0,
        save_every=50,
        initial_pos=None,
        device='cpu'
):
    """
    Generate samples using Langevin dynamics

    Args:
        potential: Potential energy object with .force() method
        n_steps: Total simulation steps
        dt: Time step
        mass: Particle mass (scalar, assumes same for both dimensions)
        kBT: Temperature in energy units
        save_every: Save every N-th sample
        initial_pos: Initial position [x, y] or None for random
        device: 'cpu' or 'cuda'

    Returns:
        samples: tensor of shape (n_samples, 2)
    """
    # Initialize
    if initial_pos is None:
        x = torch.randn(2, device=device) * 0.5
    else:
        x = torch.tensor(initial_pos, device=device, dtype=torch.float32)

    v = torch.randn(2, device=device) * np.sqrt(kBT / mass)

    # Friction coefficient
    gamma = 1.0

    # Pre-compute constants
    noise_scale = np.sqrt(2 * gamma * kBT * dt / mass)

    # Storage
    samples = []

    print(f"Running Langevin dynamics for {n_steps} steps...")
    for step in range(n_steps):
        # Compute force
        force = potential.force(x.unsqueeze(0)).squeeze(0)

        # Velocity update (Langevin integrator)
        noise = torch.randn_like(v) * noise_scale
        v = v - gamma * v * dt + (force / mass) * dt + noise

        # Position update
        x = x + v * dt

        # Save sample
        if step % save_every == 0:
            samples.append(x.cpu().clone())

        if (step + 1) % 100000 == 0:
            print(f"Step {step + 1}/{n_steps}")

    samples = torch.stack(samples)
    print(f"Generated {len(samples)} samples")

    return samples


class MolecularDataset(Dataset):
    """PyTorch Dataset for molecular configurations"""

    def __init__(self, samples):
        """
        Args:
            samples: tensor of shape (n_samples, 2) for 2D positions
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# Generate dataset
def generate_muller_brown_data(
        n_samples=100_000,
        n_steps=5_000_000,
        save_every=50,
        kBT=23.0,
        device='cpu'
):
    """
    Generate training data from Müller-Brown potential

    Returns:
        dataset: MolecularDataset object
        potential: MullerBrownPotential object (for evaluation)
    """
    potential = MullerBrownPotential(kBT=kBT)

    # Generate samples via Langevin dynamics
    samples = langevin_sampling(
        potential=potential,
        n_steps=n_steps,
        dt=0.005,
        mass=0.5,
        kBT=kBT,
        save_every=save_every,
        device=device
    )

    # Normalize to zero mean and unit variance (important for diffusion models)
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
    samples_normalized = (samples - mean) / std

    dataset = MolecularDataset(samples_normalized)

    # Store normalization stats for later use
    dataset.mean = mean
    dataset.std = std

    return dataset, potential


# Example usage
if __name__ == "__main__":
    # Generate data
    dataset, potential = generate_muller_brown_data(
        n_samples=100_000,
        n_steps=5_000_000,
        device='cpu'
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    # Visualize samples
    samples = dataset.samples[:10000]
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Normalized Müller-Brown Samples')
    plt.savefig('muller_brown_samples.png', dpi=150)
    plt.close()

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0].shape}")
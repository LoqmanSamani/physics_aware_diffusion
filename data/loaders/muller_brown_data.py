import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from typing import Optional
import matplotlib.pyplot as plt



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



class MullerBrownEvaluator:
    """evaluate consistency between iid sampling and simulation"""
    def __init__(self, potential, dataset_mean, dataset_std, kbt = 23.0):
        self.potential = potential
        self.mean = dataset_mean
        self.std = dataset_std
        self.kbt = kbt

    def unnormalize(self, x_norm):
        """convert normalized samples back to original space"""
        return x_norm * self.std + self.mean

    def normalize(self, x):
        """convert original samples to normalized space"""
        return (x - self.mean) / self.std

    #@torch.no_grad()
    def iid_sampling(self, sampler: nn.Module, n_samples = 10000, dt: float = 1e-3, device = 'cuda',
                     store_trajectory: bool = False, normalize_output: bool = False):
        """generate independent samples via diffusion denoising
        args
            sampler: sampler of trained diffusion model
            n_samples: Number of samples to generate
        returns:
            samples: tensor of shape (n_samples, 2) in original space
        """
        samples_norm = sampler.iid_sampler(
            n_samples = n_samples,
            dt = dt,
            device = device,
            store_trajectory = store_trajectory,
            normalize_output = normalize_output
        )
        samples = self.unnormalize(samples_norm)
        return samples

    #@torch.no_grad()
    def simulation_sampling(self, model: nn.Module, n_steps: int = 30000, n_parallel: int = 100,
                            dt: float = 0.005, mass: float = 0.5, gamma: float = 1.0, t_eval: float = 1e-5,
                            save_every: int = 100, device: str = 'cuda', log_freq: int = 1000):
        """
        generate samples via Langevin simulation using learned score at t = 0
        this evaluates the learned energy landscape by using:
        force = -kbt * score = (convert model output to score: model(x, t = 0))
        args:
            model: trained neural network with energy output
            n_steps: number of simulation steps
            n_parallel: number of parallel trajectories
        returns:
            samples: tensor of shape (total_samples, 2) in original space
        """
        # initialize from random positions
        x = torch.randn(n_parallel, 2, device=device)
        v = torch.randn(n_parallel, 2, device=device) * np.sqrt(self.kbt / mass)
        noise_scale = np.sqrt(2 * gamma * self.kbt * dt / mass)
        t_eval = torch.full((n_parallel,), t_eval, device = device)
        samples = []
        print(f"Running simulation for {n_steps} steps...")
        for step in range(n_steps):
            # get energy (logp) from the model and convert it to score at t = e_eval (the model  will do it internally)
            # (we will train and sample the energy model with t_min > 0 for training stability reasons)
            score = model.score(x, t_eval)
            force = -self.kbt * score / (self.std.to(device) ** 2)
            # Langevin update
            noise = torch.randn_like(v) * noise_scale
            v = v - gamma * v * dt + (force / mass) * dt + noise
            x = x + v * dt
            if step % save_every == 0 and step > n_steps // 2:
                samples.append(x.cpu().clone())
            if (step + 1) % log_freq == 0:
                print(f"Simulation step {step + 1}/{n_steps}")
        samples = torch.cat(samples, dim = 0)
        samples = self.unnormalize(samples)
        return samples

    def compute_free_energy_2d(self, samples: torch.Tensor, bins: int = 64, extent = None):
        """
        compute 2d free energy surface from samples
        args:
            samples: tensor of shape (n_samples, 2)
            bins: Number of bins for histogram
            extent: [[xmin, xmax], [ymin, ymax]] or None for auto
        returns:
            free_energy: 2d array
            x_edges: bin edges for x
            y_edges: bin edges for y
        """
        samples_np = samples.cpu().numpy()
        if extent is None:
            extent = [
                [samples_np[:, 0].min(), samples_np[:, 0].max()],
                [samples_np[:, 1].min(), samples_np[:, 1].max()]
            ]
        # compute 2d histogram
        hist, x_edges, y_edges = np.histogram2d(
            samples_np[:, 0],
            samples_np[:, 1],
            bins=bins,
            range=extent,
            density=True
        )
        # convert to free energy: F = -kBT * log(p)
        # add small constant to avoid log(0)
        hist = hist + 1e-10
        free_energy = -self.kbt * np.log(hist)
        # shift so minimum is zero
        free_energy = free_energy - free_energy.min()
        return free_energy, x_edges, y_edges

    def visualize_comparison(self, reference_samples, iid_samples, sim_samples, bins: int = 64, title_prefix="", save_path = None):
        """
        create visualization comparing distributions
        args:
            reference_samples: ground truth samples from true potential
            iid_samples: samples from iid diffusion sampling
            sim_samples: samples from simulation using learned energy (logp)
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        datasets = [
            ("Reference", reference_samples),
            ("IID Sampling", iid_samples),
            ("Simulation (t = 0)", sim_samples)
        ]
        all_samples = torch.cat([reference_samples, iid_samples, sim_samples])
        extent = [
            [all_samples[:, 0].min().item(), all_samples[:, 0].max().item()],
            [all_samples[:, 1].min().item(), all_samples[:, 1].max().item()]
        ]
        for idx, (name, samples) in enumerate(datasets):
            ax_scatter = axes[0, idx]
            samples_np = samples.cpu().numpy()[:5000]
            ax_scatter.scatter(
                samples_np[:, 0],
                samples_np[:, 1],
                alpha=0.3,
                s=1
            )
            ax_scatter.set_xlabel('x')
            ax_scatter.set_ylabel('y')
            ax_scatter.set_title(f'{name} (Scatter)')
            ax_energy = axes[1, idx]
            free_energy, x_edges, y_edges = self.compute_free_energy_2d(
                samples,
                bins = bins,
                extent = extent
            )
            im = ax_energy.imshow(
                free_energy.T,
                origin = 'lower',
                extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                aspect = 'auto',
                cmap = 'viridis',
                vmax = 10
            )
            ax_energy.set_xlabel('x')
            ax_energy.set_ylabel('y')
            ax_energy.set_title(f'{name} (Free Energy)')
            plt.colorbar(im, ax=ax_energy, label='Energy / kBT')

        plt.suptitle(f'{title_prefix}Consistency Comparison', fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
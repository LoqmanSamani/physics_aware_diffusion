import torch
import torch.nn as nn
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from ..data.loaders.muller_brown_data import langevin_sampling


class DiffusionEvaluator:
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

    @torch.no_grad()
    def iid_sampling(self, sampler: nn.Module, n_samples = 10000, device = 'cuda'):
        """generate independent samples via diffusion denoising
        args:
            sampler: sampler of trained diffusion model
            n_samples: Number of samples to generate
        returns:
            samples: tensor of shape (n_samples, 2) in original space
        """
        samples_norm = sampler(n_samples, device = device)
        samples = self.unnormalize(samples_norm)
        return samples

    # @torch.no_grad()
    def simulation_sampling(self, model: nn.Module, score_fn: Callable, n_steps: int = 30000, n_parallel: int = 100,
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
            # get energy (logp) from the model and convert it to score at t = e_eval
            # (we will train and sample the energy model with t_min > 0 for training stability reasons)
            x = x.detach()
            x.requires_grad_(True)
            logp = model(x, t_eval)
            score = score_fn(logp, x)
            # convert score to force: F = -kBT * ∇log p = -kBT * score
            # score is gradient of log density in normalized space
            # need to account for normalization when converting to forces
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


# example evaluation
def evaluate_model(model, dataset, potential, device='cpu'):
    """
    Complete evaluation pipeline

    Args:
        model: Trained diffusion model
        dataset: MolecularDataset with normalization stats
        potential: True potential for reference
    """
    evaluator = DiffusionEvaluator(potential=potential, dataset_mean=dataset.mean, dataset_std=dataset.std, kbt=23.0)

    print("Generating reference samples from true potential...")
    reference_samples = langevin_sampling(
        potential=potential,
        n_steps=500000,
        save_every=5,
        device=device
    )

    print("\nGenerating IID samples from diffusion model...")
    iid_samples = evaluator.iid_sampling(sampler=model, n_samples=10000, device=device)

    print("\nGenerating simulation samples using learned score...")
    sim_samples = evaluator.simulation_sampling(
        model=model,
        n_steps=30000,
        n_parallel=100,
        device=device
    )

    print("\nVisualizing results...")
    evaluator.visualize_comparison(
        reference_samples=reference_samples,
        iid_samples=iid_samples,
        sim_samples=sim_samples,
        title_prefix=f"{model.__class__.__name__} - ",
        save_path=f"{model.__class__.__name__}_comparison.png"
    )

    return reference_samples, iid_samples, sim_samples
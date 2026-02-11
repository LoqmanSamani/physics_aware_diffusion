import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
import os


class MullerBrownSampler(nn.Module):
    """
    Sampler for energy-based diffusion models
    Handles both iid sampling (denoising) and simulation
    """
    def __init__(self, model: nn.Module, rev: nn.Module, eps_time: float = 1e-5, store_path: str = "./mb_samples", kbt: float = 23.0):
        """
        args:
            model: MullerBrownNet
            rev: reverse vp-sde diffusion
            kbt: Temperature for force computation
        """
        super().__init__()
        self.model = model
        self.rev = rev
        self.eps_time = eps_time
        self.store_path = store_path
        self.kbt = kbt


    #@torch.no_grad()
    def iid_sampler(self, n_samples, dt: float = 1e-3, device: str = 'cuda', store_trajectory: bool = False, normalize_output: bool = False):
        """
        generate samples via denoising (iid sampling)

        Uses the reverse-time SDE:
        dx = [f(x,t) - g²(t) * score(x,t)] dt + g(t) dw

        Args:
            n_samples: number of samples to generate
            device: 'cpu' or 'cuda'
            store_trajectory: If True, store full denoising trajectory

        Returns:
            samples: dict of "x0":(n_samples, input_dim) and trajectory

        """
        self.model.eval()
        x = torch.randn(n_samples, self.model.input_dim, device = device)
        results = {"x0": None, "trajectory": []}
        if store_trajectory:
            results["trajectory"].append(x.clone())
        num_steps = int((1.0 - self.eps_time) / dt)
        t_schedule = torch.linspace(1.0, self.eps_time, num_steps + 1)
        dt = torch.tensor(dt, device=x.device, dtype=x.dtype)
        iterator = tqdm(range(num_steps), desc="Sampling")
        with self.freeze_params(self.energy_net):
            for step in iterator:
                t_current = float(t_schedule[step])
                t_batch = t_current * torch.ones(n_samples, device=device)
                score = self.model.score(x, t_batch)
                if step == 0:
                    x = self.rev(x, score, t_batch, dt, last_step=True)
                else:
                    x = self.rev(x, score, t_batch, dt)
                x = x.detach()
                if store_trajectory:
                    results['trajectory'].append(x.clone())
        if normalize_output:
            results['x0'] = torch.clamp(x, min=-1.0, max=1.0)
            if store_trajectory:
                results['trajectory'] = torch.stack(results['trajectory'], dim=0)
                results['trajectory'] = torch.clamp(results['trajectory'], min=-1.0, max=1.0)
        else:
            results['x0'] = x
            if store_trajectory:
                results['trajectory'] = torch.stack(results['trajectory'], dim=0)
        self.save_results(results, f"sample_{n_samples}_iid.pth")
        return results

    #@torch.no_grad()
    def langevin_simulator(self, n_parallel: int = 100, n_steps: int = 30000, dt: float = 0.005, mass: float = 0.5,
                           gamma: float = 1.0, device: str ='cuda', save_every: int = 100, burn_in_ratio: float = 0.5):
        """
        generate samples via Langevin simulation using learned forces at t -> 0
        this evaluates p₀(x) = exp(-U(x)/kBT) where U is the learned energy
        args:
            n_parallel: number of parallel trajectories
            n_steps: total simulation steps
            dt: Time step
            mass: particle mass
            gamma: friction coefficient
            device: 'cpu' or 'cuda'
            save_every: save every n-th sample
            burn_in_ratio: fraction of steps to discard as burn-in
        returns:
            samples: (n_saved_samples, input_dim)
        """
        self.model.eval()
        x = torch.randn(n_parallel, self.model.input_dim, device=device)
        v = torch.randn(n_parallel, self.model.input_dim, device=device) * np.sqrt(self.kbt / mass)
        noise_scale = np.sqrt(2 * gamma * self.kbt * dt / mass)
        samples = []
        burn_in_steps = int(n_steps * burn_in_ratio)
        with self.freeze_params(self.model):
            for step in tqdm(range(n_steps), desc="Simulating"):
                # get force at t -> 0
                force = self.model.force_at_t0(x = x, kbt = self.kbt, t_eval =  self.eps_time)
                noise = torch.randn_like(v) * noise_scale
                v = v - gamma * v * dt + (force / mass) * dt + noise
                x = x + v * dt
                # save samples after burn-in
                if step >= burn_in_steps and step % save_every == 0:
                    samples.append(x.cpu().clone())
        samples = torch.cat(samples, dim=0)
        self.save_results(samples, f"sample_{n_parallel}_langevin.pth")
        return samples

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

    def save_results(self, results, file_name) -> None:
        filepath = os.path.join(self.store_path, file_name)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(results, filepath)
        print(f"Results saved to {filepath}")


# Example usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    model = ImprovedEnergyNet(
        input_dim=2,
        time_embed_dim=64,
        hidden_dim=128,
        num_blocks=3
    ).to(device)

    # Create sampler
    sampler = MullerBrownSampler(
        model=model,
        beta_min=0.1,
        beta_max=20.0,
        n_timesteps=1000,
        kBT=23.0
    )

    # Test iid sampling
    print("Testing iid sampling...")
    iid_samples = sampler.iid_sampler(n_samples=1000, device=device)
    print(f"IID samples shape: {iid_samples.shape}")

    # Test simulation
    print("\nTesting simulation...")
    sim_samples = sampler.langevin_simulator(n_parallel=10, n_steps=1000, device=device)
    print(f"Simulation samples shape: {sim_samples.shape}")
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
import os




class MBSampler(nn.Module):
    """
    sampler for mueller-Brown experiment
    handles both iid sampling (denoising) and simulation
    """
    def __init__(self, model: nn.Module, rev: nn.Module, dataset_mean: torch.Tensor,
                 dataset_std: torch.Tensor, eps_time: float = 1e-5,
                 store_path: str = "./mb_samples", kbt: float = 23.0):
        """
        args:
            model: MullerBrownNet
            rev: reverse vp-sde diffusion
            dataset_mean: Mean from training data (for un-normalization)
            dataset_std: Std from training data (for un-normalization)
            eps_time: Small time value for evaluation at t~0
            store_path: Path to save results
            kbt: Temperature for force computation
        """
        super().__init__()
        self.model = model
        self.rev = rev
        self.eps_time = eps_time
        self.store_path = store_path
        self.kbt = kbt
        self.register_buffer('mean', dataset_mean)
        self.register_buffer('std', dataset_std)


    def normalize(self, x):
        """convert from original space to normalized space"""
        return (x - self.mean) / self.std

    def unnormalize(self, x_norm):
        """convert from normalized space to original space"""
        return x_norm * self.std + self.mean

    def iid_sampler(self, n_samples, dt: float = 1e-3, device: str = 'cuda',
                    store_trajectory: bool = False, return_original_space: bool = True):
        """
        generate samples via denoising (iid sampling)
        uses the reverse-time sde:
        dx = [f(x,t) - g²(t) * score(x,t)] dt + g(t) dw
        args:
            n_samples: number of samples to generate
            dt: time step
            device: 'cpu' or 'cuda'
            store_trajectory: If True, store full denoising trajectory
            return_original_space: If True, unnormalize output (recommended!)
        returns:
            samples: (n_samples, input_dim) in original space (if return_original_space=True)
        """
        self.model.to(device)
        self.model.eval()
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        x = torch.randn(n_samples, self.model.input_dim, device=device)
        results = {"x0": None, "trajectory": []}
        if store_trajectory:
            results["trajectory"].append(x.clone())
        num_steps = int((1.0 - self.eps_time) / dt)
        t_schedule = torch.linspace(1.0, self.eps_time, num_steps + 1)
        dt_tensor = torch.tensor(dt, device=x.device, dtype=x.dtype)
        iterator = tqdm(range(num_steps), desc="IID Sampling")

        with self.freeze_params(self.model):
            for step in iterator:
                t_current = float(t_schedule[step])
                t_batch = t_current * torch.ones(n_samples, device=device)
                score = self.model.score(x, t_batch)
                if step == num_steps - 1:
                    x = self.rev(x, score, t_batch, dt_tensor, last_step=True)
                else:
                    x = self.rev(x, score, t_batch, dt_tensor)
                x = x.detach()
                if store_trajectory:
                    results['trajectory'].append(x.clone())

        if return_original_space:
            x_original = self.unnormalize(x)
            results['x0'] = x_original
            if store_trajectory:
                traj_stack = torch.stack(results['trajectory'], dim=0)
                results['trajectory'] = self.unnormalize(traj_stack.view(-1, self.model.input_dim)).view(traj_stack.shape)
        else:
            results['x0'] = x
            if store_trajectory:
                results['trajectory'] = torch.stack(results['trajectory'], dim=0)
        self.save_results(results, f"sample_{n_samples}_iid.pth")

        return results['x0']

    def langevin_simulator(self, n_parallel: int = 100, n_steps: int = 30000,
                           dt: float = 0.005, mass: float = 0.5, gamma: float = 1.0,
                           device: str = 'cuda', save_every: int = 100,
                           burn_in_ratio: float = 0.5, return_original_space: bool = True):
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
            return_original_space: If True, unnormalize output (recommended!)
        returns:
            samples: (n_saved_samples, input_dim) in original space (if return_original_space=True)
        """
        self.model.to(device)
        self.model.eval()
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        x = torch.randn(n_parallel, self.model.input_dim, device=device)
        v = torch.randn(n_parallel, self.model.input_dim, device=device) * np.sqrt(self.kbt / mass)
        noise_scale = np.sqrt(2 * gamma * self.kbt * dt / mass)
        samples = []
        burn_in_steps = int(n_steps * burn_in_ratio)

        with self.freeze_params(self.model):
            for step in tqdm(range(n_steps), desc="Langevin Simulation"):
                # get score in normalized space at t -> 0
                t_eval = torch.full((n_parallel,), self.eps_time, device=device)
                score_norm = self.model.score(x, t_eval)
                force_norm = -self.kbt * score_norm
                noise = torch.randn_like(v) * noise_scale
                v = v - gamma * v * dt + (force_norm / mass) * dt + noise
                x = x + v * dt
                if step >= burn_in_steps and step % save_every == 0:
                    samples.append(x.cpu().clone())

        samples = torch.cat(samples, dim=0)
        if return_original_space:
            samples_original = self.unnormalize(samples)
            self.save_results(samples_original, f"sample_{n_parallel}_langevin.pth")
            return samples_original
        else:
            self.save_results(samples, f"sample_{n_parallel}_langevin.pth")
            return samples

    @contextmanager
    def freeze_params(self, module):
        """context manager to temporarily freeze model parameters"""
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
        """save results to disk"""
        filepath = os.path.join(self.store_path, file_name)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(results, filepath)
        print(f"Results saved to {filepath}")
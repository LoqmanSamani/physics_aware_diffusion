import torch
import torch.nn as nn
from typing import Optional, Callable, Dict
from contextlib import contextmanager
from tqdm import tqdm
import os


class MDSampler:
    """
    sampling methods for energy-based diffusion models
    supports both iid sampling (denoising) and md simulation
    """
    def __init__(self, energy_net: nn.Module, reverse_vp: nn.Module, score_fn: Callable, dist_energy_net: nn.Module = None,
                 eps_time: float = 1e-5, store_path: str = "./samples", device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.energy_net = energy_net.to(self.device)
        self.reverse_vp = reverse_vp.to(self.device)
        if dist_energy_net is not None:
            self.dist_energy_net = dist_energy_net.to(self.device)
        else:
            self.dist_energy_net = dist_energy_net
        self.score_fn = score_fn # derive score from energy (output of energy-net)
        self.eps_time = eps_time
        self.store_path = store_path

    def sample_iid(self, atom_features: torch.Tensor, edge_index: torch.Tensor,
                   dt: float = 1e-3, num_samples: int = 1, sampling_mode: str = "sde",
                   normalize_output: bool = False, store_trajectory: bool = False) -> Dict:
        """
        independent sampling via reverse-time sde/ode
        this sampler:
            - starts from Gaussian noise at t = 1
            - integrates the reverse-time dynamics down to t = eps ~ 0
            - uses a trained energy model E(x, t) whose gradient gives the score:
                  score(x, t) = ∇_x log p_t(x) = -∇_x E(x, t)
        important:
            - we cannot use torch.no_grad() because xt must have gradients
              to compute the score via autograd
        """
        results = {"x0": None, "trajectory": []}
        self.energy_net.eval()
        self.reverse_vp.eval()
        if self.dist_energy_net is not None:
            self.dist_energy_net.eval()
        # number of atom per molecules
        num_atoms = atom_features.shape[0]
        # start from pure noise
        xt = torch.randn(num_samples, num_atoms, 3, device=self.device) # (num_samples, num_atoms, 3)
        batch_idx = torch.repeat_interleave(
            torch.arange(num_samples, device=self.device), repeats=num_atoms
        ) # maps each atom to its molecule index
        # prepare batched inputs
        af_batch = atom_features.unsqueeze(0).expand(num_samples, -1, -1) # (num_samples, num_atoms, feat_dim)
        af_flat = af_batch.reshape(-1, atom_features.shape[-1]) # (num_samples * num_atoms, feat_dim)
        # batch edge indices
        edge_index_batch = []
        for i in range(num_samples):
            edge_index_batch.append(edge_index + i * num_atoms)
        edge_index_batched = torch.cat(edge_index_batch, dim=1) # (2, num_edges * num_samples)
        if store_trajectory:
            results["trajectory"].append(xt.clone())
        num_steps = int((1.0 - self.eps_time) / dt)
        assert num_steps > 0, "num_steps must be positive"
        t_schedule = torch.linspace(1.0, self.eps_time, num_steps + 1)
        dt = torch.tensor(dt, device=xt.device, dtype=xt.dtype)
        iterator = tqdm(range(num_steps), desc="Sampling")
        # freeze energy network parameters
        # we still allow gradients w.r.t. xt
        # but prevent autograd from tracking parameter gradients
        with self.freeze_params(self.energy_net):
            for step in iterator:
                t_current = float(t_schedule[step])
                # per-molecule time tensor
                t_mol = torch.full((num_samples,), t_current, dtype=torch.float32, device=self.device)
                # per-atom time
                t_atom = t_mol[batch_idx] #  (num_samples * num_atoms,)
                xt_flat = xt.reshape(-1, 3) # (num_samples * num_atoms, 3)
                xt_flat.requires_grad_(True)
                # compute energy
                if self.dist_energy_net is not None:
                    logp = self.dist_energy_net(xt_flat, af_flat, edge_index_batched, t_atom, batch_idx)
                else:
                    logp = self.energy_net(xt_flat, af_flat, edge_index_batched, t_atom, batch_idx)
                score = self.score_fn(logp, xt_flat) # score = ∇_x log p(x|t)
                assert score.shape == xt_flat.shape, "score shape mismatch"
                # take reverse sde or ode step
                if sampling_mode == "sde":
                    if step == 0:
                        xt_flat = self.reverse_vp(xt_flat, score, t_atom, dt, last_step = True)
                    else:
                        xt_flat = self.reverse_vp(xt_flat, score, t_atom, dt)
                else:
                    xt_flat = self.reverse_vp.probability_flow_ode(xt_flat, score, t_atom, dt)
                xt = xt_flat.view(num_samples, num_atoms, 3)
                xt = xt.detach()
                if store_trajectory:
                    results['trajectory'].append(xt.clone())
                #if step % 100 == 0:
                #    print("t =", t_current)
                #    print("score norm:", score.norm(dim=1).mean().item())

        if normalize_output:
            results['x0'] = torch.clamp(xt, min=-1.0, max=1.0)
            if store_trajectory:
                results['trajectory'] = torch.stack(results['trajectory'], dim=0)
                results['trajectory'] = torch.clamp(results['trajectory'], min=-1.0, max=1.0)
        else:
            results['x0'] = xt
            if store_trajectory:
                results['trajectory'] = torch.stack(results['trajectory'], dim=0)
        self.save_results(results, f"sample_{num_samples}_iid.pth")
        return results

    def get_forces(self, x: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor,
                   batch_idx: Optional[torch.Tensor] = None,* ,t_eval: float = 1e-5, kb_t: float = 1.0
    ) -> torch.Tensor:
        """
        extract forces from the score at t ≈ 0
            - s_θ(x, t=0) = ∇_x log p(x) = -∇_x U(x) / k_B T
            - so forces F = -∇_x U(x) = k_B T * s_θ(x, 0)
        arguments:
            x: (num_atoms, 3) or (batch_size*num_atoms, 3) coordinates
            atom_features: atom features
            edge_index: connectivity
            batch_idx: batch assignment (if batched)
            t_eval: small timestep to evaluate (here we use the same t as
                    used in time sampling (epsilon = 1e-5 by default))
            kb_t: boltzmann constant times temperature here we assume kB_T = 1
        returns:
            forces with same shape as x
        """
        self.energy_net.eval()
        with self.freeze_params(self.energy_net):
            # continuous diffusion time at t ≈ 0
            t_atom = torch.full((x.shape[0], ), t_eval, device=self.device) # (number of atoms, )
            x = x.detach()
            x.requires_grad_(True)
            # computes energy and score
            logp = self.energy_net(x, atom_features, edge_index, t_atom, batch_idx)
            score = self.score_fn(logp, x)
            forces = (score * kb_t).detach()
            #if torch.isnan(forces).any():
            #    raise RuntimeError("NaNs detected in forces")
        return forces

    def simulate_langevin(self, x0: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor,
                          num_steps: int, dt: float = 2.0, temp: float = 300.0, friction: float = 1.0,
                          mass: float = 1.0,  kb: float = 1.380649e-23, save_frequency: int = 100, t_eval = 1e-5) -> Dict:
        """
        langevin dynamics simulation using learned forces
        dx = v dt
        M dv = -∇U(x) dt - γM v dt + √(2γk_B T) dw
        arguments:
            x0: (num_atoms, 3) initial coordinates
            atom_features: atom features
            edge_index: connectivity
            num_steps: number of simulation steps
            dt: timestep (must be consistent with friction units)
            temp: temperature in Kelvin
            friction: friction coefficient
            mass: particle mass
            kb: boltzmann constant (J/K)
            save_frequency: save every N steps
        returns:
            trajectory: (num_saved, num_atoms, 3) coordinates
            velocities: (num_saved, num_atoms, 3) velocities
        """
        self.energy_net.eval()
        kb_t = kb * temp
        assert kb_t > 0, "temperature must be positive"
        x = x0.clone()
        kb_t = torch.tensor(kb_t, device=x.device)
        mass = torch.tensor(mass, device=x.device)
        v = torch.randn_like(x) * torch.sqrt((kb_t / mass).detach().clone()) # maxwell-boltzmann
        alpha = torch.exp(torch.tensor(-friction * dt, device=x.device))
        sigma = torch.sqrt(kb_t * (1 - alpha ** 2) / mass)
        results = {'trajectory': [x.clone()], 'velocity_trajectory': [v.clone()]}
        steps = tqdm(range(num_steps), desc="MD-Trajectory")
        for step in steps:
            with self.freeze_params(self.energy_net):
                # get forces from the trained energy model
                forces = self.get_forces(x, atom_features, edge_index, kb_t = float(kb_t), t_eval = t_eval)
                # langevin integrator
                #-----------------------------------------------------------------
                # update velocity
                v = (alpha * v + (1 - alpha) / (friction * mass) * forces + sigma * torch.randn_like(v))
                # update position
                x = x + dt * v
                #-----------------------------------------------------------------
                if (step + 1) % save_frequency == 0:
                    results['trajectory'].append(x.clone())
                    results['velocity_trajectory'].append(v.clone())
                x = x.detach()
                v = v.detach()
        results['trajectory'] = torch.stack(results['trajectory'], dim=0)
        results['velocity_trajectory'] = torch.stack(results['velocity_trajectory'], dim=0)
        self.save_results(results, f"sample_{num_steps}_steps_langevin.pth")
        return results

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
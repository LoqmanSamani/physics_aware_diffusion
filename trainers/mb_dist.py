import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler
from typing import Callable, Optional, Dict
from torch.nn.utils import clip_grad_norm_
from contextlib import contextmanager
from tqdm import tqdm
import os



class MBDistillationTrainer(nn.Module):
    """knowledge distillation trainer for muller-brown experiment
     with fokker-planck regularization applied through gating"""
    def __init__(
            self,
            t_net: nn.Module,
            s_net: nn.Module,
            fwd: nn.Module,
            data_loader,
            optim: torch.optim.Optimizer,
            fp_gate: Callable,
            fp_loss: Callable, # fokker-planck loss function
            dist_loss: Callable, # distillation loss function
            score_fn: Callable, # computes score from energy
            fp_res: Callable, # computes weak fokker-planck residuals
            epochs: int,
            device: str = 'cuda',
            grad_acc: int = 1,
            checkpoint: int = 5,
            log_freq: int = 1,
            store_path: str = "./mb_dist_train",
            warmup_steps: int = 0,
            fp_alpha: float = 5e-4,
            gate_params: Optional[Dict] = None,
            traj_freq: int = 10,
            eps_time: float = 1e-5,
            compute_trajectory_loss: bool = False,
            *args
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.t_net = t_net.to(self.device)
        self.s_net = s_net.to(self.device)
        self.fwd = fwd.to(self.device)
        self.data_loader = data_loader
        self.optim = optim
        self.fp_gate = fp_gate
        self.fp_loss = fp_loss
        self.dist_loss = dist_loss
        self.score_fn = score_fn
        self.fp_res = fp_res
        self.epochs = epochs
        self.grad_acc = grad_acc
        self.checkpoint = checkpoint
        self.log_freq = log_freq
        self.store_path = store_path
        self.warmup_steps = warmup_steps
        self.fp_alpha = fp_alpha
        self.traj_freq = traj_freq
        self.eps_time = eps_time
        self.compute_trajectory_loss = compute_trajectory_loss
        self.global_step = 0
        self.base_lr = optim.param_groups[0]['lr']
        self.best_loss = float('inf')
        self.scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-6)
        self.gate_params = gate_params or {
            "k": 1.2, "snr_min": 0.5, "t_scale": 0.1,
            "sharpness": 5.0, "eps": 0.1
        }
        self.losses = {
            'total_losses': [], 'fp_losses': [],
            'force_losses': [], 'traj_losses': []
        }

    def forward(self):
        self.t_net.eval()
        self.s_net.train()
        for epoch in range(self.epochs):
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            mean_losses = self.train_batch(pbar, epoch)
            self.losses['total_losses'].append(mean_losses[0])
            self.losses['force_losses'].append(mean_losses[1])
            self.losses['fp_losses'].append(mean_losses[2])
            self.losses['traj_losses'].append(mean_losses[3])
            if (epoch + 1) % self.log_freq == 0:
                lr = self.optim.param_groups[0]['lr']
                print(f"Epoch: {epoch + 1}/{self.epochs} | LR: {lr:.2e} | "
                      f"Total Loss: {mean_losses[0]:.4f} | Force Loss: {mean_losses[1]:.4f} | "
                      f"FP Loss: {mean_losses[2]:.4f} | Traj Loss: {mean_losses[3]:.4f}")
            if (epoch + 1) % self.checkpoint == 0:
                self.save_checkpoint(epoch + 1, mean_losses[0])
            if mean_losses[0] < self.best_loss:
                self.best_loss = mean_losses[0]
                self.save_checkpoint(epoch + 1, mean_losses[0], is_best=True)
        return self.losses

    def train_batch(self, pbar, epoch):
        """train one epoch of student model"""
        total_losses = []
        force_losses = []
        fp_losses = []
        traj_losses = []
        for step, batch in enumerate(pbar):
            x0 = batch.to(self.device)
            indices = [step]
            comp_traj = False
            if epoch % self.traj_freq == 0 and self.compute_trajectory_loss:
                traj_data = self.data_loader.dataset_ref.get_trajectories_for_batch(indices)
                t_traj = traj_data["trajectories"]
                traj_params_list = traj_data["params"]
                comp_traj = True
            else:
                t_traj = None
                traj_params_list = None
            step_losses = self.train_step(
                x0,
                t_traj=t_traj,
                traj_params_list=traj_params_list,
                comp_traj_loss=comp_traj
            )
            step_losses[0].backward()
            if (step + 1) % self.grad_acc == 0:
                clip_grad_norm_(self.s_net.parameters(), max_norm=1.0)
                self.optim.step()
                self.optim.zero_grad()
                self.global_step += 1
                self.update_learning_rate()
            pbar.set_postfix({
                'total': f'{step_losses[0].item() * self.grad_acc:.4f}',
                'force': f'{step_losses[1].item() * self.grad_acc:.4f}',
                'fp': f'{step_losses[2].item() * self.grad_acc:.4f}',
                'traj': f'{step_losses[3].item() * self.grad_acc:.4f}'
            })
            total_losses.append(step_losses[0].item() * self.grad_acc)
            force_losses.append(step_losses[1].item() * self.grad_acc)
            fp_losses.append(step_losses[2].item() * self.grad_acc)
            traj_losses.append(step_losses[3].item() * self.grad_acc)
        if self.global_step >= self.warmup_steps:
            self.scheduler.step()
        return (sum(total_losses) / len(total_losses),
                sum(force_losses) / len(force_losses),
                sum(fp_losses) / len(fp_losses),
                sum(traj_losses) / len(traj_losses))

    def train_step(self, x0: torch.Tensor, t_traj: list, traj_params_list: list, comp_traj_loss):
        # force loss
        # ------------------------------------------------
        noise = torch.randn_like(x0)
        x0 = x0.clone()
        noise = noise.clone()
        t = self.sample_time(x0.shape[0], self.eps_time)
        xt, _ = self.fwd(x0, noise, t)
        xt = xt.detach()
        with self.freeze_params(self.t_net):
            xt_t = xt.clone().requires_grad_(True)
            true_logp = self.t_net(xt_t, t)
            true_score = self.score_fn(true_logp, xt_t)
        xt_s = xt.clone().requires_grad_(True)
        pred_logp = self.s_net(xt_s, t)
        pred_score = self.score_fn(pred_logp, xt_s)
        force_loss = self.dist_loss(pred_score, true_score)
        # fp regularization
        # ---------------------------------------------------
        fp_loss = torch.tensor(0.0, device=self.device)
        fp_mask, activate_fp = self.fp_gate(
            x0, t, self.fwd.vs,
            k=self.gate_params['k'], snr_min=self.gate_params['snr_min'],
            t_scale=self.gate_params['t_scale'], sharpness=self.gate_params['sharpness'],
            eps=self.gate_params['eps']
        )
        if activate_fp:
            xt_active, t_active = self.subset_graph_data(xt_s, t, fp_mask)
            seed1 = self.global_step * 1000
            seed2 = self.global_step * 1001
            r1 = self.fp_res(self.s_net, xt_active, t_active, self.fwd.vs, seed1)
            r2 = self.fp_res(self.s_net, xt_active, t_active, self.fwd.vs, seed2)
            fp_loss = self.fp_loss(r1, r2, alpha=self.fp_alpha)
        # trajectory loss
        # -------------------------------------------------------------------
        traj_loss = torch.tensor(0.0, device=self.device)
        if comp_traj_loss and t_traj is not None:
            if self.use_cached_trajectories:
                # process each sample's trajectory separately
                traj_losses = []
                for idx in range(x0.shape[0]):
                    point = x0[idx].unsqueeze(0)
                    true_traj = t_traj[idx].to(self.device)
                    params = traj_params_list[idx]
                    pred_traj = self.simulate_langevin(
                        net = self.s_enet,
                        x0 = point,
                        num_steps = params.num_steps,
                        dt = params.dt,
                        temp = params.temp,
                        friction = params.friction,
                        mass = params.mass,
                        kb = params.kb,
                        t_eval = params.t_eval
                    )
                    mol_loss = ((pred_traj - true_traj) ** 2).mean()
                    traj_losses.append(mol_loss)
                traj_loss = torch.stack(traj_losses).mean()
        total_loss = (force_loss + fp_loss + traj_loss) / self.grad_acc
        force_loss = force_loss / self.grad_acc
        fp_loss = fp_loss / self.grad_acc
        traj_loss = traj_loss / self.grad_acc
        return total_loss, force_loss, fp_loss, traj_loss

    def simulate_langevin(self, net, x0: torch.Tensor, num_steps: int, dt: float, temp: float,
                          friction: float, mass: float, kb: float, t_eval: float):
        """langevin dynamics simulation using learned forces"""
        traj = []
        kb_t = kb * temp
        assert kb_t > 0, "temperature must be positive"
        x = x0.clone()
        kb_t = torch.tensor(kb_t, device=x.device)
        mass = torch.tensor(mass, device=x.device)
        v = torch.randn_like(x) * torch.sqrt((kb_t / mass).detach().clone())  # maxwell-boltzmann
        alpha = torch.exp(torch.tensor(-friction * dt, device=x.device))
        sigma = torch.sqrt(kb_t * (1 - alpha ** 2) / mass)
        for _ in range(num_steps):
            forces = self.get_forces(net, x, float(kb_t), t_eval)
            v = (alpha * v + (1 - alpha) / (friction * mass) * forces + sigma * torch.randn_like(v))
            x = x + dt * v
            traj.append(x.clone())
        traj_ = torch.stack(traj)
        return traj_

    def get_forces(self, net, x: torch.Tensor, t_eval, kb_t) -> torch.Tensor:
        """extract forces from the score at t ≈ 0"""
        t_atom = torch.full((x.shape[0], ), t_eval, device=self.device)
        x = x.detach()
        x.requires_grad_(True)
        logp = net(x, t_atom)
        score = self.score_fn(logp, x)
        forces = (score * kb_t).detach()
        return forces

    def subset_graph_data(self, xt, t, active_mask):
        xt_active = xt[active_mask]
        t_active = t[active_mask]
        return xt_active, t_active

    def update_learning_rate(self):
        """linear warmup"""
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
            lr = self.base_lr * (self.global_step + 1) / self.warmup_steps
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> None:
        checkpoint = {
            'current_epoch': epoch,
            'student_energy_net_state': self.s_net.state_dict(),
            'teacher_energy_net_state': self.t_net.state_dict(),
            'optimizer_state': self.optim.state_dict(),
            'current_loss': loss,
            'losses': self.losses,
            'variance_scheduler': self.fwd.vs.state_dict(),
            'max_epochs': self.epochs,
            'base_larning_rate': self.base_lr,
            'global_step': self.global_step
        }
        filename = "dist_best.pth" if is_best else f"dist_epoch_{epoch}.pth"
        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)
        if is_best:
            print(f"Best model saved at epoch {epoch} with loss {loss:.4f}")
        else:
            print(f"Checkpoint saved at epoch {epoch} with loss {loss: .4f}")

    def sample_time(self, batch_size: int, eps: float = 1e-5) -> torch.Tensor:
        return eps + (1 - eps) * torch.rand(batch_size, device=self.device)

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
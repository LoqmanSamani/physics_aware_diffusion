import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os
from typing import Callable



class MolEnergyTrainer(nn.Module):
    """energy trainer for vp-sde on a molecular dataset"""
    def __init__(self, energy_net: nn.Module, forward_vp: nn.Module, data_loader, noise_fn: Callable,
                 optimizer: torch.optim.Optimizer, loss_fn: Callable, epochs: int, device: str,
                 grad_acc: int, checkpoint: int, log_freq: int, store_path: str,
                 warmup_steps: int = 0, rotation_augmentation: bool = False,
                 lambda_t = lambda t: 1.0, mix_precision: bool = True, *args) -> None:
        super().__init__()
        self.energy_net = energy_net.to(device)
        self.forward_vp = forward_vp.to(device)
        self.noise_fn = noise_fn
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device
        self.grad_acc = grad_acc
        self.checkpoint = checkpoint
        self.log_freq = log_freq
        self.store_path = store_path
        self.warmup_steps = warmup_steps
        self.rotation_augmentation = rotation_augmentation
        self.lambda_t = lambda_t
        self.mix_precision = mix_precision
        self.global_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        self.best_loss = float('inf')
        self.use_amp = mix_precision and (device == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6
        )

    def forward(self):
        self.energy_net.train()
        losses = []
        for epoch in range(self.epochs):
            epoch_losses = []
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for step, batch in enumerate(pbar):
                x0 = batch.coords.to(self.device)
                atom_features = batch.atom_features.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                batch_idx = batch.batch.to(self.device)
                noise = torch.randn_like(x0)
                num_molecules = batch.num_graphs
                if self.rotation_augmentation:
                    num_molecules = batch.num_graphs
                    R = self.random_rotation_matrix(num_molecules, self.device)
                    # rotate positions and noise consistently per molecule
                    for mol_idx in range(num_molecules):
                        mask = batch_idx == mol_idx
                        x0[mask] = x0[mask] @ R[mol_idx].T
                        noise[mask] = noise[mask] @ R[mol_idx].T

                t_mol = self.sample_time(num_molecules)  # (num_molecules,)
                t_atom = t_mol[batch_idx]  # (num_atoms,)
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        xt, true_score = self.forward_vp(x0, noise, t_atom)
                        std_per_atom = self.forward_vp.vs.std(t_atom)
                        while std_per_atom.dim() < noise.dim():
                            std_per_atom = std_per_atom.unsqueeze(-1)
                        xt.requires_grad_(True)
                        logp = self.energy_net(xt, atom_features, edge_index, t_atom, batch_idx)
                        pred_noise = self.noise_fn(logp, xt, t_atom, self.forward_vp.vs)
                        loss_ = self.loss_fn(pred_noise, noise, batch_idx) / self.grad_acc
                        weight = self.lambda_t(t_atom.mean().item())
                        loss = weight * loss_
                    self.scaler.scale(loss).backward()
                else:
                    xt, true_score = self.forward_vp(x0, noise, t_atom)
                    std_per_atom = self.forward_vp.vs.std(t_atom)
                    while std_per_atom.dim() < noise.dim():
                        std_per_atom = std_per_atom.unsqueeze(-1)
                    xt.requires_grad_(True)
                    logp = self.energy_net(xt, atom_features, edge_index, t_atom, batch_idx)
                    pred_score = self.noise_fn(logp, xt, t_atom, self.forward_vp.vs)
                    loss_ = self.loss_fn(pred_score, true_score, batch_idx) / self.grad_acc
                    weight = self.lambda_t(t_atom.mean().item())
                    loss = weight * loss_
                    loss.backward()

                if (step + 1) % self.grad_acc == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.energy_net.parameters(), max_norm=1.0)
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    self._update_learning_rate()
                epoch_losses.append(loss.item() * self.grad_acc)
                pbar.set_postfix({'loss': f'{loss.item() * self.grad_acc:.4f}'})
            if len(self.data_loader) % self.grad_acc != 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.energy_net.parameters(), max_norm=1.0)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                self._update_learning_rate()

            mean_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(mean_loss)
            pbar.set_postfix({'loss': f'{mean_loss:.4f} (mean)'})
            pbar.close()
            if self.global_step >= self.warmup_steps:
                self.scheduler.step()
            if (epoch + 1) % self.log_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch + 1}/{self.epochs} | LR: {lr:.2e} | Train Loss: {mean_loss:.4f}")
            if (epoch + 1) % self.checkpoint == 0:
                self._save_checkpoint(epoch + 1, mean_loss, losses)
            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                self._save_checkpoint(epoch + 1, mean_loss, losses, is_best=True)
        return losses

    def _update_learning_rate(self):
        """linear warmup"""
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
            lr = self.base_lr * (self.global_step + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def random_rotation_matrix(self, batch_size: int, device: str) -> torch.Tensor:
        """generate random 3D rotation matrices using QR decomposition"""
        M = torch.randn(batch_size, 3, 3, device=device)
        Q, R = torch.linalg.qr(M)
        det = torch.det(Q)
        Q = Q * det.view(-1, 1, 1).sign()
        return Q
    def sample_time(self, batch_size: int, eps: float = 1e-5, use_epsilon: bool = False) -> torch.Tensor:
        """optionally sample time from [eps, 1-eps] instead (0, 1] which enhance stability"""
        t = torch.rand(batch_size, device=self.device)
        if use_epsilon:
            t = eps + (1.0 - 2 * eps) * t
        return t

    def _save_checkpoint(self, epoch: int, loss: float, train_losses: list, is_best: bool = False) -> None:
        checkpoint = {
            'epoch': epoch,
            'energy_net_state': self.energy_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss': loss,
            'train_losses': train_losses,
            'scheduler': self.forward_vp.vs.state_dict(),
            'epochs': self.epochs
        }
        filename = "vpe_best.pth" if is_best else f"vpe_epoch_{epoch}.pth"
        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)
        if is_best:
            print(f"Best model saved at epoch {epoch} with loss {loss:.4f}")
        else:
            print(f"Checkpoint saved at epoch {epoch}")

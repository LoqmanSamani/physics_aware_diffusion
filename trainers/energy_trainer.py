import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os
from typing import Callable



class MolEnergyTrainer(nn.Module):
    """energy trainer for VP-SDE on a molecular dataset"""
    def __init__(self, energy_net: nn.Module, forward_vp: nn.Module, data_loader, score_fn: Callable,
                 optimizer: torch.optim.Optimizer, loss_fn: Callable, epochs: int, device: str,
                 grad_acc: int, checkpoint: int, log_freq: int, store_path: str,
                 warmup_steps: int = 0, rotation_augmentation: bool = False, lambda_t = lambda t: 1.0,*args) -> None:
        super().__init__()
        self.energy_net = energy_net.to(device)
        self.forward_vp = forward_vp.to(device)
        self.score_fn = score_fn
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
        self.global_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        self.best_loss = float('inf')
        self.use_amp = (device == 'cuda')
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
                x = batch.pos.to(self.device)
                atom_features = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                batch_idx = batch.batch.to(self.device)
                noise = torch.randn_like(x)
                num_molecules = batch.num_graphs
                if self.rotation_augmentation:
                    num_molecules = batch.num_graphs
                    R = self.random_rotation_matrix(num_molecules, self.device)
                    # rotate positions and noise consistently per molecule
                    for mol_idx in range(num_molecules):
                        mask = batch_idx == mol_idx
                        x[mask] = x[mask] @ R[mol_idx].T
                        noise[mask] = noise[mask] @ R[mol_idx].T

                time_ = torch.randint(1, self.forward_vp.vs.num_steps, (num_molecules,), device=self.device)
                time_per_atom = time_[batch_idx]
                t_norm_per_atom = time_per_atom.float() / (self.forward_vp.vs.num_steps - 1)
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        noisy_x, true_score = self.forward_vp(x, noise, time_per_atom)
                        std_per_atom = self.forward_vp.vs.get_std(time_per_atom)
                        variance = self.forward_vp.vs.get_variance(time_per_atom)
                        while std_per_atom.dim() < noise.dim():
                            std_per_atom = std_per_atom.unsqueeze(-1)
                        noisy_x.requires_grad_(True)
                        logp = self.energy_net(noisy_x, atom_features, edge_index, t_norm_per_atom, batch_idx)
                        pred_noise = self.score_fn(logp, noisy_x)
                        loss_ = self.loss_fn(pred_noise=pred_noise, target_noise=noise, variance=variance, batch_idx=batch_idx) / self.grad_acc
                        weight = self.lambda_t(t_norm_per_atom.mean().item())
                        loss = weight * loss_
                    self.scaler.scale(loss).backward()
                else:
                    noisy_x, true_score = self.forward_vp(x, noise, time_per_atom)
                    std_per_atom = self.forward_vp.vs.get_std(time_per_atom)
                    while std_per_atom.dim() < noise.dim():
                        std_per_atom = std_per_atom.unsqueeze(-1)
                    noisy_x.requires_grad_(True)
                    logp = self.energy_net(noisy_x, atom_features, edge_index, t_norm_per_atom, batch_idx)
                    pred_noise = self.score_fn(logp, noisy_x)
                    loss_ = self.loss_fn(pred_noise, noise, variance, batch_idx) / self.grad_acc
                    weight = self.lambda_t(t_norm_per_atom.mean().item())
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

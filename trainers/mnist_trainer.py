import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os
from typing import Callable


class MNISTTrainer(nn.Module):
    """trainer for the variance preserving diffusion model trained on mnist dataset"""
    def __init__(self, score_net: torch.nn.Module, forward_vp: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
                 loss_fn: Callable, epochs: int, grad_acc: int, checkpoint: int,
                 log_freq: int, store_path: str, device: str, warmup_steps: int = 0, *args) -> None:
        super().__init__()
        self.score_net = score_net
        self.forward_vp = forward_vp
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.grad_acc = grad_acc
        self.checkpoint = checkpoint
        self.log_freq = log_freq
        self.store_path = store_path
        self.device = device
        self.warmup_steps = warmup_steps
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

    def _update_learning_rate(self):
        """linear warmup"""
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
            lr = self.base_lr * (self.global_step + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def forward(self) -> list:
        self.score_net.train()
        train_losses = []
        for epoch in range(self.epochs):
            train_losses_epoch = []
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for step, (x0, _) in enumerate(pbar):
                x0 = x0.to(self.device)
                noise = torch.randn_like(x0)
                time = self.sample_time(x0.shape[0])
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        xt, true_score = self.forward_vp(x0, noise, time)
                    with torch.amp.autocast('cuda', enabled=False):
                        pred_noise = self.score_net(xt, time)
                    with torch.amp.autocast('cuda'):
                        loss = self.loss_fn(pred_noise, noise) / self.grad_acc
                    self.scaler.scale(loss).backward()
                else:
                    xt, pred_score = self.forward_vp(x0, noise, time)
                    pred_noise = self.score_net(xt, time)
                    loss = self.loss_fn(pred_noise, noise) / self.grad_acc
                    #print(loss.shape)
                    loss.backward()
                if (step + 1) % self.grad_acc == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.score_net.parameters(), max_norm=1.0)
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    self._update_learning_rate()

                train_losses_epoch.append(loss.item() * self.grad_acc)
                pbar.set_postfix({'loss': loss.item() * self.grad_acc})
            if len(self.data_loader) % self.grad_acc != 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.score_net.parameters(), max_norm=1.0)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                self._update_learning_rate()

            mean_train_loss = sum(train_losses_epoch) / len(train_losses_epoch)
            train_losses.append(mean_train_loss)
            pbar.set_postfix({'loss': f'{mean_train_loss:.4f} (mean)'})
            pbar.close()

            if self.global_step >= self.warmup_steps:
                self.scheduler.step()
            if (epoch + 1) % self.log_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"\nEpoch: {epoch + 1}/{self.epochs} | LR: {lr:.2e} | Train Loss: {mean_train_loss:.4f}")
            if (epoch + 1) % self.checkpoint == 0:
                self._save_checkpoint(epoch + 1, mean_train_loss, train_losses)
            if mean_train_loss < self.best_loss:
                self.best_loss = mean_train_loss
                self._save_checkpoint(epoch + 1, mean_train_loss, train_losses, is_best=True)
        return train_losses

    def sample_time(self, batch_size: int, eps: float = 1e-3) -> torch.Tensor:
        """oversample middle timesteps where score is hardest"""
        # beta distribution concentrates sampling around t=0.5
        #t = torch.distributions.Beta(2.0, 2.0).sample((batch_size,)).to(self.device)
        #t = eps + (1.0 - 2 * eps) * t
        return torch.rand(batch_size, device=self.device)

    def _save_checkpoint(self, epoch: int, loss: float, train_losses: list, is_best: bool = False) -> None:
        checkpoint = {
            'epoch': epoch,
            'score_net_state': self.score_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss': loss,
            'train_losses': train_losses,
            'scheduler': self.forward_vp.vs.state_dict(),
            'epochs': self.epochs,
            'global_step': self.global_step
        }
        filename = "vp_best.pth" if is_best else f"vp_epoch_{epoch}.pth"
        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)
        if is_best:
            print(f"Best model saved at epoch {epoch} with loss {loss:.4f}")
        else:
            print(f"Checkpoint saved at epoch {epoch}")
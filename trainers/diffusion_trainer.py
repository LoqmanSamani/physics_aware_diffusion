import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os
from typing import Callable



class DiffusionTrainer(nn.Module):
    """trainer for the variance preserving diffusion model"""
    def __init__(self, score_net: torch.nn.Module, forward_vp: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
                 loss_fn: Callable, epochs: int, grad_acc: int, checkpoint: int,
                 log_freq: int, store_path: str, device: str, *args) -> None:
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
        self.best_loss = float('inf')
        self.use_amp = (device == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6
        )

    def forward(self) -> list:
        self.score_net.train()
        train_losses = []
        for epoch in range(self.epochs):
            train_losses_epoch = []
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for step, (x, _) in enumerate(pbar):
                x = x.to(self.device)
                noise = torch.randn_like(x)
                time_ = torch.randint(0, self.forward_vp.vs.num_steps, (x.shape[0],), device=self.device)
                t_norm = time_.float() / (self.forward_vp.vs.num_steps - 1)
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        noisy_x = self.forward_vp(x, noise, time_)
                    with torch.amp.autocast('cuda', enabled=False):
                        score = self.score_net(noisy_x, t_norm)
                    with torch.amp.autocast('cuda'):
                        loss = self.loss_fn(score, noise, time_, self.forward_vp.vs) / self.grad_acc
                    self.scaler.scale(loss).backward()
                else:
                    noisy_x = self.forward_vp(x, noise, time_)
                    score = self.score_net(noisy_x, t_norm)
                    loss = self.loss_fn(score, noise, time_, self.forward_vp.vs) / self.grad_acc
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
            mean_train_loss = sum(train_losses_epoch) / len(train_losses_epoch)
            train_losses.append(mean_train_loss)
            self.scheduler.step()
            #if (epoch + 1) % self.log_freq == 0:
                #lr = self.optimizer.param_groups[0]['lr']
                #print(f"\nEpoch: {epoch + 1}/{self.epochs} | LR: {lr:.2e} | Train Loss: {mean_train_loss:.4f}")
            if (epoch + 1) % self.checkpoint == 0:
                self._save_checkpoint(epoch + 1, mean_train_loss, train_losses)
            if mean_train_loss < self.best_loss:
                self.best_loss = mean_train_loss
                self._save_checkpoint(epoch + 1, mean_train_loss, train_losses, is_best=True)
        return train_losses

    def _save_checkpoint(self, epoch: int, loss: float, train_losses: list, is_best: bool = False) -> None:
        checkpoint = {
            'epoch': epoch,
            'score_net_state': self.score_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss': loss,
            'train_losses': train_losses,
            'scheduler': self.forward_vp.vs.state_dict(),
            'epochs': self.epochs
        }
        filename = "vp_best.pth" if is_best else f"vp_epoch_{epoch}.pth"
        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)
        if is_best:
            print(f"Best model saved at epoch {epoch} with loss {loss:.4f}")
        else:
            print(f"Checkpoint saved at epoch {epoch}")
